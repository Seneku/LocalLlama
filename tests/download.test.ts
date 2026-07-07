import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { createServer, type Server } from "node:http";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { DownloadManager } from "../server/downloadManager";
import { resetSettingsCache } from "../server/settings";

let tempDir = "";
let modelsDir = "";
let server: Server | null = null;
let baseUrl = "";
const payload = Buffer.alloc(256 * 1024, 7); // 256 KiB of bytes

const originalDataDir = process.env.LOCALLLAMA_DATA_DIR;
const originalModelsDir = process.env.LOCALLLAMA_MODELS_DIR;

beforeEach(async () => {
  tempDir = mkdtempSync(path.join(tmpdir(), "localllama-dl-"));
  modelsDir = path.join(tempDir, "models");
  process.env.LOCALLLAMA_DATA_DIR = tempDir;
  process.env.LOCALLLAMA_MODELS_DIR = modelsDir;
  resetSettingsCache();

  server = createServer((request, response) => {
    if (request.url?.includes("slow")) {
      // Dribble bytes so a cancel can land mid-stream.
      response.writeHead(200, { "content-length": String(payload.length) });
      let sent = 0;
      const timer = setInterval(() => {
        if (sent >= payload.length) {
          clearInterval(timer);
          response.end();
          return;
        }
        response.write(payload.subarray(sent, sent + 4096));
        sent += 4096;
      }, 20);
      request.on("close", () => clearInterval(timer));
      return;
    }
    response.writeHead(200, { "content-length": String(payload.length) });
    response.end(payload);
  });
  await new Promise<void>((resolve) => server!.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("test server did not bind");
  }
  baseUrl = `http://127.0.0.1:${address.port}`;
});

afterEach(async () => {
  if (server) {
    await new Promise<void>((resolve) => server!.close(() => resolve()));
    server = null;
  }
  if (originalDataDir === undefined) {
    delete process.env.LOCALLLAMA_DATA_DIR;
  } else {
    process.env.LOCALLLAMA_DATA_DIR = originalDataDir;
  }
  if (originalModelsDir === undefined) {
    delete process.env.LOCALLLAMA_MODELS_DIR;
  } else {
    process.env.LOCALLLAMA_MODELS_DIR = originalModelsDir;
  }
  resetSettingsCache();
  rmSync(tempDir, { recursive: true, force: true });
});

async function waitFor(manager: DownloadManager, states: string[], timeoutMs = 5000): Promise<string> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const state = manager.getStatus().state;
    if (states.includes(state)) {
      return state;
    }
    if (Date.now() > deadline) {
      throw new Error(`timed out waiting for ${states.join("/")}, still ${state}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
}

describe("DownloadManager", () => {
  test("rejects unsafe filenames before any fetch", async () => {
    const manager = new DownloadManager({ resolveUrl: () => baseUrl });
    await expect(manager.start({ id: "x/y", filename: "../escape.gguf" })).rejects.toThrow();
    await expect(manager.start({ id: "x/y", filename: "sub/dir.gguf" })).rejects.toThrow();
    await expect(manager.start({ id: "x/y", filename: "notamodel.txt" })).rejects.toThrow();
    expect(manager.getStatus().state).toBe("idle");
  });

  test("streams to a .part then renames on completion", async () => {
    const manager = new DownloadManager({ resolveUrl: () => baseUrl });
    await manager.start({ id: "org/repo", filename: "model-Q4_K_M.gguf" });
    const final = await waitFor(manager, ["completed", "failed"]);
    expect(final).toBe("completed");

    const dest = path.join(modelsDir, "model-Q4_K_M.gguf");
    expect(existsSync(dest)).toBe(true);
    expect(existsSync(`${dest}.part`)).toBe(false);
    expect(readFileSync(dest).length).toBe(payload.length);

    const status = manager.getStatus();
    expect(status.receivedBytes).toBe(payload.length);
    expect(status.totalBytes).toBe(payload.length);
  });

  test("rejects a second concurrent download", async () => {
    const manager = new DownloadManager({ resolveUrl: () => `${baseUrl}/slow` });
    await manager.start({ id: "org/repo", filename: "a.gguf" });
    await expect(manager.start({ id: "org/repo", filename: "b.gguf" })).rejects.toThrow(/already in progress/i);
    await manager.cancel();
    await waitFor(manager, ["cancelled", "completed"]);
  });

  test("cancel aborts the stream and removes the partial file", async () => {
    const manager = new DownloadManager({ resolveUrl: () => `${baseUrl}/slow` });
    await manager.start({ id: "org/repo", filename: "big.gguf" });
    await waitFor(manager, ["downloading"]);
    await manager.cancel();
    const final = await waitFor(manager, ["cancelled", "completed"]);
    expect(final).toBe("cancelled");
    expect(existsSync(path.join(modelsDir, "big.gguf"))).toBe(false);
    expect(existsSync(path.join(modelsDir, "big.gguf.part"))).toBe(false);
  });

  test("refuses to overwrite an existing model", async () => {
    const manager = new DownloadManager({ resolveUrl: () => baseUrl });
    await manager.start({ id: "org/repo", filename: "dup.gguf" });
    await waitFor(manager, ["completed"]);
    const again = new DownloadManager({ resolveUrl: () => baseUrl });
    await expect(again.start({ id: "org/repo", filename: "dup.gguf" })).rejects.toThrow(/already/i);
  });
});
