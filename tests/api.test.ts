import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { createApp } from "../server/app";
import { createBenchmarkStore } from "../server/benchmarkStore";
import { createProfileStore } from "../server/profileStore";

let tempDir = "";
let server: ReturnType<typeof createApp> | null = null;
let baseUrl = "";

beforeEach(async () => {
  tempDir = mkdtempSync(path.join(tmpdir(), "llama-tuner-test-"));
  server = createApp({
    store: createProfileStore(path.join(tempDir, "profiles.json")),
    benchmarkStore: createBenchmarkStore(path.join(tempDir, "benchmarks.json"))
  });
  await new Promise<void>((resolve) => server!.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("test server did not bind to a TCP port");
  }
  baseUrl = `http://127.0.0.1:${address.port}`;
});

afterEach(async () => {
  if (server) {
    await new Promise<void>((resolve) => server!.close(() => resolve()));
    server = null;
  }
  if (tempDir) {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

describe("API smoke", () => {
  test("loads seeded profiles", async () => {
    const response = await fetch(`${baseUrl}/api/profiles`);
    const profiles = await response.json();

    expect(response.status).toBe(200);
    expect(Array.isArray(profiles)).toBe(true);
    expect(profiles.length).toBeGreaterThanOrEqual(3);
  });

  test("saves a profile and previews its command", async () => {
    const profilesResponse = await fetch(`${baseUrl}/api/profiles`);
    const profiles = await profilesResponse.json();
    const profile = { ...profiles[0], id: "", name: "Smoke Custom", port: 8101 };

    const saveResponse = await fetch(`${baseUrl}/api/profiles`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(profile)
    });
    const saved = await saveResponse.json();
    expect(saveResponse.status).toBe(201);
    expect(saved.id).toBeTruthy();

    const previewResponse = await fetch(`${baseUrl}/api/preview`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ profile: saved })
    });
    const preview = await previewResponse.json();

    expect(previewResponse.status).toBe(200);
    expect(preview.endpoint).toBe("http://127.0.0.1:8101");
    expect(preview.args).toContain("8101");
  });

  test("loads benchmark history and previews a benchmark command", async () => {
    const profilesResponse = await fetch(`${baseUrl}/api/profiles`);
    const profiles = await profilesResponse.json();

    const historyResponse = await fetch(`${baseUrl}/api/benchmarks`);
    const history = await historyResponse.json();
    expect(historyResponse.status).toBe(200);
    expect(history).toEqual([]);

    const previewResponse = await fetch(`${baseUrl}/api/benchmarks/preview`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        profile: profiles[0],
        settings: {
          promptTokens: 128,
          generationTokens: 32,
          repetitions: 1,
          batchSize: 512,
          ubatchSize: 128,
          noWarmup: true,
          flashAttention: "auto"
        }
      })
    });
    const preview = await previewResponse.json();

    expect(previewResponse.status).toBe(200);
    expect(preview.args).toContain("-o");
    expect(preview.args).toContain("json");
    expect(preview.args).toContain("--no-warmup");
  });
});
