import fs from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import path from "node:path";

import { BenchmarkManager, buildBenchmarkCommand } from "./benchmark";
import { createBenchmarkStore, type BenchmarkStore } from "./benchmarkStore";
import { estimateProfileMemory } from "./estimate";
import { buildCommand } from "./llama";
import { createProfileStore, type ProfileStore } from "./profileStore";
import { createRuntimeConfig } from "./paths";
import { RuntimeManager } from "./runtime";
import { normalizeProfile } from "./normalize";
import type { BenchmarkSettings } from "../src/shared/types";

const MAX_BODY_BYTES = 1024 * 1024;

class HttpError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = "HttpError";
  }
}

interface AppDeps {
  store?: ProfileStore;
  runtime?: RuntimeManager;
  benchmarkStore?: BenchmarkStore;
  benchmark?: BenchmarkManager;
}

const MIME_TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon"
};

function sendJson(response: ServerResponse, status: number, payload: unknown): void {
  const body = JSON.stringify(payload, null, 2);
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store"
  });
  response.end(body);
}

function sendError(response: ServerResponse, status: number, message: string): void {
  sendJson(response, status, { error: message });
}

async function readJson<T>(request: IncomingMessage): Promise<T> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of request) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    total += buffer.length;
    if (total > MAX_BODY_BYTES) {
      request.destroy();
      throw new HttpError(413, "request body too large");
    }
    chunks.push(buffer);
  }
  const raw = Buffer.concat(chunks).toString("utf8");
  if (!raw) {
    return {} as T;
  }
  try {
    return JSON.parse(raw) as T;
  } catch {
    throw new HttpError(400, "request body is not valid JSON");
  }
}

function createId(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/gu, "-")
    .replace(/^-|-$/gu, "");
  const suffix = Math.random().toString(36).slice(2, 8);
  return `${slug || "profile"}-${Date.now().toString(36)}-${suffix}`;
}

async function routeApi(
  request: IncomingMessage,
  response: ServerResponse,
  pathname: string,
  store: ProfileStore,
  runtime: RuntimeManager,
  benchmarkStore: BenchmarkStore,
  benchmark: BenchmarkManager
): Promise<void> {
  if (request.method === "GET" && pathname === "/api/config") {
    sendJson(response, 200, createRuntimeConfig(fs.existsSync));
    return;
  }

  if (request.method === "GET" && pathname === "/api/profiles") {
    sendJson(response, 200, await store.load());
    return;
  }

  if (request.method === "POST" && pathname === "/api/profiles") {
    const profile = normalizeProfile(await readJson<unknown>(request));
    const profiles = await store.load();
    const id = profile.id || createId(profile.name || "Profile");
    const next = { ...profile, id };
    profiles.push(next);
    await store.save(profiles);
    sendJson(response, 201, next);
    return;
  }

  const profileMatch = pathname.match(/^\/api\/profiles\/([^/]+)$/u);
  if (profileMatch && request.method === "PUT") {
    const id = decodeURIComponent(profileMatch[1]);
    const profile = normalizeProfile(await readJson<unknown>(request));
    const profiles = await store.load();
    const index = profiles.findIndex((item) => item.id === id);
    if (index === -1) {
      sendError(response, 404, "profile not found");
      return;
    }
    profiles[index] = { ...profile, id };
    await store.save(profiles);
    sendJson(response, 200, profiles[index]);
    return;
  }

  if (profileMatch && request.method === "DELETE") {
    const id = decodeURIComponent(profileMatch[1]);
    const profiles = await store.load();
    const next = profiles.filter((item) => item.id !== id);
    if (next.length === profiles.length) {
      sendError(response, 404, "profile not found");
      return;
    }
    await store.save(next);
    sendJson(response, 200, { ok: true });
    return;
  }

  if (request.method === "POST" && pathname === "/api/preview") {
    const body = await readJson<Record<string, unknown>>(request);
    const raw = body && typeof body === "object" && "profile" in body ? body.profile : body;
    sendJson(response, 200, buildCommand(normalizeProfile(raw)));
    return;
  }

  if (request.method === "POST" && pathname === "/api/estimate") {
    const body = await readJson<Record<string, unknown>>(request);
    const raw = body && typeof body === "object" && "profile" in body ? body.profile : body;
    sendJson(response, 200, await estimateProfileMemory(normalizeProfile(raw)));
    return;
  }

  if (request.method === "GET" && pathname === "/api/benchmarks") {
    sendJson(response, 200, await benchmarkStore.load());
    return;
  }

  if (request.method === "GET" && pathname === "/api/benchmarks/status") {
    sendJson(response, 200, benchmark.getStatus());
    return;
  }

  if (request.method === "GET" && pathname === "/api/benchmarks/logs") {
    sendJson(response, 200, benchmark.getLogs());
    return;
  }

  if (request.method === "POST" && pathname === "/api/benchmarks/preview") {
    const body = await readJson<{ profile: unknown; settings?: Partial<BenchmarkSettings> }>(request);
    sendJson(response, 200, buildBenchmarkCommand(normalizeProfile(body.profile), body.settings));
    return;
  }

  if (request.method === "POST" && pathname === "/api/benchmarks/start") {
    const body = await readJson<{ profileId: string; settings?: Partial<BenchmarkSettings> }>(request);
    const runtimeStatus = await runtime.getStatus();
    if (runtimeStatus.state === "running" || runtimeStatus.state === "starting") {
      sendError(response, 409, "Stop the llama-server runtime before running a benchmark for cleaner, safer results.");
      return;
    }
    const profiles = await store.load();
    const profile = profiles.find((item) => item.id === body.profileId);
    if (!profile) {
      sendError(response, 404, "profile not found");
      return;
    }
    sendJson(response, 200, await benchmark.start(profile, body.settings));
    return;
  }

  if (request.method === "POST" && pathname === "/api/benchmarks/stop") {
    sendJson(response, 200, await benchmark.stop());
    return;
  }

  const benchmarkMatch = pathname.match(/^\/api\/benchmarks\/([^/]+)$/u);
  if (benchmarkMatch && request.method === "DELETE") {
    const deleted = await benchmarkStore.delete(decodeURIComponent(benchmarkMatch[1]));
    if (!deleted) {
      sendError(response, 404, "benchmark not found");
      return;
    }
    sendJson(response, 200, { ok: true });
    return;
  }

  if (request.method === "POST" && pathname === "/api/server/start") {
    const body = await readJson<{ profileId: string }>(request);
    const profiles = await store.load();
    const profile = profiles.find((item) => item.id === body.profileId);
    if (!profile) {
      sendError(response, 404, "profile not found");
      return;
    }
    sendJson(response, 200, runtime.start(profile));
    return;
  }

  if (request.method === "POST" && pathname === "/api/server/stop") {
    sendJson(response, 200, await runtime.stop());
    return;
  }

  if (request.method === "GET" && pathname === "/api/server/status") {
    sendJson(response, 200, await runtime.getStatus());
    return;
  }

  if (request.method === "GET" && pathname === "/api/server/logs") {
    sendJson(response, 200, runtime.getLogs());
    return;
  }

  sendError(response, 404, "route not found");
}

function serveStatic(response: ServerResponse, pathname: string): void {
  const distRoot = path.resolve(process.cwd(), "dist");
  const requested = pathname === "/" ? "/index.html" : pathname;
  const candidate = path.resolve(distRoot, `.${decodeURIComponent(requested)}`);
  // Guard against path traversal by ensuring the resolved candidate is the
  // dist root itself or strictly nested beneath it (a bare prefix check would
  // match sibling directories like "dist-cuda").
  const relative = path.relative(distRoot, candidate);
  const insideDist =
    candidate === distRoot || (!relative.startsWith("..") && !path.isAbsolute(relative));
  const filePath = insideDist && fs.existsSync(candidate)
    ? candidate
    : path.join(distRoot, "index.html");

  if (!fs.existsSync(filePath)) {
    response.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    response.end("LlamaTuner build not found. Run bun run build first, or use bun run dev.");
    return;
  }

  const extension = path.extname(filePath);
  response.writeHead(200, {
    "content-type": MIME_TYPES[extension] ?? "application/octet-stream",
    "cache-control": extension === ".html" ? "no-store" : "public, max-age=31536000, immutable"
  });
  fs.createReadStream(filePath).pipe(response);
}

export function createRequestHandler(deps: AppDeps = {}) {
  const store = deps.store ?? createProfileStore();
  const runtime = deps.runtime ?? new RuntimeManager();
  const benchmarkStore = deps.benchmarkStore ?? createBenchmarkStore();
  const benchmark = deps.benchmark ?? new BenchmarkManager(benchmarkStore);

  return async (request: IncomingMessage, response: ServerResponse) => {
    try {
      const url = new URL(request.url ?? "/", "http://127.0.0.1");
      if (url.pathname.startsWith("/api/")) {
        await routeApi(request, response, url.pathname, store, runtime, benchmarkStore, benchmark);
        return;
      }
      if (request.method !== "GET" && request.method !== "HEAD") {
        sendError(response, 405, "method not allowed");
        return;
      }
      serveStatic(response, url.pathname);
    } catch (error) {
      if (error instanceof HttpError) {
        sendError(response, error.status, error.message);
        return;
      }
      sendError(response, 500, error instanceof Error ? error.message : String(error));
    }
  };
}

export function createApp(deps: AppDeps = {}) {
  return createServer(createRequestHandler(deps));
}
