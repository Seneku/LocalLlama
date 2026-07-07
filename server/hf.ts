// Proxies to GitHub (llama.cpp releases) and Hugging Face (GGUF model search /
// file listing / downloads). Kept server-side to avoid browser CORS, to attach
// the optional HF token, and to cache the release lookup. Pure normalizers are
// exported separately so they can be unit-tested against captured fixtures.
import { getSettings } from "./settings";
import type {
  EstimateFit,
  HardwareInfo,
  LlamaCppAsset,
  LlamaCppAssetKind,
  LlamaCppRelease,
  ModelFile,
  ModelSearchResult
} from "../src/shared/types";

const GITHUB_LATEST = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest";
const HF_API = "https://huggingface.co/api";
const HF_HOST = "https://huggingface.co";
const RELEASE_TTL_MS = 60 * 60 * 1000;
const USER_AGENT = "LocalLlama";

export class HttpProxyError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = "HttpProxyError";
  }
}

function bytesToMiB(bytes: number): number {
  return bytes / 1024 / 1024;
}

export function hfHeaders(): Record<string, string> {
  const headers: Record<string, string> = { "user-agent": USER_AGENT };
  const token = getSettings().hfToken;
  if (token) {
    headers.authorization = `Bearer ${token}`;
  }
  return headers;
}

// ---- pure helpers (unit-tested) ----

export function classifyAsset(name: string): LlamaCppAssetKind {
  const lower = name.toLowerCase();
  if (lower.includes("cudart")) {
    return "cudart";
  }
  if (lower.includes("cuda")) {
    return "cuda";
  }
  if (lower.includes("vulkan")) {
    return "vulkan";
  }
  if (lower.includes("hip") || lower.includes("radeon") || lower.includes("rocm")) {
    return "hip";
  }
  if (lower.includes("sycl")) {
    return "sycl";
  }
  if (lower.includes("cpu") || lower.includes("avx") || lower.includes("noavx") || lower.includes("arm64")) {
    return "cpu";
  }
  return "other";
}

const QUANT_PATTERN = /\b(IQ\d[A-Z0-9_]*|Q\d[A-Z0-9_]*|BF16|F16|F32)\b/;

export function parseQuant(filename: string): string | null {
  const match = QUANT_PATTERN.exec(filename.toUpperCase());
  return match ? match[1] : null;
}

// Coarse pre-download fit: we only know the file size, not the GGUF tensor
// geometry, so weights (x1.08 for runtime/compute) plus a moderate KV allowance
// are compared against the card's TOTAL VRAM (minus a reserve for the OS /
// desktop compositor) — not whatever is free at this instant, which would be
// misleadingly pessimistic while other apps hold VRAM. The exact per-run number
// comes from the calibrated estimator once the model is on disk.
export function coarseFit(sizeMiB: number, hardware: HardwareInfo): EstimateFit {
  const gpu = hardware.gpus[0] ?? null;
  const total = gpu?.totalMiB ?? gpu?.freeMiB ?? null;
  if (!total || total <= 0) {
    return "unknown";
  }
  const usable = Math.max(0, total - 1024); // ~1 GiB reserved for the OS/compositor
  const need = sizeMiB * 1.08 + 768;
  const ratio = need / usable;
  if (ratio <= 0.9) {
    return "fits";
  }
  if (ratio <= 1) {
    return "tight";
  }
  return "over";
}

export function classifyRelease(json: unknown): LlamaCppRelease {
  const source = (json ?? {}) as Record<string, unknown>;
  const assets = Array.isArray(source.assets) ? source.assets : [];
  const winAssets: LlamaCppAsset[] = assets
    .map((raw) => raw as Record<string, unknown>)
    .filter((asset) => typeof asset.name === "string" && asset.name.toLowerCase().includes("win"))
    .map((asset) => ({
      name: String(asset.name),
      size: typeof asset.size === "number" ? asset.size : 0,
      url: typeof asset.browser_download_url === "string" ? asset.browser_download_url : "",
      kind: classifyAsset(String(asset.name))
    }));
  return {
    tag: typeof source.tag_name === "string" ? source.tag_name : "",
    htmlUrl: typeof source.html_url === "string" ? source.html_url : "https://github.com/ggml-org/llama.cpp/releases",
    winAssets
  };
}

export function normalizeSearch(json: unknown): ModelSearchResult[] {
  if (!Array.isArray(json)) {
    return [];
  }
  return json.map((raw) => {
    const model = raw as Record<string, unknown>;
    const id = typeof model.id === "string" ? model.id : String(model.modelId ?? "");
    return {
      id,
      author: typeof model.author === "string" ? model.author : id.split("/")[0] ?? "",
      downloads: typeof model.downloads === "number" ? model.downloads : 0,
      likes: typeof model.likes === "number" ? model.likes : 0,
      gated: model.gated === true || typeof model.gated === "string",
      pipelineTag: typeof model.pipeline_tag === "string" ? model.pipeline_tag : null,
      updatedAt: typeof model.lastModified === "string" ? model.lastModified : null
    };
  });
}

export function normalizeTree(json: unknown, hardware: HardwareInfo): ModelFile[] {
  if (!Array.isArray(json)) {
    return [];
  }
  return json
    .map((raw) => raw as Record<string, unknown>)
    .filter(
      (entry) => entry.type === "file" && typeof entry.path === "string" && entry.path.toLowerCase().endsWith(".gguf")
    )
    .map((entry) => {
      const sizeBytes = typeof entry.size === "number" ? entry.size : 0;
      const sizeMiB = bytesToMiB(sizeBytes);
      const filename = String(entry.path);
      return {
        filename,
        sizeBytes,
        sizeMiB: Math.round(sizeMiB),
        quant: parseQuant(filename),
        fit: coarseFit(sizeMiB, hardware)
      };
    })
    .sort((a, b) => a.sizeBytes - b.sizeBytes);
}

export function resolveDownloadUrl(id: string, filename: string): string {
  return `${HF_HOST}/${id}/resolve/main/${filename}`;
}

// Largest number-of-billions token in the id (e.g. "Qwen3-30B-A3B" -> 30,
// "Llama-3.1-8B" -> 8, "gemma-4-12b-it" -> 12). Returns null if none found.
export function parseParamsB(id: string): number | null {
  const matches = [...id.matchAll(/(\d+(?:\.\d+)?)\s*b\b/giu)]
    .map((match) => Number(match[1]))
    .filter((value) => Number.isFinite(value) && value > 0 && value < 2000);
  return matches.length ? Math.max(...matches) : null;
}

// Roughly the biggest model (in billions of params) that fits at a common
// Q4_K_M quant: ~0.62 GiB of weights per 1B params, plus a reserve for the
// OS and KV/compute. Uses total VRAM when a GPU is present, else system RAM.
const GIB_PER_B_Q4 = 0.62;
export function recommendedMaxParamsB(hardware: HardwareInfo): number {
  const gpu = hardware.gpus[0] ?? null;
  const budgetMiB = gpu?.totalMiB ?? gpu?.freeMiB ?? hardware.totalRamMiB;
  const reserveGiB = gpu ? 2 : 3;
  const usableGiB = Math.max(0, budgetMiB / 1024 - reserveGiB);
  return Math.max(1, Math.round(usableGiB / GIB_PER_B_Q4));
}

function isRecommendable(model: ModelSearchResult, maxParamsB: number): boolean {
  // Skip embedding / non-chat repos.
  if (model.pipelineTag && model.pipelineTag !== "text-generation") {
    return false;
  }
  const params = parseParamsB(model.id);
  return params === null || params <= maxParamsB;
}

export interface RecommendedResult {
  models: ModelSearchResult[];
  maxParamsB: number;
}

export async function getRecommendedModels(hardware: HardwareInfo): Promise<RecommendedResult> {
  const maxParamsB = recommendedMaxParamsB(hardware);
  // Most popular first; fetch extra so the size filter still leaves a full page.
  const raw = await searchModels("", "downloads", 80);
  const models = raw.filter((model) => isRecommendable(model, maxParamsB)).slice(0, 30);
  return { models, maxParamsB };
}

// ---- network ----

let releaseCache: { at: number; release: LlamaCppRelease } | null = null;

export async function getLatestLlamaCppRelease(now: number = Date.now()): Promise<LlamaCppRelease> {
  if (releaseCache && now - releaseCache.at < RELEASE_TTL_MS) {
    return releaseCache.release;
  }
  const response = await fetch(GITHUB_LATEST, {
    headers: { "user-agent": USER_AGENT, accept: "application/vnd.github+json" }
  });
  if (!response.ok) {
    throw new HttpProxyError(response.status, `GitHub releases API returned ${response.status}`);
  }
  const release = classifyRelease(await response.json());
  releaseCache = { at: now, release };
  return release;
}

export async function searchModels(query: string, sort = "downloads", limit = 30): Promise<ModelSearchResult[]> {
  const url = new URL(`${HF_API}/models`);
  url.searchParams.set("filter", "gguf");
  if (query.trim()) {
    url.searchParams.set("search", query.trim());
  }
  url.searchParams.set("sort", sort);
  url.searchParams.set("direction", "-1");
  url.searchParams.set("limit", String(limit));
  const response = await fetch(url, { headers: hfHeaders() });
  if (!response.ok) {
    throw new HttpProxyError(response.status, `Hugging Face search returned ${response.status}`);
  }
  return normalizeSearch(await response.json());
}

export async function listModelFiles(id: string, hardware: HardwareInfo): Promise<ModelFile[]> {
  const response = await fetch(`${HF_API}/models/${id}/tree/main?recursive=true`, { headers: hfHeaders() });
  if (response.status === 401 || response.status === 403) {
    throw new HttpProxyError(
      response.status,
      "This model is gated or private. Accept its license on Hugging Face and add an access token in Settings."
    );
  }
  if (!response.ok) {
    throw new HttpProxyError(response.status, `Hugging Face file listing returned ${response.status}`);
  }
  return normalizeTree(await response.json(), hardware);
}

/** Test hook: clear the cached llama.cpp release. */
export function resetReleaseCache(): void {
  releaseCache = null;
}
