import type {
  BackendMode,
  FlashAttentionMode,
  KvCacheType,
  LlamaProfile,
  ReasoningMode,
  SpecType,
  SpeculativeSettings,
  ThreadsMode
} from "../src/shared/types";

function str(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function num(value: unknown, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return fallback;
}

function bool(value: unknown, fallback = false): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function oneOf<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
  return typeof value === "string" && (allowed as readonly string[]).includes(value) ? (value as T) : fallback;
}

const BACKEND_MODES: readonly BackendMode[] = ["auto", "cuda", "cpu"];
const REASONING_MODES: readonly ReasoningMode[] = ["off", "auto"];
const THREADS_MODES: readonly ThreadsMode[] = ["auto", "manual"];
const KV_TYPES: readonly KvCacheType[] = ["", "f16", "q8_0", "q4_0", "q4_1"];
const SPEC_TYPES: readonly SpecType[] = ["draft-mtp", "none"];
const FLASH_MODES: readonly FlashAttentionMode[] = ["auto", "on", "off"];

function normalizeSpeculative(value: unknown): SpeculativeSettings {
  const raw = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  return {
    enabled: bool(raw.enabled, false),
    type: oneOf(raw.type, SPEC_TYPES, "none"),
    draftNMax: num(raw.draftNMax, 0),
    draftModelPath: str(raw.draftModelPath, ""),
    draftGpuLayers: num(raw.draftGpuLayers, 0),
    draftCacheK: oneOf(raw.draftCacheK, KV_TYPES, ""),
    draftCacheV: oneOf(raw.draftCacheV, KV_TYPES, ""),
    draftPMin: num(raw.draftPMin, 0)
  };
}

/**
 * Coerce an untrusted request body into a well-formed LlamaProfile, supplying
 * defaults for every field (especially nested objects like `speculative`) so a
 * malformed body cannot throw a TypeError deep in command/estimate code.
 */
export function normalizeProfile(input: unknown): LlamaProfile {
  const raw = input && typeof input === "object" ? (input as Record<string, unknown>) : {};
  return {
    id: str(raw.id, ""),
    name: str(raw.name, ""),
    description: str(raw.description, ""),
    modelPath: str(raw.modelPath, ""),
    modelAlias: str(raw.modelAlias, ""),
    backendMode: oneOf(raw.backendMode, BACKEND_MODES, "auto"),
    host: str(raw.host, "127.0.0.1"),
    port: num(raw.port, 8080),
    contextSize: num(raw.contextSize, 4096),
    threadsMode: oneOf(raw.threadsMode, THREADS_MODES, "auto"),
    threads: num(raw.threads, 0),
    gpuLayers: num(raw.gpuLayers, 0),
    reasoning: oneOf(raw.reasoning, REASONING_MODES, "off"),
    jinja: bool(raw.jinja, false),
    mlock: bool(raw.mlock, false),
    mmap: bool(raw.mmap, true),
    fit: bool(raw.fit, false),
    fitTargetMiB: num(raw.fitTargetMiB, 0),
    cpuMoe: bool(raw.cpuMoe, false),
    nCpuMoe: num(raw.nCpuMoe, 0),
    temperature: num(raw.temperature, 0.8),
    batchSize: num(raw.batchSize, 0),
    ubatchSize: num(raw.ubatchSize, 0),
    flashAttention: oneOf(raw.flashAttention, FLASH_MODES, "auto"),
    parallelSlots: num(raw.parallelSlots, 1),
    kvCacheK: oneOf(raw.kvCacheK, KV_TYPES, ""),
    kvCacheV: oneOf(raw.kvCacheV, KV_TYPES, ""),
    speculative: normalizeSpeculative(raw.speculative)
  };
}
