// Hoists environment facts (llama.cpp build, hardware, model identity) out of a
// benchmark run's raw llama-bench JSON so history can be grouped and charted
// without re-parsing rows[].raw. Lives in its own module because benchmark.ts
// already imports benchmarkStore.ts, and the store needs this for backfilling
// legacy runs — importing it back from benchmark.ts would create a cycle.
import type { BenchmarkEnv, BenchmarkRow } from "../src/shared/types";

function textValue(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function numberValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

// llama-bench has emitted flash_attn as a boolean, 0/1, and "0"/"1" across
// builds — accept all of them.
function boolValue(value: unknown): boolean | null {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "number") {
    return value !== 0;
  }
  if (typeof value === "string") {
    const trimmed = value.trim().toLowerCase();
    if (trimmed === "1" || trimmed === "true") {
      return true;
    }
    if (trimmed === "0" || trimmed === "false") {
      return false;
    }
  }
  return null;
}

export function extractBenchmarkEnv(rows: BenchmarkRow[]): BenchmarkEnv | null {
  const raw = rows[0]?.raw;
  if (!raw || typeof raw !== "object" || Object.keys(raw).length === 0) {
    return null;
  }
  return {
    buildNumber: numberValue(raw.build_number),
    buildCommit: textValue(raw.build_commit),
    gpuName: textValue(raw.gpu_info),
    cpuName: textValue(raw.cpu_info),
    backends: textValue(raw.backends),
    modelType: textValue(raw.model_type),
    modelSizeBytes: numberValue(raw.model_size),
    modelParams: numberValue(raw.model_n_params),
    nGpuLayers: numberValue(raw.n_gpu_layers),
    flashAttn: boolValue(raw.flash_attn)
  };
}
