// llama-bench reports a model_type string like "qwen35 2B Q4_K - Medium":
// "<arch> <size-bucket> <ftype>". llama.cpp derives the size bucket by snapping
// the parameter count to a fixed set of sizes (0.5B, 1B, 2B, 7B, 13B, …) and
// emits a wrong bucket — or "?B" — for models it doesn't recognise (e.g. an
// 11.7B model shown as "2B"). The architecture and quant parts are reliable;
// only the size is wrong. When the true parameter count is available we swap
// the bucket for the real size so the Model column matches the model.
import type { BenchmarkEnv } from "./types";

/** Round a raw parameter count to a clean size label (11.68e9 -> "12B", 3.8e9 -> "3.8B", 494e6 -> "494M"). */
export function formatParamCount(nParams: number): string {
  const billions = nParams / 1e9;
  if (billions >= 1) {
    return billions >= 10 ? `${Math.round(billions)}B` : `${Number(billions.toFixed(1))}B`;
  }
  return `${Math.round(nParams / 1e6)}M`;
}

// A size-bucket token: "2B", "13B", "0.5B", "270M", "?B", or MoE "8x7B".
function looksLikeSizeBucket(token: string): boolean {
  return /^\?B$/i.test(token) || /^[\d.]+(?:x[\d.]+)?[BM]$/i.test(token);
}

/**
 * A human label for the benchmarked model with llama.cpp's unreliable size
 * bucket corrected from the true parameter count. Returns null when there is
 * no model_type to show.
 */
export function formatModelType(env: BenchmarkEnv | null | undefined): string | null {
  const raw = env?.modelType?.trim() || null;
  if (!raw) {
    return null;
  }
  if (!env?.modelParams || env.modelParams <= 0) {
    return raw;
  }
  const size = formatParamCount(env.modelParams);
  const words = raw.split(/\s+/u);
  // model_type is "<arch> <bucket> <ftype…>"; the bucket is the second word.
  if (words.length >= 2 && looksLikeSizeBucket(words[1])) {
    words[1] = size;
    return words.join(" ");
  }
  // Unexpected shape (no arch prefix, or a bucket we don't recognise): keep the
  // original but annotate it with the real size so it is at least accurate.
  return `${raw} (${size})`;
}
