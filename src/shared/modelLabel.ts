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

// A size-bucket token: "2B", "13B", "0.5B", "270M", "?B", MoE "8x7B", or
// llama.cpp's active-expert notation "35B.A3B" / "30B-A3B".
function looksLikeSizeBucket(token: string): boolean {
  return (
    /^\?B$/i.test(token) ||
    /^[\d.]+(?:x[\d.]+)?[BM]$/i.test(token) ||
    /^[\d.]+B[.\-]A[\d.]+B$/i.test(token)
  );
}

// Billions of params implied by a bucket token: "9B" -> 9, "270M" -> 0.27,
// "8x7B" -> 56 (MoE total), "35B.A3B" -> 35 (MoE total). Null for "?B".
function bucketBillions(token: string): number | null {
  const active = /^([\d.]+)B[.\-]A[\d.]+B$/i.exec(token);
  if (active) {
    return Number(active[1]);
  }
  const moe = /^([\d.]+)x([\d.]+)B$/i.exec(token);
  if (moe) {
    return Number(moe[1]) * Number(moe[2]);
  }
  const single = /^([\d.]+)([BM])$/i.exec(token);
  if (single) {
    return single[2].toUpperCase() === "M" ? Number(single[1]) / 1000 : Number(single[1]);
  }
  return null;
}

/**
 * A human label for the benchmarked model. llama.cpp's model_type buckets the
 * parameter count into fixed sizes and mislabels ones it doesn't recognise
 * (an 11.7B model shown as "2B", or "?B"). We keep llama.cpp's bucket when it
 * roughly agrees with the true parameter count — its rounded marketing labels
 * ("7B", "9B", "8x7B") read better than a raw count — and only override the
 * size token when the bucket is clearly wrong or unknown. Returns null when
 * there is no model_type to show.
 */
export function formatModelType(env: BenchmarkEnv | null | undefined): string | null {
  const raw = env?.modelType?.trim() || null;
  if (!raw) {
    return null;
  }
  if (!env?.modelParams || env.modelParams <= 0) {
    return raw;
  }
  const realBillions = env.modelParams / 1e9;
  const words = raw.split(/\s+/u);
  // model_type is "<arch> <bucket> <ftype…>"; the bucket is the second word.
  if (words.length >= 2 && looksLikeSizeBucket(words[1])) {
    const claimed = bucketBillions(words[1]);
    // Trust the bucket when the real count is within ~±25% of it.
    if (claimed !== null && realBillions >= claimed * 0.8 && realBillions <= claimed * 1.25) {
      return raw;
    }
    words[1] = formatParamCount(env.modelParams);
    return words.join(" ");
  }
  // Unexpected shape (no arch prefix, or a bucket we don't recognise): keep the
  // original but annotate it with the real size so it is at least accurate.
  return `${raw} (${formatParamCount(env.modelParams)})`;
}
