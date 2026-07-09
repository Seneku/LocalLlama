// Model-choice guidance: quant quality ladder + per-repo recommended quant,
// MoE-aware parameter parsing, use-case classification, and requant-mirror
// dedup. All pure functions over strings/listings so they unit-test against
// fixtures (same convention as hf.ts).
import type { ModelFile, ModelSearchResult, UseCase } from "../src/shared/types";

// ---- quant quality ladder ----

export interface QuantInfo {
  rank: number;
  /** Approximate bits per weight. */
  bpw: number;
  blurb: string;
}

/**
 * Quality ordering of common GGUF quants. Rank compares quality (higher =
 * closer to the original weights); size grows with rank too, so "best quant
 * that fits" balances both.
 */
export const QUANT_LADDER: Record<string, QuantInfo> = {
  Q2_K: { rank: 10, bpw: 2.6, blurb: "Smallest usable size — noticeable quality loss; a last resort." },
  IQ2_XXS: { rank: 8, bpw: 2.1, blurb: "Extremely small; expect visible degradation." },
  IQ2_XS: { rank: 9, bpw: 2.3, blurb: "Extremely small; expect visible degradation." },
  IQ3_XXS: { rank: 12, bpw: 3.1, blurb: "Very small with surprisingly usable quality for the size." },
  IQ3_M: { rank: 14, bpw: 3.7, blurb: "Small; decent quality for tight VRAM." },
  Q3_K_S: { rank: 13, bpw: 3.5, blurb: "Small; clearly below Q4 quality." },
  Q3_K_M: { rank: 15, bpw: 3.9, blurb: "Small; a step below the Q4 sweet spot." },
  IQ4_XS: { rank: 19, bpw: 4.3, blurb: "Nearly Q4_K_M quality in slightly less space." },
  Q4_K_S: { rank: 20, bpw: 4.6, blurb: "Solid all-rounder, slightly smaller than Q4_K_M." },
  Q4_0: { rank: 18, bpw: 4.5, blurb: "Legacy 4-bit; prefer Q4_K_M when available." },
  Q4_K_M: { rank: 21, bpw: 4.8, blurb: "The usual sweet spot: near-full quality at ~half the F16 size." },
  Q5_K_S: { rank: 23, bpw: 5.5, blurb: "A notch above Q4 quality for ~15% more memory." },
  Q5_K_M: { rank: 24, bpw: 5.7, blurb: "A notch above Q4 quality for ~20% more memory." },
  Q6_K: { rank: 26, bpw: 6.6, blurb: "Very close to original quality; use when VRAM is plentiful." },
  Q8_0: { rank: 28, bpw: 8.5, blurb: "Practically lossless; twice the size of Q4." },
  BF16: { rank: 30, bpw: 16, blurb: "Original precision; needs the most memory, no quality loss." },
  F16: { rank: 30, bpw: 16, blurb: "Original precision; needs the most memory, no quality loss." },
  F32: { rank: 31, bpw: 32, blurb: "Full 32-bit weights; rarely worth the size." }
};

export function quantInfo(quant: string | null): QuantInfo | null {
  if (!quant) {
    return null;
  }
  const exact = QUANT_LADDER[quant];
  if (exact) {
    return exact;
  }
  // Unknown variant: coarse rank from the leading digit family.
  const family = /^I?Q(\d)/u.exec(quant);
  if (family) {
    const digit = Number(family[1]);
    const rank = { 1: 5, 2: 9, 3: 13, 4: 19, 5: 23, 6: 26, 8: 28 }[digit] ?? 15;
    return { rank, bpw: digit + 0.5, blurb: "Uncommon variant; quality roughly tracks its bit width." };
  }
  return null;
}

const SHARD_PATTERN = /-(\d{5})-of-(\d{5})\.gguf$/iu;

/**
 * Pick the quant to recommend from a repo's file list: the highest-quality
 * quant whose fit verdict is "fits" (falling back to "tight"), floored at the
 * Q4-class sweet spot — going above Q6_K rarely improves output but always
 * costs VRAM headroom, and dropping below Q4 is flagged as a compromise.
 */
export function recommendQuant(
  files: Array<Pick<ModelFile, "filename" | "quant" | "fit">>
): { filename: string; reason: string; compromise: boolean } | null {
  const candidates = files
    .map((file) => ({ file, info: quantInfo(file.quant), shard: SHARD_PATTERN.exec(file.filename) }))
    // For multi-part models, only shard 1 represents the set.
    .filter((entry) => entry.info !== null && (!entry.shard || entry.shard[1] === "00001"));

  type Candidate = (typeof candidates)[number];
  const pickHighest = (fits: Candidate[]) =>
    fits.reduce<Candidate | null>((best, entry) => (best === null || entry.info!.rank > best.info!.rank ? entry : best), null);
  const pickLowest = (fits: Candidate[]) =>
    fits.reduce<Candidate | null>((best, entry) => (best === null || entry.info!.rank < best.info!.rank ? entry : best), null);

  const fitting = candidates.filter((entry) => entry.file.fit === "fits");
  const tight = candidates.filter((entry) => entry.file.fit === "tight");
  const Q6_RANK = QUANT_LADDER.Q6_K.rank;
  const Q4_RANK = QUANT_LADDER.Q4_K_M.rank;

  // Prefer comfortable fits capped at Q6_K-class; above that level extra bits
  // buy almost nothing, so if only Q8/F16 fit take the smallest of them.
  const comfortable =
    pickHighest(fitting.filter((entry) => entry.info!.rank <= Q6_RANK)) ?? pickLowest(fitting);
  const chosen = comfortable ?? pickHighest(tight);
  if (!chosen) {
    return null;
  }
  const info = chosen.info!;
  const compromise = info.rank < Q4_RANK;
  const reason = compromise
    ? `${chosen.file.quant}: the largest quant that fits your hardware — below the Q4 sweet spot, so expect some quality loss. ${info.blurb}`
    : `${chosen.file.quant}: best quality that comfortably fits your hardware. ${info.blurb}`;
  return { filename: chosen.file.filename, reason, compromise };
}

/** Attach recommended/recommendReason to the file list (returns a new array). */
export function annotateRecommendedQuant(files: ModelFile[]): ModelFile[] {
  const recommendation = recommendQuant(files);
  if (!recommendation) {
    return files;
  }
  return files.map((file) =>
    file.filename === recommendation.filename
      ? { ...file, recommended: true, recommendReason: recommendation.reason }
      : file
  );
}

// ---- MoE-aware parameter parsing ----

export interface ModelParams {
  totalB: number | null;
  /** Active params per token for MoE models (e.g. Qwen3-30B-A3B -> 3). */
  activeB: number | null;
}

export function parseModelParams(id: string): ModelParams {
  // "30B-A3B" style: total + active.
  const moe = /(\d+(?:\.\d+)?)\s*b[-_.]a(\d+(?:\.\d+)?)\s*b/iu.exec(id);
  if (moe) {
    return { totalB: Number(moe[1]), activeB: Number(moe[2]) };
  }
  // "8x7B" style (Mixtral): total = n*size; ~2 experts active per token.
  const grid = /(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*b\b/iu.exec(id);
  if (grid) {
    const experts = Number(grid[1]);
    const size = Number(grid[2]);
    return { totalB: experts * size, activeB: Math.min(experts, 2) * size };
  }
  const matches = [...id.matchAll(/(\d+(?:\.\d+)?)\s*b\b/giu)]
    .map((match) => Number(match[1]))
    .filter((value) => Number.isFinite(value) && value > 0 && value < 2000);
  return { totalB: matches.length ? Math.max(...matches) : null, activeB: null };
}

// ---- use-case classification ----

/**
 * Keyword rules over repo ids — HF pipeline tags are too unreliable for
 * grouping. First match wins; extend by adding a row.
 */
export const USE_CASE_RULES: Array<{ useCase: UseCase; pattern: RegExp }> = [
  { useCase: "coding", pattern: /coder|codestral|devstral|starcoder|codellama|code-llama|deepseek-coder|codegemma|codeqwen/iu },
  { useCase: "vision", pattern: /-vl(?:-|\b)|vision|llava|pixtral|minicpm-v|internvl|moondream/iu },
  { useCase: "reasoning", pattern: /(?:^|[-_/])r1(?:[-_.]|\b)|qwq|reason|thinking/iu }
];

export function classifyUseCase(id: string, _pipelineTag: string | null): UseCase {
  // Keywords only. The image-text-to-text pipeline tag is NOT a vision signal
  // anymore — most modern flagship chat models are multimodal and carry it;
  // "vision" here means vision-specialist models (LLaVA, -VL, etc.).
  for (const rule of USE_CASE_RULES) {
    if (rule.pattern.test(id)) {
      return rule.useCase;
    }
  }
  return "chat";
}

// ---- requant-mirror dedup ----

// Packaging-only tokens. Deliberately does NOT strip "instruct"/"it"/"chat" —
// a base model and its instruct tune are different models and must not merge.
const KEY_STRIP_TOKENS = /(?:^|[-_.])(gguf|ggml|imatrix|i1|gptq|awq|exl2|hf)(?=$|[-_.])/giu;

/**
 * Canonical key for grouping requants/mirrors of the same base model.
 * Conservative on purpose: strips the author, quant tokens, and packaging
 * suffixes but keeps every other token so distinct finetunes never merge.
 */
export function baseModelKey(id: string): string {
  const name = id.split("/").pop() ?? id;
  return name
    .toUpperCase()
    .replace(/\b(IQ\d[A-Z0-9_]*|Q\d[A-Z0-9_]*|BF16|F16|F32)\b/gu, "")
    .toLowerCase()
    .replace(KEY_STRIP_TOKENS, "")
    .replace(/[-_.]+/gu, "-")
    .replace(/^-+|-+$/gu, "");
}

export interface ModelGroup {
  representative: ModelSearchResult;
  mirrorCount: number;
}

/** Collapse mirrors of the same base model, keeping the most-downloaded repo. */
export function dedupeMirrors(models: ModelSearchResult[]): ModelSearchResult[] {
  const groups = new Map<string, ModelGroup>();
  const order: string[] = [];
  for (const model of models) {
    const key = baseModelKey(model.id);
    const existing = groups.get(key);
    if (!existing) {
      groups.set(key, { representative: model, mirrorCount: 0 });
      order.push(key);
    } else {
      existing.mirrorCount += 1;
      if (model.downloads > existing.representative.downloads) {
        existing.representative = model;
      }
    }
  }
  return order.map((key) => {
    const group = groups.get(key)!;
    return { ...group.representative, mirrorCount: group.mirrorCount };
  });
}
