import { describe, expect, test } from "bun:test";

import { isRecommendable } from "../server/hf";
import {
  baseModelKey,
  classifyUseCase,
  dedupeMirrors,
  parseModelParams,
  quantInfo,
  recommendQuant
} from "../server/modelGuidance";
import type { EstimateFit, HardwareInfo, ModelSearchResult } from "../src/shared/types";

describe("parseModelParams", () => {
  test("parses total and active params for A3B-style MoE names", () => {
    expect(parseModelParams("Qwen/Qwen3-30B-A3B-GGUF")).toEqual({ totalB: 30, activeB: 3 });
    expect(parseModelParams("unsloth/Qwen3-235B-A22B-GGUF")).toEqual({ totalB: 235, activeB: 22 });
  });

  test("parses NxM MoE names with ~2 active experts", () => {
    expect(parseModelParams("Mixtral-8x7B-Instruct-v0.1-GGUF")).toEqual({ totalB: 56, activeB: 14 });
  });

  test("parses dense names with the largest B token and no active split", () => {
    expect(parseModelParams("meta-llama/Llama-3.1-8B-Instruct")).toEqual({ totalB: 8, activeB: null });
    expect(parseModelParams("google/gemma-3-12b-it")).toEqual({ totalB: 12, activeB: null });
    expect(parseModelParams("bartowski/some-model-GGUF")).toEqual({ totalB: null, activeB: null });
  });
});

const hardware: HardwareInfo = {
  totalRamMiB: 65536,
  freeRamMiB: 48000,
  gpus: [{ name: "RTX 4070 SUPER", vendor: "nvidia", totalMiB: 12281, usedMiB: 1600, freeMiB: 10681 }]
};

function model(id: string, downloads = 1000, pipelineTag: string | null = "text-generation"): ModelSearchResult {
  return { id, author: id.split("/")[0] ?? "", downloads, likes: 0, gated: false, pipelineTag, updatedAt: null };
}

describe("isRecommendable", () => {
  const maxParamsB = 16; // ~12 GB card

  test("keeps MoE models whose ACTIVE params fit even when total params exceed the GPU budget", () => {
    // This is the Qwen3-30B-A3B case the old max-token parser wrongly excluded.
    expect(isRecommendable(model("unsloth/Qwen3-30B-A3B-GGUF"), maxParamsB, hardware)).toBe(true);
  });

  test("still drops MoE models too big for combined RAM + VRAM", () => {
    expect(isRecommendable(model("unsloth/Qwen3-235B-A22B-GGUF"), maxParamsB, hardware)).toBe(false);
  });

  test("keeps small dense models and drops large ones", () => {
    expect(isRecommendable(model("meta-llama/Llama-3.1-8B-Instruct"), maxParamsB, hardware)).toBe(true);
    expect(isRecommendable(model("meta-llama/Llama-3.1-70B-Instruct"), maxParamsB, hardware)).toBe(false);
  });

  test("drops embedding repos (by tag or id) but keeps multimodal chat models", () => {
    expect(isRecommendable(model("some/embed-model-1B", 1, "feature-extraction"), maxParamsB, hardware)).toBe(false);
    expect(isRecommendable(model("ggml-org/embeddinggemma-300M-GGUF", 1, null), maxParamsB, hardware)).toBe(false);
    expect(isRecommendable(model("some/qwen-vl-7b", 1, "image-text-to-text"), maxParamsB, hardware)).toBe(true);
  });
});

describe("classifyUseCase", () => {
  test("classifies by id keywords with chat as the default", () => {
    expect(classifyUseCase("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "text-generation")).toBe("coding");
    expect(classifyUseCase("mistralai/Devstral-Small-GGUF", "text-generation")).toBe("coding");
    expect(classifyUseCase("Qwen/Qwen2-VL-7B-GGUF", "text-generation")).toBe("vision");
    expect(classifyUseCase("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "text-generation")).toBe("reasoning");
    expect(classifyUseCase("Qwen/QwQ-32B-GGUF", "text-generation")).toBe("reasoning");
    expect(classifyUseCase("meta-llama/Llama-3.1-8B-Instruct", "text-generation")).toBe("chat");
    // Multimodal chat models carry image-text-to-text but are still chat;
    // "vision" is reserved for keyword-matched vision specialists.
    expect(classifyUseCase("unsloth/gemma-4-26B-A4B-it-GGUF", "image-text-to-text")).toBe("chat");
  });
});

describe("baseModelKey + dedupeMirrors", () => {
  test("collapses requant mirrors of the same base model", () => {
    const mirrors = [
      model("bartowski/Qwen2.5-7B-Instruct-GGUF", 50000),
      model("unsloth/Qwen2.5-7B-Instruct-GGUF", 90000),
      model("mradermacher/Qwen2.5-7B-Instruct-i1-GGUF", 20000)
    ];
    const deduped = dedupeMirrors(mirrors);
    expect(deduped).toHaveLength(1);
    expect(deduped[0].id).toBe("unsloth/Qwen2.5-7B-Instruct-GGUF"); // most downloads wins
    expect(deduped[0].mirrorCount).toBe(2);
  });

  test("never merges distinct finetunes or base vs instruct", () => {
    expect(baseModelKey("a/Qwen2.5-7B-Instruct-GGUF")).not.toBe(baseModelKey("a/Qwen2.5-7B-GGUF"));
    expect(baseModelKey("a/Llama-3.1-8B-Instruct-GGUF")).not.toBe(baseModelKey("a/Llama-3.1-8B-Instruct-abliterated-GGUF"));
  });

  test("ignores quant tokens embedded in repo names", () => {
    expect(baseModelKey("a/Gemma-4-12B-it-Q4_K_M-GGUF")).toBe(baseModelKey("b/Gemma-4-12B-it-GGUF"));
  });
});

describe("recommendQuant", () => {
  function file(quant: string, fit: EstimateFit, filename = `model-${quant}.gguf`) {
    return { filename, quant, fit };
  }

  test("picks the best quality that comfortably fits, capped at Q6_K-class", () => {
    const pick = recommendQuant([
      file("Q2_K", "fits"),
      file("Q4_K_M", "fits"),
      file("Q6_K", "fits"),
      file("Q8_0", "tight"),
      file("F16", "over")
    ]);
    expect(pick?.filename).toBe("model-Q6_K.gguf");
    expect(pick?.compromise).toBe(false);
  });

  test("takes the smallest above-cap quant when only Q8/F16 fit", () => {
    const pick = recommendQuant([file("Q8_0", "fits"), file("F16", "fits")]);
    expect(pick?.filename).toBe("model-Q8_0.gguf");
  });

  test("flags a quality compromise when only sub-Q4 quants fit", () => {
    const pick = recommendQuant([file("Q2_K", "fits"), file("Q3_K_M", "fits"), file("Q4_K_M", "over")]);
    expect(pick?.filename).toBe("model-Q3_K_M.gguf");
    expect(pick?.compromise).toBe(true);
  });

  test("falls back to tight fits and skips non-first shards", () => {
    const pick = recommendQuant([
      file("Q4_K_M", "tight", "model-Q4_K_M-00002-of-00003.gguf"),
      file("Q4_K_M", "tight", "model-Q4_K_M-00001-of-00003.gguf"),
      file("Q8_0", "over")
    ]);
    expect(pick?.filename).toBe("model-Q4_K_M-00001-of-00003.gguf");
  });

  test("returns null when nothing fits", () => {
    expect(recommendQuant([file("Q4_K_M", "over"), file("Q8_0", "over")])).toBeNull();
  });
});

describe("quantInfo", () => {
  test("knows the ladder and coarsely ranks unknown variants", () => {
    expect(quantInfo("Q4_K_M")!.rank).toBeGreaterThan(quantInfo("Q3_K_M")!.rank);
    expect(quantInfo("Q8_0")!.rank).toBeGreaterThan(quantInfo("Q6_K")!.rank);
    expect(quantInfo("Q4_1")).not.toBeNull(); // unknown variant, Q4 family
    expect(quantInfo("MXFP4")).toBeNull();
    expect(quantInfo(null)).toBeNull();
  });
});
