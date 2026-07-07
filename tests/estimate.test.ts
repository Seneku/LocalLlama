import { describe, expect, test } from "bun:test";

import { appleWorkingSetMiB, estimateProfileMemory } from "../server/estimate";
import { defaultProfiles } from "../server/defaultProfiles";
import type { GgufTensorLayout } from "../server/gguf";
import type { HardwareInfo, LlamaProfile, ModelMetadata } from "../src/shared/types";

const hardware: HardwareInfo = {
  totalRamMiB: 65536,
  freeRamMiB: 48000,
  gpus: [
    {
      name: "NVIDIA GeForce RTX 4070 SUPER",
      totalMiB: 12281,
      usedMiB: 1600,
      freeMiB: 10681
    }
  ]
};

const emptyMetadata: ModelMetadata = {
  architecture: null,
  name: null,
  parameterSize: null,
  fileType: null,
  fileSizeMiB: 0,
  blockCount: null,
  contextLength: null,
  embeddingLength: null,
  headCount: null,
  headCountKv: null,
  keyLength: null,
  valueLength: null,
  slidingWindow: null,
  slidingWindowPattern: null,
  keyLengthSwa: null,
  valueLengthSwa: null,
  fullAttentionInterval: null,
  ssmStateSize: null,
  ssmInnerSize: null,
  ssmConvKernel: null,
  ssmGroupCount: null,
  nextnPredictLayers: null
};

// Plain GQA transformer (llama-3-8B-like geometry).
const denseMetadata: ModelMetadata = {
  ...emptyMetadata,
  architecture: "llama",
  name: "Dense test",
  parameterSize: "8B",
  fileType: 15,
  fileSizeMiB: 4800,
  blockCount: 32,
  contextLength: 32768,
  embeddingLength: 4096,
  headCount: 32,
  headCountKv: 8,
  keyLength: 128,
  valueLength: 128
};

// Gemma4-like: interleaved SWA (distinct tensor shapes), MQA full layers,
// tied embeddings (no output.weight tensor).
const MIB = 1024 * 1024;

function gemmaLayout(): GgufTensorLayout {
  const layers = Array.from({ length: 48 }, (_, index) => ({
    index,
    bytes: 130 * MIB,
    // Every 6th layer (5, 11, ...) is full attention: 1 KV head x 512 dims.
    // SWA layers: 8 KV heads x 256 dims.
    attnKElements: (index + 1) % 6 === 0 ? 512 : 2048,
    attnVElements: (index + 1) % 6 === 0 ? 512 : 2048,
    recurrent: false
  }));
  return {
    layers,
    tokenEmbdBytes: 780 * MIB,
    outputBytes: 0,
    otherBytes: 4 * MIB,
    totalBytes: layers.length * 130 * MIB + 784 * MIB,
    exact: true
  };
}

const gemmaMetadata: ModelMetadata = {
  ...emptyMetadata,
  architecture: "gemma4",
  name: "Gemma4 test",
  parameterSize: "12B",
  fileType: 15,
  fileSizeMiB: 7024,
  blockCount: 48,
  contextLength: 131072,
  embeddingLength: 3840,
  headCount: 16,
  headCountKv: null,
  keyLength: 512,
  valueLength: 512,
  slidingWindow: 1024,
  keyLengthSwa: 256,
  valueLengthSwa: 256
};

// Qwen35-like hybrid: 8 attention layers, 24 recurrent SSM layers, 1 MTP block.
function hybridLayout(): GgufTensorLayout {
  const layers = Array.from({ length: 33 }, (_, index) => {
    const attention = (index + 1) % 4 === 0 || index === 32;
    return {
      index,
      bytes: 146 * MIB,
      attnKElements: attention ? 1024 : null,
      attnVElements: attention ? 1024 : null,
      recurrent: !attention
    };
  });
  return {
    layers,
    tokenEmbdBytes: 346 * MIB,
    outputBytes: 340 * MIB,
    otherBytes: 8 * MIB,
    totalBytes: layers.length * 146 * MIB + 694 * MIB,
    exact: true
  };
}

const hybridMetadata: ModelMetadata = {
  ...emptyMetadata,
  architecture: "qwen35",
  name: "Ornith test",
  parameterSize: "9B",
  fileType: 15,
  fileSizeMiB: 5512,
  blockCount: 33,
  contextLength: 262144,
  embeddingLength: 4096,
  headCount: 16,
  headCountKv: 4,
  keyLength: 256,
  valueLength: 256,
  fullAttentionInterval: 4,
  ssmStateSize: 128,
  ssmInnerSize: 4096,
  ssmConvKernel: 4,
  ssmGroupCount: 16,
  nextnPredictLayers: 1
};

function makeProfile(overrides: Partial<LlamaProfile>): LlamaProfile {
  return {
    ...defaultProfiles[0],
    id: "test",
    name: "test",
    ...overrides
  };
}

describe("memory estimates", () => {
  test("estimates CUDA model, KV cache, and headroom for a dense GQA model", async () => {
    const profile = makeProfile({ backendMode: "cuda", gpuLayers: 999, contextSize: 32768, parallelSlots: 1 });
    const estimate = await estimateProfileMemory(profile, {
      hardware,
      metadata: denseMetadata,
      fileExists: () => true
    });

    expect(estimate.backend).toBe("CUDA");
    // ~90% of the file offloads (metadata fallback keeps embeddings host-side).
    expect(estimate.breakdown.gpuModelWeightsMiB).toBeGreaterThan(4000);
    // 32 layers x 32768 tokens x 8 heads x 128 dims x 2 (K+V) x 2 bytes = 4096 MiB
    expect(estimate.breakdown.kvCacheMiB).toBeGreaterThan(3900);
    expect(estimate.breakdown.kvCacheMiB).toBeLessThan(4300);
    expect(estimate.vramHeadroomMiB).not.toBeNull();
    expect(estimate.assumptions.some((item) => item.includes("full model offload"))).toBe(true);
    // Everything fits at full offload, so there is nothing better to recommend.
    expect(estimate.recommendation).toBeNull();
  });

  test("recommends the largest gpu-layer count that fits free VRAM", async () => {
    const profile = makeProfile({ backendMode: "cuda", gpuLayers: 999, contextSize: 12288, parallelSlots: 1 });
    const tightHardware: HardwareInfo = {
      ...hardware,
      gpus: [{ name: "RTX 4070 SUPER", totalMiB: 12281, usedMiB: 4600, freeMiB: 7681 }]
    };
    const estimate = await estimateProfileMemory(profile, {
      hardware: tightHardware,
      metadata: gemmaMetadata,
      layout: gemmaLayout(),
      fileExists: () => true
    });

    expect(estimate.fit).toBe("over");
    expect(estimate.recommendation).not.toBeNull();
    expect(estimate.recommendation!.gpuLayers).toBeGreaterThan(0);
    expect(estimate.recommendation!.gpuLayers).toBeLessThan(49);
    expect(estimate.recommendation!.fullOffload).toBe(false);
    // The recommended setting must itself fit the free VRAM.
    expect(estimate.recommendation!.estimatedVramMiB).toBeLessThanOrEqual(7681 - 128);
    expect(estimate.recommendation!.vramHeadroomMiB).toBeGreaterThanOrEqual(0);
  });

  test("recommends offloading more when there is headroom", async () => {
    const profile = makeProfile({ backendMode: "cuda", gpuLayers: 10, contextSize: 12288, parallelSlots: 1 });
    const estimate = await estimateProfileMemory(profile, {
      hardware,
      metadata: gemmaMetadata,
      layout: gemmaLayout(),
      fileExists: () => true
    });

    expect(estimate.recommendation).not.toBeNull();
    expect(estimate.recommendation!.gpuLayers).toBeGreaterThan(10);
  });

  test("sliding-window attention caps SWA layer KV at the window size", async () => {
    const profile = makeProfile({ backendMode: "cuda", gpuLayers: 999, contextSize: 12288, parallelSlots: 1 });
    const estimate = await estimateProfileMemory(profile, {
      hardware,
      metadata: gemmaMetadata,
      layout: gemmaLayout(),
      fileExists: () => true
    });

    // Measured on real hardware: 192 MiB full-attention + 360 MiB SWA = 552 MiB.
    expect(estimate.breakdown.kvCacheMiB).toBeGreaterThan(500);
    expect(estimate.breakdown.kvCacheMiB).toBeLessThan(620);
    expect(estimate.confidence).toBe("high");
    expect(estimate.assumptions.some((item) => item.includes("Sliding-window"))).toBe(true);
    // Tied embeddings: offloaded output head duplicates token_embd on GPU.
    expect(estimate.breakdown.gpuModelWeightsMiB).toBeGreaterThan(6900);
  });

  test("hybrid SSM models only cache KV on attention layers", async () => {
    const profile = makeProfile({ backendMode: "cuda", gpuLayers: 999, contextSize: 32768, parallelSlots: 4 });
    const estimate = await estimateProfileMemory(profile, {
      hardware,
      metadata: hybridMetadata,
      layout: hybridLayout(),
      fileExists: () => true
    });

    // Measured on real hardware: 1024 MiB KV (8 layers) + ~200 MiB SSM state.
    expect(estimate.breakdown.kvCacheMiB).toBeGreaterThan(1150);
    expect(estimate.breakdown.kvCacheMiB).toBeLessThan(1350);
    expect(estimate.assumptions.some((item) => item.includes("Hybrid"))).toBe(true);
  });

  test("keeps CPU profiles mostly out of VRAM", async () => {
    const profile = makeProfile({ backendMode: "cpu", gpuLayers: 0 });
    const estimate = await estimateProfileMemory(profile, {
      hardware,
      metadata: denseMetadata,
      fileExists: () => true
    });

    expect(estimate.backend).toBe("CPU");
    expect(estimate.estimatedVramMiB).toBe(0);
    expect(estimate.estimatedSystemRamMiB).toBeGreaterThan(1000);
    expect(estimate.warnings.some((item) => item.includes("CPU profiles"))).toBe(true);
  });
});

describe("appleWorkingSetMiB", () => {
  test("honors an explicit iogpu.wired_limit_mb override", () => {
    expect(appleWorkingSetMiB(32768, 28000)).toBe(28000);
  });

  test("uses ~2/3 of RAM on smaller machines, ~3/4 on larger", () => {
    expect(appleWorkingSetMiB(16384, null)).toBe(Math.floor(16384 * 0.67));
    expect(appleWorkingSetMiB(65536, null)).toBe(Math.floor(65536 * 0.75));
    expect(appleWorkingSetMiB(65536, 0)).toBe(Math.floor(65536 * 0.75)); // 0 = default, not an override
  });
});
