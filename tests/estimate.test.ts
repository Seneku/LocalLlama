import { describe, expect, test } from "bun:test";

import { estimateProfileMemory } from "../server/estimate";
import { defaultProfiles } from "../server/defaultProfiles";
import type { HardwareInfo, ModelMetadata } from "../src/shared/types";

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

const metadata: ModelMetadata = {
  architecture: "qwen35",
  name: "Orinth test",
  parameterSize: "9B",
  fileType: 15,
  fileSizeMiB: 5800,
  blockCount: 36,
  contextLength: 32768,
  embeddingLength: 4096,
  headCount: 32,
  headCountKv: 8,
  keyLength: 128,
  valueLength: 128
};

describe("memory estimates", () => {
  test("estimates CUDA model, KV cache, and headroom", async () => {
    const profile = defaultProfiles.find((item) => item.id === "orinth9b-mtp-coding")!;
    const estimate = await estimateProfileMemory(profile, {
      hardware,
      metadata,
      fileExists: () => true
    });

    expect(estimate.backend).toBe("CUDA");
    expect(estimate.breakdown.gpuModelWeightsMiB).toBe(5800);
    expect(estimate.breakdown.kvCacheMiB).toBeGreaterThan(2000);
    expect(estimate.estimatedVramMiB).toBeGreaterThan(8000);
    expect(estimate.vramHeadroomMiB).not.toBeNull();
    expect(estimate.assumptions.some((item) => item.includes("full model offload"))).toBe(true);
  });

  test("keeps CPU profiles mostly out of VRAM", async () => {
    const profile = {
      ...defaultProfiles[0],
      backendMode: "cpu" as const,
      gpuLayers: 0
    };
    const estimate = await estimateProfileMemory(profile, {
      hardware,
      metadata,
      fileExists: () => true
    });

    expect(estimate.backend).toBe("CPU");
    expect(estimate.estimatedVramMiB).toBe(0);
    expect(estimate.estimatedSystemRamMiB).toBeGreaterThan(1000);
    expect(estimate.warnings.some((item) => item.includes("CPU profiles"))).toBe(true);
  });
});
