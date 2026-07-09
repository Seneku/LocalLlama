import { describe, expect, test } from "bun:test";

import {
  classifyAsset,
  classifyRelease,
  coarseFit,
  normalizeSearch,
  normalizeTree,
  parseParamsB,
  parseQuant,
  recommendedMaxParamsB,
  resolveDownloadUrl
} from "../server/hf";
import type { HardwareInfo } from "../src/shared/types";

function hardware(freeMiB: number | null, totalMiB = 12281): HardwareInfo {
  return {
    totalRamMiB: 65536,
    freeRamMiB: 48000,
    gpus:
      freeMiB === null
        ? []
        : [{ name: "RTX 4070 SUPER", vendor: "nvidia" as const, totalMiB, usedMiB: totalMiB - freeMiB, freeMiB }]
  };
}

describe("classifyAsset", () => {
  test("distinguishes cudart from cuda and the rest", () => {
    expect(classifyAsset("cudart-llama-bin-win-cuda-12.4-x64.zip")).toBe("cudart");
    expect(classifyAsset("llama-b9894-bin-win-cuda-12.4-x64.zip")).toBe("cuda");
    expect(classifyAsset("llama-b9894-bin-win-cpu-x64.zip")).toBe("cpu");
    expect(classifyAsset("llama-b9894-bin-win-vulkan-x64.zip")).toBe("vulkan");
    expect(classifyAsset("llama-b9894-bin-win-hip-radeon-x64.zip")).toBe("hip");
    expect(classifyAsset("llama-b9894-bin-win-sycl-x64.zip")).toBe("sycl");
    expect(classifyAsset("llama-b9894-bin-win-openvino-2026.2.1-x64.zip")).toBe("other");
  });
});

describe("parseQuant", () => {
  test("extracts the quant token from a filename", () => {
    expect(parseQuant("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")).toBe("Q4_K_M");
    expect(parseQuant("model-IQ3_XS.gguf")).toBe("IQ3_XS");
    expect(parseQuant("model.Q8_0.gguf")).toBe("Q8_0");
    expect(parseQuant("model-f16.gguf")).toBe("F16");
    expect(parseQuant("model-BF16.gguf")).toBe("BF16");
    expect(parseQuant("some-model-name.gguf")).toBeNull();
  });
});

describe("coarseFit", () => {
  test("measures against total VRAM, not what is free right now", () => {
    // need ~= sizeMiB * 1.08 + 768; usable = total - 1024.
    // Free VRAM is only 6000 MiB but a 12 GB card still fits a 5 GB model.
    expect(coarseFit(5000, hardware(6000, 12000))).toBe("fits"); // 6168 / 10976 = 0.56
    expect(coarseFit(8064, hardware(6000, 11000))).toBe("tight"); // 9477 / 9976 = 0.95
    expect(coarseFit(12000, hardware(6000, 11000))).toBe("over"); // 13728 / 9976 = 1.38
    expect(coarseFit(5000, hardware(null))).toBe("unknown");
  });

  test("a 9B Q4 (~5.2 GB) fits a 12 GB card", () => {
    expect(coarseFit(5366, hardware(6890, 12281))).toBe("fits");
  });
});

describe("normalizeSearch", () => {
  test("maps HF model objects and derives author from id", () => {
    const results = normalizeSearch([
      { id: "bartowski/Model-GGUF", downloads: 1000, likes: 12, pipeline_tag: "text-generation" },
      { modelId: "org/Other-GGUF", gated: "manual", author: "org", downloads: 5, likes: 0 }
    ]);
    expect(results).toHaveLength(2);
    expect(results[0]).toMatchObject({ id: "bartowski/Model-GGUF", author: "bartowski", downloads: 1000, gated: false });
    expect(results[1]).toMatchObject({ id: "org/Other-GGUF", author: "org", gated: true });
  });

  test("tolerates non-array input", () => {
    expect(normalizeSearch(null)).toEqual([]);
  });
});

describe("normalizeTree", () => {
  test("keeps only .gguf files, computes fit, and sorts by size", () => {
    const files = normalizeTree(
      [
        { type: "file", path: "README.md", size: 1000 },
        { type: "file", path: "model-Q8_0.gguf", size: 12_000_000_000 },
        { type: "file", path: "model-Q4_K_M.gguf", size: 4_000_000_000 },
        { type: "directory", path: "sub" }
      ],
      hardware(11000)
    );
    expect(files.map((file) => file.filename)).toEqual(["model-Q4_K_M.gguf", "model-Q8_0.gguf"]);
    expect(files[0].quant).toBe("Q4_K_M");
    expect(files[0].sizeMiB).toBeGreaterThan(3500);
    expect(files[0].fit).toBe("fits");
    expect(files[1].fit).toBe("over");
  });
});

describe("classifyRelease", () => {
  test("filters to win assets and classifies kinds", () => {
    const release = classifyRelease({
      tag_name: "b9894",
      html_url: "https://github.com/ggml-org/llama.cpp/releases/tag/b9894",
      assets: [
        { name: "llama-b9894-bin-win-cuda-12.4-x64.zip", size: 100, browser_download_url: "https://x/cuda.zip" },
        { name: "cudart-llama-bin-win-cuda-12.4-x64.zip", size: 200, browser_download_url: "https://x/cudart.zip" },
        { name: "llama-b9894-bin-macos-arm64.tar.gz", size: 50, browser_download_url: "https://x/mac.tgz" }
      ]
    });
    expect(release.tag).toBe("b9894");
    expect(release.winAssets).toHaveLength(2);
    expect(release.winAssets.map((asset) => asset.kind).sort()).toEqual(["cuda", "cudart"]);
  });
});

describe("parseParamsB", () => {
  test("extracts the largest billions-of-params token", () => {
    expect(parseParamsB("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")).toBe(8);
    expect(parseParamsB("unsloth/gemma-4-12b-it-GGUF")).toBe(12);
    expect(parseParamsB("Qwen/Qwen3-30B-A3B-GGUF")).toBe(30); // total, not active
    expect(parseParamsB("org/TinyModel-0.5B-GGUF")).toBe(0.5);
    expect(parseParamsB("org/SmolLM2-135M-GGUF")).toBeNull(); // M, not B
    expect(parseParamsB("org/some-random-model-GGUF")).toBeNull();
  });
});

describe("recommendedMaxParamsB", () => {
  test("scales with total VRAM (12 GB card -> ~16B)", () => {
    expect(recommendedMaxParamsB(hardware(6000, 12281))).toBe(16);
    expect(recommendedMaxParamsB(hardware(3000, 8192))).toBeLessThan(12);
    expect(recommendedMaxParamsB(hardware(20000, 24576))).toBeGreaterThan(30);
  });

  test("falls back to system RAM with no GPU", () => {
    const cpuOnly = { totalRamMiB: 32768, freeRamMiB: 16000, gpus: [] };
    expect(recommendedMaxParamsB(cpuOnly)).toBeGreaterThan(20);
  });
});

describe("resolveDownloadUrl", () => {
  test("builds the HF resolve URL", () => {
    expect(resolveDownloadUrl("org/repo-GGUF", "model-Q4_K_M.gguf")).toBe(
      "https://huggingface.co/org/repo-GGUF/resolve/main/model-Q4_K_M.gguf"
    );
  });
});
