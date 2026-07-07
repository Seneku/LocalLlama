import { availableParallelism } from "node:os";
import path from "node:path";

import { getSettings } from "./settings";
import type { RuntimeConfig } from "../src/shared/types";

export interface RuntimePaths {
  llamaRoot: string;
  cudaServerPath: string;
  cpuServerPath: string;
  cudaBenchPath: string;
  cpuBenchPath: string;
  dataPath: string;
}

const DEFAULT_LLAMA_ROOT = "E:\\Projects\\llama.cpp";

// Precedence: saved settings > environment variable > default derived from llamaRoot.
export function getRuntimePaths(): RuntimePaths {
  const settings = getSettings();
  const llamaRoot = settings.llamaRoot || process.env.LOCALLLAMA_LLAMA_ROOT || DEFAULT_LLAMA_ROOT;
  const dataPath = process.env.LOCALLLAMA_DATA_DIR ?? path.resolve(process.cwd(), "data");

  return {
    llamaRoot,
    cudaServerPath:
      settings.cudaServerPath ||
      process.env.LOCALLLAMA_CUDA_SERVER ||
      path.join(llamaRoot, "dist-cuda", "llama-server.exe"),
    cpuServerPath:
      settings.cpuServerPath ||
      process.env.LOCALLLAMA_CPU_SERVER ||
      path.join(llamaRoot, "build", "bin", "llama-server.exe"),
    cudaBenchPath:
      settings.cudaBenchPath ||
      process.env.LOCALLLAMA_CUDA_BENCH ||
      path.join(llamaRoot, "dist-cuda", "llama-bench.exe"),
    cpuBenchPath:
      settings.cpuBenchPath ||
      process.env.LOCALLLAMA_CPU_BENCH ||
      path.join(llamaRoot, "build", "bin", "llama-bench.exe"),
    dataPath
  };
}

// Where downloaded GGUF models live. Precedence: settings > env > default.
export function getModelsDir(): string {
  const settings = getSettings();
  if (settings.modelsDir) {
    return settings.modelsDir;
  }
  if (process.env.LOCALLLAMA_MODELS_DIR) {
    return process.env.LOCALLLAMA_MODELS_DIR;
  }
  return path.join(getRuntimePaths().dataPath, "models");
}

export function getDefaultThreads(): number {
  try {
    return Math.max(1, availableParallelism());
  } catch {
    return 1;
  }
}

export function createRuntimeConfig(fileExists: (filePath: string) => boolean): RuntimeConfig {
  const paths = getRuntimePaths();

  return {
    ...paths,
    modelsDir: getModelsDir(),
    defaultThreads: getDefaultThreads(),
    detected: {
      cudaServer: fileExists(paths.cudaServerPath),
      cpuServer: fileExists(paths.cpuServerPath),
      cudaBench: fileExists(paths.cudaBenchPath),
      cpuBench: fileExists(paths.cpuBenchPath)
    }
  };
}
