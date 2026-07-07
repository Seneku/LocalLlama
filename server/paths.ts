import { availableParallelism, homedir } from "node:os";
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

// llama.cpp binaries carry a .exe suffix only on Windows.
const EXE = process.platform === "win32" ? ".exe" : "";
const LLAMA_SERVER = `llama-server${EXE}`;
const LLAMA_BENCH = `llama-bench${EXE}`;

// A neutral, cross-platform guess (e.g. C:\Users\you\llama.cpp or ~/llama.cpp).
// Users set their real path in Settings (with the folder picker) on first run.
const DEFAULT_LLAMA_ROOT = path.join(homedir(), "llama.cpp");

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
      path.join(llamaRoot, "dist-cuda", LLAMA_SERVER),
    cpuServerPath:
      settings.cpuServerPath ||
      process.env.LOCALLLAMA_CPU_SERVER ||
      path.join(llamaRoot, "build", "bin", LLAMA_SERVER),
    cudaBenchPath:
      settings.cudaBenchPath ||
      process.env.LOCALLLAMA_CUDA_BENCH ||
      path.join(llamaRoot, "dist-cuda", LLAMA_BENCH),
    cpuBenchPath:
      settings.cpuBenchPath ||
      process.env.LOCALLLAMA_CPU_BENCH ||
      path.join(llamaRoot, "build", "bin", LLAMA_BENCH),
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
