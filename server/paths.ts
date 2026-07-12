import fs from "node:fs";
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

// Common layouts under the llama.cpp root. Official release zips extract the
// binaries to the folder ROOT; source builds put them in build/bin; dist-cuda
// / cuda / cpu / dist-cpu are folder-per-build conventions for people who keep
// a GPU and a CPU build side by side. Probe order puts the deliberate
// per-build folders first so they win over a generic root binary; the
// `fallback` entry is what Settings displays when nothing exists yet (root for
// GPU — matching the "extract the release zip" guide — and build/bin for CPU).
const GPU_LAYOUT = { subdirs: [["dist-cuda"], ["cuda"], [], ["build", "bin"]], fallback: [] as string[] };
const CPU_LAYOUT = { subdirs: [["build", "bin"], ["cpu"], ["dist-cpu"], []], fallback: ["build", "bin"] };

/**
 * Resolve a binary under the llama.cpp root by probing the common layouts and
 * returning the first that exists on disk; falls back to the expected location
 * so Settings can show where the file should go. Exported for tests.
 */
export function resolveBinary(
  llamaRoot: string,
  binary: string,
  layout: { subdirs: string[][]; fallback: string[] },
  fileExists: (filePath: string) => boolean
): string {
  const found = layout.subdirs
    .map((parts) => path.join(llamaRoot, ...parts, binary))
    .find(fileExists);
  return found ?? path.join(llamaRoot, ...layout.fallback, binary);
}

// Precedence: saved settings > environment variable > first existing common
// layout under llamaRoot (release-zip root, build/bin, dist-cuda, ...).
export function getRuntimePaths(fileExists: (filePath: string) => boolean = fs.existsSync): RuntimePaths {
  const settings = getSettings();
  const llamaRoot = settings.llamaRoot || process.env.LOCALLLAMA_LLAMA_ROOT || DEFAULT_LLAMA_ROOT;
  const dataPath = process.env.LOCALLLAMA_DATA_DIR ?? path.resolve(process.cwd(), "data");

  return {
    llamaRoot,
    cudaServerPath:
      settings.cudaServerPath ||
      process.env.LOCALLLAMA_CUDA_SERVER ||
      resolveBinary(llamaRoot, LLAMA_SERVER, GPU_LAYOUT, fileExists),
    cpuServerPath:
      settings.cpuServerPath ||
      process.env.LOCALLLAMA_CPU_SERVER ||
      resolveBinary(llamaRoot, LLAMA_SERVER, CPU_LAYOUT, fileExists),
    cudaBenchPath:
      settings.cudaBenchPath ||
      process.env.LOCALLLAMA_CUDA_BENCH ||
      resolveBinary(llamaRoot, LLAMA_BENCH, GPU_LAYOUT, fileExists),
    cpuBenchPath:
      settings.cpuBenchPath ||
      process.env.LOCALLLAMA_CPU_BENCH ||
      resolveBinary(llamaRoot, LLAMA_BENCH, CPU_LAYOUT, fileExists),
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
