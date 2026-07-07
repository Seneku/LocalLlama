import os from "node:os";

// Apple Silicon (arm64 macOS) has a Metal GPU backed by unified memory — llama.cpp
// offloads to it with -ngl just like CUDA, but there is no separate VRAM pool.
// Intel Macs fall through to the CPU path (no unified-memory model here).
export function isAppleSilicon(): boolean {
  return process.platform === "darwin" && os.arch() === "arm64";
}
