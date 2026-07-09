// Pre-download fit: run the calibrated VRAM estimator against a remote GGUF
// by reading only its header over HTTP Range (ggufRemote.ts). A small
// concurrency gate keeps the model browser from firing dozens of ranged
// requests at Hugging Face at once.
import { estimateProfileMemory } from "./estimate";
import { readRemoteGgufInfo, type FetchLike } from "./ggufRemote";
import { listModelFiles } from "./hf";
import { normalizeProfile } from "./normalize";
import type { HardwareInfo, RemoteModelEstimate } from "../src/shared/types";

const MAX_CONCURRENT = 2;
let active = 0;
const waiters: Array<() => void> = [];

function acquire(): Promise<void> {
  if (active < MAX_CONCURRENT) {
    active += 1;
    return Promise.resolve();
  }
  return new Promise<void>((resolve) => waiters.push(resolve)).then(() => {
    active += 1;
  });
}

function release(): void {
  active -= 1;
  waiters.shift()?.();
}

const SPLIT_PATTERN = /^(.*)-(\d{5})-of-(\d{5})\.gguf$/iu;

export interface RemoteEstimateOptions {
  fetchImpl?: FetchLike;
  /** Injected for tests; defaults to the real HF tree listing (split models only). */
  listFiles?: typeof listModelFiles;
}

export async function estimateRemoteModel(
  id: string,
  filename: string,
  contextSize: number,
  sizeBytes: number,
  hardware: HardwareInfo,
  options: RemoteEstimateOptions = {}
): Promise<RemoteModelEstimate> {
  await acquire();
  try {
    const warnings: string[] = [];
    let headerFilename = filename;
    let headerSizeBytes = sizeBytes;
    let totalBytes = sizeBytes;
    const split = SPLIT_PATTERN.exec(filename);
    if (split) {
      // Multi-part GGUF: the first shard holds the metadata; weights span all
      // shards, so per-layer tensor math would be misleading — sum the sizes
      // and let the estimator use its metadata heuristics.
      headerFilename = `${split[1]}-00001-of-${split[3]}.gguf`;
      const listFiles = options.listFiles ?? listModelFiles;
      const files = await listFiles(id, hardware);
      const shards = files.filter((file) => {
        const match = SPLIT_PATTERN.exec(file.filename);
        return match !== null && match[1] === split[1] && match[3] === split[3];
      });
      totalBytes = shards.reduce((sum, file) => sum + file.sizeBytes, 0) || sizeBytes;
      headerSizeBytes = shards.find((file) => file.filename === headerFilename)?.sizeBytes ?? sizeBytes;
      warnings.push("Multi-part model: sizes are summed across shards and the estimate uses metadata heuristics.");
    }

    const info = await readRemoteGgufInfo(id, headerFilename, headerSizeBytes, options.fetchImpl);
    const metadata = { ...info.metadata, fileSizeMiB: totalBytes / 1024 / 1024 };
    const layout = split ? null : info.layout;

    // Synthetic profile at full offload: the estimate's own recommendation
    // then says how many layers actually fit free VRAM.
    const profile = normalizeProfile({
      id: "remote-estimate",
      name: id,
      modelPath: filename,
      contextSize,
      gpuLayers: 999
    });
    const estimateOptions = { metadata, layout, hardware, fileExists: () => false };
    const base = await estimateProfileMemory(profile, estimateOptions);

    const isMoe = layout?.layers.some((layer) => layer.expertBytes > 0) ?? false;
    let cpuMoe: RemoteModelEstimate["cpuMoe"] = null;
    if (isMoe) {
      const moeEstimate = await estimateProfileMemory({ ...profile, cpuMoe: true }, estimateOptions);
      cpuMoe = {
        fit: moeEstimate.fit,
        estimatedVramMiB: moeEstimate.estimatedVramMiB,
        estimatedSystemRamMiB: moeEstimate.estimatedSystemRamMiB
      };
    }

    return {
      id,
      filename,
      contextSize: profile.contextSize,
      confidence: base.confidence,
      fit: base.fit,
      estimatedVramMiB: base.estimatedVramMiB,
      estimatedSystemRamMiB: base.estimatedSystemRamMiB,
      recommendation: base.recommendation,
      cpuMoe,
      maxGpuLayers: metadata.blockCount ? metadata.blockCount + 1 : null,
      split: split !== null,
      warnings: [...warnings, ...base.warnings]
    };
  } finally {
    release();
  }
}
