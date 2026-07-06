import { execFile } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import { promisify } from "node:util";

import { readGgufMetadata } from "./gguf";
import type {
  GpuInfo,
  HardwareInfo,
  KvCacheType,
  LlamaProfile,
  MemoryEstimate,
  ModelMetadata,
  ResolvedBackend
} from "../src/shared/types";

const execFileAsync = promisify(execFile);

const EMPTY_MODEL: ModelMetadata = {
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
  valueLength: null
};

export interface EstimateOptions {
  hardware?: HardwareInfo;
  metadata?: ModelMetadata;
  fileExists?: (filePath: string) => boolean;
}

function bytesToMiB(bytes: number): number {
  return bytes / 1024 / 1024;
}

function roundMiB(value: number): number {
  return Math.max(0, Math.round(value));
}

function roundSignedMiB(value: number): number {
  return Math.round(value);
}

function positiveInteger(value: number, fallback: number): number {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return Math.max(1, Math.floor(value));
}

function cacheTypeBytes(type: KvCacheType): number {
  switch (type || "f16") {
    case "q8_0":
      return 1.0625;
    case "q4_0":
      return 0.5625;
    case "q4_1":
      return 0.625;
    case "f16":
    default:
      return 2;
  }
}

function resolveBackend(profile: LlamaProfile, hardware: HardwareInfo): ResolvedBackend {
  if (profile.backendMode === "cuda") {
    return "CUDA";
  }
  if (profile.backendMode === "cpu") {
    return "CPU";
  }
  return hardware.gpus.length > 0 ? "CUDA" : "CPU";
}

function offloadRatio(profile: LlamaProfile, model: ModelMetadata, backend: ResolvedBackend): number {
  if (backend !== "CUDA" || profile.gpuLayers <= 0) {
    return 0;
  }
  const layerCount = model.blockCount ?? 0;
  if (layerCount <= 0) {
    return profile.gpuLayers >= 99 ? 1 : 0.75;
  }
  if (profile.gpuLayers >= layerCount || profile.gpuLayers >= 99) {
    return 1;
  }

  const nonLayerFraction = 0.12;
  const layerFraction = 1 - nonLayerFraction;
  return Math.min(1, nonLayerFraction + layerFraction * (profile.gpuLayers / layerCount));
}

function kvCacheMiB(profile: LlamaProfile, model: ModelMetadata, backend: ResolvedBackend): number {
  if (backend !== "CUDA" || profile.gpuLayers <= 0) {
    return 0;
  }

  const layerCount = model.blockCount ?? 0;
  const embeddingLength = model.embeddingLength ?? 0;
  const headCount = model.headCount ?? 0;
  const headCountKv = model.headCountKv ?? headCount;
  if (layerCount <= 0 || embeddingLength <= 0 || headCount <= 0 || headCountKv <= 0) {
    return profile.contextSize * layerCount * 0.25 / 1024;
  }

  const headDim = embeddingLength / headCount;
  const keyLength = model.keyLength ?? headDim;
  const valueLength = model.valueLength ?? headDim;
  const keyElements = headCountKv * keyLength;
  const valueElements = headCountKv * valueLength;
  const keyBytes = cacheTypeBytes(profile.kvCacheK);
  const valueBytes = cacheTypeBytes(profile.kvCacheV);
  const bytes = profile.contextSize * layerCount * (keyElements * keyBytes + valueElements * valueBytes);
  return bytesToMiB(bytes);
}

function computeOverheadMiB(profile: LlamaProfile, model: ModelMetadata, backend: ResolvedBackend): number {
  if (backend !== "CUDA" || profile.gpuLayers <= 0) {
    return 0;
  }
  const contextFactor = Math.sqrt(positiveInteger(profile.contextSize, 4096) / 4096);
  const modelFactor = Math.sqrt(Math.max(model.fileSizeMiB, 1024) / 1024);
  const parallelFactor = Math.max(1, profile.parallelSlots);
  return 384 + 96 * contextFactor + 64 * modelFactor + 64 * (parallelFactor - 1);
}

function estimateDraftMiB(profile: LlamaProfile, backend: ResolvedBackend, fileExists: (filePath: string) => boolean): number {
  if (backend !== "CUDA" || !profile.speculative.enabled || !profile.speculative.draftModelPath.trim()) {
    return 0;
  }
  try {
    if (!fileExists(profile.speculative.draftModelPath)) {
      return 0;
    }
    return bytesToMiB(fs.statSync(profile.speculative.draftModelPath).size);
  } catch {
    return 0;
  }
}

function fitStatus(estimatedVramMiB: number, availableVramMiB: number | null, totalVramMiB: number | null) {
  const capacity = availableVramMiB ?? totalVramMiB;
  if (!capacity) {
    return "unknown" as const;
  }
  const ratio = estimatedVramMiB / capacity;
  if (ratio <= 0.82) {
    return "fits" as const;
  }
  if (ratio <= 1) {
    return "tight" as const;
  }
  return "over" as const;
}

const HARDWARE_CACHE_TTL_MS = 3000;
let hardwareCache: { info: HardwareInfo; at: number } | null = null;

export async function getHardwareInfo(): Promise<HardwareInfo> {
  const now = Date.now();
  if (hardwareCache && now - hardwareCache.at < HARDWARE_CACHE_TTL_MS) {
    return hardwareCache.info;
  }

  const gpus: GpuInfo[] = [];
  try {
    const { stdout } = await execFileAsync("nvidia-smi.exe", [
      "--query-gpu=name,memory.total,memory.used,memory.free",
      "--format=csv,noheader,nounits"
    ]);
    for (const line of stdout.trim().split(/\r?\n/u)) {
      const [name, total, used, free] = line.split(",").map((part) => part.trim());
      if (!name) {
        continue;
      }
      gpus.push({
        name,
        totalMiB: Number.isFinite(Number(total)) ? Number(total) : null,
        usedMiB: Number.isFinite(Number(used)) ? Number(used) : null,
        freeMiB: Number.isFinite(Number(free)) ? Number(free) : null
      });
    }
  } catch {
    // Non-NVIDIA systems still get system RAM estimates.
  }

  const info: HardwareInfo = {
    totalRamMiB: bytesToMiB(os.totalmem()),
    freeRamMiB: bytesToMiB(os.freemem()),
    gpus
  };
  hardwareCache = { info, at: now };
  return info;
}

export async function estimateProfileMemory(
  profile: LlamaProfile,
  options: EstimateOptions = {}
): Promise<MemoryEstimate> {
  const fileExists = options.fileExists ?? fs.existsSync;
  const hardware = options.hardware ?? (await getHardwareInfo());
  const warnings: string[] = [];
  const assumptions: string[] = [];
  let metadata = options.metadata ?? null;

  if (!metadata) {
    try {
      metadata = await readGgufMetadata(profile.modelPath);
    } catch (error) {
      const fileSizeMiB = fileExists(profile.modelPath) ? bytesToMiB(fs.statSync(profile.modelPath).size) : 0;
      metadata = { ...EMPTY_MODEL, fileSizeMiB };
      warnings.push(error instanceof Error ? error.message : "Unable to read GGUF metadata.");
    }
  }

  const backend = resolveBackend(profile, hardware);
  const primaryGpu = hardware.gpus[0] ?? null;
  const ratio = offloadRatio(profile, metadata, backend);
  const modelWeightsMiB = metadata.fileSizeMiB;
  const gpuModelWeightsMiB = modelWeightsMiB * ratio;
  const cpuModelWeightsMiB = modelWeightsMiB * (1 - ratio);
  const kvMiB = kvCacheMiB(profile, metadata, backend);
  const overheadMiB = computeOverheadMiB(profile, metadata, backend);
  const draftMiB = estimateDraftMiB(profile, backend, fileExists);
  const preMarginVram = gpuModelWeightsMiB + kvMiB + overheadMiB + draftMiB;
  const safetyMarginMiB = backend === "CUDA" && profile.gpuLayers > 0 ? Math.max(512, preMarginVram * 0.08) : 0;
  const estimatedVramMiB = preMarginVram + safetyMarginMiB;
  const estimatedSystemRamMiB =
    cpuModelWeightsMiB + (profile.mlock ? modelWeightsMiB * 0.35 : modelWeightsMiB * 0.12) + 512;

  assumptions.push("KV cache is estimated as GPU-resident for CUDA profiles with GPU layers enabled.");
  assumptions.push("Runtime overhead includes graph/workspace allocation and an 8% or 512 MiB safety margin.");
  if (profile.gpuLayers >= 99) {
    assumptions.push("GPU layers 99+ is treated as full model offload.");
  }
  if (profile.speculative.enabled && !profile.speculative.draftModelPath.trim()) {
    assumptions.push("Bundled MTP heads are treated as part of the main GGUF file.");
  }
  if (!metadata.blockCount || !metadata.embeddingLength || !metadata.headCount) {
    warnings.push("Model metadata is incomplete, so KV/offload estimates use conservative fallbacks.");
  }
  if (profile.contextSize > (metadata.contextLength ?? Number.POSITIVE_INFINITY)) {
    warnings.push("Profile context size is above the model metadata context length.");
  }
  if (backend === "CPU") {
    warnings.push("CPU profiles are estimated to use little VRAM, but they still need enough system RAM for model residency.");
  }

  const availableVramMiB = primaryGpu?.freeMiB ?? null;
  const totalVramMiB = primaryGpu?.totalMiB ?? null;
  const confidence = metadata.blockCount && metadata.embeddingLength && metadata.headCount ? "medium" : "low";

  return {
    backend,
    fit: fitStatus(estimatedVramMiB, availableVramMiB, totalVramMiB),
    confidence,
    totalVramMiB,
    availableVramMiB,
    estimatedVramMiB: roundMiB(estimatedVramMiB),
    estimatedSystemRamMiB: roundMiB(estimatedSystemRamMiB),
    vramHeadroomMiB: availableVramMiB === null ? null : roundSignedMiB(availableVramMiB - estimatedVramMiB),
    model: {
      ...metadata,
      fileSizeMiB: roundMiB(metadata.fileSizeMiB)
    },
    hardware: {
      totalRamMiB: roundMiB(hardware.totalRamMiB),
      freeRamMiB: roundMiB(hardware.freeRamMiB),
      gpus: hardware.gpus
    },
    breakdown: {
      modelWeightsMiB: roundMiB(modelWeightsMiB),
      gpuModelWeightsMiB: roundMiB(gpuModelWeightsMiB),
      cpuModelWeightsMiB: roundMiB(cpuModelWeightsMiB),
      kvCacheMiB: roundMiB(kvMiB),
      computeOverheadMiB: roundMiB(overheadMiB),
      draftModelMiB: roundMiB(draftMiB),
      safetyMarginMiB: roundMiB(safetyMarginMiB)
    },
    assumptions,
    warnings
  };
}
