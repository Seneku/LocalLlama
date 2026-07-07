import { execFile } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import { promisify } from "node:util";

import { readGgufModelInfo, type GgufTensorLayout } from "./gguf";
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

// Calibrated against llama.cpp (build 2026-06) on RTX 4070 SUPER:
// - SWA caches allocate n_seq * (window + pad) cells, pad measured at 128.
// - A CUDA-backend process holds ~200 MiB of driver/context memory beyond
//   the buffers llama.cpp itself reports.
// - The output layer is offloaded first, then the top (ngl - 1) blocks.
const SWA_CELL_PAD = 128;
const CUDA_CONTEXT_MIB = 200;
const HOST_COMPUTE_MIB = 40;

// Interleave patterns for SWA architectures whose layers share a single KV
// geometry (every Nth layer is full-attention). Gemma4-style models encode
// distinct SWA tensor shapes instead, which we detect directly.
const SWA_PATTERNS: Record<string, number> = {
  gemma2: 2,
  gemma3: 6,
  gemma3n: 6,
  gemma4: 6,
  cohere2: 4,
  llama4: 4
};

export interface EstimateOptions {
  hardware?: HardwareInfo;
  metadata?: ModelMetadata;
  layout?: GgufTensorLayout | null;
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

interface LayerModel {
  /** Attention layers: bytes stored per cached token (already scaled by KV quant type). */
  kvBytesPerCell: number;
  /** Recurrent layers: constant state bytes per parallel sequence. */
  stateBytesPerSeq: number;
  kind: "full" | "swa" | "recurrent";
  weightBytes: number;
  offloaded: boolean;
}

interface ModelPlan {
  layers: LayerModel[];
  gpuWeightsMiB: number;
  cpuWeightsMiB: number;
  fromTensorLayout: boolean;
  notes: string[];
}

function recurrentStateBytes(metadata: ModelMetadata): number {
  const state = metadata.ssmStateSize ?? 128;
  const inner = metadata.ssmInnerSize ?? metadata.embeddingLength ?? 4096;
  const conv = metadata.ssmConvKernel ?? 4;
  const groups = metadata.ssmGroupCount ?? 1;
  // f32 SSM state + conv tail, per sequence.
  return (state * inner + (conv - 1) * (inner + 2 * groups * state)) * 4;
}

function classifySwa(
  attnIndices: number[],
  kvGeometry: Map<number, { k: number; v: number }>,
  metadata: ModelMetadata,
  notes: string[]
): Set<number> {
  const swaLayers = new Set<number>();
  if (!metadata.slidingWindow || attnIndices.length === 0) {
    return swaLayers;
  }

  // Distinct SWA tensor shapes (gemma4 style): the minority geometry group is
  // the full-attention set, everything else slides.
  const groups = new Map<string, number[]>();
  for (const index of attnIndices) {
    const geometry = kvGeometry.get(index);
    const key = geometry ? `${geometry.k}:${geometry.v}` : "unknown";
    const group = groups.get(key) ?? [];
    group.push(index);
    groups.set(key, group);
  }
  if (groups.size > 1) {
    const sorted = [...groups.values()].sort((a, b) => a.length - b.length);
    for (const group of sorted.slice(1)) {
      for (const index of group) {
        swaLayers.add(index);
      }
    }
    notes.push(
      `Sliding-window attention detected from tensor shapes: ${swaLayers.size} of ${attnIndices.length} layers cache only ~${metadata.slidingWindow} tokens.`
    );
    return swaLayers;
  }

  // Uniform geometry: fall back to the arch interleave pattern.
  const pattern =
    metadata.slidingWindowPattern ??
    (metadata.architecture ? SWA_PATTERNS[metadata.architecture] : undefined) ??
    (metadata.architecture?.startsWith("gemma") ? 6 : null);
  if (!pattern || pattern <= 1) {
    notes.push("Model reports a sliding window but the layer pattern is unknown; KV cache is estimated as full-context (may overestimate).");
    return swaLayers;
  }
  attnIndices.forEach((index, position) => {
    if ((position + 1) % pattern !== 0) {
      swaLayers.add(index);
    }
  });
  notes.push(
    `Sliding-window attention assumed for ${swaLayers.size} of ${attnIndices.length} layers (1-in-${pattern} full-attention pattern).`
  );
  return swaLayers;
}

function buildModelPlan(
  profile: LlamaProfile,
  metadata: ModelMetadata,
  layout: GgufTensorLayout | null,
  backend: ResolvedBackend
): ModelPlan {
  const notes: string[] = [];
  const keyBytes = cacheTypeBytes(profile.kvCacheK);
  const valueBytes = cacheTypeBytes(profile.kvCacheV);
  const fileBytes = metadata.fileSizeMiB * 1024 * 1024;
  const gpuLayers = backend === "CUDA" ? Math.max(0, profile.gpuLayers) : 0;

  interface BlockInfo {
    index: number;
    bytes: number;
    kElements: number | null;
    vElements: number | null;
    recurrent: boolean;
  }

  let blocks: BlockInfo[];
  let tokenEmbdBytes: number;
  let outputBytes: number;
  let otherBytes: number;
  let trailingBytes = 0; // MTP/nextn blocks past block_count stay on CPU
  const fromTensorLayout = Boolean(layout && layout.layers.length > 0);

  if (layout && layout.layers.length > 0) {
    const blockCount = metadata.blockCount ?? layout.layers.length;
    blocks = layout.layers
      .filter((layer) => layer.index < blockCount)
      .map((layer) => ({
        index: layer.index,
        bytes: layer.bytes,
        kElements: layer.attnKElements,
        vElements: layer.attnVElements ?? layer.attnKElements,
        recurrent: layer.recurrent
      }));
    trailingBytes = layout.layers
      .filter((layer) => layer.index >= blockCount)
      .reduce((sum, layer) => sum + layer.bytes, 0);
    tokenEmbdBytes = layout.tokenEmbdBytes;
    outputBytes = layout.outputBytes;
    otherBytes = layout.otherBytes;
  } else {
    // Metadata-only fallback: uniform blocks, ~10% of the file assumed to be
    // embeddings/output that stay CPU-side.
    const blockCount = Math.max(1, metadata.blockCount ?? 32);
    const headCount = metadata.headCount ?? 32;
    const headCountKv = metadata.headCountKv ?? headCount;
    const headDim =
      metadata.keyLength ?? (metadata.embeddingLength && metadata.headCount ? metadata.embeddingLength / metadata.headCount : 128);
    const kElements = headCountKv * headDim;
    const vElements = headCountKv * (metadata.valueLength ?? headDim);
    const interval = metadata.fullAttentionInterval;
    const layerBytes = (fileBytes * 0.9) / blockCount;
    blocks = Array.from({ length: blockCount }, (_, index) => ({
      index,
      bytes: layerBytes,
      kElements,
      vElements,
      recurrent: interval ? (index + 1) % interval !== 0 : false
    }));
    tokenEmbdBytes = fileBytes * 0.1;
    outputBytes = 0;
    otherBytes = 0;
    if (!metadata.headCountKv && !metadata.slidingWindow) {
      notes.push("KV head count missing from metadata; assuming one KV head per attention head (may overestimate).");
    }
  }

  // MTP/nextn prediction blocks sit inside block_count but allocate no cache.
  const cacheBlockLimit = (metadata.blockCount ?? blocks.length) - (metadata.nextnPredictLayers ?? 0);

  // Hybrid SSM models flag attention layers via full_attention_interval when
  // tensor names are unavailable.
  const attnIndices = blocks
    .filter((block) => !block.recurrent && (block.kElements ?? 0) !== 0 && block.index < cacheBlockLimit)
    .map((block) => block.index);
  const kvGeometry = new Map<number, { k: number; v: number }>();
  for (const block of blocks) {
    if (block.kElements) {
      kvGeometry.set(block.index, { k: block.kElements, v: block.vElements ?? block.kElements });
    }
  }
  const swaLayers = classifySwa(attnIndices, kvGeometry, metadata, notes);
  if (blocks.some((block) => block.recurrent)) {
    const recurrentCount = blocks.filter((block) => block.recurrent).length;
    notes.push(`Hybrid model: ${recurrentCount} of ${blocks.length} layers are recurrent (constant-size state, no KV cache).`);
  }

  // llama.cpp (2026 builds) offloads the output layer first, then the top
  // (ngl - 1) blocks; token embeddings always stay host-side.
  const entities = blocks.length + 1;
  const offloadCount = Math.min(gpuLayers, entities);
  const outputOffloaded = offloadCount >= 1;
  const offloadedBlockCount = Math.max(0, offloadCount - 1);
  const offloadThreshold = blocks.length - offloadedBlockCount;

  const stateBytes = recurrentStateBytes(metadata);
  const layers: LayerModel[] = blocks.map((block) => {
    const offloaded = gpuLayers > 0 && block.index >= offloadThreshold;
    if (block.index >= cacheBlockLimit) {
      // MTP/nextn block: weights load, but no KV cache or recurrent state.
      return { kind: "recurrent" as const, kvBytesPerCell: 0, stateBytesPerSeq: 0, weightBytes: block.bytes, offloaded };
    }
    if (block.recurrent || !block.kElements) {
      return { kind: "recurrent" as const, kvBytesPerCell: 0, stateBytesPerSeq: stateBytes, weightBytes: block.bytes, offloaded };
    }
    const kind = swaLayers.has(block.index) ? ("swa" as const) : ("full" as const);
    const kvBytesPerCell = block.kElements * keyBytes + (block.vElements ?? block.kElements) * valueBytes;
    return { kind, kvBytesPerCell, stateBytesPerSeq: 0, weightBytes: block.bytes, offloaded };
  });

  // Tied-embedding models have no separate output.weight; llama.cpp
  // materialises the output head from token_embd on the GPU when offloaded.
  const outputHeadBytes = outputBytes > 0 ? outputBytes : tokenEmbdBytes;
  let gpuWeights = layers.filter((layer) => layer.offloaded).reduce((sum, layer) => sum + layer.weightBytes, 0);
  if (gpuLayers > 0) {
    gpuWeights += (outputOffloaded ? outputHeadBytes : 0) + otherBytes;
  }
  // Token embeddings always stay host-side (mmapped), even at full offload.
  const cpuWeights =
    layers.filter((layer) => !layer.offloaded).reduce((sum, layer) => sum + layer.weightBytes, 0) +
    (fromTensorLayout ? tokenEmbdBytes + trailingBytes : Math.max(0, fileBytes * 0.1)) +
    (gpuLayers > 0 ? 0 : outputBytes + otherBytes);

  return {
    layers,
    gpuWeightsMiB: bytesToMiB(gpuWeights),
    cpuWeightsMiB: bytesToMiB(cpuWeights),
    fromTensorLayout,
    notes
  };
}

interface KvEstimate {
  gpuMiB: number;
  cpuMiB: number;
  totalMiB: number;
}

function estimateKv(profile: LlamaProfile, metadata: ModelMetadata, plan: ModelPlan): KvEstimate {
  const ctx = Math.max(0, profile.contextSize);
  const slots = Math.max(1, profile.parallelSlots);
  const window = metadata.slidingWindow ?? 0;
  const swaCells = window > 0 ? Math.min(ctx, slots * (window + SWA_CELL_PAD)) : ctx;

  let gpu = 0;
  let cpu = 0;
  for (const layer of plan.layers) {
    let bytes = 0;
    if (layer.kind === "recurrent") {
      bytes = layer.stateBytesPerSeq * slots;
    } else {
      const cells = layer.kind === "swa" ? swaCells : ctx;
      bytes = layer.kvBytesPerCell * cells;
    }
    if (layer.offloaded) {
      gpu += bytes;
    } else {
      cpu += bytes;
    }
  }
  return { gpuMiB: bytesToMiB(gpu), cpuMiB: bytesToMiB(cpu), totalMiB: bytesToMiB(gpu + cpu) };
}

function computeOverheadMiB(profile: LlamaProfile, metadata: ModelMetadata, backend: ResolvedBackend): number {
  if (backend !== "CUDA" || profile.gpuLayers <= 0) {
    return 0;
  }
  const embedding = metadata.embeddingLength ?? 4096;
  const slots = Math.max(1, profile.parallelSlots);
  // CUDA driver context + scheduler compute buffer, calibrated against
  // measured llama-server allocations (96-136 MiB compute at 4k-32k ctx).
  const computeBuffer = 80 + embedding * 0.013 + profile.contextSize * 0.0006 + (slots - 1) * 8;
  return CUDA_CONTEXT_MIB + computeBuffer;
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

function vramForGpuLayers(
  profile: LlamaProfile,
  metadata: ModelMetadata,
  layout: GgufTensorLayout | null,
  gpuLayers: number,
  draftMiB: number
): number {
  const candidate = { ...profile, gpuLayers };
  const plan = buildModelPlan(candidate, metadata, layout, "CUDA");
  const kv = estimateKv(candidate, metadata, plan);
  const overhead = computeOverheadMiB(candidate, metadata, "CUDA");
  const preMargin = plan.gpuWeightsMiB + kv.gpuMiB + overhead + draftMiB;
  return gpuLayers > 0 ? preMargin + Math.max(384, preMargin * 0.04) : 0;
}

// Invert the estimator: find the largest gpu-layer count whose estimated VRAM
// fits the currently free VRAM (with a small extra reserve). VRAM is monotonic
// in the layer count, so a top-down scan finds the answer in <= blocks+2 steps.
function recommendGpuLayers(
  profile: LlamaProfile,
  metadata: ModelMetadata,
  layout: GgufTensorLayout | null,
  backend: ResolvedBackend,
  availableVramMiB: number | null,
  draftMiB: number
) {
  if (backend !== "CUDA" || !availableVramMiB || availableVramMiB <= 0) {
    return null;
  }
  const blockCount = metadata.blockCount ?? layout?.layers.length ?? 0;
  if (blockCount <= 0 || metadata.fileSizeMiB <= 0) {
    return null;
  }
  const maxNgl = blockCount + 1; // blocks + output layer
  const budget = availableVramMiB - 128;

  for (let ngl = maxNgl; ngl >= 0; ngl -= 1) {
    const vram = vramForGpuLayers(profile, metadata, layout, ngl, draftMiB);
    if (vram > budget) {
      continue;
    }
    const effectiveCurrent = Math.min(Math.max(0, profile.gpuLayers), maxNgl);
    if (ngl === effectiveCurrent) {
      return null; // already optimal
    }
    return {
      gpuLayers: ngl,
      estimatedVramMiB: roundMiB(vram),
      vramHeadroomMiB: roundSignedMiB(availableVramMiB - vram),
      fullOffload: ngl >= maxNgl
    };
  }
  return null;
}

function fitStatus(estimatedVramMiB: number, availableVramMiB: number | null, totalVramMiB: number | null) {
  const capacity = availableVramMiB ?? totalVramMiB;
  if (!capacity) {
    return "unknown" as const;
  }
  const ratio = estimatedVramMiB / capacity;
  if (ratio <= 0.9) {
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
    // nvidia-smi is `nvidia-smi.exe` on Windows, `nvidia-smi` elsewhere. On
    // machines without an NVIDIA GPU (incl. Apple Silicon) this throws and we
    // fall back to system-RAM-only estimates.
    const nvidiaSmi = process.platform === "win32" ? "nvidia-smi.exe" : "nvidia-smi";
    const { stdout } = await execFileAsync(nvidiaSmi, [
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
  let layout = options.layout ?? null;

  if (!metadata) {
    try {
      const info = await readGgufModelInfo(profile.modelPath);
      metadata = info.metadata;
      layout = info.layout;
    } catch (error) {
      const fileSizeMiB = fileExists(profile.modelPath) ? bytesToMiB(fs.statSync(profile.modelPath).size) : 0;
      metadata = { ...EMPTY_MODEL, fileSizeMiB };
      warnings.push(error instanceof Error ? error.message : "Unable to read GGUF metadata.");
    }
  }

  const backend = resolveBackend(profile, hardware);
  const primaryGpu = hardware.gpus[0] ?? null;
  const plan = buildModelPlan(profile, metadata, layout, backend);
  const kv = estimateKv(profile, metadata, plan);
  const overheadMiB = computeOverheadMiB(profile, metadata, backend);
  const draftMiB = estimateDraftMiB(profile, backend, fileExists);

  const gpuActive = backend === "CUDA" && profile.gpuLayers > 0;
  const preMarginVram = gpuActive ? plan.gpuWeightsMiB + kv.gpuMiB + overheadMiB + draftMiB : 0;
  const safetyMarginMiB = gpuActive ? Math.max(384, preMarginVram * 0.04) : 0;
  const estimatedVramMiB = preMarginVram + safetyMarginMiB;

  const baseRam = plan.cpuWeightsMiB + kv.cpuMiB + HOST_COMPUTE_MIB + 300;
  const estimatedSystemRamMiB = profile.mlock
    ? Math.max(baseRam, metadata.fileSizeMiB + kv.cpuMiB + 300)
    : baseRam;

  assumptions.push(...plan.notes);
  if (plan.fromTensorLayout) {
    assumptions.push("Weights and KV sizes are computed from the GGUF tensor table (exact per-layer sizes).");
  } else {
    assumptions.push("GGUF tensor table unavailable; using metadata heuristics for per-layer sizes.");
  }
  assumptions.push("Includes ~200 MiB CUDA context overhead and a 4% (min 384 MiB) safety margin.");
  if (profile.gpuLayers >= 99) {
    assumptions.push("GPU layers 99+ is treated as full model offload.");
  }
  if (profile.speculative.enabled && !profile.speculative.draftModelPath.trim()) {
    assumptions.push("Bundled MTP heads are treated as part of the main GGUF file.");
  }
  if (profile.mlock) {
    assumptions.push("Mlock pins the entire model file in system RAM.");
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
  const metadataComplete = Boolean(metadata.blockCount && metadata.embeddingLength && metadata.headCount);
  const confidence = plan.fromTensorLayout && metadataComplete ? "high" : metadataComplete ? "medium" : "low";
  const recommendation = recommendGpuLayers(profile, metadata, layout, backend, availableVramMiB, draftMiB);

  return {
    backend,
    recommendation,
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
      modelWeightsMiB: roundMiB(metadata.fileSizeMiB),
      gpuModelWeightsMiB: roundMiB(gpuActive ? plan.gpuWeightsMiB : 0),
      cpuModelWeightsMiB: roundMiB(plan.cpuWeightsMiB),
      kvCacheMiB: roundMiB(gpuActive ? kv.gpuMiB : kv.totalMiB),
      computeOverheadMiB: roundMiB(overheadMiB),
      draftModelMiB: roundMiB(draftMiB),
      safetyMarginMiB: roundMiB(safetyMarginMiB)
    },
    assumptions,
    warnings
  };
}
