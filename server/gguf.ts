import type { FileHandle } from "node:fs/promises";
import { open, stat } from "node:fs/promises";

import type { ModelMetadata } from "../src/shared/types";

const GGUF_MAGIC = "GGUF";
const INITIAL_READ_BYTES = 1 * 1024 * 1024;
const MAX_READ_BYTES = 64 * 1024 * 1024;

type MetadataValue = string | number | boolean | Array<string | number | boolean> | null;

export interface GgufLayerInfo {
  index: number;
  /** Total weight bytes of all tensors in this block. */
  bytes: number;
  /** Output rows of blk.N.attn_k.weight — elements stored per token in the K cache. */
  attnKElements: number | null;
  attnVElements: number | null;
  /** True when the block contains SSM/recurrent tensors instead of attention. */
  recurrent: boolean;
}

export interface GgufTensorLayout {
  /** Blocks indexed 0..N, including trailing MTP/nextn blocks past block_count. */
  layers: GgufLayerInfo[];
  tokenEmbdBytes: number;
  outputBytes: number;
  otherBytes: number;
  totalBytes: number;
  /** True when every tensor type was recognised; false means bytes are approximate. */
  exact: boolean;
}

export interface GgufModelInfo {
  metadata: ModelMetadata;
  layout: GgufTensorLayout | null;
}

interface CacheEntry {
  mtimeMs: number;
  size: number;
  info: GgufModelInfo;
}

const metadataCache = new Map<string, CacheEntry>();

// GGML tensor type -> [bytes per block, elements per block].
const GGML_TYPE_SIZES: Record<number, [number, number]> = {
  0: [4, 1], // F32
  1: [2, 1], // F16
  2: [18, 32], // Q4_0
  3: [20, 32], // Q4_1
  6: [22, 32], // Q5_0
  7: [24, 32], // Q5_1
  8: [34, 32], // Q8_0
  9: [36, 32], // Q8_1
  10: [84, 256], // Q2_K
  11: [110, 256], // Q3_K
  12: [144, 256], // Q4_K
  13: [176, 256], // Q5_K
  14: [210, 256], // Q6_K
  15: [292, 256], // Q8_K
  16: [66, 256], // IQ2_XXS
  17: [74, 256], // IQ2_XS
  18: [98, 256], // IQ3_XXS
  19: [50, 256], // IQ1_S
  20: [18, 32], // IQ4_NL
  21: [110, 256], // IQ3_S
  22: [82, 256], // IQ2_S
  23: [136, 256], // IQ4_XS
  24: [1, 1], // I8
  25: [2, 1], // I16
  26: [4, 1], // I32
  27: [8, 1], // I64
  28: [8, 1], // F64
  29: [56, 256], // IQ1_M
  30: [2, 1] // BF16
};

// Unknown/exotic quant types fall back to ~4.5 bits per weight.
const FALLBACK_BYTES_PER_ELEMENT = 0.5625;

// Signals that the current header window was exhausted and needs to grow.
class WindowOverflowError extends Error {
  constructor(public readonly required: number) {
    super("GGUF metadata exceeds the inspected header window");
    this.name = "WindowOverflowError";
  }
}

class BufferReader {
  private offset = 0;

  constructor(private readonly buffer: Buffer) {}

  position(): number {
    return this.offset;
  }

  seek(offset: number): void {
    this.offset = offset;
  }

  skip(bytes: number): void {
    this.ensure(bytes);
    this.offset += bytes;
  }

  string(bytes: number): string {
    this.ensure(bytes);
    const value = this.buffer.toString("utf8", this.offset, this.offset + bytes);
    this.offset += bytes;
    return value;
  }

  u8(): number {
    this.ensure(1);
    return this.buffer.readUInt8(this.offset++);
  }

  i8(): number {
    this.ensure(1);
    return this.buffer.readInt8(this.offset++);
  }

  u16(): number {
    this.ensure(2);
    const value = this.buffer.readUInt16LE(this.offset);
    this.offset += 2;
    return value;
  }

  i16(): number {
    this.ensure(2);
    const value = this.buffer.readInt16LE(this.offset);
    this.offset += 2;
    return value;
  }

  u32(): number {
    this.ensure(4);
    const value = this.buffer.readUInt32LE(this.offset);
    this.offset += 4;
    return value;
  }

  i32(): number {
    this.ensure(4);
    const value = this.buffer.readInt32LE(this.offset);
    this.offset += 4;
    return value;
  }

  f32(): number {
    this.ensure(4);
    const value = this.buffer.readFloatLE(this.offset);
    this.offset += 4;
    return value;
  }

  u64(): number {
    this.ensure(8);
    const value = this.buffer.readBigUInt64LE(this.offset);
    this.offset += 8;
    return Number(value);
  }

  i64(): number {
    this.ensure(8);
    const value = this.buffer.readBigInt64LE(this.offset);
    this.offset += 8;
    return Number(value);
  }

  f64(): number {
    this.ensure(8);
    const value = this.buffer.readDoubleLE(this.offset);
    this.offset += 8;
    return value;
  }

  ggufString(): string {
    const length = this.u64();
    this.checkLength(length);
    return this.string(length);
  }

  // Reject implausible string/array lengths before looping so a corrupt file
  // cannot hang the parser or allocate absurd amounts of memory. A genuine
  // header value can never exceed the maximum header window we ever read.
  checkLength(count: number): void {
    if (!Number.isFinite(count) || count < 0 || count > MAX_READ_BYTES) {
      throw new Error("GGUF file declares an implausible string/array length");
    }
  }

  private ensure(bytes: number): void {
    if (this.offset + bytes > this.buffer.length) {
      throw new WindowOverflowError(this.offset + bytes);
    }
  }
}

function readScalar(reader: BufferReader, type: number): MetadataValue {
  switch (type) {
    case 0:
      return reader.u8();
    case 1:
      return reader.i8();
    case 2:
      return reader.u16();
    case 3:
      return reader.i16();
    case 4:
      return reader.u32();
    case 5:
      return reader.i32();
    case 6:
      return reader.f32();
    case 7:
      return Boolean(reader.u8());
    case 8:
      return reader.ggufString();
    case 10:
      return reader.u64();
    case 11:
      return reader.i64();
    case 12:
      return reader.f64();
    default:
      throw new Error(`unsupported GGUF metadata type ${type}`);
  }
}

function skipScalar(reader: BufferReader, type: number): void {
  switch (type) {
    case 0:
    case 1:
    case 7:
      reader.skip(1);
      return;
    case 2:
    case 3:
      reader.skip(2);
      return;
    case 4:
    case 5:
    case 6:
      reader.skip(4);
      return;
    case 8:
      reader.skip(reader.u64());
      return;
    case 10:
    case 11:
    case 12:
      reader.skip(8);
      return;
    default:
      throw new Error(`unsupported GGUF metadata array type ${type}`);
  }
}

function readValue(reader: BufferReader, type: number): MetadataValue {
  if (type !== 9) {
    return readScalar(reader, type);
  }

  const itemType = reader.u32();
  const length = reader.u64();
  reader.checkLength(length);
  if (length > 64 || itemType === 8) {
    for (let index = 0; index < length; index += 1) {
      skipScalar(reader, itemType);
    }
    return null;
  }

  const values: Array<string | number | boolean> = [];
  for (let index = 0; index < length; index += 1) {
    const value = readScalar(reader, itemType);
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      values.push(value);
    }
  }
  return values;
}

function numberMeta(metadata: Map<string, MetadataValue>, key: string): number | null {
  const value = metadata.get(key);
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringMeta(metadata: Map<string, MetadataValue>, key: string): string | null {
  const value = metadata.get(key);
  return typeof value === "string" ? value : null;
}

function tensorBytes(elements: number, type: number): { bytes: number; exact: boolean } {
  const size = GGML_TYPE_SIZES[type];
  if (!size) {
    return { bytes: elements * FALLBACK_BYTES_PER_ELEMENT, exact: false };
  }
  const [blockBytes, blockElements] = size;
  return { bytes: Math.ceil(elements / blockElements) * blockBytes, exact: true };
}

function parseTensorLayout(reader: BufferReader, tensorCount: number): GgufTensorLayout {
  const layers = new Map<number, GgufLayerInfo>();
  const layout: GgufTensorLayout = {
    layers: [],
    tokenEmbdBytes: 0,
    outputBytes: 0,
    otherBytes: 0,
    totalBytes: 0,
    exact: true
  };

  for (let index = 0; index < tensorCount; index += 1) {
    const name = reader.ggufString();
    const dimCount = reader.u32();
    if (dimCount > 8) {
      throw new Error("GGUF tensor declares an implausible dimension count");
    }
    const dims: number[] = [];
    for (let dim = 0; dim < dimCount; dim += 1) {
      dims.push(reader.u64());
    }
    const elements = dims.reduce((product, value) => product * value, 1);
    const type = reader.u32();
    reader.u64(); // data offset

    const { bytes, exact } = tensorBytes(elements, type);
    layout.exact = layout.exact && exact;
    layout.totalBytes += bytes;

    const blockMatch = /^blk\.(\d+)\.(.+)$/.exec(name);
    if (blockMatch) {
      const blockIndex = Number(blockMatch[1]);
      const tensorName = blockMatch[2];
      let layer = layers.get(blockIndex);
      if (!layer) {
        layer = { index: blockIndex, bytes: 0, attnKElements: null, attnVElements: null, recurrent: false };
        layers.set(blockIndex, layer);
      }
      layer.bytes += bytes;
      if (tensorName === "attn_k.weight") {
        // ne[1] = output rows = elements stored per token in the K cache.
        layer.attnKElements = dims[1] ?? null;
      }
      if (tensorName === "attn_v.weight") {
        layer.attnVElements = dims[1] ?? null;
      }
      if (tensorName.startsWith("ssm_")) {
        layer.recurrent = true;
      }
    } else if (name === "token_embd.weight") {
      layout.tokenEmbdBytes += bytes;
    } else if (name === "output.weight") {
      layout.outputBytes += bytes;
    } else {
      layout.otherBytes += bytes;
    }
  }

  layout.layers = [...layers.values()].sort((a, b) => a.index - b.index);
  return layout;
}

function parseFile(buffer: Buffer, fileSizeBytes: number): GgufModelInfo {
  const reader = new BufferReader(buffer);
  const magic = reader.string(4);
  if (magic !== GGUF_MAGIC) {
    throw new Error("model is not a GGUF file");
  }

  reader.u32();
  const tensorCount = reader.u64();
  reader.checkLength(tensorCount);
  const metadataCount = reader.u64();
  reader.checkLength(metadataCount);
  const metadata = new Map<string, MetadataValue>();

  for (let index = 0; index < metadataCount; index += 1) {
    const key = reader.ggufString();
    const type = reader.u32();
    metadata.set(key, readValue(reader, type));
  }

  const architecture = stringMeta(metadata, "general.architecture");
  const prefix = architecture ? `${architecture}.` : "";

  const model: ModelMetadata = {
    architecture,
    name: stringMeta(metadata, "general.name"),
    parameterSize: stringMeta(metadata, "general.size_label"),
    fileType: numberMeta(metadata, "general.file_type"),
    fileSizeMiB: fileSizeBytes / 1024 / 1024,
    blockCount: numberMeta(metadata, `${prefix}block_count`),
    contextLength: numberMeta(metadata, `${prefix}context_length`),
    embeddingLength: numberMeta(metadata, `${prefix}embedding_length`),
    headCount: numberMeta(metadata, `${prefix}attention.head_count`),
    headCountKv: numberMeta(metadata, `${prefix}attention.head_count_kv`),
    keyLength: numberMeta(metadata, `${prefix}attention.key_length`),
    valueLength: numberMeta(metadata, `${prefix}attention.value_length`),
    slidingWindow: numberMeta(metadata, `${prefix}attention.sliding_window`),
    slidingWindowPattern: numberMeta(metadata, `${prefix}attention.sliding_window_pattern`),
    keyLengthSwa: numberMeta(metadata, `${prefix}attention.key_length_swa`),
    valueLengthSwa: numberMeta(metadata, `${prefix}attention.value_length_swa`),
    fullAttentionInterval: numberMeta(metadata, `${prefix}full_attention_interval`),
    ssmStateSize: numberMeta(metadata, `${prefix}ssm.state_size`),
    ssmInnerSize: numberMeta(metadata, `${prefix}ssm.inner_size`),
    ssmConvKernel: numberMeta(metadata, `${prefix}ssm.conv_kernel`),
    ssmGroupCount: numberMeta(metadata, `${prefix}ssm.group_count`),
    nextnPredictLayers: numberMeta(metadata, `${prefix}nextn_predict_layers`)
  };

  // Tensor infos follow the metadata section. A window overflow must propagate
  // so the caller grows the read window; any other layout failure degrades to
  // metadata-only (the estimator falls back to metadata heuristics).
  let layout: GgufTensorLayout | null = null;
  try {
    layout = parseTensorLayout(reader, tensorCount);
  } catch (error) {
    if (error instanceof WindowOverflowError) {
      throw error;
    }
    layout = null;
  }

  return { metadata: model, layout };
}

async function readWindow(file: FileHandle, size: number): Promise<Buffer> {
  const buffer = Buffer.alloc(size);
  await file.read(buffer, 0, size, 0);
  return buffer;
}

export async function readGgufModelInfo(modelPath: string): Promise<GgufModelInfo> {
  const fileStat = await stat(modelPath);
  const cached = metadataCache.get(modelPath);
  if (cached && cached.mtimeMs === fileStat.mtimeMs && cached.size === fileStat.size) {
    return cached.info;
  }

  const file = await open(modelPath, "r");
  try {
    // Read an initial small window and grow it (doubling, up to the 64 MiB
    // cap) only if the header genuinely overflows the current window.
    let windowSize = Math.min(INITIAL_READ_BYTES, fileStat.size);
    let buffer = await readWindow(file, windowSize);

    for (;;) {
      try {
        const info = parseFile(buffer, fileStat.size);
        metadataCache.set(modelPath, {
          mtimeMs: fileStat.mtimeMs,
          size: fileStat.size,
          info
        });
        return info;
      } catch (error) {
        if (!(error instanceof WindowOverflowError)) {
          throw error;
        }
        if (windowSize >= fileStat.size || windowSize >= MAX_READ_BYTES) {
          throw new Error("GGUF metadata exceeds the inspected header window");
        }
        windowSize = Math.min(Math.max(windowSize * 2, error.required), MAX_READ_BYTES, fileStat.size);
        buffer = await readWindow(file, windowSize);
      }
    }
  } finally {
    await file.close();
  }
}

export async function readGgufMetadata(modelPath: string): Promise<ModelMetadata> {
  return (await readGgufModelInfo(modelPath)).metadata;
}
