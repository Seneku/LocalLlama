import type { FileHandle } from "node:fs/promises";
import { open, stat } from "node:fs/promises";

import type { ModelMetadata } from "../src/shared/types";

const GGUF_MAGIC = "GGUF";
const INITIAL_READ_BYTES = 1 * 1024 * 1024;
const MAX_READ_BYTES = 64 * 1024 * 1024;

type MetadataValue = string | number | boolean | Array<string | number | boolean> | null;

interface CacheEntry {
  mtimeMs: number;
  size: number;
  metadata: ModelMetadata;
}

const metadataCache = new Map<string, CacheEntry>();

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

function parseHeader(buffer: Buffer, fileSizeBytes: number): ModelMetadata {
  const reader = new BufferReader(buffer);
  const magic = reader.string(4);
  if (magic !== GGUF_MAGIC) {
    throw new Error("model is not a GGUF file");
  }

  reader.u32();
  reader.u64();
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

  return {
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
    valueLength: numberMeta(metadata, `${prefix}attention.value_length`)
  };
}

async function readWindow(file: FileHandle, size: number): Promise<Buffer> {
  const buffer = Buffer.alloc(size);
  await file.read(buffer, 0, size, 0);
  return buffer;
}

export async function readGgufMetadata(modelPath: string): Promise<ModelMetadata> {
  const fileStat = await stat(modelPath);
  const cached = metadataCache.get(modelPath);
  if (cached && cached.mtimeMs === fileStat.mtimeMs && cached.size === fileStat.size) {
    return cached.metadata;
  }

  const file = await open(modelPath, "r");
  try {
    // Read an initial small window and grow it (doubling, up to the old 64 MiB
    // cap) only if the header genuinely overflows the current window.
    let windowSize = Math.min(INITIAL_READ_BYTES, fileStat.size);
    let buffer = await readWindow(file, windowSize);

    for (;;) {
      try {
        const metadata = parseHeader(buffer, fileStat.size);
        metadataCache.set(modelPath, {
          mtimeMs: fileStat.mtimeMs,
          size: fileStat.size,
          metadata
        });
        return metadata;
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
