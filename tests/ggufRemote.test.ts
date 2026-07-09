import { describe, expect, test } from "bun:test";

import { fetchGgufPrefix, readRemoteGgufInfo, resetRemoteGgufCache, RemoteGgufError, type FetchLike } from "../server/ggufRemote";
import { estimateRemoteModel } from "../server/hfEstimate";
import type { HardwareInfo, ModelFile } from "../src/shared/types";

// ---- minimal GGUF v3 fixture builder ----

function u32(value: number): Buffer {
  const buffer = Buffer.alloc(4);
  buffer.writeUInt32LE(value);
  return buffer;
}

function u64(value: number): Buffer {
  const buffer = Buffer.alloc(8);
  buffer.writeBigUInt64LE(BigInt(value));
  return buffer;
}

function ggufString(value: string): Buffer {
  const bytes = Buffer.from(value, "utf8");
  return Buffer.concat([u64(bytes.length), bytes]);
}

function kvString(key: string, value: string): Buffer {
  return Buffer.concat([ggufString(key), u32(8), ggufString(value)]);
}

function kvU32(key: string, value: number): Buffer {
  return Buffer.concat([ggufString(key), u32(4), u32(value)]);
}

function tensor(name: string, dims: number[], type = 1): Buffer {
  return Buffer.concat([ggufString(name), u32(dims.length), ...dims.map(u64), u32(type), u64(0)]);
}

function buildGguf(kvs: Buffer[], tensors: Buffer[] = []): Buffer {
  return Buffer.concat([
    Buffer.from("GGUF", "utf8"),
    u32(3), // version
    u64(tensors.length),
    u64(kvs.length),
    ...kvs,
    ...tensors
  ]);
}

const denseKvs = [
  kvString("general.architecture", "llama"),
  kvString("general.name", "Test dense"),
  kvU32("llama.block_count", 32),
  kvU32("llama.context_length", 32768),
  kvU32("llama.embedding_length", 4096),
  kvU32("llama.attention.head_count", 32),
  kvU32("llama.attention.head_count_kv", 8),
  kvU32("llama.attention.key_length", 128),
  kvU32("llama.attention.value_length", 128)
];

/** fetch stub that serves Range slices of a fixture buffer with HTTP 206. */
function rangeFetch(fixture: Buffer, calls: Array<{ range: string | undefined }> = []): FetchLike {
  return async (_url, init) => {
    const headers = (init?.headers ?? {}) as Record<string, string>;
    calls.push({ range: headers.range });
    const match = /bytes=0-(\d+)/u.exec(headers.range ?? "");
    const end = match ? Math.min(Number(match[1]), fixture.length - 1) : fixture.length - 1;
    return new Response(new Uint8Array(fixture.subarray(0, end + 1)), { status: 206 });
  };
}

const hardware: HardwareInfo = {
  totalRamMiB: 65536,
  freeRamMiB: 48000,
  gpus: [{ name: "RTX 4070 SUPER", vendor: "nvidia", totalMiB: 12281, usedMiB: 1600, freeMiB: 10681 }]
};

describe("fetchGgufPrefix", () => {
  test("sends a Range header and returns the 206 body", async () => {
    const calls: Array<{ range: string | undefined }> = [];
    const fixture = buildGguf(denseKvs);
    const buffer = await fetchGgufPrefix("http://x/model.gguf", 128, rangeFetch(fixture, calls));
    expect(calls[0].range).toBe("bytes=0-127");
    expect(buffer.length).toBe(128);
    expect(buffer.toString("utf8", 0, 4)).toBe("GGUF");
  });

  test("truncates a 200 response that ignored the Range header", async () => {
    const fixture = buildGguf(denseKvs);
    const fetchImpl: FetchLike = async () => new Response(new Uint8Array(fixture), { status: 200 });
    const buffer = await fetchGgufPrefix("http://x/model.gguf", 16, fetchImpl);
    expect(buffer.length).toBe(16);
    expect(buffer.toString("utf8", 0, 4)).toBe("GGUF");
  });

  test("maps 403 to a gated-repo error", async () => {
    const fetchImpl: FetchLike = async () => new Response("denied", { status: 403 });
    await expect(fetchGgufPrefix("http://x/model.gguf", 16, fetchImpl)).rejects.toThrow(RemoteGgufError);
    await expect(fetchGgufPrefix("http://x/model.gguf", 16, fetchImpl)).rejects.toThrow(/gated or private/u);
  });
});

describe("readRemoteGgufInfo", () => {
  test("parses metadata from a ranged header read and caches it", async () => {
    resetRemoteGgufCache();
    const fixture = buildGguf(denseKvs);
    const calls: Array<{ range: string | undefined }> = [];
    const fetchImpl = rangeFetch(fixture, calls);

    const info = await readRemoteGgufInfo("author/dense", "dense.gguf", 5_000_000_000, fetchImpl);
    expect(info.metadata.architecture).toBe("llama");
    expect(info.metadata.blockCount).toBe(32);
    expect(info.metadata.contextLength).toBe(32768);
    expect(calls.length).toBe(1);

    // Second call served from the cache — no network.
    await readRemoteGgufInfo("author/dense", "dense.gguf", 5_000_000_000, fetchImpl);
    expect(calls.length).toBe(1);
  });

  test("grows the window when the header overflows the first read", async () => {
    resetRemoteGgufCache();
    // A ~1.5 MiB padding string pushes the header past the 1 MiB initial window.
    const padded = buildGguf([...denseKvs, kvString("general.padding", "x".repeat(1_500_000))]);
    const calls: Array<{ range: string | undefined }> = [];
    const info = await readRemoteGgufInfo("author/padded", "padded.gguf", 5_000_000_000, rangeFetch(padded, calls));
    expect(info.metadata.blockCount).toBe(32);
    expect(calls.length).toBe(2);
    const firstEnd = Number(/bytes=0-(\d+)/u.exec(calls[0].range ?? "")![1]);
    const secondEnd = Number(/bytes=0-(\d+)/u.exec(calls[1].range ?? "")![1]);
    expect(firstEnd).toBe(1024 * 1024 - 1);
    expect(secondEnd).toBeGreaterThan(firstEnd);
  });
});

describe("estimateRemoteModel", () => {
  test("estimates a dense model at the requested context", async () => {
    resetRemoteGgufCache();
    const fixture = buildGguf(denseKvs);
    const estimate = await estimateRemoteModel("author/dense", "dense-q4.gguf", 16384, 5_000_000_000, hardware, {
      fetchImpl: rangeFetch(fixture)
    });

    expect(estimate.contextSize).toBe(16384);
    expect(estimate.split).toBe(false);
    expect(estimate.maxGpuLayers).toBe(33);
    expect(estimate.cpuMoe).toBeNull();
    expect(estimate.estimatedVramMiB).toBeGreaterThan(0);
    expect(["fits", "tight", "over", "unknown"]).toContain(estimate.fit);
  });

  test("offers a --cpu-moe variant for MoE models", async () => {
    resetRemoteGgufCache();
    const moeKvs = [
      kvString("general.architecture", "llama"),
      kvU32("llama.block_count", 2),
      kvU32("llama.context_length", 32768),
      kvU32("llama.embedding_length", 4096),
      kvU32("llama.attention.head_count", 32),
      kvU32("llama.attention.head_count_kv", 8),
      kvU32("llama.attention.key_length", 128),
      kvU32("llama.attention.value_length", 128)
    ];
    const moeTensors = [
      tensor("token_embd.weight", [4096, 32000]),
      tensor("blk.0.attn_k.weight", [4096, 1024]),
      tensor("blk.0.attn_v.weight", [4096, 1024]),
      tensor("blk.0.ffn_gate_exps.weight", [4096, 14336, 8]),
      tensor("blk.1.attn_k.weight", [4096, 1024]),
      tensor("blk.1.attn_v.weight", [4096, 1024]),
      tensor("blk.1.ffn_gate_exps.weight", [4096, 14336, 8])
    ];
    const fixture = buildGguf(moeKvs, moeTensors);
    const estimate = await estimateRemoteModel("author/moe", "moe-q4.gguf", 8192, 30_000_000_000, hardware, {
      fetchImpl: rangeFetch(fixture)
    });

    expect(estimate.cpuMoe).not.toBeNull();
    expect(estimate.cpuMoe!.estimatedVramMiB).toBeLessThan(estimate.estimatedVramMiB);
    expect(estimate.cpuMoe!.estimatedSystemRamMiB).toBeGreaterThan(0);
  });

  test("sums shard sizes for split GGUFs and reads shard 1's header", async () => {
    resetRemoteGgufCache();
    const fixture = buildGguf(denseKvs);
    const requested: string[] = [];
    const fetchImpl: FetchLike = async (url, init) => {
      requested.push(String(url));
      return rangeFetch(fixture)(url, init);
    };
    const shardFiles: ModelFile[] = [1, 2, 3].map((part) => ({
      filename: `big-model-0000${part}-of-00003.gguf`,
      sizeBytes: 10_000_000_000,
      sizeMiB: 9537,
      quant: "Q4_K_M",
      fit: "over"
    }));

    const estimate = await estimateRemoteModel(
      "author/big",
      "big-model-00002-of-00003.gguf",
      8192,
      10_000_000_000,
      hardware,
      {
        fetchImpl,
        listFiles: async () => shardFiles
      }
    );

    expect(estimate.split).toBe(true);
    expect(requested[0]).toContain("big-model-00001-of-00003.gguf");
    // 30 GB of summed weights cannot fully offload to a 12 GB card, but a
    // partial-offload recommendation should exist.
    expect(estimate.fit).toBe("over");
    expect(estimate.recommendation).not.toBeNull();
    expect(estimate.recommendation!.gpuLayers).toBeLessThan(33);
    expect(estimate.warnings.some((warning) => warning.includes("Multi-part"))).toBe(true);
  });
});
