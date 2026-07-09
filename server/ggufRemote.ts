// Remote GGUF header reads over HTTP Range requests. GGUF headers (metadata +
// tensor table) are usually 1–25 MiB, so the calibrated estimator can run
// against a Hugging Face file BEFORE downloading tens of gigabytes. Reuses
// gguf.ts's grow-on-overflow prefix reader through the GgufByteSource seam.
import { readGgufInfoFromSource, type GgufModelInfo } from "./gguf";
import { hfHeaders, resolveDownloadUrl } from "./hf";

export class RemoteGgufError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = "RemoteGgufError";
  }
}

// Parsed header info per repo file. Keyed by size too: HF replaces files in
// place, but a changed file almost always changes size.
const remoteInfoCache = new Map<string, GgufModelInfo>();
const REMOTE_CACHE_MAX = 200;

export type FetchLike = (url: string, init?: RequestInit) => Promise<Response>;

export async function fetchGgufPrefix(url: string, byteLength: number, fetchImpl: FetchLike = fetch): Promise<Buffer> {
  const response = await fetchImpl(url, {
    // The 302 hop to the CDN is followed automatically; the runtime drops the
    // Authorization header cross-origin, which is correct — the signed CDN
    // URL needs no auth.
    headers: { ...hfHeaders(), range: `bytes=0-${byteLength - 1}` },
    redirect: "follow"
  });

  if (response.status === 401 || response.status === 403) {
    throw new RemoteGgufError(
      response.status,
      "This model is gated or private on Hugging Face. Add an access token in Settings to check its fit."
    );
  }
  if (response.status === 429) {
    throw new RemoteGgufError(429, "Hugging Face rate limit reached; fit check unavailable right now.");
  }
  if (response.status === 206) {
    return Buffer.from(await response.arrayBuffer());
  }
  if (response.status === 200) {
    // Server ignored the Range header: stream only the needed prefix, then
    // cancel the rest of the (multi-GB) body.
    const reader = response.body?.getReader();
    if (!reader) {
      throw new RemoteGgufError(502, "Hugging Face returned no response body for the model header.");
    }
    const chunks: Buffer[] = [];
    let total = 0;
    while (total < byteLength) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      chunks.push(Buffer.from(value));
      total += value.length;
    }
    await reader.cancel().catch(() => undefined);
    return Buffer.concat(chunks).subarray(0, byteLength);
  }
  throw new RemoteGgufError(
    response.status,
    `Hugging Face returned ${response.status} while reading the model header.`
  );
}

export async function readRemoteGgufInfo(
  id: string,
  filename: string,
  sizeBytes: number,
  fetchImpl: FetchLike = fetch
): Promise<GgufModelInfo> {
  const key = `${id}/${filename}@${sizeBytes}`;
  const cached = remoteInfoCache.get(key);
  if (cached) {
    return cached;
  }

  const url = resolveDownloadUrl(id, filename);
  const info = await readGgufInfoFromSource({
    size: sizeBytes,
    readPrefix: (byteLength) => fetchGgufPrefix(url, Math.min(byteLength, sizeBytes), fetchImpl)
  });

  if (remoteInfoCache.size >= REMOTE_CACHE_MAX) {
    const oldest = remoteInfoCache.keys().next().value;
    if (oldest !== undefined) {
      remoteInfoCache.delete(oldest);
    }
  }
  remoteInfoCache.set(key, info);
  return info;
}

export function resetRemoteGgufCache(): void {
  remoteInfoCache.clear();
}
