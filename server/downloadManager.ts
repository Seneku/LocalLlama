import { once } from "node:events";
import { createWriteStream, existsSync } from "node:fs";
import { mkdir, rm } from "node:fs/promises";
import path from "node:path";

import { getModelsDir } from "./paths";
import { hfHeaders, resolveDownloadUrl } from "./hf";
import type { DownloadStatus } from "../src/shared/types";

const IDLE: DownloadStatus = {
  state: "idle",
  modelId: null,
  filename: null,
  dest: null,
  totalBytes: null,
  receivedBytes: 0,
  startedAt: null,
  completedAt: null,
  error: null
};

export interface StartDownloadInput {
  id: string;
  filename: string;
}

// One active GGUF download at a time (mirrors the single-run benchmark manager).
// Streams straight to a `<file>.part` and renames on success, so a cancel or
// crash never leaves a truncated file that looks complete.
export interface DownloadManagerOptions {
  /** Override how a model id + filename maps to a download URL (for tests). */
  resolveUrl?: (id: string, filename: string) => string;
}

export class DownloadManager {
  private status: DownloadStatus = { ...IDLE };
  private controller: AbortController | null = null;
  private partPath: string | null = null;
  private readonly resolveUrl: (id: string, filename: string) => string;

  constructor(options: DownloadManagerOptions = {}) {
    this.resolveUrl = options.resolveUrl ?? resolveDownloadUrl;
  }

  getStatus(): DownloadStatus {
    return this.status;
  }

  async start(input: StartDownloadInput): Promise<DownloadStatus> {
    if (this.status.state === "downloading") {
      throw new Error("A download is already in progress. Wait for it to finish or cancel it first.");
    }

    const id = input.id?.trim();
    const filename = input.filename?.trim();
    if (!id || !filename) {
      throw new Error("A model id and filename are required.");
    }
    // Guard against path traversal: only a bare .gguf basename is allowed.
    if (path.basename(filename) !== filename || !filename.toLowerCase().endsWith(".gguf")) {
      throw new Error("Invalid model filename.");
    }

    const modelsDir = getModelsDir();
    const dest = path.join(modelsDir, filename);
    if (existsSync(dest)) {
      throw new Error(`${filename} is already in your models folder.`);
    }

    await mkdir(modelsDir, { recursive: true });
    const partPath = `${dest}.part`;
    await rm(partPath, { force: true });

    this.controller = new AbortController();
    this.partPath = partPath;
    this.status = {
      state: "downloading",
      modelId: id,
      filename,
      dest,
      totalBytes: null,
      receivedBytes: 0,
      startedAt: new Date().toISOString(),
      completedAt: null,
      error: null
    };

    // Run the transfer in the background; callers poll getStatus().
    void this.run(id, filename, dest, partPath, this.controller.signal);
    return this.status;
  }

  private async run(
    id: string,
    filename: string,
    dest: string,
    partPath: string,
    signal: AbortSignal
  ): Promise<void> {
    try {
      const response = await fetch(this.resolveUrl(id, filename), { headers: hfHeaders(), signal });
      if (response.status === 401 || response.status === 403) {
        throw new Error("This model is gated or private. Accept its license on Hugging Face and add a token in Settings.");
      }
      if (!response.ok || !response.body) {
        throw new Error(`Download failed: Hugging Face returned ${response.status}.`);
      }

      const contentLength = Number(response.headers.get("content-length"));
      this.status = { ...this.status, totalBytes: Number.isFinite(contentLength) && contentLength > 0 ? contentLength : null };

      const fileStream = createWriteStream(partPath);
      const reader = response.body.getReader();
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) {
            break;
          }
          this.status = { ...this.status, receivedBytes: this.status.receivedBytes + value.length };
          if (!fileStream.write(Buffer.from(value))) {
            await once(fileStream, "drain");
          }
        }
      } finally {
        await new Promise<void>((resolve, reject) => {
          fileStream.end((error?: Error | null) => (error ? reject(error) : resolve()));
        });
      }

      const { rename } = await import("node:fs/promises");
      await rename(partPath, dest);
      this.status = { ...this.status, state: "completed", completedAt: new Date().toISOString() };
    } catch (error) {
      const aborted = signal.aborted;
      await rm(partPath, { force: true }).catch(() => undefined);
      this.status = {
        ...this.status,
        state: aborted ? "cancelled" : "failed",
        completedAt: new Date().toISOString(),
        error: aborted ? null : error instanceof Error ? error.message : String(error)
      };
    } finally {
      this.controller = null;
      this.partPath = null;
    }
  }

  async cancel(): Promise<DownloadStatus> {
    if (this.status.state === "downloading" && this.controller) {
      this.controller.abort();
    }
    return this.status;
  }

  /** Abort and remove any partial file — called on server shutdown. */
  async dispose(): Promise<void> {
    this.controller?.abort();
    if (this.partPath) {
      await rm(this.partPath, { force: true }).catch(() => undefined);
    }
  }
}
