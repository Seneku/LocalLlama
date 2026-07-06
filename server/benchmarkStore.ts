import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";

import { getRuntimePaths } from "./paths";
import type { BenchmarkRun } from "../src/shared/types";

export interface BenchmarkStore {
  benchmarksFile: string;
  load(): Promise<BenchmarkRun[]>;
  save(runs: BenchmarkRun[]): Promise<void>;
  upsert(run: BenchmarkRun): Promise<void>;
  delete(id: string): Promise<boolean>;
}

export function createBenchmarkStore(benchmarksFile?: string): BenchmarkStore {
  const filePath = benchmarksFile ?? path.join(getRuntimePaths().dataPath, "benchmarks.json");

  let cache: BenchmarkRun[] | null = null;
  let seeded = false;
  let writeChain: Promise<void> = Promise.resolve();

  async function ensureFile(): Promise<void> {
    if (seeded) {
      return;
    }
    await mkdir(path.dirname(filePath), { recursive: true });
    try {
      await readFile(filePath, "utf8");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        throw error;
      }
      await writeFile(filePath, "[]\n", "utf8");
    }
    seeded = true;
  }

  async function loadRuns(): Promise<BenchmarkRun[]> {
    if (cache) {
      return cache;
    }
    await ensureFile();
    const raw = await readFile(filePath, "utf8");
    const parsed = JSON.parse(raw) as BenchmarkRun[];
    if (!Array.isArray(parsed)) {
      throw new Error("benchmarks.json must contain an array of benchmark runs");
    }
    cache = parsed;
    return cache;
  }

  async function writeFileAtomic(contents: string): Promise<void> {
    await mkdir(path.dirname(filePath), { recursive: true });
    const tempPath = `${filePath}.${process.pid}.${Date.now().toString(36)}.tmp`;
    await writeFile(tempPath, contents, "utf8");
    await rename(tempPath, filePath);
  }

  function saveRuns(runs: BenchmarkRun[]): Promise<void> {
    const run = writeChain.then(async () => {
      await writeFileAtomic(`${JSON.stringify(runs, null, 2)}\n`);
      cache = runs;
      seeded = true;
    });
    writeChain = run.catch(() => undefined);
    return run;
  }

  return {
    benchmarksFile: filePath,
    load() {
      return loadRuns();
    },
    save(runs) {
      return saveRuns(runs);
    },
    async upsert(run) {
      // Serialize the read-modify-write through the write chain so concurrent
      // upserts/deletes cannot interleave.
      const result = writeChain.then(async () => {
        const runs = cache ?? (await loadRuns());
        const index = runs.findIndex((item) => item.id === run.id);
        const next = runs.slice();
        if (index === -1) {
          next.unshift(run);
        } else {
          next[index] = run;
        }
        const trimmed = next.slice(0, 200);
        await writeFileAtomic(`${JSON.stringify(trimmed, null, 2)}\n`);
        cache = trimmed;
        seeded = true;
      });
      writeChain = result.catch(() => undefined);
      await result;
    },
    async delete(id) {
      let deleted = false;
      const result = writeChain.then(async () => {
        const runs = cache ?? (await loadRuns());
        const next = runs.filter((item) => item.id !== id);
        if (next.length === runs.length) {
          return;
        }
        deleted = true;
        await writeFileAtomic(`${JSON.stringify(next, null, 2)}\n`);
        cache = next;
        seeded = true;
      });
      writeChain = result.catch(() => undefined);
      await result;
      return deleted;
    }
  };
}
