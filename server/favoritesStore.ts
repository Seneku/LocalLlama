import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";

import { getRuntimePaths } from "./paths";
import type { FavoriteModel, ModelSearchResult } from "../src/shared/types";

export interface FavoritesStore {
  file: string;
  load(): Promise<FavoriteModel[]>;
  add(model: ModelSearchResult): Promise<FavoriteModel[]>;
  remove(id: string): Promise<FavoriteModel[]>;
}

function sanitize(raw: unknown): ModelSearchResult {
  const source = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const str = (value: unknown) => (typeof value === "string" ? value : "");
  const num = (value: unknown) => (typeof value === "number" && Number.isFinite(value) ? value : 0);
  const id = str(source.id);
  return {
    id,
    author: str(source.author) || id.split("/")[0] || "",
    downloads: num(source.downloads),
    likes: num(source.likes),
    gated: source.gated === true,
    pipelineTag: typeof source.pipelineTag === "string" ? source.pipelineTag : null,
    updatedAt: typeof source.updatedAt === "string" ? source.updatedAt : null
  };
}

export function createFavoritesStore(favoritesFile?: string): FavoritesStore {
  const filePath = favoritesFile ?? path.join(getRuntimePaths().dataPath, "favorites.json");

  let cache: FavoriteModel[] | null = null;
  let writeChain: Promise<void> = Promise.resolve();

  async function read(): Promise<FavoriteModel[]> {
    if (cache) {
      return cache;
    }
    try {
      const parsed = JSON.parse(await readFile(filePath, "utf8")) as unknown;
      cache = Array.isArray(parsed) ? (parsed as FavoriteModel[]) : [];
    } catch {
      cache = [];
    }
    return cache;
  }

  async function write(list: FavoriteModel[]): Promise<void> {
    const run = writeChain.then(async () => {
      await mkdir(path.dirname(filePath), { recursive: true });
      const tempPath = `${filePath}.${process.pid}.${Date.now().toString(36)}.tmp`;
      await writeFile(tempPath, `${JSON.stringify(list, null, 2)}\n`, "utf8");
      await rename(tempPath, filePath);
      cache = list;
    });
    writeChain = run.catch(() => undefined);
    await run;
  }

  return {
    file: filePath,
    async load() {
      return read();
    },
    async add(model) {
      const clean = sanitize(model);
      if (!clean.id) {
        throw new Error("a model id is required");
      }
      const list = await read();
      // Newest first; replace any existing entry for the same id.
      const next: FavoriteModel[] = [
        { ...clean, addedAt: new Date().toISOString() },
        ...list.filter((item) => item.id !== clean.id)
      ];
      await write(next);
      return next;
    },
    async remove(id) {
      const list = await read();
      const next = list.filter((item) => item.id !== id);
      await write(next);
      return next;
    }
  };
}
