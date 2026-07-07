import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { createFavoritesStore } from "../server/favoritesStore";
import type { ModelSearchResult } from "../src/shared/types";

let tempDir = "";
let file = "";

function model(id: string, extra: Partial<ModelSearchResult> = {}): ModelSearchResult {
  return { id, author: id.split("/")[0], downloads: 100, likes: 5, gated: false, pipelineTag: "text-generation", updatedAt: null, ...extra };
}

beforeEach(() => {
  tempDir = mkdtempSync(path.join(tmpdir(), "localllama-fav-"));
  file = path.join(tempDir, "favorites.json");
});

afterEach(() => {
  rmSync(tempDir, { recursive: true, force: true });
});

describe("favorites store", () => {
  test("adds newest-first with an addedAt stamp", async () => {
    const store = createFavoritesStore(file);
    await store.add(model("org/A-GGUF"));
    const list = await store.add(model("org/B-GGUF"));
    expect(list.map((f) => f.id)).toEqual(["org/B-GGUF", "org/A-GGUF"]);
    expect(typeof list[0].addedAt).toBe("string");
  });

  test("adding an existing id de-duplicates and moves it to the front", async () => {
    const store = createFavoritesStore(file);
    await store.add(model("org/A-GGUF"));
    await store.add(model("org/B-GGUF"));
    const list = await store.add(model("org/A-GGUF", { likes: 99 }));
    expect(list).toHaveLength(2);
    expect(list[0].id).toBe("org/A-GGUF");
    expect(list[0].likes).toBe(99);
  });

  test("removes by id and persists to disk", async () => {
    const store = createFavoritesStore(file);
    await store.add(model("org/A-GGUF"));
    await store.add(model("org/B-GGUF"));
    const list = await store.remove("org/A-GGUF");
    expect(list.map((f) => f.id)).toEqual(["org/B-GGUF"]);

    const reloaded = createFavoritesStore(file);
    expect((await reloaded.load()).map((f) => f.id)).toEqual(["org/B-GGUF"]);
    // File is valid JSON on disk.
    expect(Array.isArray(JSON.parse(readFileSync(file, "utf8")))).toBe(true);
  });

  test("sanitizes junk fields and rejects a missing id", async () => {
    const store = createFavoritesStore(file);
    const list = await store.add({ id: "org/C-GGUF", downloads: "lots", likes: 3, gated: "yes" } as never);
    expect(list[0]).toMatchObject({ id: "org/C-GGUF", author: "org", downloads: 0, gated: false });
    await expect(store.add({} as never)).rejects.toThrow();
  });
});
