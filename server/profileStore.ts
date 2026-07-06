import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";

import { defaultProfiles } from "./defaultProfiles";
import { getRuntimePaths } from "./paths";
import type { LlamaProfile } from "../src/shared/types";

export interface ProfileStore {
  profilesFile: string;
  load(): Promise<LlamaProfile[]>;
  save(profiles: LlamaProfile[]): Promise<void>;
}

export function createProfileStore(profilesFile?: string): ProfileStore {
  const filePath = profilesFile ?? path.join(getRuntimePaths().dataPath, "profiles.json");

  let cache: LlamaProfile[] | null = null;
  let seeded = false;
  let writeChain: Promise<void> = Promise.resolve();

  async function ensureSeedFile(): Promise<void> {
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
      await writeFile(filePath, `${JSON.stringify(defaultProfiles, null, 2)}\n`, "utf8");
    }
    seeded = true;
  }

  async function writeFileAtomic(contents: string): Promise<void> {
    await mkdir(path.dirname(filePath), { recursive: true });
    const tempPath = `${filePath}.${process.pid}.${Date.now().toString(36)}.tmp`;
    await writeFile(tempPath, contents, "utf8");
    await rename(tempPath, filePath);
  }

  return {
    profilesFile: filePath,
    async load() {
      if (cache) {
        return cache;
      }
      await ensureSeedFile();
      const raw = await readFile(filePath, "utf8");
      const parsed = JSON.parse(raw) as LlamaProfile[];
      if (!Array.isArray(parsed)) {
        throw new Error("profiles.json must contain an array of profiles");
      }
      cache = parsed;
      return cache;
    },
    async save(profiles) {
      const run = writeChain.then(async () => {
        await writeFileAtomic(`${JSON.stringify(profiles, null, 2)}\n`);
        cache = profiles;
        seeded = true;
      });
      writeChain = run.catch(() => undefined);
      await run;
    }
  };
}
