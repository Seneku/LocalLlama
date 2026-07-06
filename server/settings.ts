import fs from "node:fs";
import path from "node:path";

import type { AppSettings } from "../src/shared/types";

export const EMPTY_SETTINGS: AppSettings = {
  llamaRoot: "",
  cudaServerPath: "",
  cpuServerPath: "",
  cudaBenchPath: "",
  cpuBenchPath: ""
};

// The settings file lives in the data directory, which is intentionally NOT
// settings-overridable (the settings file must be findable before settings load).
function settingsFile(): string {
  const dataPath = process.env.LLAMATUNER_DATA_DIR ?? path.resolve(process.cwd(), "data");
  return path.join(dataPath, "settings.json");
}

let cache: AppSettings | null = null;

function sanitize(raw: unknown): AppSettings {
  const source = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const text = (value: unknown) => (typeof value === "string" ? value.trim() : "");
  return {
    llamaRoot: text(source.llamaRoot),
    cudaServerPath: text(source.cudaServerPath),
    cpuServerPath: text(source.cpuServerPath),
    cudaBenchPath: text(source.cudaBenchPath),
    cpuBenchPath: text(source.cpuBenchPath)
  };
}

export function getSettings(): AppSettings {
  if (cache) {
    return cache;
  }
  try {
    cache = sanitize(JSON.parse(fs.readFileSync(settingsFile(), "utf8")));
  } catch {
    cache = { ...EMPTY_SETTINGS };
  }
  return cache;
}

export function saveSettings(update: unknown): AppSettings {
  const merged = sanitize({
    ...getSettings(),
    ...(update && typeof update === "object" ? (update as Record<string, unknown>) : {})
  });
  const file = settingsFile();
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(merged, null, 2));
  fs.renameSync(tmp, file);
  cache = merged;
  return merged;
}

/** Test hook: force the next getSettings() to re-read from disk. */
export function resetSettingsCache(): void {
  cache = null;
}
