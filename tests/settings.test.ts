import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { getRuntimePaths } from "../server/paths";
import { getSettings, resetSettingsCache, saveSettings } from "../server/settings";

const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "localllama-settings-"));
const originalDataDir = process.env.LOCALLLAMA_DATA_DIR;

beforeAll(() => {
  process.env.LOCALLLAMA_DATA_DIR = tempDir;
  resetSettingsCache();
});

afterAll(() => {
  if (originalDataDir === undefined) {
    delete process.env.LOCALLLAMA_DATA_DIR;
  } else {
    process.env.LOCALLLAMA_DATA_DIR = originalDataDir;
  }
  resetSettingsCache();
  fs.rmSync(tempDir, { recursive: true, force: true });
});

describe("settings", () => {
  test("defaults to empty overrides", () => {
    const settings = getSettings();
    expect(settings.llamaRoot).toBe("");
    expect(settings.cudaServerPath).toBe("");
  });

  test("saved llamaRoot drives derived binary paths", () => {
    const root = path.join(tempDir, "llama.cpp");
    saveSettings({ llamaRoot: root });

    // Binaries are suffixed with .exe only on Windows. Nothing exists on
    // disk, so the display fallbacks apply: release-zip root for GPU,
    // build/bin for CPU.
    const exe = process.platform === "win32" ? ".exe" : "";
    const paths = getRuntimePaths(() => false);
    expect(paths.llamaRoot).toBe(root);
    expect(paths.cudaServerPath).toBe(path.join(root, `llama-server${exe}`));
    expect(paths.cpuBenchPath).toBe(path.join(root, "build", "bin", `llama-bench${exe}`));
  });

  test("binary paths probe common llama.cpp layouts and use what exists", () => {
    const root = path.join(tempDir, "llama.cpp");
    saveSettings({ llamaRoot: root });
    const exe = process.platform === "win32" ? ".exe" : "";
    const server = `llama-server${exe}`;

    // Release zip extracted to the root: both slots find the root binary.
    const rootOnly = new Set([path.join(root, server)]);
    let paths = getRuntimePaths((p) => rootOnly.has(p));
    expect(paths.cudaServerPath).toBe(path.join(root, server));
    expect(paths.cpuServerPath).toBe(path.join(root, server));

    // Side-by-side builds: the deliberate per-build folders win over the root.
    const sideBySide = new Set([
      path.join(root, server),
      path.join(root, "dist-cuda", server),
      path.join(root, "build", "bin", server)
    ]);
    paths = getRuntimePaths((p) => sideBySide.has(p));
    expect(paths.cudaServerPath).toBe(path.join(root, "dist-cuda", server));
    expect(paths.cpuServerPath).toBe(path.join(root, "build", "bin", server));

    // Source build only: the GPU slot falls through to build/bin too.
    const sourceBuild = new Set([path.join(root, "build", "bin", server)]);
    paths = getRuntimePaths((p) => sourceBuild.has(p));
    expect(paths.cudaServerPath).toBe(path.join(root, "build", "bin", server));
  });

  test("explicit binary overrides beat derived defaults", () => {
    const custom = path.join(tempDir, "custom-server.exe");
    saveSettings({ cudaServerPath: custom });
    expect(getRuntimePaths().cudaServerPath).toBe(custom);
  });

  test("settings persist to disk and reload", () => {
    resetSettingsCache();
    const reloaded = getSettings();
    expect(reloaded.llamaRoot).toBe(path.join(tempDir, "llama.cpp"));
    expect(reloaded.cudaServerPath).toBe(path.join(tempDir, "custom-server.exe"));
  });

  test("non-string values are sanitised away", () => {
    const settings = saveSettings({ llamaRoot: 42, cpuServerPath: "  padded  " });
    expect(settings.llamaRoot).toBe("");
    expect(settings.cpuServerPath).toBe("padded");
  });
});
