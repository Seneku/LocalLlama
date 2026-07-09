import { describe, expect, test } from "bun:test";

import {
  classifyVendor,
  detectLinuxSysfsGpus,
  filterGpus,
  looksIntegrated,
  parseNvidiaSmi,
  parseRocmSmiJson,
  parseWindowsGpuJson,
  type SysfsReader
} from "../server/gpu";
import type { GpuInfo } from "../src/shared/types";

describe("classifyVendor", () => {
  test("recognizes the major vendors from adapter names", () => {
    expect(classifyVendor("NVIDIA GeForce RTX 4070 SUPER")).toBe("nvidia");
    expect(classifyVendor("AMD Radeon RX 7900 XT")).toBe("amd");
    expect(classifyVendor("Radeon RX 580 Series")).toBe("amd");
    expect(classifyVendor("Intel(R) Arc(TM) A770 Graphics")).toBe("intel");
    expect(classifyVendor("Intel(R) UHD Graphics 770")).toBe("intel");
    expect(classifyVendor("Moore Threads MTT S80")).toBe("unknown");
  });
});

describe("looksIntegrated", () => {
  test("flags iGPUs but not discrete cards", () => {
    expect(looksIntegrated("Intel(R) UHD Graphics 770")).toBe(true);
    expect(looksIntegrated("Intel(R) Iris(R) Xe Graphics")).toBe(true);
    expect(looksIntegrated("AMD Radeon(TM) Graphics")).toBe(true);
    expect(looksIntegrated("AMD Radeon RX 7900 XT")).toBe(false);
    expect(looksIntegrated("NVIDIA GeForce RTX 4070 SUPER")).toBe(false);
    // Intel Arc is discrete despite the "Graphics" suffix.
    expect(looksIntegrated("Intel(R) Arc(TM) A770 Graphics")).toBe(false);
  });
});

describe("parseNvidiaSmi", () => {
  test("parses multi-GPU CSV output", () => {
    const stdout = "NVIDIA GeForce RTX 4070 SUPER, 12282, 1600, 10682\nNVIDIA GeForce GTX 1660, 6144, 300, 5844\n";
    const gpus = parseNvidiaSmi(stdout);
    expect(gpus).toHaveLength(2);
    expect(gpus[0]).toEqual({
      name: "NVIDIA GeForce RTX 4070 SUPER",
      vendor: "nvidia",
      totalMiB: 12282,
      usedMiB: 1600,
      freeMiB: 10682
    });
  });

  test("tolerates unparsable numbers and blank lines", () => {
    const gpus = parseNvidiaSmi("NVIDIA T400, [N/A], [N/A], [N/A]\n\n");
    expect(gpus).toHaveLength(1);
    expect(gpus[0].totalMiB).toBeNull();
    expect(gpus[0].freeMiB).toBeNull();
  });
});

describe("parseWindowsGpuJson", () => {
  test("parses an array of adapters and converts bytes to MiB", () => {
    const json = JSON.stringify([
      { name: "AMD Radeon RX 7800 XT", bytes: 17163091968 },
      { name: "Intel(R) UHD Graphics 770", bytes: 134217728 }
    ]);
    const gpus = parseWindowsGpuJson(json);
    expect(gpus).toHaveLength(2);
    expect(gpus[0].vendor).toBe("amd");
    expect(gpus[0].totalMiB).toBe(16368);
    expect(gpus[0].freeMiB).toBeNull();
    expect(gpus[1].vendor).toBe("intel");
  });

  test("handles the single-object shape ConvertTo-Json emits for one adapter", () => {
    const gpus = parseWindowsGpuJson(JSON.stringify({ name: "AMD Radeon RX 7600", bytes: 8589934592 }));
    expect(gpus).toHaveLength(1);
    expect(gpus[0].totalMiB).toBe(8192);
  });

  test("dedupes ghost adapter instances repeating the same DriverDesc", () => {
    const json = JSON.stringify([
      { name: "AMD Radeon RX 7600", bytes: 8589934592 },
      { name: "AMD Radeon RX 7600", bytes: 8589934592 }
    ]);
    expect(parseWindowsGpuJson(json)).toHaveLength(1);
  });

  test("drops zero-byte entries, empty names, and garbage input", () => {
    expect(parseWindowsGpuJson(JSON.stringify([{ name: "Ghost", bytes: 0 }, { name: "", bytes: 123 }]))).toHaveLength(0);
    expect(parseWindowsGpuJson("")).toHaveLength(0);
    expect(parseWindowsGpuJson("not json")).toHaveLength(0);
    expect(parseWindowsGpuJson("null")).toHaveLength(0);
  });
});

describe("parseRocmSmiJson", () => {
  test("parses vram totals, used memory, and product name per card", () => {
    const json = JSON.stringify({
      card0: {
        "VRAM Total Memory (B)": "17163091968",
        "VRAM Total Used Memory (B)": "3221225472",
        "Card series": "Radeon RX 7800 XT"
      }
    });
    const gpus = parseRocmSmiJson(json);
    expect(gpus).toHaveLength(1);
    expect(gpus[0]).toEqual({
      name: "Radeon RX 7800 XT",
      vendor: "amd",
      totalMiB: 16368,
      usedMiB: 3072,
      freeMiB: 16368 - 3072
    });
  });

  test("falls back to a generic card name and ignores non-card keys", () => {
    const json = JSON.stringify({
      card1: { "VRAM Total Memory (B)": "8589934592" },
      system: { "Driver version": "6.1" }
    });
    const gpus = parseRocmSmiJson(json);
    expect(gpus).toHaveLength(1);
    expect(gpus[0].name).toBe("AMD GPU (card1)");
    expect(gpus[0].usedMiB).toBeNull();
    expect(gpus[0].freeMiB).toBeNull();
  });

  test("returns empty on garbage", () => {
    expect(parseRocmSmiJson("nope")).toHaveLength(0);
  });
});

describe("detectLinuxSysfsGpus", () => {
  function fakeSysfs(files: Record<string, string>, dirs: Record<string, string[]>): SysfsReader {
    return {
      listDir: (dirPath) => dirs[dirPath] ?? [],
      readFile: (filePath) => files[filePath] ?? null
    };
  }

  test("reads amdgpu vram counters", () => {
    const sysfs = fakeSysfs(
      {
        "/sys/class/drm/card0/device/vendor": "0x1002\n",
        "/sys/class/drm/card0/device/mem_info_vram_total": "17163091968\n",
        "/sys/class/drm/card0/device/mem_info_vram_used": "1073741824\n"
      },
      { "/sys/class/drm": ["card0", "card0-DP-1", "renderD128"] }
    );
    const gpus = detectLinuxSysfsGpus(sysfs);
    expect(gpus).toHaveLength(1);
    expect(gpus[0].vendor).toBe("amd");
    expect(gpus[0].totalMiB).toBe(16368);
    expect(gpus[0].freeMiB).toBe(16368 - 1024);
  });

  test("skips integrated Intel (no lmem) but keeps discrete Arc", () => {
    const sysfs = fakeSysfs(
      {
        "/sys/class/drm/card0/device/vendor": "0x8086\n",
        "/sys/class/drm/card1/device/vendor": "0x8086\n",
        "/sys/class/drm/card1/device/lmem_total_bytes": "17179869184\n"
      },
      { "/sys/class/drm": ["card0", "card1"] }
    );
    const gpus = detectLinuxSysfsGpus(sysfs);
    expect(gpus).toHaveLength(1);
    expect(gpus[0].vendor).toBe("intel");
    expect(gpus[0].totalMiB).toBe(16384);
  });

  test("ignores nvidia cards (covered by nvidia-smi)", () => {
    const sysfs = fakeSysfs(
      { "/sys/class/drm/card0/device/vendor": "0x10de\n" },
      { "/sys/class/drm": ["card0"] }
    );
    expect(detectLinuxSysfsGpus(sysfs)).toHaveLength(0);
  });
});

describe("filterGpus", () => {
  const discrete: GpuInfo = { name: "AMD Radeon RX 7800 XT", vendor: "amd", totalMiB: 16368, usedMiB: null, freeMiB: null };
  const integrated: GpuInfo = { name: "Intel(R) UHD Graphics 770", vendor: "intel", totalMiB: 2048, usedMiB: null, freeMiB: null };
  const ghost: GpuInfo = { name: "Microsoft Basic Display", vendor: "unknown", totalMiB: 128, usedMiB: null, freeMiB: null };

  test("drops iGPUs and sub-1GiB adapters when a discrete card exists", () => {
    expect(filterGpus([integrated, discrete, ghost])).toEqual([discrete]);
  });

  test("keeps an iGPU when it is the only usable adapter", () => {
    expect(filterGpus([integrated, ghost])).toEqual([integrated]);
  });

  test("always keeps the Apple unified-memory GPU", () => {
    const apple: GpuInfo = { name: "Apple M3 (Metal)", vendor: "apple", totalMiB: 24576, usedMiB: 0, freeMiB: 24576 };
    expect(filterGpus([apple])).toEqual([apple]);
  });
});
