// Hardware detection. NVIDIA GPUs come from nvidia-smi (the only path with
// live free-VRAM numbers). AMD/Intel cards are invisible to nvidia-smi, so on
// Windows we read the display-driver registry key
// HardwareInformation.qwMemorySize (a QWORD — unlike WMI's AdapterRAM it is
// not capped at 4 GiB) and on Linux we ask rocm-smi, falling back to the
// amdgpu sysfs vram counters. Apple Silicon models unified memory as a Metal
// working-set budget. Pure parsers are exported for unit tests; only
// getHardwareInfo touches the real system.
import { execFile } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";

import { isAppleSilicon } from "./platform";
import type { GpuInfo, GpuVendor, HardwareInfo } from "../src/shared/types";

const execFileAsync = promisify(execFile);

function bytesToMiB(bytes: number): number {
  return bytes / 1024 / 1024;
}

function parseNum(value: string | undefined): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function classifyVendor(name: string): GpuVendor {
  const lower = name.toLowerCase();
  if (/nvidia|geforce|quadro|\brtx\b|\bgtx\b/u.test(lower)) {
    return "nvidia";
  }
  if (/\bamd\b|radeon|firepro|\bati\b/u.test(lower)) {
    return "amd";
  }
  if (/intel|\barc\b|\biris\b|uhd graphics/u.test(lower)) {
    return "intel";
  }
  return "unknown";
}

// Integrated GPUs share system RAM and are useless for offload sizing; drop
// them whenever a discrete card is present. Intel Arc is discrete despite the
// "Graphics" suffix, so it is explicitly exempted.
export function looksIntegrated(name: string): boolean {
  const lower = name.toLowerCase();
  if (lower.includes("arc")) {
    return false;
  }
  return /uhd graphics|\biris\b|\bhd graphics\b|integrated|radeon\(tm\) graphics|vega \d+ graphics/u.test(lower);
}

// Adapters under 1 GiB dedicated memory are ghosts or display-only devices;
// integrated GPUs are dropped only when a discrete card exists.
export function filterGpus(gpus: GpuInfo[]): GpuInfo[] {
  const usable = gpus.filter((gpu) => gpu.vendor === "apple" || (gpu.totalMiB ?? 0) >= 1024);
  const hasDiscrete = usable.some((gpu) => !looksIntegrated(gpu.name));
  return hasDiscrete ? usable.filter((gpu) => !looksIntegrated(gpu.name)) : usable;
}

// ---- NVIDIA (nvidia-smi CSV) ----

export function parseNvidiaSmi(stdout: string): GpuInfo[] {
  const gpus: GpuInfo[] = [];
  for (const line of stdout.trim().split(/\r?\n/u)) {
    const [name, total, used, free] = line.split(",").map((part) => part.trim());
    if (!name) {
      continue;
    }
    gpus.push({
      name,
      vendor: "nvidia",
      totalMiB: parseNum(total),
      usedMiB: parseNum(used),
      freeMiB: parseNum(free)
    });
  }
  return gpus;
}

// ---- Windows (display-driver registry) ----

// The {4d36e968-…} class GUID is "Display adapters"; each 00NN subkey is one
// adapter instance. Emits [{ name, bytes }] as compact JSON.
export const WINDOWS_GPU_QUERY = [
  "$keys = Get-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' -ErrorAction SilentlyContinue",
  "$rows = @($keys | ForEach-Object { [pscustomobject]@{ name = $_.DriverDesc; bytes = [int64]$_.'HardwareInformation.qwMemorySize' } } | Where-Object { $_.name -and $_.bytes -gt 0 })",
  "ConvertTo-Json -InputObject $rows -Compress"
].join("; ");

export function parseWindowsGpuJson(json: string): GpuInfo[] {
  const trimmed = json.trim();
  if (!trimmed) {
    return [];
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return [];
  }
  const entries = Array.isArray(parsed) ? parsed : [parsed];
  const seen = new Set<string>();
  const gpus: GpuInfo[] = [];
  for (const entry of entries) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const name = String((entry as { name?: unknown }).name ?? "").trim();
    const bytes = Number((entry as { bytes?: unknown }).bytes ?? 0);
    // Ghost/disabled adapter instances repeat the same DriverDesc — keep one.
    if (!name || !Number.isFinite(bytes) || bytes <= 0 || seen.has(name)) {
      continue;
    }
    seen.add(name);
    gpus.push({
      name,
      vendor: classifyVendor(name),
      totalMiB: Math.round(bytesToMiB(bytes)),
      usedMiB: null,
      // The registry only stores capacity; free VRAM needs vendor tooling we
      // don't have here. Consumers must handle freeMiB: null.
      freeMiB: null
    });
  }
  return gpus;
}

// ---- Linux AMD (rocm-smi, then amdgpu sysfs) ----

// `rocm-smi --showmeminfo vram --showproductname --json` returns
// { "card0": { "VRAM Total Memory (B)": "17163091968", "VRAM Total Used Memory (B)": "…", "Card series": "…" } }
export function parseRocmSmiJson(json: string): GpuInfo[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json.trim());
  } catch {
    return [];
  }
  if (!parsed || typeof parsed !== "object") {
    return [];
  }
  const gpus: GpuInfo[] = [];
  for (const [card, value] of Object.entries(parsed as Record<string, unknown>)) {
    if (!/^card\d+$/u.test(card) || !value || typeof value !== "object") {
      continue;
    }
    const entry = value as Record<string, unknown>;
    let totalBytes: number | null = null;
    let usedBytes: number | null = null;
    let name: string | null = null;
    for (const [key, raw] of Object.entries(entry)) {
      const lower = key.toLowerCase();
      if (/vram total memory/u.test(lower)) {
        totalBytes = parseNum(String(raw));
      } else if (/vram total used/u.test(lower)) {
        usedBytes = parseNum(String(raw));
      } else if (/card (series|model)/u.test(lower) && !name && String(raw).trim()) {
        name = String(raw).trim();
      }
    }
    if (totalBytes === null || totalBytes <= 0) {
      continue;
    }
    const totalMiB = Math.round(bytesToMiB(totalBytes));
    const usedMiB = usedBytes === null ? null : Math.round(bytesToMiB(usedBytes));
    gpus.push({
      name: name ?? `AMD GPU (${card})`,
      vendor: "amd",
      totalMiB,
      usedMiB,
      freeMiB: usedMiB === null ? null : Math.max(0, totalMiB - usedMiB)
    });
  }
  return gpus;
}

export interface SysfsReader {
  listDir(dirPath: string): string[];
  readFile(filePath: string): string | null;
}

const realSysfs: SysfsReader = {
  listDir(dirPath) {
    try {
      return fs.readdirSync(dirPath);
    } catch {
      return [];
    }
  },
  readFile(filePath) {
    try {
      return fs.readFileSync(filePath, "utf8");
    } catch {
      return null;
    }
  }
};

const PCI_VENDOR_AMD = "0x1002";
const PCI_VENDOR_INTEL = "0x8086";

// amdgpu exposes mem_info_vram_total/used per card; Intel discrete (i915/xe)
// exposes lmem totals on some kernels. NVIDIA cards are skipped here —
// nvidia-smi already covers them and nouveau has no vram counters.
export function detectLinuxSysfsGpus(sysfs: SysfsReader = realSysfs, root = "/sys/class/drm"): GpuInfo[] {
  const gpus: GpuInfo[] = [];
  for (const entry of sysfs.listDir(root)) {
    if (!/^card\d+$/u.test(entry)) {
      continue;
    }
    const device = path.posix.join(root, entry, "device");
    const vendorId = sysfs.readFile(path.posix.join(device, "vendor"))?.trim() ?? "";
    if (vendorId === PCI_VENDOR_AMD) {
      const totalBytes = parseNum(sysfs.readFile(path.posix.join(device, "mem_info_vram_total"))?.trim());
      if (!totalBytes || totalBytes <= 0) {
        continue;
      }
      const usedBytes = parseNum(sysfs.readFile(path.posix.join(device, "mem_info_vram_used"))?.trim());
      const totalMiB = Math.round(bytesToMiB(totalBytes));
      const usedMiB = usedBytes === null ? null : Math.round(bytesToMiB(usedBytes));
      gpus.push({
        name: `AMD GPU (${entry})`,
        vendor: "amd",
        totalMiB,
        usedMiB,
        freeMiB: usedMiB === null ? null : Math.max(0, totalMiB - usedMiB)
      });
    } else if (vendorId === PCI_VENDOR_INTEL) {
      const lmem =
        parseNum(sysfs.readFile(path.posix.join(device, "lmem_total_bytes"))?.trim()) ??
        parseNum(sysfs.readFile(path.posix.join(root, entry, "lmem_total_bytes"))?.trim());
      // No lmem => integrated Intel graphics; skip (shares system RAM).
      if (!lmem || lmem <= 0) {
        continue;
      }
      gpus.push({
        name: `Intel GPU (${entry})`,
        vendor: "intel",
        totalMiB: Math.round(bytesToMiB(lmem)),
        usedMiB: null,
        freeMiB: null
      });
    }
  }
  return gpus;
}

// ---- Apple Silicon (Metal unified memory) ----

// Apple's Metal recommendedMaxWorkingSetSize (the share of unified memory the
// GPU may use) can't be read from a shell, so approximate it: honor an explicit
// iogpu.wired_limit_mb override, else ~2/3 of RAM on smaller machines rising to
// ~3/4 on larger ones — roughly matching macOS defaults.
export function appleWorkingSetMiB(totalRamMiB: number, overrideMiB: number | null): number {
  if (overrideMiB && overrideMiB > 0) {
    return overrideMiB;
  }
  const fraction = totalRamMiB <= 36 * 1024 ? 0.67 : 0.75;
  return Math.floor(totalRamMiB * fraction);
}

async function detectAppleSiliconGpu(totalRamMiB: number, freeRamMiB: number): Promise<GpuInfo | null> {
  if (!isAppleSilicon()) {
    return null;
  }
  let overrideMiB: number | null = null;
  try {
    const { stdout } = await execFileAsync("sysctl", ["-n", "iogpu.wired_limit_mb"]);
    const parsed = Number(stdout.trim());
    if (Number.isFinite(parsed)) {
      overrideMiB = parsed;
    }
  } catch {
    // Key absent on older macOS; fall back to the fraction.
  }
  let name = "Apple Silicon GPU";
  try {
    const { stdout } = await execFileAsync("sysctl", ["-n", "machdep.cpu.brand_string"]);
    if (stdout.trim()) {
      name = `${stdout.trim()} (Metal)`;
    }
  } catch {
    // Keep the generic name.
  }
  const budgetMiB = appleWorkingSetMiB(totalRamMiB, overrideMiB);
  // Unified memory: free GPU ≈ free system RAM, capped to the working-set budget.
  const freeMiB = Math.min(budgetMiB, freeRamMiB);
  return { name, vendor: "apple", totalMiB: budgetMiB, usedMiB: Math.max(0, budgetMiB - freeMiB), freeMiB };
}

// ---- Aggregation ----

const HARDWARE_CACHE_TTL_MS = 3000;
let hardwareCache: { info: HardwareInfo; at: number } | null = null;

async function detectNvidiaGpus(): Promise<GpuInfo[]> {
  try {
    // nvidia-smi is `nvidia-smi.exe` on Windows, `nvidia-smi` elsewhere. On
    // machines without an NVIDIA GPU (incl. Apple Silicon) this throws.
    const nvidiaSmi = process.platform === "win32" ? "nvidia-smi.exe" : "nvidia-smi";
    const { stdout } = await execFileAsync(nvidiaSmi, [
      "--query-gpu=name,memory.total,memory.used,memory.free",
      "--format=csv,noheader,nounits"
    ]);
    return parseNvidiaSmi(stdout);
  } catch {
    return [];
  }
}

async function detectWindowsRegistryGpus(): Promise<GpuInfo[]> {
  try {
    const { stdout } = await execFileAsync(
      "powershell.exe",
      ["-NoProfile", "-NonInteractive", "-Command", WINDOWS_GPU_QUERY],
      { windowsHide: true }
    );
    return parseWindowsGpuJson(stdout);
  } catch {
    return [];
  }
}

async function detectLinuxAmdIntelGpus(): Promise<GpuInfo[]> {
  try {
    const { stdout } = await execFileAsync("rocm-smi", ["--showmeminfo", "vram", "--showproductname", "--json"]);
    const fromRocm = parseRocmSmiJson(stdout);
    if (fromRocm.length > 0) {
      // rocm-smi only reports AMD; still scan sysfs for Intel discrete cards.
      return [...fromRocm, ...detectLinuxSysfsGpus().filter((gpu) => gpu.vendor === "intel")];
    }
  } catch {
    // rocm-smi absent; sysfs below covers amdgpu boxes without ROCm installed.
  }
  return detectLinuxSysfsGpus();
}

export async function getHardwareInfo(): Promise<HardwareInfo> {
  const now = Date.now();
  if (hardwareCache && now - hardwareCache.at < HARDWARE_CACHE_TTL_MS) {
    return hardwareCache.info;
  }

  const totalRamMiB = bytesToMiB(os.totalmem());
  const freeRamMiB = bytesToMiB(os.freemem());

  const gpus = await detectNvidiaGpus();
  if (process.platform === "win32") {
    // nvidia-smi already reported NVIDIA cards with live free VRAM; keep
    // registry rows for the vendors it can't see (or everything if it failed).
    const nvidiaSeen = gpus.length > 0;
    const names = new Set(gpus.map((gpu) => gpu.name));
    for (const gpu of await detectWindowsRegistryGpus()) {
      if ((nvidiaSeen && gpu.vendor === "nvidia") || names.has(gpu.name)) {
        continue;
      }
      gpus.push(gpu);
    }
  } else if (process.platform === "linux") {
    gpus.push(...(await detectLinuxAmdIntelGpus()));
  }

  if (gpus.length === 0) {
    const apple = await detectAppleSiliconGpu(totalRamMiB, freeRamMiB);
    if (apple) {
      gpus.push(apple);
    }
  }

  const info: HardwareInfo = { totalRamMiB, freeRamMiB, gpus: filterGpus(gpus) };
  hardwareCache = { info, at: now };
  return info;
}
