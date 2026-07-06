import fs from "node:fs";

import { getDefaultThreads, getRuntimePaths, type RuntimePaths } from "./paths";
import type { CommandPreview, LlamaProfile, ResolvedBackend } from "../src/shared/types";

export interface BuildCommandOptions {
  paths?: RuntimePaths;
  defaultThreads?: number;
  fileExists?: (filePath: string) => boolean;
}

export interface LaunchValidation {
  command: CommandPreview;
  errors: string[];
}

function positiveInteger(value: number, fallback: number): number {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return Math.max(1, Math.floor(value));
}

function quoteArg(arg: string): string {
  if (arg.length === 0) {
    return "\"\"";
  }
  if (!/[\s"]/u.test(arg)) {
    return arg;
  }
  return `"${arg.replaceAll("\"", "\\\"")}"`;
}

function resolveBackend(
  mode: LlamaProfile["backendMode"],
  paths: RuntimePaths,
  fileExists: (filePath: string) => boolean
): ResolvedBackend {
  if (mode === "cuda") {
    return "CUDA";
  }
  if (mode === "cpu") {
    return "CPU";
  }
  return fileExists(paths.cudaServerPath) ? "CUDA" : "CPU";
}

export function buildCommand(profile: LlamaProfile, options: BuildCommandOptions = {}): CommandPreview {
  const paths = options.paths ?? getRuntimePaths();
  const defaultThreads = options.defaultThreads ?? getDefaultThreads();
  const fileExists = options.fileExists ?? fs.existsSync;
  const backend = resolveBackend(profile.backendMode, paths, fileExists);
  const executable = backend === "CUDA" ? paths.cudaServerPath : paths.cpuServerPath;
  const threads =
    profile.threadsMode === "manual" ? positiveInteger(profile.threads, defaultThreads) : defaultThreads;
  const warnings: string[] = [];
  const args: string[] = [
    "-m",
    profile.modelPath,
    "-c",
    String(positiveInteger(profile.contextSize, 4096)),
    "--host",
    profile.host || "127.0.0.1",
    "--port",
    String(positiveInteger(profile.port, 8080)),
    "--reasoning",
    profile.reasoning,
    "--threads",
    String(threads),
    "--threads-batch",
    String(threads)
  ];

  if (profile.modelAlias.trim()) {
    args.splice(2, 0, "-a", profile.modelAlias.trim());
  }
  if (profile.jinja) {
    args.push("--jinja");
  }
  if (profile.mlock) {
    args.push("--mlock");
  }
  if (backend === "CUDA" && profile.gpuLayers > 0) {
    args.push("-ngl", String(Math.floor(profile.gpuLayers)));
  }
  if (backend === "CPU" && profile.gpuLayers > 0) {
    warnings.push("GPU layers are ignored when the CPU backend is selected.");
  }
  if (profile.parallelSlots > 0) {
    args.push("-np", String(positiveInteger(profile.parallelSlots, 1)));
  }
  if (profile.kvCacheK) {
    args.push("-ctk", profile.kvCacheK);
  }
  if (profile.kvCacheV) {
    args.push("-ctv", profile.kvCacheV);
  }
  if (profile.speculative.enabled) {
    args.push("--spec-type", profile.speculative.type);
    if (profile.speculative.draftModelPath.trim()) {
      args.push("--spec-draft-model", profile.speculative.draftModelPath.trim());
    }
    if (profile.speculative.draftNMax > 0) {
      args.push("--spec-draft-n-max", String(Math.floor(profile.speculative.draftNMax)));
    }
    if (profile.speculative.draftGpuLayers > 0) {
      args.push("--spec-draft-ngl", String(Math.floor(profile.speculative.draftGpuLayers)));
    }
  }

  const display = [executable, ...args].map(quoteArg).join(" ");

  return {
    executable,
    args,
    display,
    backend,
    endpoint: `http://${profile.host || "127.0.0.1"}:${positiveInteger(profile.port, 8080)}`,
    serverExists: fileExists(executable),
    modelExists: fileExists(profile.modelPath),
    warnings
  };
}

export function validateProfileForLaunch(
  profile: LlamaProfile,
  options: BuildCommandOptions = {}
): LaunchValidation {
  const command = buildCommand(profile, options);
  const errors: string[] = [];

  if (!command.serverExists) {
    errors.push(`llama-server not found: ${command.executable}`);
  }
  if (!command.modelExists) {
    errors.push(`model not found: ${profile.modelPath}`);
  }
  if (!profile.host.trim()) {
    errors.push("host is required");
  }
  if (!Number.isInteger(profile.port) || profile.port < 1 || profile.port > 65535) {
    errors.push("port must be between 1 and 65535");
  }
  if (!Number.isInteger(profile.contextSize) || profile.contextSize < 1) {
    errors.push("context size must be a positive integer");
  }

  return { command, errors };
}
