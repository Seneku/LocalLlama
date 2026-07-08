import { describe, expect, test } from "bun:test";

import { exampleProfiles } from "../server/defaultProfiles";
import { buildCommand } from "../server/llama";
import type { RuntimePaths } from "../server/paths";

const paths: RuntimePaths = {
  llamaRoot: "C:\\llama.cpp",
  cudaServerPath: "C:\\llama.cpp\\dist-cuda\\llama-server.exe",
  cpuServerPath: "C:\\llama.cpp\\build\\bin\\llama-server.exe",
  cudaBenchPath: "C:\\llama.cpp\\dist-cuda\\llama-bench.exe",
  cpuBenchPath: "C:\\llama.cpp\\build\\bin\\llama-bench.exe",
  dataPath: "C:\\LocalLlama\\data"
};

const exists = (filePath: string) =>
  [
    paths.cudaServerPath,
    paths.cpuServerPath,
    paths.cudaBenchPath,
    paths.cpuBenchPath,
    ...exampleProfiles.map((profile) => profile.modelPath)
  ].includes(filePath);

describe("buildCommand", () => {
  test("generates the Gemma4 CUDA coding command", () => {
    const profile = exampleProfiles.find((item) => item.id === "gemma4-coding")!;
    const preview = buildCommand(profile, { paths, defaultThreads: 12, fileExists: exists });

    expect(preview.backend).toBe("CUDA");
    expect(preview.args).toContain("-ngl");
    expect(preview.args).toContain("999");
    expect(preview.args).toContain("--reasoning");
    expect(preview.args).toContain("off");
    expect(preview.args).toContain("--threads");
    expect(preview.args).toContain("12");
  });

  test("falls back to CPU without GPU layer args", () => {
    const profile = exampleProfiles.find((item) => item.id === "gemma4-general")!;
    const preview = buildCommand(profile, {
      paths,
      defaultThreads: 8,
      fileExists: (filePath) => filePath !== paths.cudaServerPath && exists(filePath)
    });

    expect(preview.backend).toBe("CPU");
    expect(preview.args).not.toContain("-ngl");
    expect(preview.warnings).toContain("GPU layers are ignored when the CPU backend is selected.");
    expect(preview.args).toContain("auto");
  });

  test("generates the Orinth bundled MTP command", () => {
    const profile = exampleProfiles.find((item) => item.id === "orinth9b-mtp-coding")!;
    const preview = buildCommand(profile, { paths, defaultThreads: 16, fileExists: exists });

    expect(preview.args).toContain("--jinja");
    expect(preview.args).toContain("-ctk");
    expect(preview.args).toContain("q8_0");
    expect(preview.args).toContain("-ctv");
    expect(preview.args).toContain("--spec-type");
    expect(preview.args).toContain("draft-mtp");
    expect(preview.args).toContain("--spec-draft-n-max");
    expect(preview.args).toContain("3");
    // Draft-model KV cache quant.
    expect(preview.args).toContain("-ctkd");
    expect(preview.args).toContain("-ctvd");
    // Temperature is always emitted (defaults to the llama.cpp default).
    expect(preview.args).toContain("--temp");
    expect(preview.args).toContain("0.8");
  });

  test("emits fit, no-mmap, temperature, and draft-p-min flags", () => {
    const orinth = exampleProfiles.find((item) => item.id === "orinth9b-mtp-coding")!;
    const profile = {
      ...orinth,
      mmap: false,
      fit: true,
      fitTargetMiB: 512,
      temperature: 0,
      speculative: { ...orinth.speculative, draftPMin: 0.75 }
    };
    const line = buildCommand(profile, { paths, defaultThreads: 8, fileExists: exists }).args.join(" ");

    expect(line).toContain("--no-mmap");
    expect(line).toContain("--fit on");
    expect(line).toContain("--fit-target 512");
    expect(line).toContain("--temp 0");
    expect(line).toContain("--spec-draft-p-min 0.75");
  });

  test("emits MoE expert-offload flags", () => {
    const gemma = exampleProfiles.find((item) => item.id === "gemma4-coding")!;
    const all = buildCommand({ ...gemma, cpuMoe: true }, { paths, defaultThreads: 12, fileExists: exists }).args.join(" ");
    expect(all).toContain("--cpu-moe");

    const firstN = buildCommand(
      { ...gemma, cpuMoe: false, nCpuMoe: 20 },
      { paths, defaultThreads: 12, fileExists: exists }
    ).args.join(" ");
    expect(firstN).toContain("--n-cpu-moe 20");

    // --cpu-moe wins over --n-cpu-moe when both are set.
    const both = buildCommand(
      { ...gemma, cpuMoe: true, nCpuMoe: 20 },
      { paths, defaultThreads: 12, fileExists: exists }
    ).args.join(" ");
    expect(both).toContain("--cpu-moe");
    expect(both).not.toContain("--n-cpu-moe");
  });

  test("omits fit-target and no-mmap at their defaults", () => {
    const profile = exampleProfiles.find((item) => item.id === "gemma4-coding")!;
    const line = buildCommand(profile, { paths, defaultThreads: 12, fileExists: exists }).args.join(" ");

    expect(line).not.toContain("--no-mmap"); // mmap defaults on
    expect(line).not.toContain("--fit"); // fit defaults off
  });

  test("uses manual overrides for custom tuning", () => {
    const profile = {
      ...exampleProfiles[0],
      host: "0.0.0.0",
      port: 8099,
      contextSize: 65536,
      threadsMode: "manual" as const,
      threads: 6,
      parallelSlots: 2
    };
    const preview = buildCommand(profile, { paths, defaultThreads: 16, fileExists: exists });

    expect(preview.endpoint).toBe("http://0.0.0.0:8099");
    expect(preview.args).toContain("65536");
    expect(preview.args).toContain("6");
    expect(preview.args).toContain("-np");
    expect(preview.args).toContain("2");
  });
});
