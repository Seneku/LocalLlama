import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";

import { buildCommand, validateProfileForLaunch } from "./llama";
import type { CommandPreview, LlamaProfile, RuntimeLog, RuntimeStatus } from "../src/shared/types";

const MAX_LOGS = 800;
const HEALTH_CACHE_TTL_MS = 2000;

/**
 * Reliably terminate a process tree on Windows via `taskkill /T /F`, awaiting
 * completion so the caller knows the child is reaped. Failures are reported
 * through the provided logger instead of throwing.
 */
export function killTree(pid: number, log: (message: string) => void): Promise<void> {
  return new Promise((resolve) => {
    const child = spawn("taskkill.exe", ["/PID", String(pid), "/T", "/F"], {
      shell: false,
      windowsHide: true
    });
    child.once("error", (error) => {
      log(`taskkill failed for PID ${pid}: ${error.message}`);
      resolve();
    });
    child.once("exit", (code) => {
      if (code !== 0) {
        log(`taskkill exited with code ${code ?? "null"} for PID ${pid}.`);
      }
      resolve();
    });
  });
}

export class RuntimeManager {
  private process: ChildProcessWithoutNullStreams | null = null;
  private status: RuntimeStatus = {
    state: "stopped",
    pid: null,
    profileId: null,
    profileName: null,
    startedAt: null,
    exitedAt: null,
    exitCode: null,
    signal: null,
    endpoint: null,
    health: "unknown",
    command: null
  };
  private logs: RuntimeLog[] = [];
  private nextLogId = 1;
  private streamBuffers: Record<"stdout" | "stderr", string> = {
    stdout: "",
    stderr: ""
  };
  private healthCheckedAt = 0;

  getLogs(): RuntimeLog[] {
    return this.logs;
  }

  async getStatus(): Promise<RuntimeStatus> {
    if (this.status.state === "running" || this.status.state === "starting") {
      // Cache the health probe so polling does not hammer the server with a
      // network request on every status call.
      const now = Date.now();
      if (now - this.healthCheckedAt >= HEALTH_CACHE_TTL_MS) {
        this.status.health = await this.checkHealth(this.status.endpoint);
        this.healthCheckedAt = now;
      }
    }
    return this.status;
  }

  start(profile: LlamaProfile): RuntimeStatus {
    if (this.process && !this.process.killed) {
      throw new Error(`A server is already running with profile ${this.status.profileName ?? "unknown"}.`);
    }

    const validation = validateProfileForLaunch(profile);
    if (validation.errors.length > 0) {
      throw new Error(validation.errors.join("\n"));
    }

    const command = validation.command;
    this.appendLog("system", `Starting ${profile.name}`);
    this.appendLog("system", command.display);
    this.process = spawn(command.executable, command.args, {
      cwd: process.env.LLAMATUNER_LLAMA_ROOT ?? "E:\\Projects\\llama.cpp",
      shell: false,
      windowsHide: true
    });

    this.status = {
      state: "starting",
      pid: this.process.pid ?? null,
      profileId: profile.id,
      profileName: profile.name,
      startedAt: new Date().toISOString(),
      exitedAt: null,
      exitCode: null,
      signal: null,
      endpoint: command.endpoint,
      health: "unknown",
      command
    };

    this.process.stdout.on("data", (chunk: Buffer) => this.appendChunk("stdout", chunk));
    this.process.stderr.on("data", (chunk: Buffer) => this.appendChunk("stderr", chunk));
    this.process.once("spawn", () => {
      this.status.state = "running";
      this.appendLog("system", `Process started with PID ${this.process?.pid ?? "unknown"}.`);
    });
    this.process.once("error", (error) => {
      this.status.state = "exited";
      this.status.exitedAt = new Date().toISOString();
      this.appendLog("system", `Start failed: ${error.message}`);
      this.process = null;
    });
    this.process.once("exit", (code, signal) => {
      this.flushBuffers();
      this.status.state = "exited";
      this.status.exitedAt = new Date().toISOString();
      this.status.exitCode = code;
      this.status.signal = signal;
      this.status.health = "unknown";
      this.appendLog("system", `Process exited with code ${code ?? "null"} signal ${signal ?? "null"}.`);
      this.process = null;
    });

    return this.status;
  }

  async stop(): Promise<RuntimeStatus> {
    if (!this.process || this.process.killed) {
      this.status.state = this.status.state === "exited" ? "exited" : "stopped";
      return this.status;
    }

    const pid = this.process.pid;
    this.appendLog("system", `Stopping process ${pid ?? "unknown"}.`);
    this.process.kill();

    // On Windows plain kill() does not terminate the child process tree, and
    // `.killed` only means a signal was delivered. Always run taskkill /T /F to
    // reliably reap the whole tree so VRAM is released.
    if (process.platform === "win32" && pid) {
      await killTree(pid, (message) => this.appendLog("system", message));
    }

    return this.status;
  }

  preview(profile: LlamaProfile): CommandPreview {
    return buildCommand(profile);
  }

  private appendChunk(stream: "stdout" | "stderr", chunk: Buffer): void {
    const text = this.streamBuffers[stream] + chunk.toString("utf8");
    const lines = text.split(/\r?\n/u);
    this.streamBuffers[stream] = lines.pop() ?? "";
    for (const line of lines) {
      if (line.trim().length > 0) {
        this.appendLog(stream, line);
      }
    }
  }

  private flushBuffers(): void {
    for (const stream of ["stdout", "stderr"] as const) {
      const line = this.streamBuffers[stream];
      if (line.trim().length > 0) {
        this.appendLog(stream, line);
      }
      this.streamBuffers[stream] = "";
    }
  }

  private appendLog(stream: RuntimeLog["stream"], line: string): void {
    this.logs.push({
      id: this.nextLogId++,
      time: new Date().toISOString(),
      stream,
      line
    });
    if (this.logs.length > MAX_LOGS) {
      this.logs.splice(0, this.logs.length - MAX_LOGS);
    }
  }

  private async checkHealth(endpoint: string | null): Promise<RuntimeStatus["health"]> {
    if (!endpoint) {
      return "unknown";
    }
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 600);
      const response = await fetch(`${endpoint}/health`, { signal: controller.signal });
      clearTimeout(timeout);
      return response.status < 500 ? "ok" : "unreachable";
    } catch {
      return "unreachable";
    }
  }
}
