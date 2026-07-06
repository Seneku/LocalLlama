import { spawn } from "node:child_process";

const commands = [
  {
    name: "api",
    command: "bun",
    args: ["--watch", "server/index.ts"],
    env: { LLAMATUNER_PORT: "3174" }
  },
  {
    name: "ui",
    command: "bun",
    args: ["x", "vite", "--host", "127.0.0.1"]
  }
];

const children = commands.map(({ name, command, args, env }) => {
  const child = spawn(command, args, {
    stdio: "pipe",
    shell: false,
    windowsHide: true,
    env: { ...process.env, ...env }
  });

  child.stdout.on("data", (chunk) => process.stdout.write(`[${name}] ${chunk}`));
  child.stderr.on("data", (chunk) => process.stderr.write(`[${name}] ${chunk}`));
  child.on("exit", (code) => {
    if (code && code !== 0) {
      process.exitCode = code;
      stopAll();
    }
  });

  return child;
});

function stopAll() {
  for (const child of children) {
    if (!child.killed) {
      child.kill();
    }
  }
}

process.on("SIGINT", () => {
  stopAll();
  process.exit(0);
});
process.on("SIGTERM", () => {
  stopAll();
  process.exit(0);
});
