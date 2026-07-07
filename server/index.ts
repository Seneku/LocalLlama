import { createApp } from "./app";
import { BenchmarkManager } from "./benchmark";
import { createBenchmarkStore } from "./benchmarkStore";
import { DownloadManager } from "./downloadManager";
import { RuntimeManager } from "./runtime";

const port = Number(process.env.LOCALLLAMA_PORT ?? 4187);
const host = process.env.LOCALLLAMA_HOST ?? "127.0.0.1";

const runtime = new RuntimeManager();
const benchmarkStore = createBenchmarkStore();
const benchmark = new BenchmarkManager(benchmarkStore);
const downloads = new DownloadManager();

let shuttingDown = false;

async function shutdown(signal: NodeJS.Signals): Promise<void> {
  if (shuttingDown) {
    return;
  }
  shuttingDown = true;
  console.log(`\nReceived ${signal}, stopping child processes...`);
  try {
    // Kill the llama-server / llama-bench process trees so they do not linger
    // holding VRAM after the manager exits, and drop any partial download.
    await Promise.allSettled([runtime.stop(), benchmark.stop(), downloads.dispose()]);
  } finally {
    process.exit(0);
  }
}

process.on("SIGINT", () => {
  void shutdown("SIGINT");
});
process.on("SIGTERM", () => {
  void shutdown("SIGTERM");
});

function listen(candidatePort: number, retries = 0): void {
  const server = createApp({ runtime, benchmarkStore, benchmark, downloads });
  server.once("error", (error: NodeJS.ErrnoException) => {
    if (error.code === "EADDRINUSE" && !process.env.LOCALLLAMA_PORT && retries < 10) {
      listen(candidatePort + 1, retries + 1);
      return;
    }
    console.error(error.message);
    process.exit(1);
  });
  server.listen(candidatePort, host, () => {
    console.log(`LocalLlama listening at http://${host}:${candidatePort}`);
  });
}

listen(port);
