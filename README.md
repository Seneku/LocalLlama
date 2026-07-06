# LlamaTuner

A local web UI for tuning, launching, and benchmarking [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` configurations. Build named profiles, see the exact command before you run it, get a GGUF-accurate VRAM estimate for your GPU, run `llama-bench` sweeps, and compare the results — without hand-editing `.cmd` files.

> **Platform:** Windows-focused. LlamaTuner shells out to `nvidia-smi`, `taskkill`, and `.exe` binaries. It runs against a local llama.cpp build you already have.

## Features

- **Profiles** — save reusable server configs (model, context, GPU layers, KV-cache quant, parallel slots, speculative decoding, etc.), with a live-updating preview of the exact `llama-server` command.
- **Accurate VRAM estimation** — parses the GGUF tensor table for exact per-layer weights and KV geometry, and models the things simple heuristics miss: interleaved **sliding-window attention** (Gemma-family), **hybrid SSM** layers (Qwen3.5/recurrent), **MTP/nextn** blocks, tied-embedding output heads, and llama.cpp's offload order. Calibrated against measured `llama-server` allocations.
- **GPU-layer auto-recommend** — inverts the estimator to suggest the largest `-ngl` that fits your currently-free VRAM (or to offload *more* when there's headroom), with one-click apply.
- **Benchmarking** — run `llama-bench` from a profile, watch live logs, and keep a history of results with prompt/generation throughput, latency, and a blended score. Filter by profile/backend/status and compare runs side by side.
- **Runtime management** — start/stop/restart the server, stream stdout/stderr, health checks, and child-process-tree cleanup so llama.cpp doesn't linger holding VRAM.
- **Settings** — point LlamaTuner at your llama.cpp install (root + optional per-binary overrides) from the UI; no env vars required.

## Requirements

- [Bun](https://bun.sh) (the runtime, package manager, and test runner)
- A local llama.cpp build with `llama-server.exe` and `llama-bench.exe`
- Optional: an NVIDIA GPU with `nvidia-smi` on `PATH` for VRAM fit estimates (the app still works CPU-only without it)

## Getting started

```sh
bun install
bun run dev
```

`bun run dev` starts the API (port `3174`) and the Vite dev server (port `5173`) with hot reload. Open **http://127.0.0.1:5173**.

On first launch, if llama.cpp isn't detected you'll see a banner prompting you to open **Settings** and set your llama.cpp root. Then create a profile, point it at a `.gguf`, and hit **Start**.

### Production / standalone

```sh
bun run build     # type-check + bundle the frontend into dist/
bun run start     # serve the built app + API from a single process (port 4187)
```

`bun run restart` (or `scripts/restart-llamatuner.cmd`) stops a running standalone server and its llama.cpp children, rebuilds, and relaunches — handy since `start` runs without `--watch`.

## Configuration

Paths resolve in order: **Settings (saved in the UI) → environment variables → defaults derived from the llama.cpp root.** Settings are stored in `data/settings.json` (git-ignored).

| Variable | Purpose |
| --- | --- |
| `LLAMATUNER_LLAMA_ROOT` | Root of your llama.cpp build (binary paths derive from it) |
| `LLAMATUNER_CUDA_SERVER` / `LLAMATUNER_CPU_SERVER` | Override the `llama-server` binary paths |
| `LLAMATUNER_CUDA_BENCH` / `LLAMATUNER_CPU_BENCH` | Override the `llama-bench` binary paths |
| `LLAMATUNER_DATA_DIR` | Where profiles, benchmark history, and settings are stored (default `./data`) |
| `LLAMATUNER_PORT` / `LLAMATUNER_HOST` | Standalone server bind address (default `127.0.0.1:4187`) |

## Development

```sh
bun test          # run the test suite
bunx tsc --noEmit # type-check
```

- `server/` — `node:http` API: profile/benchmark stores (atomic JSON), GGUF parsing, VRAM estimation, and llama.cpp process management.
- `src/` — React 19 + Vite frontend.
- `tests/` — command building, benchmark parsing, VRAM estimation (dense / SWA / hybrid), and settings resolution.

Your local `data/` (profiles, benchmark history, settings) is git-ignored, so it stays on your machine.

## License

MIT
