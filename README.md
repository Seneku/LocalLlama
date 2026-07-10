# LocalLlama

A local web UI for tuning, launching, and benchmarking [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` configurations. Build named profiles, see the exact command before you run it, get a GGUF-accurate VRAM estimate for your GPU, run `llama-bench` sweeps, and compare the results — without hand-editing `.cmd` files.

> **Platform:** Windows, Linux, and macOS. LocalLlama runs against a local llama.cpp build you already have. GPU/VRAM detection covers NVIDIA (`nvidia-smi`, live free VRAM), AMD and Intel discrete GPUs (Windows driver registry / Linux `rocm-smi` & sysfs), and Apple Silicon's Metal GPU modeled as unified memory on macOS (an approximate working-set budget; Intel Macs fall back to CPU).

## Download

Grab the binary for your platform from the [Releases page](https://github.com/Seneku/LocalLlama/releases). Each is a single self-contained executable — no install, no Bun, no dependencies. Run it and it starts the server and opens the app in your browser. You still need a local [llama.cpp](https://github.com/ggml-org/llama.cpp) build; point LocalLlama at it from the **Settings** screen on first run.

| Platform | Asset |
| --- | --- |
| Windows x64 | `LocalLlama-*-win-x64.exe` |
| Linux x64 | `LocalLlama-*-linux-x64` |
| macOS (Apple Silicon) | `LocalLlama-*-macos-arm64` |
| macOS (Intel) | `LocalLlama-*-macos-x64` |

> ⚠️ **The binaries are unsigned.** First-run steps per OS:
> - **Windows** — SmartScreen shows *"Windows protected your PC"*: click **More info → Run anyway**.
> - **macOS** — Gatekeeper blocks unsigned apps. Clear the quarantine flag with `xattr -d com.apple.quarantine ./LocalLlama-*-macos-*` (or right-click → **Open** once), then `chmod +x` it.
> - **Linux** — make it executable: `chmod +x ./LocalLlama-*-linux-x64`, then run it.
>
> If you'd rather not trust a prebuilt binary, [build it yourself from source](#package-as-a-standalone-executable) with `bun run package` (host platform) or `bun run package:all` (all platforms) — the result is reproducible from this repo.

## Features

- **Get started in-app** — a setup guide that recommends the right prebuilt llama.cpp Windows build for your hardware (with direct release links), plus a **model browser** that searches Hugging Face for GGUF models and downloads them straight into your models folder with live progress. Recommendations are grouped by use case (chat / coding / reasoning / vision), MoE-aware, and deduped across requant mirrors, with a ★ badge on the quant that best fits your hardware.
- **Pre-download fit checks** — reads just the GGUF header from Hugging Face (a few MB) and runs the full estimator *before* you download: exact verdicts at your chosen context size, including partial-offload and MoE `--cpu-moe` alternatives.
- **One-click Optimize** — benchmarks a short, smart series of setting combinations (GPU layers, flash attention, batch/micro-batch, KV-cache quant), skips configs that won't fit VRAM, ranks results with noise awareness, and applies the winner to your profile in one click.
- **Profiles** — save reusable server configs (model, context, GPU layers, batch sizes, flash attention, KV-cache quant, parallel slots, speculative decoding, etc.), with a live-updating preview of the exact `llama-server` command.
- **Accurate VRAM estimation** — parses the GGUF tensor table for exact per-layer weights and KV geometry, and models the things simple heuristics miss: interleaved **sliding-window attention** (Gemma-family), **hybrid SSM** layers (Qwen3.5/recurrent), **MTP/nextn** blocks, tied-embedding output heads, and llama.cpp's offload order. Calibrated against measured `llama-server` allocations.
- **GPU-layer auto-recommend** — inverts the estimator to suggest the largest `-ngl` that fits your currently-free VRAM (or to offload *more* when there's headroom), with one-click apply.
- **Benchmarking** — run `llama-bench` from a profile, watch live logs, and keep a history of results with prompt/generation throughput, latency, and a blended score. Filter by profile/backend/status and compare runs side by side.
- **Runtime management** — start/stop/restart the server, stream stdout/stderr, health checks, and child-process-tree cleanup so llama.cpp doesn't linger holding VRAM.
- **Settings** — point LocalLlama at your llama.cpp install (root + optional per-binary overrides) from the UI; no env vars required.

## Requirements

- [Bun](https://bun.sh) (the runtime, package manager, and test runner)
- A local llama.cpp build with `llama-server` / `llama-bench` (`.exe` on Windows)
- Optional: a GPU for VRAM fit estimates — NVIDIA (`nvidia-smi` on `PATH`), AMD/Intel discrete (detected via the Windows driver registry or Linux `rocm-smi`/sysfs), or Apple Silicon (the app still works CPU-only without one)

## Getting started

```sh
bun install
bun run dev
```

`bun run dev` starts the API (port `3174`) and the Vite dev server (port `5173`) with hot reload. Open **http://127.0.0.1:5173**.

On first launch, if llama.cpp isn't detected you'll see a banner prompting you to open **Settings** and set your llama.cpp root. Then create a profile, point it at a `.gguf`, and hit **Start**.

### Run from source (single process)

```sh
bun run build     # type-check + bundle the frontend into dist/
bun run start     # serve the built app + API from one process (port 4187)
```

`bun run restart` (or `scripts/restart-localllama.cmd`) stops a running server and its llama.cpp children, rebuilds, and relaunches — handy since `start` runs without `--watch`.

### Package as a standalone executable

```sh
bun run package       # builds the UI, inlines it, and compiles a binary for THIS platform
bun run package:all   # cross-compiles binaries for Windows, Linux, and macOS (arm64 + x64)
```

`bun run package` produces a single self-contained binary (~110 MB — it bundles the Bun runtime and the whole UI) in `dist/`. `bun run package:all` cross-compiles all four release targets from any host (`dist/LocalLlama-win-x64.exe`, `-linux-x64`, `-macos-arm64`, `-macos-x64`). It needs no installed Bun, no `dist/` folder, and no source. Run it and it:

1. starts the API + UI server on port `4187` (falls back to the next free port if taken),
2. opens the app in your default browser.

Close the console window or press `Ctrl+C` to stop it. Profiles, benchmark history, and settings are stored in a `data/` folder next to the executable (override with `LOCALLLAMA_DATA_DIR`). Set `LOCALLLAMA_NO_OPEN=1` to skip auto-opening the browser.

You still need a local llama.cpp build; point the app at it from the **Settings** screen on first run.

## Configuration

Paths resolve in order: **Settings (saved in the UI) → environment variables → defaults derived from the llama.cpp root.** Settings are stored in `data/settings.json` (git-ignored).

| Variable | Purpose |
| --- | --- |
| `LOCALLLAMA_LLAMA_ROOT` | Root of your llama.cpp build (binary paths derive from it) |
| `LOCALLLAMA_CUDA_SERVER` / `LOCALLLAMA_CPU_SERVER` | Override the `llama-server` binary paths |
| `LOCALLLAMA_CUDA_BENCH` / `LOCALLLAMA_CPU_BENCH` | Override the `llama-bench` binary paths |
| `LOCALLLAMA_DATA_DIR` | Where profiles, benchmark history, and settings are stored (default `./data`) |
| `LOCALLLAMA_MODELS_DIR` | Where downloaded GGUF models are saved (default `<data>/models`) |
| `LOCALLLAMA_PORT` / `LOCALLLAMA_HOST` | Standalone server bind address (default `127.0.0.1:4187`) |

## Development

```sh
bun test          # run the test suite
bunx tsc --noEmit # type-check
```

- `server/` — `node:http` API: profile/benchmark stores (atomic JSON), GGUF parsing, VRAM estimation, and llama.cpp process management.
- `src/` — React 19 + Vite frontend.
- `tests/` — command building, benchmark parsing, VRAM estimation (dense / SWA / hybrid), and settings resolution.

Your local `data/` (profiles, benchmark history, settings) is git-ignored, so it stays on your machine.

### Releasing

Pushing a `v*` tag triggers the [release workflow](.github/workflows/release.yml), which runs the tests, cross-compiles the Windows/Linux/macOS binaries on a Linux runner, and publishes a GitHub Release with all four attached:

```sh
git tag v0.3.0
git push origin v0.3.0
```

## License

MIT
