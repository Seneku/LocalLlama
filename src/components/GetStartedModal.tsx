import {
  Check,
  Cpu,
  Download,
  ExternalLink,
  HardDrive,
  Loader,
  Package,
  Search,
  Sparkles,
  Star,
  Trash2,
  X,
  XCircle
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "../api";
import type {
  DownloadStatus,
  EstimateFit,
  FavoriteModel,
  HardwareInfo,
  LlamaCppRelease,
  LocalModel,
  ModelFile,
  ModelSearchResult,
  RemoteModelEstimate,
  RuntimeConfig,
  UseCase
} from "../shared/types";
import type { Notify } from "./Toasts";
import { ConfirmButton } from "./ui";

type Tab = "llama" | "models";

interface GetStartedModalProps {
  open: boolean;
  initialTab: Tab;
  config: RuntimeConfig | null;
  onClose(): void;
  onUseModel(path: string): void;
  notify: Notify;
}

function formatGb(bytes: number | null | undefined): string {
  if (!bytes || !Number.isFinite(bytes)) {
    return "-";
  }
  return `${(bytes / 1024 / 1024 / 1024).toLocaleString(undefined, { maximumFractionDigits: 2 })} GB`;
}

const CONTEXT_CHOICES = [4096, 8192, 16384, 32768] as const;

/**
 * Turn an accurate remote estimate into pill text. Full offload first; when
 * that is over budget, a partial-offload or --cpu-moe configuration can still
 * make the model usable, so say so instead of a blunt "too big".
 */
function describeEstimate(est: RemoteModelEstimate): { cls: EstimateFit; label: string; title: string } {
  const gb = (mib: number) => `${(mib / 1024).toFixed(1)} GB`;
  const ctx = `${Math.round(est.contextSize / 1024)}k context`;
  if (est.fit === "fits" || est.fit === "tight") {
    return {
      cls: est.fit,
      label: est.fit === "fits" ? "Fits" : "Tight",
      title: `Full offload: ~${gb(est.estimatedVramMiB)} VRAM at ${ctx} (${est.confidence} confidence).`
    };
  }
  if (est.cpuMoe && est.cpuMoe.fit !== "over") {
    return {
      cls: "tight",
      label: "Fits · CPU experts",
      title: `MoE model: with --cpu-moe it needs ~${gb(est.cpuMoe.estimatedVramMiB)} VRAM + ~${gb(est.cpuMoe.estimatedSystemRamMiB)} RAM at ${ctx}.`
    };
  }
  if (est.recommendation && est.recommendation.gpuLayers > 0) {
    return {
      cls: "tight",
      label: `Fits · ${est.recommendation.gpuLayers}${est.maxGpuLayers ? `/${est.maxGpuLayers}` : ""} layers`,
      title: `Partial offload: ${est.recommendation.gpuLayers} GPU layers ≈ ${gb(est.recommendation.estimatedVramMiB)} VRAM at ${ctx}; the rest runs from ~${gb(est.estimatedSystemRamMiB)} system RAM (slower).`
    };
  }
  return {
    cls: "over",
    label: "Too big",
    title: `Needs ~${gb(est.estimatedVramMiB)} VRAM at ${ctx} even at full offload.`
  };
}

function fitLabel(fit: EstimateFit): string {
  switch (fit) {
    case "fits":
      return "Fits";
    case "tight":
      return "Tight";
    case "over":
      return "Too big";
    default:
      return "Unknown";
  }
}

export function GetStartedModal({ open, initialTab, config, onClose, onUseModel, notify }: GetStartedModalProps) {
  const [tab, setTab] = useState<Tab>(initialTab);

  useEffect(() => {
    if (open) {
      setTab(initialTab);
    }
  }, [open, initialTab]);

  useEffect(() => {
    if (!open) {
      return;
    }
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open) {
    return null;
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal get-started" role="dialog" aria-label="Get started" onClick={(event) => event.stopPropagation()}>
        <div className="panel-title">
          <Package size={18} />
          <span>Get Started</span>
          <button className="icon-button" title="Close" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        <div className="tab-row">
          <button className={`tab ${tab === "llama" ? "active" : ""}`} onClick={() => setTab("llama")}>
            <Cpu size={15} />
            Install llama.cpp
          </button>
          <button className={`tab ${tab === "models" ? "active" : ""}`} onClick={() => setTab("models")}>
            <Download size={15} />
            Get Models
          </button>
        </div>

        {tab === "llama" ? (
          <LlamaCppGuide config={config} notify={notify} />
        ) : (
          <ModelBrowser onUseModel={onUseModel} notify={notify} />
        )}
      </div>
    </div>
  );
}

// ---------- llama.cpp guide ----------

function LlamaCppGuide({ config, notify }: { config: RuntimeConfig | null; notify: Notify }) {
  const [release, setRelease] = useState<LlamaCppRelease | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api
      .llamaCppRelease()
      .then(setRelease)
      .catch((error) => notify(error.message, "error"))
      .finally(() => setLoading(false));
  }, [notify]);

  const detected = config?.detected;
  const alreadyInstalled = Boolean(detected && (detected.cudaServer || detected.cpuServer));

  // Recommend a build kind from the release assets. We can't detect the exact
  // GPU vendor here, so present the common paths clearly.
  const assetFor = (kind: string) => release?.winAssets.find((asset) => asset.kind === kind) ?? null;
  const cudaAsset = assetFor("cuda");
  const cudartAsset = assetFor("cudart");
  const cpuAsset = assetFor("cpu");
  const hipAsset = assetFor("hip");
  const vulkanAsset = assetFor("vulkan");

  return (
    <div className="guide">
      {alreadyInstalled ? (
        <div className="guide-ok">
          <Check size={16} />
          <span>llama.cpp is already detected at <code>{config?.llamaRoot}</code>. You're set — head to the Models tab.</span>
        </div>
      ) : null}

      <p className="modal-hint">
        LocalLlama runs your local <strong>llama.cpp</strong> build — it doesn't bundle one. Grab a prebuilt Windows
        release below, unzip it, and point LocalLlama at the folder in <strong>Settings</strong>.
        {release ? <> Current release: <strong>{release.tag}</strong>.</> : null}
      </p>

      <ol className="guide-steps">
        <li>
          <strong>Pick the build for your hardware</strong> and download it from the release page:
          <div className="build-options">
            <BuildOption
              title="NVIDIA GPU (CUDA)"
              note="Fastest on NVIDIA. Also download the matching cudart runtime package unless you have the CUDA toolkit installed."
              asset={cudaAsset}
              extra={cudartAsset}
            />
            <BuildOption title="AMD GPU (HIP/ROCm)" note="For Radeon GPUs." asset={hipAsset} />
            <BuildOption title="Any GPU (Vulkan)" note="Vendor-neutral GPU acceleration." asset={vulkanAsset} />
            <BuildOption title="CPU only" note="No GPU acceleration; works everywhere." asset={cpuAsset} />
          </div>
        </li>
        <li>
          <strong>Unzip it</strong> somewhere permanent, e.g. <code>C:\llama.cpp</code>. The zip contains{" "}
          <code>llama-server.exe</code> and <code>llama-bench.exe</code> at its root. For a CUDA build, extract the{" "}
          <code>cudart-*</code> package into the <em>same folder</em> so the DLLs sit next to the exe.
        </li>
        <li>
          <strong>Point LocalLlama at it</strong>: open Settings and set the llama.cpp root to that folder. The pills
          turn green when the binaries are found.
        </li>
      </ol>

      <div className="guide-links">
        <a href={release?.htmlUrl ?? "https://github.com/ggml-org/llama.cpp/releases"} target="_blank" rel="noreferrer noopener">
          <ExternalLink size={14} /> llama.cpp releases
        </a>
        <a href="https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md" target="_blank" rel="noreferrer noopener">
          <ExternalLink size={14} /> Build / setup docs
        </a>
      </div>
      {loading ? <div className="muted-line">Fetching the latest release…</div> : null}
    </div>
  );
}

function BuildOption({
  title,
  note,
  asset,
  extra
}: {
  title: string;
  note: string;
  asset: LlamaCppRelease["winAssets"][number] | null;
  extra?: LlamaCppRelease["winAssets"][number] | null;
}) {
  return (
    <div className="build-option">
      <div className="build-option-head">
        <strong>{title}</strong>
        {asset ? <small>{formatGb(asset.size)}</small> : null}
      </div>
      <p>{note}</p>
      {asset ? (
        <a href={asset.url} target="_blank" rel="noreferrer noopener">
          <Download size={13} /> {asset.name}
        </a>
      ) : (
        <span className="muted-line">Not in this release</span>
      )}
      {extra ? (
        <a href={extra.url} target="_blank" rel="noreferrer noopener">
          <Download size={13} /> {extra.name}
        </a>
      ) : null}
    </div>
  );
}

// ---------- model browser ----------

function ModelBrowser({ onUseModel, notify }: { onUseModel(path: string): void; notify: Notify }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<ModelSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [selected, setSelected] = useState<ModelSearchResult | null>(null);
  const [files, setFiles] = useState<ModelFile[]>([]);
  const [hardware, setHardware] = useState<HardwareInfo | null>(null);
  const [filesLoading, setFilesLoading] = useState(false);
  const [download, setDownload] = useState<DownloadStatus | null>(null);
  const [local, setLocal] = useState<LocalModel[]>([]);
  const [favorites, setFavorites] = useState<FavoriteModel[]>([]);
  const [recommended, setRecommended] = useState<ModelSearchResult[]>([]);
  const [maxParamsB, setMaxParamsB] = useState<number | null>(null);
  const [view, setView] = useState<"recommended" | "search" | "favorites">("recommended");
  const [useCaseFilter, setUseCaseFilter] = useState<UseCase | "all">("all");
  const [context, setContext] = useState<number>(8192);
  const [estimates, setEstimates] = useState<Record<string, RemoteModelEstimate | "loading" | "error">>({});
  const estimateInFlight = useRef(new Set<string>());

  const fetchEstimate = useCallback(
    (modelId: string, file: ModelFile, ctx: number) => {
      const key = `${modelId}/${file.filename}@${ctx}`;
      if (estimateInFlight.current.has(key)) {
        return;
      }
      estimateInFlight.current.add(key);
      setEstimates((prev) => (prev[file.filename] ? prev : { ...prev, [file.filename]: "loading" }));
      api
        .remoteEstimate(modelId, file.filename, ctx, file.sizeBytes)
        .then((estimate) => setEstimates((prev) => ({ ...prev, [file.filename]: estimate })))
        .catch(() => setEstimates((prev) => ({ ...prev, [file.filename]: "error" })))
        .finally(() => estimateInFlight.current.delete(key));
    },
    []
  );

  // Upgrade the coarse size-only pills to real estimator verdicts for the
  // plausible download candidates (coarse fits/tight, capped at 6 — the
  // server additionally limits ranged reads to 2 in flight).
  const prefetchEstimates = useCallback(
    (modelId: string, fileList: ModelFile[], ctx: number) => {
      fileList
        .filter((file) => file.fit === "fits" || file.fit === "tight")
        .slice(0, 6)
        .forEach((file) => fetchEstimate(modelId, file, ctx));
    },
    [fetchEstimate]
  );

  const runSearch = useCallback(
    async (term: string) => {
      setSearching(true);
      try {
        setResults(await api.searchModels(term));
      } catch (error) {
        notify(error instanceof Error ? error.message : String(error), "error");
      } finally {
        setSearching(false);
      }
    },
    [notify]
  );

  const refreshLocal = useCallback(() => {
    api.localModels().then(setLocal).catch(() => undefined);
  }, []);

  // Initial hardware-tailored recommendations + popular search + favourites + local.
  useEffect(() => {
    api
      .recommendedModels()
      .then((response) => {
        setRecommended(response.models);
        setMaxParamsB(response.maxParamsB);
        setHardware(response.hardware);
      })
      .catch(() => undefined);
    void runSearch("");
    refreshLocal();
    api.favorites().then(setFavorites).catch(() => undefined);
  }, [runSearch, refreshLocal]);

  const isFavorite = (id: string) => favorites.some((favorite) => favorite.id === id);

  async function toggleFavorite(model: ModelSearchResult) {
    try {
      setFavorites(isFavorite(model.id) ? await api.removeFavorite(model.id) : await api.addFavorite(model));
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), "error");
    }
  }

  // Poll download progress while a download is active.
  const downloading = download?.state === "downloading";
  useEffect(() => {
    if (!downloading) {
      return;
    }
    const timer = window.setInterval(() => {
      api
        .downloadStatus()
        .then((status) => {
          setDownload(status);
          if (status.state === "completed") {
            notify(`Downloaded ${status.filename}.`, "success");
            refreshLocal();
          } else if (status.state === "failed" && status.error) {
            notify(status.error, "error");
          }
        })
        .catch(() => undefined);
    }, 700);
    return () => window.clearInterval(timer);
  }, [downloading, notify, refreshLocal]);

  async function selectModel(model: ModelSearchResult) {
    setSelected(model);
    setFiles([]);
    setEstimates({});
    setFilesLoading(true);
    try {
      const response = await api.modelFiles(model.id);
      setFiles(response.files);
      setHardware(response.hardware);
      prefetchEstimates(model.id, response.files, context);
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), "error");
    } finally {
      setFilesLoading(false);
    }
  }

  function changeContext(ctx: number) {
    setContext(ctx);
    setEstimates({});
    if (selected) {
      // Header bytes are cached server-side, so re-estimating at a new
      // context is metadata math, not another download.
      prefetchEstimates(selected.id, files, ctx);
    }
  }

  async function startDownload(file: ModelFile) {
    if (!selected) {
      return;
    }
    try {
      setDownload(await api.startDownload(selected.id, file.filename));
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), "error");
    }
  }

  async function cancelDownload() {
    try {
      setDownload(await api.cancelDownload());
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), "error");
    }
  }

  async function deleteLocal(name: string) {
    try {
      await api.deleteLocalModel(name);
      refreshLocal();
      notify(`Deleted ${name}.`, "success");
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), "error");
    }
  }

  const gpu = hardware?.gpus[0];
  const listModels =
    view === "recommended"
      ? recommended.filter((model) => useCaseFilter === "all" || model.useCase === useCaseFilter)
      : view === "favorites"
        ? favorites
        : results;
  const useCases = [...new Set(recommended.map((model) => model.useCase).filter(Boolean))] as UseCase[];

  return (
    <div className="model-browser">
      <form
        className="model-search"
        onSubmit={(event) => {
          event.preventDefault();
          setView("search");
          void runSearch(query);
        }}
      >
        <Search size={16} />
        <input
          value={query}
          placeholder="Search Hugging Face GGUF models (e.g. llama 3.1 8b, qwen coder)…"
          spellCheck={false}
          onChange={(event) => setQuery(event.target.value)}
        />
        <button type="submit" disabled={searching}>
          {searching ? "Searching…" : "Search"}
        </button>
      </form>

      {gpu ? (
        <div className="hw-line">
          <HardDrive size={14} />
          <span>
            {gpu.name} — {formatGb((gpu.totalMiB ?? 0) * 1024 * 1024)} VRAM
            {gpu.freeMiB !== null ? <> ({formatGb(gpu.freeMiB * 1024 * 1024)} free now)</> : null}. Pills marked ~ are
            size-only guesses; selecting a model reads its header from Hugging Face and upgrades them to exact
            estimates at your chosen context size.
          </span>
        </div>
      ) : null}

      <div className="log-toolbar model-view-toggle">
        <button className={`chip ${view === "recommended" ? "active" : ""}`} onClick={() => setView("recommended")}>
          <Sparkles size={13} />
          Recommended
        </button>
        <button className={`chip ${view === "search" ? "active" : ""}`} onClick={() => setView("search")}>
          Search results
        </button>
        <button className={`chip ${view === "favorites" ? "active" : ""}`} onClick={() => setView("favorites")}>
          <Star size={13} fill={view === "favorites" ? "currentColor" : "none"} />
          Favourites <small>{favorites.length}</small>
        </button>
      </div>

      {view === "recommended" ? (
        <>
          <div className="recommend-note">
            Popular GGUF models that should fit your {gpu ? gpu.name : "hardware"}
            {maxParamsB ? <> — up to ~{maxParamsB}B active params at a Q4 quant</> : null}. Mirrors of the same base
            model are collapsed; sorted by downloads.
          </div>
          {useCases.length > 1 ? (
            <div className="log-toolbar usecase-chips">
              <button className={`chip ${useCaseFilter === "all" ? "active" : ""}`} onClick={() => setUseCaseFilter("all")}>
                All
              </button>
              {useCases.map((useCase) => (
                <button
                  key={useCase}
                  className={`chip ${useCaseFilter === useCase ? "active" : ""}`}
                  onClick={() => setUseCaseFilter(useCase)}
                >
                  {useCase}
                  <small>{recommended.filter((model) => model.useCase === useCase).length}</small>
                </button>
              ))}
            </div>
          ) : null}
        </>
      ) : null}

      <div className="model-columns">
        <div className="model-list">
          {listModels.length === 0 ? (
            <div className="empty">
              {view === "favorites"
                ? "No favourites yet — tap the star on a model to save it here."
                : view === "recommended"
                  ? "No recommendations right now."
                  : searching
                    ? "Searching…"
                    : "No models found."}
            </div>
          ) : (
            listModels.map((model) => (
              <div key={model.id} className={`model-row-wrap ${selected?.id === model.id ? "active" : ""}`}>
                <button className="model-row" title={model.id} onClick={() => selectModel(model)}>
                  <span className="model-name">
                    <span className="id-text">{model.id}</span>
                    {model.gated ? <em className="gated-tag">gated</em> : null}
                  </span>
                  <small>
                    ↓ {model.downloads.toLocaleString()} · ♥ {model.likes.toLocaleString()}
                    {model.pipelineTag ? ` · ${model.pipelineTag}` : ""}
                    {model.mirrorCount ? ` · +${model.mirrorCount} other uploader${model.mirrorCount > 1 ? "s" : ""}` : ""}
                  </small>
                </button>
                <button
                  className={`star-btn ${isFavorite(model.id) ? "on" : ""}`}
                  title={isFavorite(model.id) ? "Remove from favourites" : "Add to favourites"}
                  onClick={() => toggleFavorite(model)}
                >
                  <Star size={16} fill={isFavorite(model.id) ? "currentColor" : "none"} />
                </button>
              </div>
            ))
          )}
        </div>

        <div className="model-files">
          {selected ? (
            <div className="files-header">
              <span className="fh-name" title={selected.id}>
                {selected.id}
              </span>
              <a
                className="hf-link"
                href={`https://huggingface.co/${selected.id}`}
                target="_blank"
                rel="noreferrer noopener"
                title="Open this model's page on Hugging Face"
              >
                <ExternalLink size={13} /> Hugging Face
              </a>
            </div>
          ) : null}
          {selected && files.length > 0 ? (
            <div className="context-chips" title="Fit estimates account for the KV cache at this context size">
              <span>Fit at</span>
              {CONTEXT_CHOICES.map((ctx) => (
                <button
                  key={ctx}
                  className={`chip ${context === ctx ? "active" : ""}`}
                  onClick={() => changeContext(ctx)}
                >
                  {ctx / 1024}k
                </button>
              ))}
            </div>
          ) : null}
          {!selected ? (
            <div className="empty">Select a model to see its GGUF files.</div>
          ) : filesLoading ? (
            <div className="empty">Loading files…</div>
          ) : files.length === 0 ? (
            <div className="empty">No .gguf files in this repo.</div>
          ) : (
            files.map((file) => {
              const isActive = download?.filename === file.filename && download.modelId === selected.id;
              const pct =
                download && download.totalBytes
                  ? Math.min(100, Math.round((download.receivedBytes / download.totalBytes) * 100))
                  : 0;
              const estimate = estimates[file.filename];
              const accurate = estimate && estimate !== "loading" && estimate !== "error" ? describeEstimate(estimate) : null;
              const fitClass = accurate ? accurate.cls : file.fit;
              return (
                <div key={file.filename} className={`file-row ${fitClass}`} title={file.filename}>
                  <div className="file-head">
                    <span className="file-name">
                      {file.quant ? (
                        <em
                          className={`quant-tag ${file.recommended ? "recommended" : ""}`}
                          title={file.recommendReason ?? undefined}
                        >
                          {file.recommended ? "★ " : ""}
                          {file.quant}
                        </em>
                      ) : null}
                      <span className="fname-text">{file.filename}</span>
                    </span>
                    {accurate ? (
                      <span className={`fit-pill ${accurate.cls}`} title={accurate.title}>
                        {accurate.label}
                      </span>
                    ) : estimate === "loading" ? (
                      <span className={`fit-pill ${file.fit}`} title="Reading the model header from Hugging Face…">
                        checking…
                      </span>
                    ) : (
                      <button
                        className={`fit-pill ${file.fit} approx`}
                        title="Approximate (file size only). Click to read the model header and compute an exact estimate at the selected context."
                        onClick={() => fetchEstimate(selected.id, file, context)}
                      >
                        ~{fitLabel(file.fit)}
                      </button>
                    )}
                  </div>
                  <div className="file-meta">
                    <span>{formatGb(file.sizeBytes)}</span>
                    {isActive && download?.state === "downloading" ? (
                      <div className="dl-progress">
                        <div className="dl-bar">
                          <i style={{ width: `${pct}%` }} />
                        </div>
                        <span>{download.totalBytes ? `${pct}%` : formatGb(download.receivedBytes)}</span>
                        <button className="icon-button" title="Cancel download" onClick={cancelDownload}>
                          <XCircle size={15} />
                        </button>
                      </div>
                    ) : (
                      <button
                        className="download-btn"
                        onClick={() => startDownload(file)}
                        disabled={download?.state === "downloading"}
                      >
                        {download?.state === "downloading" ? <Loader size={14} /> : <Download size={14} />}
                        Download
                      </button>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {local.length > 0 ? (
        <div className="local-models">
          <div className="panel-title compact">
            <HardDrive size={16} />
            <span>Downloaded models</span>
          </div>
          {local.map((model) => (
            <div key={model.path} className="local-row">
              <span className="file-name">{model.name}</span>
              <span className="local-size">{formatGb(model.sizeBytes)}</span>
              <button className="use-btn" onClick={() => onUseModel(model.path)}>
                <Check size={14} /> Use in profile
              </button>
              <ConfirmButton
                className="icon-button"
                title="Delete downloaded model"
                confirmLabel="Sure?"
                onConfirm={() => deleteLocal(model.name)}
              >
                <Trash2 size={14} />
              </ConfirmButton>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
