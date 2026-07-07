import {
  Check,
  Cpu,
  Download,
  ExternalLink,
  HardDrive,
  Loader,
  Package,
  Search,
  Trash2,
  X,
  XCircle
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "../api";
import type {
  DownloadStatus,
  EstimateFit,
  HardwareInfo,
  LlamaCppRelease,
  LocalModel,
  ModelFile,
  ModelSearchResult,
  RuntimeConfig
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

  // Initial popular list + local models.
  useEffect(() => {
    void runSearch("");
    refreshLocal();
  }, [runSearch, refreshLocal]);

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
    setFilesLoading(true);
    try {
      const response = await api.modelFiles(model.id);
      setFiles(response.files);
      setHardware(response.hardware);
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), "error");
    } finally {
      setFilesLoading(false);
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

  return (
    <div className="model-browser">
      <form
        className="model-search"
        onSubmit={(event) => {
          event.preventDefault();
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
            {gpu.name} — {formatGb((gpu.totalMiB ?? 0) * 1024 * 1024)} VRAM (
            {formatGb((gpu.freeMiB ?? 0) * 1024 * 1024)} free now). Fit badges compare model size against your card's
            total VRAM; the exact per-run estimate appears once a model is downloaded and used in a profile.
          </span>
        </div>
      ) : null}

      <div className="model-columns">
        <div className="model-list">
          {results.length === 0 ? (
            <div className="empty">{searching ? "Searching…" : "No models found."}</div>
          ) : (
            results.map((model) => (
              <button
                key={model.id}
                className={`model-row ${selected?.id === model.id ? "active" : ""}`}
                title={model.id}
                onClick={() => selectModel(model)}
              >
                <span className="model-name">
                  <span className="id-text">{model.id}</span>
                  {model.gated ? <em className="gated-tag">gated</em> : null}
                </span>
                <small>
                  ↓ {model.downloads.toLocaleString()} · ♥ {model.likes.toLocaleString()}
                  {model.pipelineTag ? ` · ${model.pipelineTag}` : ""}
                </small>
              </button>
            ))
          )}
        </div>

        <div className="model-files">
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
              return (
                <div key={file.filename} className={`file-row ${file.fit}`} title={file.filename}>
                  <div className="file-head">
                    <span className="file-name">
                      {file.quant ? <em className="quant-tag">{file.quant}</em> : null}
                      <span className="fname-text">{file.filename}</span>
                    </span>
                    <span className={`fit-pill ${file.fit}`}>{fitLabel(file.fit)}</span>
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
