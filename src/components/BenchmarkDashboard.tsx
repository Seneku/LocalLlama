import { BarChart3, Gauge, GitCompareArrows, Play, Square, Trash2, X, Zap } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { api } from "../api";
import type {
  BenchmarkCommandPreview,
  BenchmarkProfileSnapshot,
  BenchmarkRun,
  BenchmarkSettings,
  BenchmarkStatus,
  FlashAttentionMode,
  LlamaProfile,
  RuntimeLog
} from "../shared/types";
import { LogView } from "./LogView";
import type { Notify } from "./Toasts";
import { ConfirmButton, CopyButton, NumberField, SelectField, ToggleField } from "./ui";

const defaultSettings: BenchmarkSettings = {
  promptTokens: 512,
  generationTokens: 128,
  repetitions: 3,
  batchSize: 2048,
  ubatchSize: 512,
  noWarmup: false,
  flashAttention: "auto"
};

const SNAPSHOT_FIELDS: Array<{ key: keyof BenchmarkProfileSnapshot; label: string }> = [
  { key: "backendMode", label: "Backend mode" },
  { key: "contextSize", label: "Context" },
  { key: "threadsMode", label: "Threads mode" },
  { key: "threads", label: "Threads" },
  { key: "gpuLayers", label: "GPU layers" },
  { key: "parallelSlots", label: "Parallel slots" },
  { key: "kvCacheK", label: "KV cache K" },
  { key: "kvCacheV", label: "KV cache V" },
  { key: "speculativeEnabled", label: "Speculative" },
  { key: "modelPath", label: "Model" }
];

const SETTING_FIELDS: Array<{ key: keyof BenchmarkSettings; label: string }> = [
  { key: "promptTokens", label: "Prompt tokens" },
  { key: "generationTokens", label: "Gen tokens" },
  { key: "repetitions", label: "Repetitions" },
  { key: "batchSize", label: "Batch" },
  { key: "ubatchSize", label: "Ubatch" },
  { key: "flashAttention", label: "Flash attention" },
  { key: "noWarmup", label: "No warmup" }
];

interface BenchmarkDashboardProps {
  draft: LlamaProfile | null;
  selectedProfile: LlamaProfile | null;
  isDirty: boolean;
  runtimeRunning: boolean;
  busy: boolean;
  saveDraft(): Promise<LlamaProfile | null>;
  onMessage: Notify;
}

function formatNumber(value: number | null, digits = 1): string {
  if (value === null || !Number.isFinite(value)) {
    return "-";
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function formatDelta(value: number | null, previous: number | null): string {
  if (value === null || previous === null || previous === 0) {
    return "";
  }
  const delta = ((value - previous) / previous) * 100;
  const sign = delta > 0 ? "+" : "";
  return `${sign}${delta.toFixed(1)}%`;
}

function sameRuns(a: BenchmarkRun[], b: BenchmarkRun[]): boolean {
  return (
    a.length === b.length &&
    a.every((run, index) => run.id === b[index].id && run.status === b[index].status && run.completedAt === b[index].completedAt)
  );
}

export function BenchmarkDashboard({
  draft,
  selectedProfile,
  isDirty,
  runtimeRunning,
  busy,
  saveDraft,
  onMessage
}: BenchmarkDashboardProps) {
  const [settings, setSettings] = useState<BenchmarkSettings>(defaultSettings);
  const [preview, setPreview] = useState<BenchmarkCommandPreview | null>(null);
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [status, setStatus] = useState<BenchmarkStatus>({
    state: "idle",
    activeRunId: null,
    profileId: null,
    profileName: null,
    startedAt: null,
    command: null
  });
  const [logs, setLogs] = useState<RuntimeLog[]>([]);
  const [benchmarkBusy, setBenchmarkBusy] = useState(false);
  const [compareIds, setCompareIds] = useState<string[]>([]);

  const running = status.state === "running";

  const selectedRuns = useMemo(
    () => runs.filter((run) => run.profileId === selectedProfile?.id && run.status === "completed"),
    [runs, selectedProfile]
  );
  const latestRun = selectedRuns[0] ?? runs.find((run) => run.status === "completed") ?? null;
  const previousRun = selectedRuns[1] ?? null;
  const maxPrompt = Math.max(1, ...runs.map((run) => run.metrics.promptTokensPerSecond ?? 0));
  const maxGeneration = Math.max(1, ...runs.map((run) => run.metrics.generationTokensPerSecond ?? 0));
  const compareRuns = useMemo(
    () => compareIds.map((id) => runs.find((run) => run.id === id)).filter((run): run is BenchmarkRun => Boolean(run)),
    [compareIds, runs]
  );

  // Poll faster while a benchmark runs, back off when idle, skip hidden tabs.
  useEffect(() => {
    let disposed = false;
    const refresh = async () => {
      if (document.hidden) {
        return;
      }
      try {
        const [nextRuns, nextStatus, nextLogs] = await Promise.all([
          api.benchmarks(),
          api.benchmarkStatus(),
          api.benchmarkLogs()
        ]);
        if (disposed) {
          return;
        }
        setRuns((prev) => (sameRuns(prev, nextRuns) ? prev : nextRuns));
        setStatus((prev) => (JSON.stringify(prev) === JSON.stringify(nextStatus) ? prev : nextStatus));
        setLogs((prev) =>
          prev.length === nextLogs.length && prev[prev.length - 1]?.id === nextLogs[nextLogs.length - 1]?.id
            ? prev
            : nextLogs
        );
      } catch {
        // Polling stays quiet; direct actions surface their own errors.
      }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), running ? 1500 : 5000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [running]);

  useEffect(() => {
    if (!draft) {
      setPreview(null);
      return;
    }
    const timeout = window.setTimeout(() => {
      api.previewBenchmark(draft, settings).then(setPreview).catch((error) => onMessage(error.message, "error"));
    }, 180);
    return () => window.clearTimeout(timeout);
  }, [draft, settings, onMessage]);

  async function refreshBenchmarks() {
    try {
      const [nextRuns, nextStatus, nextLogs] = await Promise.all([
        api.benchmarks(),
        api.benchmarkStatus(),
        api.benchmarkLogs()
      ]);
      setRuns(nextRuns);
      setStatus(nextStatus);
      setLogs(nextLogs);
    } catch {
      // Ignore; the next poll will retry.
    }
  }

  async function runBenchmark() {
    if (!draft) {
      return;
    }
    if (runtimeRunning) {
      onMessage("Stop the llama-server runtime before running a benchmark for cleaner, safer results.", "error");
      return;
    }
    setBenchmarkBusy(true);
    try {
      let profileId = draft.id;
      if (isDirty || !draft.id) {
        const saved = await saveDraft();
        if (!saved) {
          return;
        }
        profileId = saved.id;
      }
      await api.startBenchmark(profileId, settings);
      await refreshBenchmarks();
      onMessage("Benchmark started.", "info");
    } catch (error) {
      onMessage(error instanceof Error ? error.message : String(error), "error");
    } finally {
      setBenchmarkBusy(false);
    }
  }

  async function stopBenchmark() {
    setBenchmarkBusy(true);
    try {
      await api.stopBenchmark();
      await refreshBenchmarks();
      onMessage("Benchmark stop requested.", "info");
    } catch (error) {
      onMessage(error instanceof Error ? error.message : String(error), "error");
    } finally {
      setBenchmarkBusy(false);
    }
  }

  async function deleteRun(id: string) {
    setBenchmarkBusy(true);
    try {
      await api.deleteBenchmark(id);
      setCompareIds((current) => current.filter((compareId) => compareId !== id));
      await refreshBenchmarks();
      onMessage("Benchmark result deleted.", "success");
    } catch (error) {
      onMessage(error instanceof Error ? error.message : String(error), "error");
    } finally {
      setBenchmarkBusy(false);
    }
  }

  function toggleCompare(id: string) {
    setCompareIds((current) => {
      if (current.includes(id)) {
        return current.filter((compareId) => compareId !== id);
      }
      return [...current, id].slice(-3);
    });
  }

  function updateSetting<K extends keyof BenchmarkSettings>(key: K, value: BenchmarkSettings[K]) {
    setSettings((current) => ({ ...current, [key]: value }));
  }

  const controlsDisabled = busy || benchmarkBusy || !draft;

  return (
    <section className="benchmark-band">
      <div className="dashboard-header">
        <div className="panel-title">
          <BarChart3 size={18} />
          <span>Benchmarks</span>
        </div>
        <div className="status-strip benchmark-status">
          <span className={`state-chip ${running ? "running" : "stopped"}`}>
            <i className={`runtime-dot ${running ? "live" : ""}`} />
            {running ? "running" : "idle"}
          </span>
          {status.profileName ? <span className="status-name">{status.profileName}</span> : null}
        </div>
      </div>

      <div className="benchmark-grid">
        <div className="benchmark-controls">
          <div className="benchmark-actions">
            <button className="primary" title="Run benchmark" onClick={runBenchmark} disabled={controlsDisabled || running}>
              <Play size={17} />
              Run
            </button>
            <button title="Stop benchmark" onClick={stopBenchmark} disabled={controlsDisabled || !running}>
              <Square size={17} />
              Stop
            </button>
          </div>

          <div className="benchmark-settings">
            <NumberField
              label="Prompt tokens"
              value={settings.promptTokens}
              min={1}
              onChange={(value) => updateSetting("promptTokens", value)}
            />
            <NumberField
              label="Gen tokens"
              value={settings.generationTokens}
              min={1}
              onChange={(value) => updateSetting("generationTokens", value)}
            />
            <NumberField
              label="Repetitions"
              value={settings.repetitions}
              min={1}
              max={20}
              onChange={(value) => updateSetting("repetitions", value)}
            />
            <NumberField
              label="Batch"
              value={settings.batchSize}
              min={1}
              onChange={(value) => updateSetting("batchSize", value)}
            />
            <NumberField
              label="Ubatch"
              value={settings.ubatchSize}
              min={1}
              onChange={(value) => updateSetting("ubatchSize", value)}
            />
            <SelectField<FlashAttentionMode>
              label="Flash attention"
              value={settings.flashAttention}
              options={["auto", "on", "off"]}
              onChange={(value) => updateSetting("flashAttention", value)}
            />
            <ToggleField
              label="No warmup"
              checked={settings.noWarmup}
              onChange={(value) => updateSetting("noWarmup", value)}
            />
          </div>

          <div className="panel-title compact">
            <Zap size={18} />
            <span>Benchmark Command</span>
            <CopyButton text={preview?.display} title="Copy benchmark command" />
          </div>
          <pre className="command-preview benchmark-command">{preview?.display ?? ""}</pre>
          {preview?.warnings.length ? (
            <div className="warnings">
              {preview.warnings.map((warning) => (
                <span key={warning}>{warning}</span>
              ))}
            </div>
          ) : null}
        </div>

        <div className="benchmark-kpis">
          <MetricCard
            label="Prompt eval"
            value={formatNumber(latestRun?.metrics.promptTokensPerSecond ?? null)}
            unit="tok/s"
            delta={formatDelta(
              latestRun?.metrics.promptTokensPerSecond ?? null,
              previousRun?.metrics.promptTokensPerSecond ?? null
            )}
          />
          <MetricCard
            label="Generation"
            value={formatNumber(latestRun?.metrics.generationTokensPerSecond ?? null)}
            unit="tok/s"
            delta={formatDelta(
              latestRun?.metrics.generationTokensPerSecond ?? null,
              previousRun?.metrics.generationTokensPerSecond ?? null
            )}
          />
          <MetricCard
            label="Latency"
            value={formatNumber(latestRun?.metrics.generationMsPerToken ?? null, 2)}
            unit="ms/token"
            delta={formatDelta(
              latestRun?.metrics.generationMsPerToken ?? null,
              previousRun?.metrics.generationMsPerToken ?? null
            )}
            inverse
          />
          <MetricCard
            label="Score"
            value={formatNumber(latestRun?.metrics.score ?? null)}
            unit="blend"
            delta={formatDelta(latestRun?.metrics.score ?? null, previousRun?.metrics.score ?? null)}
          />
        </div>

        <div className="benchmark-history">
          <div className="panel-title compact">
            <Gauge size={18} />
            <span>Results</span>
            <small className="panel-hint">check runs to compare</small>
          </div>
          <div className="result-table">
            <div className="result-row header">
              <span />
              <span>Run</span>
              <span>PP tok/s</span>
              <span>TG tok/s</span>
              <span>Latency</span>
              <span>Backend</span>
              <span />
            </div>
            {runs.length === 0 ? (
              <div className="empty result-empty">No benchmark results yet.</div>
            ) : (
              runs.map((run) => (
                <div
                  key={run.id}
                  className={`result-row ${run.status} ${compareIds.includes(run.id) ? "compared" : ""}`}
                  title={run.error ?? undefined}
                >
                  <input
                    type="checkbox"
                    className="compare-check"
                    title="Select for comparison"
                    checked={compareIds.includes(run.id)}
                    disabled={run.status !== "completed"}
                    onChange={() => toggleCompare(run.id)}
                  />
                  <span>
                    <strong>{run.profileName}</strong>
                    <small>{new Date(run.createdAt).toLocaleString()}</small>
                    {run.status !== "completed" ? <small className={`run-status ${run.status}`}>{run.status}</small> : null}
                  </span>
                  <span>
                    {formatNumber(run.metrics.promptTokensPerSecond)}
                    <i style={{ width: `${((run.metrics.promptTokensPerSecond ?? 0) / maxPrompt) * 100}%` }} />
                  </span>
                  <span>
                    {formatNumber(run.metrics.generationTokensPerSecond)}
                    <i style={{ width: `${((run.metrics.generationTokensPerSecond ?? 0) / maxGeneration) * 100}%` }} />
                  </span>
                  <span>{formatNumber(run.metrics.generationMsPerToken, 2)}</span>
                  <span>{run.backend}</span>
                  <ConfirmButton
                    className="icon-button"
                    title="Delete benchmark result"
                    confirmLabel="Sure?"
                    onConfirm={() => deleteRun(run.id)}
                    disabled={benchmarkBusy || run.status === "running"}
                  >
                    <Trash2 size={15} />
                  </ConfirmButton>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {compareRuns.length >= 2 ? (
        <CompareTable runs={compareRuns} onClose={() => setCompareIds([])} />
      ) : null}

      <div className="benchmark-log-panel">
        <div className="panel-title compact">
          <span>Benchmark Logs</span>
        </div>
        <LogView logs={logs} height={150} emptyText="No benchmark logs yet." />
      </div>
    </section>
  );
}

interface CompareTableProps {
  runs: BenchmarkRun[];
  onClose(): void;
}

function CompareTable({ runs, onClose }: CompareTableProps) {
  const metricRows: Array<{
    label: string;
    unit: string;
    values: Array<number | null>;
    higherIsBetter: boolean;
    digits?: number;
  }> = [
    {
      label: "Prompt eval",
      unit: "tok/s",
      values: runs.map((run) => run.metrics.promptTokensPerSecond),
      higherIsBetter: true
    },
    {
      label: "Generation",
      unit: "tok/s",
      values: runs.map((run) => run.metrics.generationTokensPerSecond),
      higherIsBetter: true
    },
    {
      label: "Latency",
      unit: "ms/token",
      values: runs.map((run) => run.metrics.generationMsPerToken),
      higherIsBetter: false,
      digits: 2
    },
    {
      label: "Score",
      unit: "blend",
      values: runs.map((run) => run.metrics.score),
      higherIsBetter: true
    }
  ];

  const differingSnapshot = SNAPSHOT_FIELDS.filter(
    ({ key }) => new Set(runs.map((run) => String(run.profile[key]))).size > 1
  );
  const differingSettings = SETTING_FIELDS.filter(
    ({ key }) => new Set(runs.map((run) => String(run.settings[key]))).size > 1
  );

  function bestIndex(values: Array<number | null>, higherIsBetter: boolean): number {
    let best = -1;
    values.forEach((value, index) => {
      if (value === null) {
        return;
      }
      const current = best === -1 ? null : values[best];
      if (current === null || (higherIsBetter ? value > current : value < current)) {
        best = index;
      }
    });
    return best;
  }

  return (
    <div className="compare-panel">
      <div className="panel-title compact">
        <GitCompareArrows size={18} />
        <span>Compare Runs</span>
        <button className="icon-button" title="Close comparison" onClick={onClose}>
          <X size={15} />
        </button>
      </div>
      <div className="compare-scroll">
        <table className="compare-table">
          <thead>
            <tr>
              <th />
              {runs.map((run) => (
                <th key={run.id}>
                  <strong>{run.profileName}</strong>
                  <small>{new Date(run.createdAt).toLocaleString()}</small>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metricRows.map((row) => {
              const best = bestIndex(row.values, row.higherIsBetter);
              return (
                <tr key={row.label}>
                  <td>
                    {row.label} <small>{row.unit}</small>
                  </td>
                  {row.values.map((value, index) => (
                    <td key={runs[index].id} className={index === best && runs.length > 1 ? "best" : ""}>
                      {formatNumber(value, row.digits ?? 1)}
                    </td>
                  ))}
                </tr>
              );
            })}
            {differingSnapshot.map(({ key, label }) => (
              <tr key={`snapshot-${key}`} className="config-row">
                <td>{label}</td>
                {runs.map((run) => (
                  <td key={run.id}>{String(run.profile[key])}</td>
                ))}
              </tr>
            ))}
            {differingSettings.map(({ key, label }) => (
              <tr key={`setting-${key}`} className="config-row">
                <td>{label}</td>
                {runs.map((run) => (
                  <td key={run.id}>{String(run.settings[key])}</td>
                ))}
              </tr>
            ))}
            {differingSnapshot.length === 0 && differingSettings.length === 0 ? (
              <tr className="config-row">
                <td>Configuration</td>
                <td colSpan={runs.length}>identical across selected runs</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}

interface MetricCardProps {
  label: string;
  value: string;
  unit: string;
  delta: string;
  inverse?: boolean;
}

function MetricCard({ label, value, unit, delta, inverse = false }: MetricCardProps) {
  const good = delta ? (inverse ? delta.startsWith("-") : delta.startsWith("+")) : false;
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{unit}</small>
      {delta ? <em className={good ? "good" : "bad"}>{delta}</em> : null}
    </div>
  );
}
