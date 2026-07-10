import {
  BarChart3,
  Check,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Gauge,
  GitCompareArrows,
  LineChart,
  Play,
  SlidersHorizontal,
  Square,
  Trash2,
  Trophy,
  Wand2,
  X,
  Zap
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { api } from "../api";
import { LogView } from "../components/LogView";
import type { Notify } from "../components/Toasts";
import { TrendChart } from "../components/TrendChart";
import { ConfirmButton, CopyButton, NumberField, SelectField, ToggleField } from "../components/ui";
import { buildTrendSeries, type TrendGroupBy, type TrendMetric } from "../shared/chartMath";
import { formatModelType } from "../shared/modelLabel";
import type {
  BenchmarkCommandPreview,
  BenchmarkEnv,
  BenchmarkProfileSnapshot,
  BenchmarkRun,
  BenchmarkSettings,
  BenchmarkStatus,
  FlashAttentionMode,
  LlamaProfile,
  ResolvedBackend,
  RuntimeLog,
  SweepResult,
  SweepStatus
} from "../shared/types";

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

const ENV_FIELDS: Array<{ key: keyof BenchmarkEnv; label: string }> = [
  { key: "buildNumber", label: "llama.cpp build" },
  { key: "gpuName", label: "GPU" },
  { key: "modelType", label: "Model type" }
];

type SortKey = "date" | "pp" | "tg" | "latency";

interface BenchmarksViewProps {
  draft: LlamaProfile | null;
  selectedProfile: LlamaProfile | null;
  isDirty: boolean;
  runtimeRunning: boolean;
  busy: boolean;
  saveDraft(): Promise<LlamaProfile | null>;
  onOptimize(): void;
  onApplySettings(settings: Partial<LlamaProfile>): void;
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

function sortValue(run: BenchmarkRun, key: SortKey): number {
  switch (key) {
    case "pp":
      return run.metrics.promptTokensPerSecond ?? -1;
    case "tg":
      return run.metrics.generationTokensPerSecond ?? -1;
    case "latency":
      return run.metrics.generationMsPerToken ?? Number.POSITIVE_INFINITY;
    default:
      return Date.parse(run.createdAt);
  }
}

const idleSweep: SweepStatus = {
  state: "idle",
  sweepId: null,
  profileId: null,
  profileName: null,
  completedRuns: 0,
  totalRuns: 0,
  currentCandidate: null,
  startedAt: null
};

export function BenchmarksView({
  draft,
  selectedProfile,
  isDirty,
  runtimeRunning,
  busy,
  saveDraft,
  onOptimize,
  onApplySettings,
  onMessage
}: BenchmarksViewProps) {
  const [settings, setSettings] = useState<BenchmarkSettings>(defaultSettings);
  const [preview, setPreview] = useState<BenchmarkCommandPreview | null>(null);
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [loaded, setLoaded] = useState(false);
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
  const [sweepStatus, setSweepStatus] = useState<SweepStatus>(idleSweep);
  const [sweepResult, setSweepResult] = useState<SweepResult | null>(null);
  const [sweepDismissed, setSweepDismissed] = useState<string | null>(null);
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [profileFilter, setProfileFilter] = useState<string>("all");
  const [backendFilter, setBackendFilter] = useState<ResolvedBackend | "all">("all");
  const [statusFilter, setStatusFilter] = useState<"all" | "completed" | "incomplete">("all");
  const [configOpen, setConfigOpen] = useState<boolean | null>(null);
  const [metric, setMetric] = useState<TrendMetric>("tg");
  const [groupBy, setGroupBy] = useState<TrendGroupBy>("profile");
  const [errorBars, setErrorBars] = useState(false);
  const [highlightRunId, setHighlightRunId] = useState<string | null>(null);
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: "date", dir: -1 });
  const tableRef = useRef<HTMLDivElement | null>(null);

  const running = status.state === "running";
  const sweepRunning = sweepStatus.state === "running";
  // Config strip: open until the first run exists, then collapsed unless toggled.
  const showConfig = configOpen ?? (loaded && runs.length === 0);

  const selectedRuns = useMemo(
    () => runs.filter((run) => run.profileId === selectedProfile?.id && run.status === "completed"),
    [runs, selectedProfile]
  );
  const latestRun = selectedRuns[0] ?? runs.find((run) => run.status === "completed") ?? null;
  const previousRun = selectedRuns[1] ?? null;

  const profileNames = useMemo(() => {
    const names: string[] = [];
    for (const run of runs) {
      if (!names.includes(run.profileName)) {
        names.push(run.profileName);
      }
    }
    return names;
  }, [runs]);
  const hasCuda = runs.some((run) => run.backend === "CUDA");
  const hasCpu = runs.some((run) => run.backend === "CPU");
  const showBackendFilter = hasCuda && hasCpu;
  const hasIncomplete = runs.some((run) => run.status !== "completed");
  const hasEnvColumns = runs.some((run) => run.env?.modelType || run.env?.buildNumber);

  const matchesProfile = (run: BenchmarkRun) => profileFilter === "all" || run.profileName === profileFilter;
  const matchesBackend = (run: BenchmarkRun) =>
    !showBackendFilter || backendFilter === "all" || run.backend === backendFilter;
  const matchesStatus = (run: BenchmarkRun) => {
    if (!hasIncomplete || statusFilter === "all") {
      return true;
    }
    return statusFilter === "completed" ? run.status === "completed" : run.status !== "completed";
  };

  const filteredRuns = useMemo(
    () => runs.filter((run) => matchesProfile(run) && matchesBackend(run) && matchesStatus(run)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [runs, profileFilter, backendFilter, statusFilter, showBackendFilter, hasIncomplete]
  );

  const sortedRuns = useMemo(() => {
    const next = [...filteredRuns];
    next.sort((a, b) => (sortValue(a, sort.key) - sortValue(b, sort.key)) * sort.dir);
    return next;
  }, [filteredRuns, sort]);

  const trendSeries = useMemo(
    () => buildTrendSeries(filteredRuns, metric, groupBy),
    [filteredRuns, metric, groupBy]
  );

  const maxPrompt = Math.max(1, ...filteredRuns.map((run) => run.metrics.promptTokensPerSecond ?? 0));
  const maxGeneration = Math.max(1, ...filteredRuns.map((run) => run.metrics.generationTokensPerSecond ?? 0));
  const compareRuns = useMemo(
    () => compareIds.map((id) => runs.find((run) => run.id === id)).filter((run): run is BenchmarkRun => Boolean(run)),
    [compareIds, runs]
  );

  // Poll faster while a benchmark runs, back off when idle, skip hidden tabs
  // (but always load once, and refresh immediately when the tab is shown).
  useEffect(() => {
    let disposed = false;
    const refresh = async (force = false) => {
      if (document.hidden && !force) {
        return;
      }
      try {
        const [nextRuns, nextStatus, nextLogs, nextSweep] = await Promise.all([
          api.benchmarks(),
          api.benchmarkStatus(),
          api.benchmarkLogs(),
          api.sweepStatus().catch(() => idleSweep)
        ]);
        if (disposed) {
          return;
        }
        setRuns((prev) => (sameRuns(prev, nextRuns) ? prev : nextRuns));
        setStatus((prev) => (JSON.stringify(prev) === JSON.stringify(nextStatus) ? prev : nextStatus));
        setSweepStatus((prev) => (JSON.stringify(prev) === JSON.stringify(nextSweep) ? prev : nextSweep));
        setLogs((prev) =>
          prev.length === nextLogs.length && prev[prev.length - 1]?.id === nextLogs[nextLogs.length - 1]?.id
            ? prev
            : nextLogs
        );
        setLoaded(true);
      } catch {
        // Polling stays quiet; direct actions surface their own errors.
      }
    };
    void refresh(true);
    const timer = window.setInterval(() => void refresh(), running || sweepStatus.state === "running" ? 1500 : 5000);
    const onVisible = () => {
      if (!document.hidden) {
        void refresh();
      }
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      disposed = true;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, sweepStatus.state]);

  // Winner-card data. A separate effect keyed on the idle transition: the
  // polling effect above re-runs (and cancels its own async work) on every
  // sweep state change, so a fetch started inside it at completion time would
  // be torn down by the very transition that scheduled it — leaving a stale
  // card from a previous sweep on screen.
  useEffect(() => {
    if (sweepRunning) {
      // A new sweep invalidates whatever card was showing.
      setSweepResult(null);
      return;
    }
    let stale = false;
    // After a completed sweep the status still carries its id; on a fresh
    // mount it is null and the server returns the latest stored sweep.
    api
      .sweepResult(sweepStatus.sweepId ?? undefined)
      .then((result) => {
        if (!stale) {
          setSweepResult(result);
        }
      })
      .catch(() => undefined);
    return () => {
      stale = true;
    };
  }, [sweepRunning, sweepStatus.sweepId]);

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

  async function stopSweep() {
    setBenchmarkBusy(true);
    try {
      await api.stopSweep();
      onMessage("Sweep stop requested — partial results will still be ranked.", "info");
    } catch (error) {
      onMessage(error instanceof Error ? error.message : String(error), "error");
    } finally {
      setBenchmarkBusy(false);
    }
  }

  function applySweepSettings() {
    if (!sweepResult || Object.keys(sweepResult.bestSettings).length === 0) {
      return;
    }
    if (draft && draft.id !== sweepResult.profileId) {
      onMessage(`This sweep tuned "${sweepResult.profileName}" — select that profile to apply its settings.`, "error");
      return;
    }
    onApplySettings(sweepResult.bestSettings);
    onMessage("Winning settings applied to the profile draft — save the profile to keep them.", "success");
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

  function toggleSort(key: SortKey) {
    setSort((current) =>
      current.key === key ? { key, dir: current.dir === 1 ? -1 : 1 } : { key, dir: key === "date" ? -1 : -1 }
    );
  }

  function focusRun(runId: string) {
    setHighlightRunId(runId);
    // Let the highlight class land, then bring the row into view.
    window.setTimeout(() => {
      tableRef.current
        ?.querySelector(`[data-run-id="${runId}"]`)
        ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }, 30);
  }

  function sortIndicator(key: SortKey) {
    if (sort.key !== key) {
      return null;
    }
    return sort.dir === 1 ? <ChevronUp size={12} /> : <ChevronDown size={12} />;
  }

  const controlsDisabled = busy || benchmarkBusy || !draft;
  const metricUnit = metric === "score" ? "score" : "tok/s";

  return (
    <section className="bench-page">
      <div className="bench-header">
        <div className="panel-title">
          <BarChart3 size={18} />
          <span>Benchmarks</span>
          <span className={`state-chip ${running ? "running" : "stopped"}`}>
            <i className={`runtime-dot ${running ? "live" : ""}`} />
            {running ? "running" : "idle"}
          </span>
          {status.profileName ? <span className="status-name">{status.profileName}</span> : null}
        </div>
        <div className="bench-header-actions">
          <button
            title="Find the fastest settings for this profile automatically"
            onClick={onOptimize}
            disabled={controlsDisabled || running || sweepRunning}
          >
            <Wand2 size={17} />
            Optimize
          </button>
          <button
            className="primary"
            title="Run benchmark"
            onClick={runBenchmark}
            disabled={controlsDisabled || running || sweepRunning}
          >
            <Play size={17} />
            Run benchmark
          </button>
          <button title="Stop benchmark" onClick={stopBenchmark} disabled={controlsDisabled || !running || sweepRunning}>
            <Square size={17} />
            Stop
          </button>
        </div>
      </div>

      {sweepRunning ? (
        <div className="bench-panel sweep-progress">
          <div className="panel-title compact">
            <Wand2 size={18} />
            <span>Optimizing {sweepStatus.profileName ?? ""}</span>
            <small className="panel-hint">
              run {Math.min(sweepStatus.completedRuns + 1, Math.max(1, sweepStatus.totalRuns))} of ~{sweepStatus.totalRuns}
              {sweepStatus.currentCandidate ? ` · ${sweepStatus.currentCandidate}` : ""}
            </small>
            <button title="Stop the sweep" onClick={stopSweep} disabled={benchmarkBusy}>
              <Square size={15} />
              Stop sweep
            </button>
          </div>
          <div className="sweep-progress-bar">
            <i
              style={{
                width: `${Math.min(100, (sweepStatus.completedRuns / Math.max(1, sweepStatus.totalRuns)) * 100)}%`
              }}
            />
          </div>
        </div>
      ) : null}

      {!sweepRunning && sweepResult && sweepResult.sweepId !== sweepDismissed && sweepResult.ranked.length > 0 ? (
        <SweepResultCard
          result={sweepResult}
          runs={runs}
          profileMismatch={Boolean(draft && draft.id !== sweepResult.profileId)}
          onApply={applySweepSettings}
          onFocusRun={focusRun}
          onDismiss={() => setSweepDismissed(sweepResult.sweepId)}
        />
      ) : null}

      <div className="bench-panel">
        <button className="bench-config-toggle" onClick={() => setConfigOpen(!showConfig)}>
          {showConfig ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
          <SlidersHorizontal size={15} />
          <span>Run configuration</span>
          <small>
            pp{settings.promptTokens} · tg{settings.generationTokens} · ×{settings.repetitions}
            {draft ? ` · ${draft.name}` : ""}
          </small>
        </button>
        {showConfig ? (
          <div className="bench-config-body">
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
        ) : null}
      </div>

      <div className="benchmark-kpis bench-kpi-row">
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

      <div className="bench-panel">
        <div className="panel-title compact bench-trend-title">
          <LineChart size={18} />
          <span>Performance over time</span>
        </div>
        <div className="log-toolbar bench-trend-toolbar">
          <div className="metric-toggle">
            {(
              [
                { key: "tg", label: "Generation" },
                { key: "pp", label: "Prompt" },
                { key: "score", label: "Score" }
              ] as const
            ).map((entry) => (
              <button
                key={entry.key}
                className={`chip ${metric === entry.key ? "active" : ""}`}
                onClick={() => setMetric(entry.key)}
              >
                {entry.label}
              </button>
            ))}
          </div>
          <label className="trend-select">
            <span>Group by</span>
            <select value={groupBy} onChange={(event) => setGroupBy(event.target.value as TrendGroupBy)}>
              <option value="profile">Profile</option>
              <option value="model">Model</option>
            </select>
          </label>
          <label className={`trend-errorbars ${metric === "score" ? "disabled" : ""}`}>
            <input
              type="checkbox"
              checked={errorBars && metric !== "score"}
              disabled={metric === "score"}
              onChange={(event) => setErrorBars(event.target.checked)}
            />
            <span>± stddev</span>
          </label>
        </div>

        <TrendChart
          series={trendSeries}
          height={260}
          yUnit={metricUnit}
          showErrorBars={errorBars && metric !== "score"}
          highlightRunId={highlightRunId}
          onPointClick={focusRun}
          tooltip={(point, series) => {
            const run = runs.find((entry) => entry.id === point.runId);
            return (
              <>
                <strong>{series.label}</strong>
                <span>{new Date(point.x).toLocaleString()}</span>
                <span>
                  {formatNumber(point.y)} {metricUnit}
                  {point.stddev ? ` ± ${formatNumber(point.stddev)}` : ""}
                </span>
                {run?.env?.buildNumber ? <span>build {run.env.buildNumber}</span> : null}
              </>
            );
          }}
        />

        {trendSeries.length > 0 ? (
          <div className="trend-legend">
            {trendSeries.map((series) => (
              <span key={series.key} className="legend-item">
                <i style={{ background: `var(--chart-${(series.colorIndex % 4) + 1})` }} />
                {series.label}
                <small>{formatNumber(series.points[series.points.length - 1]?.y ?? null)}</small>
              </span>
            ))}
          </div>
        ) : null}
      </div>

      <div className="bench-panel benchmark-history">
        <div className="panel-title compact">
          <Gauge size={18} />
          <span>Results</span>
          <small className="panel-hint">check runs to compare · click headers to sort</small>
        </div>
        {runs.length > 0 ? (
          <>
            <div className="log-toolbar">
              {profileNames.length > 6 ? (
                <label className="field">
                  <select value={profileFilter} onChange={(event) => setProfileFilter(event.target.value)}>
                    <option value="all">All profiles ({runs.length})</option>
                    {profileNames.map((name) => (
                      <option key={name} value={name}>
                        {name} ({runs.filter((run) => run.profileName === name).length})
                      </option>
                    ))}
                  </select>
                </label>
              ) : (
                <>
                  <button
                    className={`chip ${profileFilter === "all" ? "active" : ""}`}
                    onClick={() => setProfileFilter("all")}
                  >
                    All
                    <small>{runs.length}</small>
                  </button>
                  {profileNames.map((name) => (
                    <button
                      key={name}
                      className={`chip ${profileFilter === name ? "active" : ""}`}
                      onClick={() => setProfileFilter(name)}
                    >
                      {name}
                      <small>{runs.filter((run) => run.profileName === name).length}</small>
                    </button>
                  ))}
                </>
              )}
              {showBackendFilter
                ? (["all", "CUDA", "CPU"] as const).map((key) => (
                    <button
                      key={key}
                      className={`chip ${backendFilter === key ? "active" : ""}`}
                      onClick={() => setBackendFilter(key)}
                    >
                      {key === "all" ? "All backends" : key}
                    </button>
                  ))
                : null}
              {hasIncomplete
                ? (["all", "completed", "incomplete"] as const).map((key) => (
                    <button
                      key={key}
                      className={`chip ${statusFilter === key ? "active" : ""}`}
                      onClick={() => setStatusFilter(key)}
                    >
                      {key === "all" ? "All statuses" : key === "completed" ? "completed" : "failed+cancelled"}
                    </button>
                  ))
                : null}
            </div>
          </>
        ) : null}
        <div className={`result-table ${hasEnvColumns ? "with-env" : ""}`} ref={tableRef}>
          <div className="result-row header">
            <span />
            <button className="sort-header" onClick={() => toggleSort("date")}>
              Run {sortIndicator("date")}
            </button>
            {hasEnvColumns ? <span>Model</span> : null}
            {hasEnvColumns ? <span>Build</span> : null}
            <button className="sort-header" onClick={() => toggleSort("pp")}>
              PP tok/s {sortIndicator("pp")}
            </button>
            <button className="sort-header" onClick={() => toggleSort("tg")}>
              TG tok/s {sortIndicator("tg")}
            </button>
            <button className="sort-header" onClick={() => toggleSort("latency")}>
              Latency {sortIndicator("latency")}
            </button>
            <span>Backend</span>
            <span />
          </div>
          {runs.length === 0 ? (
            <div className="empty result-empty">
              <BarChart3 size={26} />
              <strong>No benchmark results yet</strong>
              <span>Run your first benchmark to start tracking performance over time.</span>
            </div>
          ) : sortedRuns.length === 0 ? (
            <div className="empty result-empty">No runs match the current filters.</div>
          ) : (
            sortedRuns.map((run) => (
              <div
                key={run.id}
                data-run-id={run.id}
                className={`result-row ${run.status} ${compareIds.includes(run.id) ? "compared" : ""} ${
                  highlightRunId === run.id ? "highlighted" : ""
                }`}
                title={run.error ?? undefined}
                onClick={() => setHighlightRunId((current) => (current === run.id ? null : run.id))}
              >
                <input
                  type="checkbox"
                  className="compare-check"
                  title="Select for comparison"
                  checked={compareIds.includes(run.id)}
                  disabled={run.status !== "completed"}
                  onClick={(event) => event.stopPropagation()}
                  onChange={() => toggleCompare(run.id)}
                />
                <span>
                  <strong>{run.profileName}</strong>
                  <small>{new Date(run.createdAt).toLocaleString()}</small>
                  {run.sweepLabel ? <small className="sweep-tag">⚡ {run.sweepLabel}</small> : null}
                  {run.status !== "completed" ? <small className={`run-status ${run.status}`}>{run.status}</small> : null}
                </span>
                {hasEnvColumns ? (
                  <span className="env-cell" title={run.env?.modelType ?? undefined}>
                    {formatModelType(run.env) ?? "-"}
                  </span>
                ) : null}
                {hasEnvColumns ? <span className="env-cell">{run.env?.buildNumber ?? "-"}</span> : null}
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

      {compareRuns.length >= 2 ? <CompareTable runs={compareRuns} onClose={() => setCompareIds([])} /> : null}

      <div className="bench-panel benchmark-log-panel">
        <div className="panel-title compact">
          <span>Benchmark Logs</span>
        </div>
        <LogView logs={logs} height={150} emptyText="No benchmark logs yet." />
      </div>
    </section>
  );
}

interface SweepResultCardProps {
  result: SweepResult;
  runs: BenchmarkRun[];
  /** True when the selected profile is not the one this sweep tuned. */
  profileMismatch: boolean;
  onApply(): void;
  onFocusRun(runId: string): void;
  onDismiss(): void;
}

function SweepResultCard({ result, runs, profileMismatch, onApply, onFocusRun, onDismiss }: SweepResultCardProps) {
  const winner = result.ranked.find((entry) => entry.runId === result.winnerRunId) ?? result.ranked[0];
  const baselineRun = runs.find((run) => run.id === result.baselineRunId) ?? null;
  const baselineScore = baselineRun?.metrics.score ?? null;
  const improvement =
    winner?.score !== null && winner !== undefined && baselineScore ? ((winner.score - baselineScore) / baselineScore) * 100 : null;
  const hasSettings = Object.keys(result.bestSettings).length > 0;

  return (
    <div className="bench-panel sweep-result">
      <div className="panel-title compact">
        <Trophy size={18} />
        <span>Optimize result — {result.profileName}</span>
        {result.status !== "completed" ? <span className={`state-chip stopped`}>{result.status}</span> : null}
        <button className="icon-button" title="Dismiss" onClick={onDismiss}>
          <X size={15} />
        </button>
      </div>
      <div className="sweep-verdict">
        <div className="sweep-winner">
          <strong>{winner?.label ?? "no result"}</strong>
          <small>
            benchmark-measured score {formatNumber(winner?.score ?? null)}
            {improvement !== null && Math.abs(improvement) >= 0.05
              ? ` · ${improvement > 0 ? "+" : ""}${improvement.toFixed(1)}% vs baseline`
              : " · matches the baseline"}
          </small>
        </div>
        <button
          className="primary"
          onClick={onApply}
          disabled={!hasSettings || profileMismatch}
          title={
            profileMismatch
              ? `This sweep tuned "${result.profileName}" — select that profile to apply its settings.`
              : "Merge the winning settings into the profile draft"
          }
        >
          <Check size={16} />
          Apply best settings
        </button>
      </div>
      <div className="sweep-ranked">
        {result.ranked.slice(0, 8).map((entry, index) => (
          <button key={entry.runId} className="sweep-ranked-row" onClick={() => onFocusRun(entry.runId)}>
            <span className="rank">{index + 1}.</span>
            <span className="label">
              {entry.label}
              {entry.runId === result.winnerRunId ? <Trophy size={12} /> : null}
            </span>
            <span className="score">
              {formatNumber(entry.score)}
              {entry.scoreStddev ? <small> ± {formatNumber(entry.scoreStddev)}</small> : null}
            </span>
            {entry.withinNoiseOfBest ? <span className="pill muted">statistically tied</span> : null}
          </button>
        ))}
      </div>
      {result.notes.length > 0 ? (
        <div className="warnings">
          {result.notes.map((note) => (
            <span key={note}>{note}</span>
          ))}
        </div>
      ) : null}
    </div>
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
  const envDisplay = (run: BenchmarkRun, key: keyof BenchmarkEnv): string =>
    key === "modelType" ? formatModelType(run.env) ?? "-" : String(run.env?.[key] ?? "-");
  const differingEnv = ENV_FIELDS.filter(
    ({ key }) => new Set(runs.map((run) => envDisplay(run, key))).size > 1
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

  const noDiffs = differingSnapshot.length === 0 && differingSettings.length === 0 && differingEnv.length === 0;

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
            {differingEnv.map(({ key, label }) => (
              <tr key={`env-${key}`} className="config-row">
                <td>{label}</td>
                {runs.map((run) => (
                  <td key={run.id}>{envDisplay(run, key)}</td>
                ))}
              </tr>
            ))}
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
            {noDiffs ? (
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
