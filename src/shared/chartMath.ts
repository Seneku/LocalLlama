// Pure, DOM-free series/axis math for the benchmark trend chart. Kept apart
// from the SVG component so it can be unit-tested like the other shared
// normalizers (hf.ts, connect.ts).
import type { BenchmarkRun } from "./types";

export type TrendMetric = "pp" | "tg" | "score";
export type TrendGroupBy = "profile" | "model";

export interface TrendPoint {
  x: number; // createdAt epoch ms
  y: number; // metric value (tok/s or score)
  stddev: number | null;
  runId: string;
}

export interface TrendSeries {
  key: string;
  label: string;
  colorIndex: number;
  points: TrendPoint[];
}

function modelFileName(modelPath: string): string {
  const parts = modelPath.split(/[\\/]/);
  return parts[parts.length - 1] || modelPath;
}

function metricValue(run: BenchmarkRun, metric: TrendMetric): number | null {
  switch (metric) {
    case "pp":
      return run.metrics.promptTokensPerSecond;
    case "tg":
      return run.metrics.generationTokensPerSecond;
    default:
      return run.metrics.score;
  }
}

function metricStddev(run: BenchmarkRun, metric: TrendMetric): number | null {
  switch (metric) {
    case "pp":
      return run.metrics.promptStddev;
    case "tg":
      return run.metrics.generationStddev;
    default:
      return null; // score is a blend; a stddev would be meaningless
  }
}

function groupKey(run: BenchmarkRun, groupBy: TrendGroupBy): string {
  if (groupBy === "model") {
    return run.env?.modelType ?? modelFileName(run.profile.modelPath);
  }
  return run.profileName;
}

/**
 * One series per group (profile or model), completed runs only, points sorted
 * ascending by createdAt. Runs whose metric is null are skipped.
 */
export function buildTrendSeries(
  runs: BenchmarkRun[],
  metric: TrendMetric,
  groupBy: TrendGroupBy
): TrendSeries[] {
  const groups = new Map<string, TrendPoint[]>();
  for (const run of runs) {
    if (run.status !== "completed") {
      continue;
    }
    const value = metricValue(run, metric);
    if (value === null || !Number.isFinite(value)) {
      continue;
    }
    const x = Date.parse(run.createdAt);
    if (!Number.isFinite(x)) {
      continue;
    }
    const key = groupKey(run, groupBy);
    const points = groups.get(key) ?? [];
    points.push({ x, y: value, stddev: metricStddev(run, metric), runId: run.id });
    groups.set(key, points);
  }
  return [...groups.entries()].map(([key, points], index) => ({
    key,
    label: key,
    colorIndex: index,
    points: points.sort((a, b) => a.x - b.x)
  }));
}

/** Round "nice" tick values covering [min, max] — 1/2/5 × 10^n steps. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || count < 2) {
    return [];
  }
  if (max <= min) {
    return [min];
  }
  const rawStep = (max - min) / (count - 1);
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const residual = rawStep / magnitude;
  // Heckbert's "nice number" rounding: 1 / 2 / 5 / 10 × 10^n.
  const step = (residual < 1.5 ? 1 : residual < 3 ? 2 : residual < 7 ? 5 : 10) * magnitude;
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let tick = start; tick <= max + step * 1e-9; tick += step) {
    // Snap floating-point drift (e.g. 0.30000000000000004).
    ticks.push(Number(tick.toPrecision(12)));
  }
  return ticks;
}

const DAY_MS = 24 * 60 * 60 * 1000;

/** Day-aligned (or multi-day-stepped) tick positions for a time axis. */
export function timeTicks(minMs: number, maxMs: number, count = 5): number[] {
  if (!Number.isFinite(minMs) || !Number.isFinite(maxMs) || maxMs <= minMs || count < 2) {
    return Number.isFinite(minMs) ? [minMs] : [];
  }
  const span = maxMs - minMs;
  if (span < DAY_MS) {
    // Sub-day range: plain even spacing is fine (hours).
    const step = span / (count - 1);
    return Array.from({ length: count }, (_, index) => Math.round(minMs + index * step));
  }
  const stepDays = Math.max(1, Math.ceil(span / DAY_MS / (count - 1)));
  const step = stepDays * DAY_MS;
  const first = Math.ceil(minMs / DAY_MS) * DAY_MS;
  const ticks: number[] = [];
  for (let tick = first; tick <= maxMs; tick += step) {
    ticks.push(tick);
  }
  return ticks.length ? ticks : [minMs];
}

/** SVG path "d" for a polyline through already-projected pixel points. */
export function buildLinePath(points: Array<{ px: number; py: number }>): string {
  if (points.length === 0) {
    return "";
  }
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"}${point.px.toFixed(1)} ${point.py.toFixed(1)}`)
    .join(" ");
}

/** Index of the point whose x is nearest to xValue (points sorted by x). */
export function nearestPointIndex(points: TrendPoint[], xValue: number): number {
  if (points.length === 0) {
    return -1;
  }
  let lo = 0;
  let hi = points.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (points[mid].x < xValue) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return Math.abs(points[lo].x - xValue) <= Math.abs(points[hi].x - xValue) ? lo : hi;
}
