import { describe, expect, test } from "bun:test";

import {
  buildLinePath,
  buildTrendSeries,
  nearestPointIndex,
  niceTicks,
  timeTicks,
  type TrendPoint
} from "../src/shared/chartMath";
import type { BenchmarkRun } from "../src/shared/types";

function run(partial: {
  id: string;
  profileName?: string;
  createdAt: string;
  status?: BenchmarkRun["status"];
  pp?: number | null;
  tg?: number | null;
  modelType?: string | null;
  modelPath?: string;
}): BenchmarkRun {
  return {
    id: partial.id,
    profileId: "p1",
    profileName: partial.profileName ?? "Profile A",
    createdAt: partial.createdAt,
    completedAt: partial.createdAt,
    status: partial.status ?? "completed",
    exitCode: 0,
    signal: null,
    backend: "CUDA",
    settings: {} as BenchmarkRun["settings"],
    command: {} as BenchmarkRun["command"],
    profile: { modelPath: partial.modelPath ?? "C:\\m\\model-a.gguf" } as BenchmarkRun["profile"],
    rows: [],
    metrics: {
      promptTokensPerSecond: partial.pp === undefined ? 1000 : partial.pp,
      generationTokensPerSecond: partial.tg === undefined ? 75 : partial.tg,
      generationMsPerToken: 13,
      promptStddev: 10,
      generationStddev: 1,
      totalSeconds: 10,
      score: 273
    },
    env: partial.modelType === undefined ? null : ({ modelType: partial.modelType } as BenchmarkRun["env"]),
    stdout: "",
    stderr: "",
    error: null
  };
}

describe("buildTrendSeries", () => {
  test("groups by profile, points ascending by createdAt", () => {
    const series = buildTrendSeries(
      [
        run({ id: "b", profileName: "A", createdAt: "2026-07-02T10:00:00Z", tg: 80 }),
        run({ id: "a", profileName: "A", createdAt: "2026-07-01T10:00:00Z", tg: 70 }),
        run({ id: "c", profileName: "B", createdAt: "2026-07-03T10:00:00Z", tg: 60 })
      ],
      "tg",
      "profile"
    );
    expect(series).toHaveLength(2);
    const a = series.find((entry) => entry.key === "A")!;
    expect(a.points.map((point) => point.runId)).toEqual(["a", "b"]);
    expect(a.points.map((point) => point.y)).toEqual([70, 80]);
    expect(series.find((entry) => entry.key === "B")!.points).toHaveLength(1);
  });

  test("groups by model via env.modelType, falling back to the model filename", () => {
    const series = buildTrendSeries(
      [
        run({ id: "a", createdAt: "2026-07-01T10:00:00Z", modelType: "gemma 12B Q4_K - Medium" }),
        run({ id: "b", createdAt: "2026-07-02T10:00:00Z", modelType: "gemma 12B Q4_K - Medium" }),
        run({ id: "c", createdAt: "2026-07-03T10:00:00Z", modelPath: "D:\\models\\other-model.gguf" })
      ],
      "tg",
      "model"
    );
    expect(series.map((entry) => entry.key).sort()).toEqual(["gemma 12B Q4_K - Medium", "other-model.gguf"]);
  });

  test("skips non-completed runs and null metrics", () => {
    const series = buildTrendSeries(
      [
        run({ id: "a", createdAt: "2026-07-01T10:00:00Z", status: "failed" }),
        run({ id: "b", createdAt: "2026-07-02T10:00:00Z", tg: null }),
        run({ id: "c", createdAt: "2026-07-03T10:00:00Z", tg: 75 })
      ],
      "tg",
      "profile"
    );
    expect(series).toHaveLength(1);
    expect(series[0].points.map((point) => point.runId)).toEqual(["c"]);
  });

  test("score series carries no stddev", () => {
    const series = buildTrendSeries([run({ id: "a", createdAt: "2026-07-01T10:00:00Z" })], "score", "profile");
    expect(series[0].points[0].stddev).toBeNull();
  });
});

describe("niceTicks", () => {
  test("produces round 1/2/5 steps covering the range", () => {
    expect(niceTicks(0, 100, 5)).toEqual([0, 20, 40, 60, 80, 100]);
    expect(niceTicks(0, 87, 5)).toEqual([0, 20, 40, 60, 80]);
    const small = niceTicks(0, 1.3, 5);
    expect(small[0]).toBe(0);
    expect(small.every((tick, index) => index === 0 || tick > small[index - 1])).toBe(true);
  });

  test("degenerate ranges", () => {
    expect(niceTicks(5, 5, 5)).toEqual([5]);
    expect(niceTicks(Number.NaN, 10, 5)).toEqual([]);
  });
});

describe("timeTicks", () => {
  const DAY = 24 * 60 * 60 * 1000;

  test("day-aligned ticks for multi-day spans", () => {
    const start = Date.parse("2026-07-01T05:00:00Z");
    const end = start + 6 * DAY;
    const ticks = timeTicks(start, end, 5);
    expect(ticks.length).toBeGreaterThan(1);
    for (const tick of ticks) {
      expect(tick % DAY).toBe(0); // aligned to UTC midnight
    }
  });

  test("sub-day spans space evenly", () => {
    const start = Date.parse("2026-07-01T05:00:00Z");
    const ticks = timeTicks(start, start + 60 * 60 * 1000, 3);
    expect(ticks).toHaveLength(3);
    expect(ticks[0]).toBe(start);
  });
});

describe("buildLinePath", () => {
  test("emits M/L path for known points", () => {
    expect(buildLinePath([{ px: 10, py: 20 }, { px: 30.25, py: 40 }])).toBe("M10.0 20.0 L30.3 40.0");
    expect(buildLinePath([])).toBe("");
  });
});

describe("nearestPointIndex", () => {
  const points: TrendPoint[] = [
    { x: 100, y: 1, stddev: null, runId: "a" },
    { x: 200, y: 2, stddev: null, runId: "b" },
    { x: 400, y: 3, stddev: null, runId: "c" }
  ];

  test("edges and exact hits", () => {
    expect(nearestPointIndex(points, 0)).toBe(0); // before first
    expect(nearestPointIndex(points, 1000)).toBe(2); // after last
    expect(nearestPointIndex(points, 200)).toBe(1); // exact
    expect(nearestPointIndex(points, 290)).toBe(1); // nearer to 200
    expect(nearestPointIndex(points, 310)).toBe(2); // nearer to 400
    expect(nearestPointIndex([], 5)).toBe(-1);
  });
});
