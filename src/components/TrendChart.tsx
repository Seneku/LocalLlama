// Hand-rolled SVG time-series chart for benchmark history — zero dependencies,
// themed entirely through the --chart-* CSS variables. Pure math lives in
// src/shared/chartMath.ts; this component only projects and renders.
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import {
  buildLinePath,
  nearestPointIndex,
  niceTicks,
  timeTicks,
  type TrendPoint,
  type TrendSeries
} from "../shared/chartMath";

const MARGIN = { top: 12, right: 16, bottom: 28, left: 48 };
const SERIES_COLORS = 4; // --chart-1..4
const MAX_MARKER_POINTS = 150; // above this, draw paths only (hover still works)

interface TrendChartProps {
  series: TrendSeries[];
  height?: number;
  yUnit?: string;
  showErrorBars?: boolean;
  highlightRunId?: string | null;
  onPointClick?(runId: string): void;
  tooltip?(point: TrendPoint, series: TrendSeries): ReactNode;
}

interface Hover {
  seriesIndex: number;
  pointIndex: number;
  px: number;
  py: number;
}

function formatTick(value: number): string {
  if (Math.abs(value) >= 1000) {
    return `${(value / 1000).toLocaleString(undefined, { maximumFractionDigits: 1 })}k`;
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
}

function formatDateTick(ms: number, spanMs: number): string {
  const date = new Date(ms);
  if (spanMs < 24 * 60 * 60 * 1000) {
    return date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  }
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function TrendChart({
  series,
  height = 260,
  yUnit = "tok/s",
  showErrorBars = false,
  highlightRunId = null,
  onPointClick,
  tooltip
}: TrendChartProps) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(720);
  const [hover, setHover] = useState<Hover | null>(null);

  useEffect(() => {
    const element = wrapRef.current;
    if (!element) {
      return;
    }
    const observer = new ResizeObserver((entries) => {
      const next = Math.floor(entries[0]?.contentRect.width ?? 0);
      if (next > 0) {
        setWidth(next);
      }
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const plot = useMemo(() => {
    const visible = series.filter((entry) => entry.points.length > 0);
    const allPoints = visible.flatMap((entry) => entry.points);
    if (allPoints.length === 0) {
      return null;
    }
    let minX = Math.min(...allPoints.map((point) => point.x));
    let maxX = Math.max(...allPoints.map((point) => point.x));
    if (maxX === minX) {
      // Single moment in time: pad ±12h so the point sits centred.
      minX -= 12 * 60 * 60 * 1000;
      maxX += 12 * 60 * 60 * 1000;
    }
    // Throughput reads best from zero; pad the top a little.
    const maxY = Math.max(...allPoints.map((point) => point.y + (showErrorBars ? point.stddev ?? 0 : 0)));
    const domainMaxY = maxY <= 0 ? 1 : maxY * 1.08;

    const innerW = Math.max(40, width - MARGIN.left - MARGIN.right);
    const innerH = Math.max(40, height - MARGIN.top - MARGIN.bottom);
    const sx = (x: number) => MARGIN.left + ((x - minX) / (maxX - minX)) * innerW;
    const sy = (y: number) => MARGIN.top + innerH - (y / domainMaxY) * innerH;

    return {
      visible,
      totalPoints: allPoints.length,
      minX,
      maxX,
      domainMaxY,
      innerW,
      innerH,
      sx,
      sy,
      yTicks: niceTicks(0, domainMaxY, 5),
      xTicks: timeTicks(minX, maxX, Math.max(3, Math.min(7, Math.floor(innerW / 110))))
    };
  }, [series, width, height, showErrorBars]);

  if (!plot) {
    return (
      <div className="trend-chart" ref={wrapRef}>
        <div className="empty">
          <strong>No data to chart yet</strong>
          <span>Completed benchmark runs will plot here over time.</span>
        </div>
      </div>
    );
  }

  const { visible, totalPoints, minX, maxX, sx, sy, innerH, yTicks, xTicks } = plot;
  const showMarkers = totalPoints <= MAX_MARKER_POINTS;
  const hoverSeries = hover ? visible[hover.seriesIndex] : null;
  const hoverPoint = hoverSeries ? hoverSeries.points[hover!.pointIndex] : null;

  function locate(clientX: number, clientY: number): Hover | null {
    const svgRect = wrapRef.current?.querySelector("svg")?.getBoundingClientRect();
    if (!svgRect) {
      return null;
    }
    const px = clientX - svgRect.left;
    const py = clientY - svgRect.top;
    const xValue = minX + ((px - MARGIN.left) / Math.max(1, plot!.innerW)) * (maxX - minX);
    let best: Hover | null = null;
    let bestDist = Number.POSITIVE_INFINITY;
    visible.forEach((entry, seriesIndex) => {
      const pointIndex = nearestPointIndex(entry.points, xValue);
      if (pointIndex < 0) {
        return;
      }
      const point = entry.points[pointIndex];
      const dx = sx(point.x) - px;
      const dy = sy(point.y) - py;
      const dist = dx * dx + dy * dy;
      if (dist < bestDist) {
        bestDist = dist;
        best = { seriesIndex, pointIndex, px: sx(point.x), py: sy(point.y) };
      }
    });
    return best;
  }

  return (
    <div className="trend-chart" ref={wrapRef}>
      <svg width={width} height={height} role="img" aria-label="Benchmark trend chart">
        {/* gridlines + y labels */}
        {yTicks.map((tick) => (
          <g key={`y${tick}`}>
            <line x1={MARGIN.left} x2={width - MARGIN.right} y1={sy(tick)} y2={sy(tick)} className="grid" />
            <text x={MARGIN.left - 8} y={sy(tick) + 3} textAnchor="end" className="tick">
              {formatTick(tick)}
            </text>
          </g>
        ))}
        {/* x labels */}
        {xTicks.map((tick) => (
          <text key={`x${tick}`} x={sx(tick)} y={height - 8} textAnchor="middle" className="tick">
            {formatDateTick(tick, maxX - minX)}
          </text>
        ))}
        <text x={MARGIN.left - 8} y={MARGIN.top - 2} textAnchor="end" className="tick unit">
          {yUnit}
        </text>

        {/* series */}
        {visible.map((entry, seriesIndex) => {
          const colorVar = `var(--chart-${(entry.colorIndex % SERIES_COLORS) + 1})`;
          const projected = entry.points.map((point) => ({ px: sx(point.x), py: sy(point.y) }));
          return (
            <g key={entry.key}>
              {showErrorBars
                ? entry.points.map((point) =>
                    point.stddev ? (
                      <line
                        key={`e${point.runId}`}
                        x1={sx(point.x)}
                        x2={sx(point.x)}
                        y1={sy(point.y - point.stddev)}
                        y2={sy(point.y + point.stddev)}
                        stroke={colorVar}
                        strokeOpacity={0.45}
                        strokeWidth={1.5}
                      />
                    ) : null
                  )
                : null}
              <path d={buildLinePath(projected)} fill="none" stroke={colorVar} strokeWidth={2} />
              {showMarkers || entry.points.length === 1
                ? entry.points.map((point, pointIndex) => {
                    const isHighlight = highlightRunId === point.runId;
                    const isHover = hover?.seriesIndex === seriesIndex && hover.pointIndex === pointIndex;
                    return (
                      <circle
                        key={point.runId}
                        cx={sx(point.x)}
                        cy={sy(point.y)}
                        r={isHighlight || isHover ? 5 : 3}
                        fill={colorVar}
                        stroke={isHighlight ? "var(--text-bright)" : "var(--bg-deep)"}
                        strokeWidth={isHighlight ? 2 : 1}
                      />
                    );
                  })
                : null}
            </g>
          );
        })}

        {/* hover crosshair */}
        {hover && hoverPoint ? (
          <g>
            <line x1={hover.px} x2={hover.px} y1={MARGIN.top} y2={MARGIN.top + innerH} className="crosshair" />
            <circle
              cx={hover.px}
              cy={hover.py}
              r={5}
              fill="none"
              stroke={`var(--chart-${(visible[hover.seriesIndex].colorIndex % SERIES_COLORS) + 1})`}
              strokeWidth={2}
            />
          </g>
        ) : null}

        {/* interaction overlay */}
        <rect
          x={MARGIN.left}
          y={MARGIN.top}
          width={plot.innerW}
          height={innerH}
          fill="transparent"
          style={{ cursor: onPointClick ? "pointer" : "crosshair" }}
          onMouseMove={(event) => setHover(locate(event.clientX, event.clientY))}
          onMouseLeave={() => setHover(null)}
          onClick={(event) => {
            const located = locate(event.clientX, event.clientY);
            if (located && onPointClick) {
              onPointClick(visible[located.seriesIndex].points[located.pointIndex].runId);
            }
          }}
        />
      </svg>

      {hover && hoverPoint && hoverSeries && tooltip ? (
        <div
          className="trend-tooltip"
          style={{
            left: Math.min(hover.px + 12, width - 220),
            top: Math.max(4, hover.py - 12)
          }}
        >
          {tooltip(hoverPoint, hoverSeries)}
        </div>
      ) : null}
    </div>
  );
}
