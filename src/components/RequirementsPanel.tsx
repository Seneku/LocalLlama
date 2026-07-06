import { Gauge, HardDrive, Info, TriangleAlert } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { api } from "../api";
import type { LlamaProfile, MemoryEstimate } from "../shared/types";
import type { Notify } from "./Toasts";

interface RequirementsPanelProps {
  profile: LlamaProfile | null;
  onMessage: Notify;
}

function formatGb(value: number | null, digits = 2): string {
  if (value === null || !Number.isFinite(value)) {
    return "-";
  }
  const absolute = Math.abs(value);
  const gbValue = value / 1024;
  if (absolute >= 1024) {
    return `${gbValue.toLocaleString(undefined, {
      maximumFractionDigits: digits,
      minimumFractionDigits: gbValue < 10 ? 2 : 1
    })} GB`;
  }
  return `${gbValue.toLocaleString(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 })} GB`;
}

function fitLabel(estimate: MemoryEstimate | null): string {
  if (!estimate) {
    return "unknown";
  }
  switch (estimate.fit) {
    case "fits":
      return "Fits";
    case "tight":
      return "Tight";
    case "over":
      return "Over";
    case "unknown":
    default:
      return "unknown";
  }
}

export function RequirementsPanel({ profile, onMessage }: RequirementsPanelProps) {
  const [estimate, setEstimate] = useState<MemoryEstimate | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!profile) {
      setEstimate(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    const timeout = window.setTimeout(() => {
      api
        .estimate(profile)
        .then(setEstimate)
        .catch((error) => {
          setEstimate(null);
          onMessage(error.message, "error");
        })
        .finally(() => setLoading(false));
    }, 260);
    return () => window.clearTimeout(timeout);
  }, [profile, onMessage]);

  const vramPercent = useMemo(() => {
    if (!estimate?.totalVramMiB) {
      return 0;
    }
    return Math.min(100, Math.round((estimate.estimatedVramMiB / estimate.totalVramMiB) * 100));
  }, [estimate]);
  const freePercent = useMemo(() => {
    if (!estimate?.totalVramMiB || !estimate.availableVramMiB) {
      return 0;
    }
    return Math.min(100, Math.round((estimate.availableVramMiB / estimate.totalVramMiB) * 100));
  }, [estimate]);
  const shortageMiB =
    estimate?.vramHeadroomMiB !== null && estimate?.vramHeadroomMiB !== undefined && estimate.vramHeadroomMiB < 0
      ? Math.abs(estimate.vramHeadroomMiB)
      : null;
  const fitMessage = estimate
    ? estimate.fit === "over"
      ? `${formatGb(shortageMiB)} over current free VRAM`
      : estimate.fit === "tight"
        ? `${formatGb(estimate.vramHeadroomMiB)} free after load`
        : estimate.fit === "fits"
          ? `${formatGb(estimate.vramHeadroomMiB)} headroom`
          : "GPU capacity unknown"
    : loading
      ? "Reading model and GPU data"
      : "Waiting for profile estimate";

  return (
    <div className="requirements-panel">
      <div className="panel-title compact">
        <Gauge size={18} />
        <span>Requirements</span>
        <span className={`fit-pill ${loading ? "estimating" : estimate?.fit ?? "unknown"}`}>
          {loading ? "Estimating" : fitLabel(estimate)}
        </span>
      </div>

      <div className="requirement-meter">
        <div className="requirement-meter-head">
          <div>
            <span>Estimated VRAM Required</span>
            <strong>{formatGb(estimate?.estimatedVramMiB ?? null)}</strong>
          </div>
          <small className={`fit-summary ${estimate?.fit ?? "unknown"}`}>{fitMessage}</small>
        </div>
        <div className="meter-track" aria-label="Estimated VRAM compared with total GPU memory">
          <i className="meter-used" style={{ width: `${vramPercent}%` }} />
          <b className="meter-free" style={{ width: `${freePercent}%` }} />
        </div>
        <div className="meter-legend">
          <span>
            <i className="legend-required" />
            Required {formatGb(estimate?.estimatedVramMiB ?? null)}
          </span>
          <span>
            <i className="legend-free" />
            Currently free {formatGb(estimate?.availableVramMiB ?? null)}
          </span>
          <span>Total {formatGb(estimate?.totalVramMiB ?? null)}</span>
        </div>
      </div>

      <div className="requirements-grid">
        <Stat label="Model on GPU" value={formatGb(estimate?.breakdown.gpuModelWeightsMiB ?? null)} />
        <Stat label="Context cache" value={formatGb(estimate?.breakdown.kvCacheMiB ?? null)} />
        <Stat label="Runtime overhead" value={formatGb(estimate?.breakdown.computeOverheadMiB ?? null)} />
        <Stat label="Safety buffer" value={formatGb(estimate?.breakdown.safetyMarginMiB ?? null)} />
        <Stat label="System RAM Needed" value={formatGb(estimate?.estimatedSystemRamMiB ?? null)} />
        <Stat label="VRAM Headroom" value={formatGb(estimate?.vramHeadroomMiB ?? null)} tone={estimate?.fit === "over" ? "bad" : "normal"} />
      </div>

      {estimate?.fit === "over" ? (
        <div className="requirement-advice">
          <TriangleAlert size={15} />
          <span>Try lowering context size, GPU layers, parallel slots, or closing other GPU-heavy apps.</span>
        </div>
      ) : null}

      {estimate ? (
        <>
          <div className="hardware-line">
            <HardDrive size={15} />
            <span>{estimate.hardware.gpus[0]?.name ?? "No CUDA GPU detected"}</span>
          </div>
          <div className="hardware-line">
            <HardDrive size={15} />
            <span>
              RAM {formatGb(estimate.hardware.freeRamMiB)} free / {formatGb(estimate.hardware.totalRamMiB)}
            </span>
          </div>
        </>
      ) : (
        <div className="estimate-loading">
          <Info size={14} />
          <span>{loading ? "Estimating requirements for the selected profile..." : "Select a profile to estimate requirements."}</span>
        </div>
      )}

      {estimate?.model.name ? (
        <div className="model-line">
          <span>{estimate.model.name}</span>
          <small>
            {estimate.model.architecture ?? "gguf"} | {formatGb(estimate.model.fileSizeMiB)} file |{" "}
            {estimate.confidence} confidence
          </small>
        </div>
      ) : null}

      {estimate?.assumptions.length ? (
        <details className="estimate-assumptions">
          <summary>
            <Info size={13} />
            <span>How this was estimated</span>
          </summary>
          <ul>
            {estimate.assumptions.map((assumption) => (
              <li key={assumption}>{assumption}</li>
            ))}
          </ul>
        </details>
      ) : (
        <div className="estimate-note">
          <Info size={13} />
          <span>Estimate includes model weights, KV cache, runtime overhead, and a safety buffer.</span>
        </div>
      )}

      {estimate?.warnings.length ? (
        <div className="requirement-warnings">
          {estimate.warnings.map((warning) => (
            <span key={warning}>
              <TriangleAlert size={13} />
              {warning}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

interface StatProps {
  label: string;
  value: string;
  tone?: "normal" | "bad";
}

function Stat({ label, value, tone = "normal" }: StatProps) {
  return (
    <div className={`requirement-stat ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
