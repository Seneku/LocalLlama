import { Play, Wand2, X } from "lucide-react";
import { useEffect, useState } from "react";

import { api } from "../api";
import type { SweepAxisId, SweepPlan } from "../shared/types";
import type { Notify } from "./Toasts";

const AXIS_LABELS: Record<SweepAxisId, string> = {
  gpuLayers: "GPU layers",
  flashAttention: "Flash attention",
  ubatchSize: "Micro-batch",
  batchSize: "Batch",
  kvCache: "KV cache",
  threads: "Threads"
};

const DEFAULT_AXES: SweepAxisId[] = ["gpuLayers", "flashAttention", "ubatchSize", "batchSize", "kvCache"];
const ALL_AXES: SweepAxisId[] = [...DEFAULT_AXES, "threads"];

interface SweepModalProps {
  open: boolean;
  profileId: string | null;
  profileName: string | null;
  onClose(): void;
  /** Called after the sweep starts, so the caller can jump to the Benchmarks view. */
  onStarted(): void;
  notify: Notify;
}

export function SweepModal({ open, profileId, profileName, onClose, onStarted, notify }: SweepModalProps) {
  const [plan, setPlan] = useState<SweepPlan | null>(null);
  const [axes, setAxes] = useState<SweepAxisId[]>(DEFAULT_AXES);
  const [quick, setQuick] = useState(false);
  const [loading, setLoading] = useState(false);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    if (!open || !profileId) {
      return;
    }
    setLoading(true);
    api
      .sweepPlan(profileId, { axes, quick })
      .then(setPlan)
      .catch((error) => notify(error instanceof Error ? error.message : String(error), "error"))
      .finally(() => setLoading(false));
  }, [open, profileId, axes, quick, notify]);

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

  function toggleAxis(axis: SweepAxisId) {
    setAxes((current) => (current.includes(axis) ? current.filter((item) => item !== axis) : [...current, axis]));
  }

  async function start() {
    if (!profileId || !plan) {
      return;
    }
    setStarting(true);
    try {
      await api.startSweep(profileId, plan);
      notify("Optimize sweep started — results appear in the Benchmarks view.", "info");
      onStarted();
      onClose();
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), "error");
    } finally {
      setStarting(false);
    }
  }

  // Rough wall-clock guess: a full-precision llama-bench run is typically 1–4
  // minutes depending on model size; surface a range rather than fake precision.
  const runs = plan?.estimatedRuns ?? 0;
  const timeHint = runs > 0 ? `${Math.max(1, Math.round(runs * (quick ? 0.7 : 1.5)))}–${Math.round(runs * (quick ? 2 : 4))} min` : "";

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal sweep-modal" role="dialog" aria-label="Optimize profile" onClick={(event) => event.stopPropagation()}>
        <div className="panel-title">
          <Wand2 size={18} />
          <span>Optimize {profileName ?? "profile"}</span>
          <button className="icon-button" title="Close" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        <p className="modal-hint">
          Runs a short series of benchmarks that tune one setting at a time — GPU layers first, then flash
          attention, batch sizes, and KV cache — carrying the best result forward. Configurations that would not
          fit VRAM are skipped. When it finishes you can apply the winning settings to the profile with one click.
        </p>

        <div className="log-toolbar sweep-axes">
          {ALL_AXES.map((axis) => (
            <button
              key={axis}
              className={`chip ${axes.includes(axis) ? "active" : ""}`}
              title={axis === "threads" ? "Off by default: thread count rarely matters with full GPU offload." : undefined}
              onClick={() => toggleAxis(axis)}
            >
              {AXIS_LABELS[axis]}
            </button>
          ))}
          <label className="trend-errorbars" title="2 repetitions with shorter prompts — faster but noisier">
            <input type="checkbox" checked={quick} onChange={(event) => setQuick(event.target.checked)} />
            <span>Quick mode</span>
          </label>
        </div>

        {loading ? (
          <div className="empty">Planning sweep…</div>
        ) : plan ? (
          <>
            <div className="sweep-stages">
              {plan.stages.map((stage) => (
                <div key={stage.axis} className="sweep-stage">
                  <strong>{stage.title}</strong>
                  <div className="sweep-candidates">
                    {stage.candidates.map((candidate) => (
                      <span
                        key={candidate.label}
                        className={`pill ${candidate.prunedReason ? "muted" : "ok"}`}
                        title={candidate.prunedReason ? `Skipped: ${candidate.prunedReason}` : undefined}
                      >
                        {candidate.label}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
            {plan.notes.length > 0 ? (
              <div className="warnings">
                {plan.notes.map((note) => (
                  <span key={note}>{note}</span>
                ))}
              </div>
            ) : null}
            <div className="modal-actions">
              <small className="panel-hint">
                ~{plan.estimatedRuns} benchmark runs{timeHint ? ` · roughly ${timeHint}` : ""} — the model reloads
                for every run, so larger models take longer.
              </small>
              <button className="primary" onClick={start} disabled={starting || plan.estimatedRuns < 2}>
                <Play size={16} />
                Start sweep
              </button>
            </div>
          </>
        ) : (
          <div className="empty">Could not build a sweep plan.</div>
        )}
      </div>
    </div>
  );
}
