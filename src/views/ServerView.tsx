import { Activity, Database, Play, Plug, Plus, RefreshCw, Server, Square, Terminal } from "lucide-react";

import { LogView } from "../components/LogView";
import { ProfileEditor } from "../components/ProfileEditor";
import { RequirementsPanel } from "../components/RequirementsPanel";
import type { Notify } from "../components/Toasts";
import { CopyButton } from "../components/ui";
import type { CommandPreview, LlamaProfile, RuntimeConfig, RuntimeLog, RuntimeStatus } from "../shared/types";

function modelFileName(modelPath: string): string {
  const parts = modelPath.split(/[\\/]/);
  return parts[parts.length - 1] || modelPath;
}

interface ServerViewProps {
  config: RuntimeConfig | null;
  profiles: LlamaProfile[];
  selectedId: string;
  draft: LlamaProfile | null;
  preview: CommandPreview | null;
  status: RuntimeStatus;
  logs: RuntimeLog[];
  busy: boolean;
  isDirty: boolean;
  running: boolean;
  onSelectProfile(profile: LlamaProfile): void;
  onNewProfile(): void;
  onDuplicateProfile(): void;
  updateDraft<K extends keyof LlamaProfile>(key: K, value: LlamaProfile[K]): void;
  updateSpec<K extends keyof LlamaProfile["speculative"]>(key: K, value: LlamaProfile["speculative"][K]): void;
  saveDraft(): Promise<LlamaProfile | null>;
  revertDraft(): void;
  deleteSelected(): Promise<void>;
  startSelected(): Promise<void>;
  stopServer(): Promise<void>;
  restartServer(): Promise<void>;
  onOpenConnect(): void;
  notify: Notify;
}

export function ServerView({
  config,
  profiles,
  selectedId,
  draft,
  preview,
  status,
  logs,
  busy,
  isDirty,
  running,
  onSelectProfile,
  onNewProfile,
  onDuplicateProfile,
  updateDraft,
  updateSpec,
  saveDraft,
  revertDraft,
  deleteSelected,
  startSelected,
  stopServer,
  restartServer,
  onOpenConnect,
  notify
}: ServerViewProps) {
  return (
    <>
      <div className="env-strip">
        <span className="path">{config?.llamaRoot ?? "Loading llama.cpp path"}</span>
        <span className={config?.detected.cudaServer ? "pill ok" : "pill muted"}>CUDA</span>
        <span className={config?.detected.cpuServer ? "pill ok" : "pill muted"}>CPU</span>
        <span className={config?.detected.cudaBench ? "pill ok" : "pill muted"}>CUDA bench</span>
        <span className={config?.detected.cpuBench ? "pill ok" : "pill muted"}>CPU bench</span>
      </div>

      <section className="workspace">
        <aside className="profiles-panel">
          <div className="panel-title">
            <Server size={18} />
            <span>Profiles</span>
            <button
              className="icon-button"
              title="New profile (based on the current one)"
              onClick={onNewProfile}
              disabled={!draft || busy}
            >
              <Plus size={17} />
            </button>
          </div>
          <div className="profile-list">
            {profiles.map((profile) => {
              const active = profile.id === selectedId && Boolean(draft?.id);
              const isLive = running && status.profileId === profile.id;
              return (
                <button
                  key={profile.id}
                  className={`profile-card ${active ? "active" : ""}`}
                  onClick={() => onSelectProfile(profile)}
                >
                  <span className="profile-name">
                    {profile.name}
                    {isLive ? <i className="runtime-dot live" title="Running" /> : null}
                    {active && isDirty ? <i className="dirty-dot" title="Unsaved changes" /> : null}
                  </span>
                  <small>{modelFileName(profile.modelPath)}</small>
                  <span className="profile-meta">
                    <em>{profile.backendMode}</em>
                    <em>{profile.contextSize.toLocaleString()} ctx</em>
                  </span>
                </button>
              );
            })}
            {draft && !draft.id ? (
              <div className="profile-card active unsaved">
                <span className="profile-name">
                  {draft.name || "New Profile"}
                  <i className="dirty-dot" title="Not saved yet" />
                </span>
                <small>unsaved</small>
              </div>
            ) : null}
          </div>
        </aside>

        <ProfileEditor
          draft={draft}
          isDirty={isDirty}
          busy={busy}
          canDelete={profiles.length > 1}
          updateDraft={updateDraft}
          updateSpec={updateSpec}
          saveDraft={saveDraft}
          revertDraft={revertDraft}
          deleteSelected={deleteSelected}
          onDuplicate={onDuplicateProfile}
          notify={notify}
        />

        <aside className="runtime-panel">
          <div className="panel-title">
            <Activity size={18} />
            <span>Runtime</span>
          </div>
          <div className="runtime-actions">
            <button className="primary" title="Start server" onClick={startSelected} disabled={busy || running || !draft}>
              <Play size={17} />
              Start
            </button>
            <button title="Stop server" onClick={stopServer} disabled={busy || !running}>
              <Square size={17} />
              Stop
            </button>
            <button title="Restart server" onClick={restartServer} disabled={busy || !draft}>
              <RefreshCw size={17} />
              Restart
            </button>
          </div>
          <dl className="status-list">
            <div>
              <dt>State</dt>
              <dd>
                <span className={`state-chip ${status.state}`}>{status.state}</span>
                {status.state === "exited" && status.exitCode !== null ? (
                  <span className="exit-code">exit {status.exitCode}</span>
                ) : null}
              </dd>
            </div>
            <div>
              <dt>PID</dt>
              <dd>{status.pid ?? "-"}</dd>
            </div>
            <div>
              <dt>Endpoint</dt>
              <dd>
                {status.endpoint ?? preview?.endpoint ? (
                  <a href={status.endpoint ?? preview?.endpoint ?? "#"} target="_blank" rel="noreferrer">
                    {status.endpoint ?? preview?.endpoint}
                  </a>
                ) : (
                  "-"
                )}
                <button className="connect-link" title="Connect an external tool to this server" onClick={onOpenConnect}>
                  <Plug size={13} /> Connect…
                </button>
              </dd>
            </div>
            <div>
              <dt>Backend</dt>
              <dd>{preview?.backend ?? "-"}</dd>
            </div>
            <div>
              <dt>Model</dt>
              <dd className={preview && !preview.modelExists ? "bad" : ""}>
                {preview ? (preview.modelExists ? "found" : "missing") : "-"}
              </dd>
            </div>
          </dl>

          <div className="panel-title compact">
            <Terminal size={18} />
            <span>Command</span>
            <CopyButton text={preview?.display} title="Copy command" />
          </div>
          <pre className="command-preview">{preview?.display ?? ""}</pre>
          {preview?.warnings.length ? (
            <div className="warnings">
              {preview.warnings.map((warning) => (
                <span key={warning}>{warning}</span>
              ))}
            </div>
          ) : null}

          <RequirementsPanel
            profile={draft}
            onMessage={notify}
            onApplyGpuLayers={(gpuLayers) => {
              updateDraft("gpuLayers", gpuLayers);
              notify(`GPU layers set to ${gpuLayers}. Save the profile to keep it.`, "info");
            }}
          />
        </aside>
      </section>

      <section className="logs-band">
        <div className="panel-title">
          <Database size={18} />
          <span>Server Logs</span>
        </div>
        <LogView logs={logs} height={220} emptyText="No logs yet. Start the server to see output here." />
      </section>
    </>
  );
}
