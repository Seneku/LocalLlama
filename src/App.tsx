import {
  Activity,
  Copy,
  Cpu,
  Database,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  Server,
  Square,
  Terminal,
  Trash2
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { api } from "./api";
import { BenchmarkDashboard } from "./components/BenchmarkDashboard";
import { LogView } from "./components/LogView";
import { RequirementsPanel } from "./components/RequirementsPanel";
import { ToastStack, useToasts } from "./components/Toasts";
import { ConfirmButton, CopyButton, NumberField, SelectField, TextField, ToggleField } from "./components/ui";
import type {
  BackendMode,
  CommandPreview,
  KvCacheType,
  LlamaProfile,
  ReasoningMode,
  RuntimeConfig,
  RuntimeLog,
  RuntimeStatus,
  SpecType,
  ThreadsMode
} from "./shared/types";

const emptyStatus: RuntimeStatus = {
  state: "stopped",
  pid: null,
  profileId: null,
  profileName: null,
  startedAt: null,
  exitedAt: null,
  exitCode: null,
  signal: null,
  endpoint: null,
  health: "unknown",
  command: null
};

function cloneProfile(profile: LlamaProfile): LlamaProfile {
  return JSON.parse(JSON.stringify(profile)) as LlamaProfile;
}

function createProfileFrom(base: LlamaProfile, name: string): LlamaProfile {
  const copy = cloneProfile(base);
  copy.id = "";
  copy.name = name;
  return copy;
}

function modelFileName(modelPath: string): string {
  const parts = modelPath.split(/[\\/]/);
  return parts[parts.length - 1] || modelPath;
}

function formatUptime(startedAt: string | null): string | null {
  if (!startedAt) {
    return null;
  }
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(startedAt).getTime()) / 1000));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = seconds % 60;
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${String(rest).padStart(2, "0")}s`;
  }
  return `${rest}s`;
}

export default function App() {
  const [config, setConfig] = useState<RuntimeConfig | null>(null);
  const [profiles, setProfiles] = useState<LlamaProfile[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [draft, setDraft] = useState<LlamaProfile | null>(null);
  const [preview, setPreview] = useState<CommandPreview | null>(null);
  const [status, setStatus] = useState<RuntimeStatus>(emptyStatus);
  const [logs, setLogs] = useState<RuntimeLog[]>([]);
  const [busy, setBusy] = useState(false);
  const { toasts, notify, dismiss } = useToasts();

  const selectedProfile = useMemo(
    () => profiles.find((profile) => profile.id === selectedId) ?? null,
    [profiles, selectedId]
  );
  const isDirty = useMemo(
    () => Boolean(draft && selectedProfile && JSON.stringify(draft) !== JSON.stringify(selectedProfile)),
    [draft, selectedProfile]
  );

  const running = status.state === "running" || status.state === "starting";

  const saveRef = useRef<() => void>(() => undefined);
  saveRef.current = () => {
    if (draft && isDirty && !busy) {
      void saveDraft();
    }
  };

  useEffect(() => {
    void loadInitial();
  }, []);

  useEffect(() => {
    if (!selectedProfile) {
      return;
    }
    setDraft(cloneProfile(selectedProfile));
  }, [selectedProfile]);

  useEffect(() => {
    if (!draft) {
      setPreview(null);
      return;
    }
    const timeout = window.setTimeout(() => {
      api.preview(draft).then(setPreview).catch((error) => notify(error.message, "error"));
    }, 180);
    return () => window.clearTimeout(timeout);
  }, [draft, notify]);

  // Poll faster while the server is active, back off when idle, skip hidden tabs.
  useEffect(() => {
    let disposed = false;
    const refresh = () => {
      if (document.hidden) {
        return;
      }
      api
        .status()
        .then((next) => {
          if (!disposed) {
            setStatus((prev) => (JSON.stringify(prev) === JSON.stringify(next) ? prev : next));
          }
        })
        .catch(() => undefined);
      api
        .logs()
        .then((next) => {
          if (!disposed) {
            setLogs((prev) =>
              prev.length === next.length && prev[prev.length - 1]?.id === next[next.length - 1]?.id ? prev : next
            );
          }
        })
        .catch(() => undefined);
    };
    refresh();
    const timer = window.setInterval(refresh, running ? 1200 : 4000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [running]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        saveRef.current();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  async function loadInitial() {
    setBusy(true);
    try {
      const [nextConfig, nextProfiles, nextStatus, nextLogs] = await Promise.all([
        api.config(),
        api.profiles(),
        api.status(),
        api.logs()
      ]);
      setConfig(nextConfig);
      setProfiles(nextProfiles);
      setSelectedId(nextProfiles[0]?.id ?? "");
      setStatus(nextStatus);
      setLogs(nextLogs);
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), "error");
    } finally {
      setBusy(false);
    }
  }

  async function saveDraft(): Promise<LlamaProfile | null> {
    if (!draft) {
      return null;
    }
    setBusy(true);
    try {
      const saved = draft.id ? await api.updateProfile(draft) : await api.createProfile(draft);
      setProfiles((current) => {
        const index = current.findIndex((profile) => profile.id === saved.id);
        if (index === -1) {
          return [...current, saved];
        }
        const next = [...current];
        next[index] = saved;
        return next;
      });
      setSelectedId(saved.id);
      setDraft(cloneProfile(saved));
      notify("Profile saved.", "success");
      return saved;
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), "error");
      return null;
    } finally {
      setBusy(false);
    }
  }

  function revertDraft() {
    if (selectedProfile) {
      setDraft(cloneProfile(selectedProfile));
      notify("Changes reverted.", "info");
    }
  }

  async function deleteSelected() {
    if (!selectedProfile || profiles.length <= 1) {
      return;
    }
    setBusy(true);
    try {
      await api.deleteProfile(selectedProfile.id);
      const next = profiles.filter((profile) => profile.id !== selectedProfile.id);
      setProfiles(next);
      setSelectedId(next[0]?.id ?? "");
      notify("Profile deleted.", "success");
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), "error");
    } finally {
      setBusy(false);
    }
  }

  async function startSelected() {
    if (!draft) {
      return;
    }
    let profileId = draft.id;
    if (isDirty || !draft.id) {
      const saved = await saveDraft();
      if (!saved) {
        return;
      }
      profileId = saved.id;
    }
    if (!profileId) {
      return;
    }
    setBusy(true);
    try {
      setStatus(await api.start(profileId));
      notify("Server start requested.", "info");
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), "error");
    } finally {
      setBusy(false);
    }
  }

  async function stopServer() {
    setBusy(true);
    try {
      setStatus(await api.stop());
      notify("Server stop requested.", "info");
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), "error");
    } finally {
      setBusy(false);
    }
  }

  async function restartServer() {
    if (!draft) {
      return;
    }
    setBusy(true);
    try {
      await api.stop();
      // Wait until the process actually exits before starting again.
      const deadline = Date.now() + 8000;
      for (;;) {
        const next = await api.status();
        setStatus(next);
        if (next.state === "stopped" || next.state === "exited" || Date.now() > deadline) {
          break;
        }
        await new Promise((resolve) => window.setTimeout(resolve, 300));
      }
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), "error");
      setBusy(false);
      return;
    }
    setBusy(false);
    await startSelected();
  }

  function updateDraft<K extends keyof LlamaProfile>(key: K, value: LlamaProfile[K]) {
    setDraft((current) => (current ? { ...current, [key]: value } : current));
  }

  function updateSpec<K extends keyof LlamaProfile["speculative"]>(
    key: K,
    value: LlamaProfile["speculative"][K]
  ) {
    setDraft((current) =>
      current ? { ...current, speculative: { ...current.speculative, [key]: value } } : current
    );
  }

  const uptime = status.state === "running" ? formatUptime(status.startedAt) : null;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>LlamaTuner</h1>
          <div className="subline">
            <span className="path">{config?.llamaRoot ?? "Loading llama.cpp path"}</span>
            <span className={config?.detected.cudaServer ? "pill ok" : "pill muted"}>CUDA</span>
            <span className={config?.detected.cpuServer ? "pill ok" : "pill muted"}>CPU</span>
            <span className={config?.detected.cudaBench ? "pill ok" : "pill muted"}>CUDA bench</span>
            <span className={config?.detected.cpuBench ? "pill ok" : "pill muted"}>CPU bench</span>
          </div>
        </div>
        <div className="status-strip">
          <span className={`state-chip ${status.state}`}>
            <i className={`runtime-dot ${running ? "live" : ""}`} />
            {status.state}
          </span>
          {status.profileName ? <span className="status-name">{status.profileName}</span> : null}
          {uptime ? <span className="status-uptime">up {uptime}</span> : null}
          {status.health === "unreachable" ? <span className="state-chip unreachable">unreachable</span> : null}
        </div>
      </header>

      <ToastStack toasts={toasts} onDismiss={dismiss} />

      <section className="workspace">
        <aside className="profiles-panel">
          <div className="panel-title">
            <Server size={18} />
            <span>Profiles</span>
            <button
              className="icon-button"
              title="New profile (based on the current one)"
              onClick={() => draft && setDraft(createProfileFrom(draft, "New Profile"))}
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
                  onClick={() => {
                    if (profile.id === selectedId) {
                      // Re-selecting the same profile restores it (e.g. after "+" starts a new draft).
                      setDraft(cloneProfile(profile));
                    } else {
                      setSelectedId(profile.id);
                    }
                  }}
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

        <section className="editor-panel">
          {draft ? (
            <>
              <div className="panel-title editor-title">
                <Cpu size={18} />
                <span>Profile Editor</span>
                {isDirty ? <span className="dirty-chip">unsaved changes</span> : null}
                <div className="button-row">
                  {isDirty && draft.id ? (
                    <button className="ghost" title="Discard changes" onClick={revertDraft} disabled={busy}>
                      <RotateCcw size={16} />
                      Revert
                    </button>
                  ) : null}
                  <button title="Save profile (Ctrl+S)" onClick={saveDraft} disabled={busy || !isDirty}>
                    <Save size={16} />
                    Save
                  </button>
                  <button
                    title="Duplicate profile"
                    onClick={() => setDraft(createProfileFrom(draft, `${draft.name} Copy`))}
                    disabled={busy}
                  >
                    <Copy size={16} />
                    Duplicate
                  </button>
                  <ConfirmButton
                    className="danger"
                    title="Delete profile"
                    confirmLabel="Delete?"
                    onConfirm={deleteSelected}
                    disabled={busy || profiles.length <= 1 || !draft.id}
                  >
                    <Trash2 size={16} />
                    Delete
                  </ConfirmButton>
                </div>
              </div>

              <FormSection title="Model">
                <TextField label="Name" value={draft.name} onChange={(value) => updateDraft("name", value)} />
                <TextField
                  label="Model alias"
                  value={draft.modelAlias}
                  placeholder="optional"
                  onChange={(value) => updateDraft("modelAlias", value)}
                />
                <TextField
                  label="Model path"
                  value={draft.modelPath}
                  wide
                  placeholder="E:\Models\model.gguf"
                  onChange={(value) => updateDraft("modelPath", value)}
                />
                <TextField
                  label="Description"
                  value={draft.description}
                  wide
                  placeholder="What is this profile for?"
                  onChange={(value) => updateDraft("description", value)}
                />
              </FormSection>

              <FormSection title="Server">
                <SelectField<BackendMode>
                  label="Backend"
                  value={draft.backendMode}
                  options={["auto", "cuda", "cpu"]}
                  onChange={(value) => updateDraft("backendMode", value)}
                />
                <SelectField<ReasoningMode>
                  label="Reasoning"
                  value={draft.reasoning}
                  options={["off", "auto"]}
                  onChange={(value) => updateDraft("reasoning", value)}
                />
                <TextField label="Host" value={draft.host} onChange={(value) => updateDraft("host", value)} />
                <NumberField
                  label="Port"
                  value={draft.port}
                  min={1}
                  max={65535}
                  onChange={(value) => updateDraft("port", value)}
                />
                <ToggleField
                  label="Jinja templates"
                  checked={draft.jinja}
                  onChange={(value) => updateDraft("jinja", value)}
                />
              </FormSection>

              <FormSection title="Performance">
                <NumberField
                  label="Context size"
                  value={draft.contextSize}
                  min={256}
                  step={256}
                  onChange={(value) => updateDraft("contextSize", value)}
                />
                <NumberField
                  label="GPU layers"
                  value={draft.gpuLayers}
                  min={0}
                  max={999}
                  onChange={(value) => updateDraft("gpuLayers", value)}
                />
                <SelectField<ThreadsMode>
                  label="Threads"
                  value={draft.threadsMode}
                  options={["auto", "manual"]}
                  onChange={(value) => updateDraft("threadsMode", value)}
                />
                <NumberField
                  label="Thread count"
                  value={draft.threads}
                  min={1}
                  max={256}
                  disabled={draft.threadsMode === "auto"}
                  onChange={(value) => updateDraft("threads", value)}
                />
                <NumberField
                  label="Parallel slots"
                  value={draft.parallelSlots}
                  min={1}
                  max={64}
                  onChange={(value) => updateDraft("parallelSlots", value)}
                />
                <ToggleField
                  label="Mlock"
                  hint="Pin model in RAM"
                  checked={draft.mlock}
                  onChange={(value) => updateDraft("mlock", value)}
                />
                <SelectField<KvCacheType>
                  label="KV cache K"
                  value={draft.kvCacheK}
                  options={["", "f16", "q8_0", "q4_0", "q4_1"]}
                  labels={{ "": "default" }}
                  onChange={(value) => updateDraft("kvCacheK", value)}
                />
                <SelectField<KvCacheType>
                  label="KV cache V"
                  value={draft.kvCacheV}
                  options={["", "f16", "q8_0", "q4_0", "q4_1"]}
                  labels={{ "": "default" }}
                  onChange={(value) => updateDraft("kvCacheV", value)}
                />
              </FormSection>

              <section className="form-section">
                <header>
                  <h3>Speculative decoding</h3>
                  <label className="switch-inline">
                    <input
                      type="checkbox"
                      className="switch"
                      checked={draft.speculative.enabled}
                      onChange={(event) => updateSpec("enabled", event.target.checked)}
                    />
                    <span>{draft.speculative.enabled ? "Enabled" : "Disabled"}</span>
                  </label>
                </header>
                {draft.speculative.enabled ? (
                  <div className="form-grid">
                    <SelectField<SpecType>
                      label="Spec type"
                      value={draft.speculative.type}
                      options={["draft-mtp", "none"]}
                      onChange={(value) => updateSpec("type", value)}
                    />
                    <NumberField
                      label="Spec n-max"
                      value={draft.speculative.draftNMax}
                      min={0}
                      onChange={(value) => updateSpec("draftNMax", value)}
                    />
                    <NumberField
                      label="Draft GPU layers"
                      value={draft.speculative.draftGpuLayers}
                      min={0}
                      onChange={(value) => updateSpec("draftGpuLayers", value)}
                    />
                    <TextField
                      label="Draft model"
                      value={draft.speculative.draftModelPath}
                      wide
                      placeholder="Path to the draft .gguf"
                      onChange={(value) => updateSpec("draftModelPath", value)}
                    />
                  </div>
                ) : null}
              </section>
            </>
          ) : (
            <div className="empty">No profile loaded.</div>
          )}
        </section>

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

          <RequirementsPanel profile={draft} onMessage={notify} />
        </aside>
      </section>

      <BenchmarkDashboard
        draft={draft}
        selectedProfile={selectedProfile}
        isDirty={isDirty}
        runtimeRunning={running}
        busy={busy}
        saveDraft={saveDraft}
        onMessage={notify}
      />

      <section className="logs-band">
        <div className="panel-title">
          <Database size={18} />
          <span>Server Logs</span>
        </div>
        <LogView logs={logs} height={220} emptyText="No logs yet. Start the server to see output here." />
      </section>
    </main>
  );
}

interface FormSectionProps {
  title: string;
  children: ReactNode;
}

function FormSection({ title, children }: FormSectionProps) {
  return (
    <section className="form-section">
      <header>
        <h3>{title}</h3>
      </header>
      <div className="form-grid">{children}</div>
    </section>
  );
}
