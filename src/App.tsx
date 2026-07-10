import { FolderCog, Package } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { api } from "./api";
import { ConnectToolsModal } from "./components/ConnectToolsModal";
import { GetStartedModal } from "./components/GetStartedModal";
import { SettingsModal } from "./components/SettingsModal";
import { SweepModal } from "./components/SweepModal";
import { ToastStack, useToasts } from "./components/Toasts";
import { TopBar, type AppView } from "./components/TopBar";
import { BenchmarksView } from "./views/BenchmarksView";
import { ServerView } from "./views/ServerView";
import type {
  CommandPreview,
  LlamaProfile,
  RuntimeConfig,
  RuntimeLog,
  RuntimeStatus
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

const VIEW_STORAGE_KEY = "localllama.view";

function initialView(): AppView {
  try {
    const stored = localStorage.getItem(VIEW_STORAGE_KEY);
    return stored === "benchmarks" ? "benchmarks" : "server";
  } catch {
    return "server";
  }
}

function cloneProfile(profile: LlamaProfile): LlamaProfile {
  return JSON.parse(JSON.stringify(profile)) as LlamaProfile;
}

function createProfileFrom(base: LlamaProfile, name: string): LlamaProfile {
  const copy = cloneProfile(base);
  copy.id = "";
  copy.name = name;
  return copy;
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
  const [view, setView] = useState<AppView>(initialView);
  const [config, setConfig] = useState<RuntimeConfig | null>(null);
  const [profiles, setProfiles] = useState<LlamaProfile[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [draft, setDraft] = useState<LlamaProfile | null>(null);
  const [preview, setPreview] = useState<CommandPreview | null>(null);
  const [status, setStatus] = useState<RuntimeStatus>(emptyStatus);
  const [logs, setLogs] = useState<RuntimeLog[]>([]);
  const [busy, setBusy] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [getStartedTab, setGetStartedTab] = useState<"llama" | "models" | null>(null);
  const [connectOpen, setConnectOpen] = useState(false);
  const [sweepOpen, setSweepOpen] = useState(false);
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
    try {
      localStorage.setItem(VIEW_STORAGE_KEY, view);
    } catch {
      // Storage unavailable (private mode); the view simply won't persist.
    }
  }, [view]);

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

  // Poll faster while the server is active, back off when idle, skip hidden
  // tabs (but always load once, and refresh immediately when the tab is shown).
  useEffect(() => {
    let disposed = false;
    const refresh = (force = false) => {
      if (document.hidden && !force) {
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
    refresh(true);
    const timer = window.setInterval(() => refresh(), running ? 1200 : 4000);
    const onVisible = () => {
      if (!document.hidden) {
        refresh();
      }
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      disposed = true;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
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

  // Persist an Optimize sweep's winning settings. Operates on the profile the
  // sweep tuned (by id), independent of the current selection, so it works from
  // the Benchmarks view regardless of which profile is drafted. "overwrite"
  // updates that profile in place; "copy" creates a new "(optimized)" profile.
  async function saveOptimizedProfile(
    profileId: string,
    settings: Partial<LlamaProfile>,
    mode: "overwrite" | "copy"
  ): Promise<boolean> {
    const base = profiles.find((profile) => profile.id === profileId);
    if (!base) {
      notify("The profile this sweep tuned no longer exists.", "error");
      return false;
    }
    setBusy(true);
    try {
      if (mode === "overwrite") {
        const saved = await api.updateProfile({ ...base, ...settings });
        setProfiles((current) => current.map((profile) => (profile.id === saved.id ? saved : profile)));
        if (selectedId === saved.id) {
          setDraft(cloneProfile(saved));
        }
        notify(`Optimized settings saved to “${saved.name}”.`, "success");
      } else {
        const saved = await api.createProfile(createProfileFrom({ ...base, ...settings }, `${base.name} (optimized)`));
        setProfiles((current) => [...current, saved]);
        setSelectedId(saved.id);
        setDraft(cloneProfile(saved));
        setView("server");
        notify(`Created “${saved.name}” with the optimized settings.`, "success");
      }
      return true;
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), "error");
      return false;
    } finally {
      setBusy(false);
    }
  }

  // The sweep runs against the saved profile, so persist the draft first.
  async function openOptimize() {
    if (!draft) {
      return;
    }
    if (running) {
      notify("Stop the llama-server runtime before optimizing for cleaner, safer results.", "error");
      return;
    }
    if (isDirty || !draft.id) {
      const saved = await saveDraft();
      if (!saved) {
        return;
      }
    }
    setSweepOpen(true);
  }

  function useDownloadedModel(path: string) {
    setGetStartedTab(null);
    if (draft) {
      updateDraft("modelPath", path);
      notify("Model set on the current profile — save to keep it.", "info");
    } else {
      notify("Select or create a profile first, then use the model.", "error");
    }
  }

  function updateSpec<K extends keyof LlamaProfile["speculative"]>(
    key: K,
    value: LlamaProfile["speculative"][K]
  ) {
    setDraft((current) =>
      current ? { ...current, speculative: { ...current.speculative, [key]: value } } : current
    );
  }

  function selectProfile(profile: LlamaProfile) {
    if (profile.id === selectedId) {
      // Re-selecting the same profile restores it (e.g. after "+" starts a new draft).
      setDraft(cloneProfile(profile));
    } else {
      setSelectedId(profile.id);
    }
  }

  const uptime = status.state === "running" ? formatUptime(status.startedAt) : null;

  return (
    <main className="app-shell">
      <TopBar
        view={view}
        status={status}
        running={running}
        uptime={uptime}
        onViewChange={setView}
        onOpenGetStarted={() => setGetStartedTab("models")}
        onOpenConnect={() => setConnectOpen(true)}
        onOpenSettings={() => setSettingsOpen(true)}
      />

      <ToastStack toasts={toasts} onDismiss={dismiss} />
      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onConfigChanged={setConfig}
        notify={notify}
      />
      <GetStartedModal
        open={getStartedTab !== null}
        initialTab={getStartedTab ?? "models"}
        config={config}
        onClose={() => setGetStartedTab(null)}
        onUseModel={useDownloadedModel}
        notify={notify}
      />
      <ConnectToolsModal
        open={connectOpen}
        status={status}
        preview={preview}
        draft={draft}
        onClose={() => setConnectOpen(false)}
        onSetAlias={(alias) => updateDraft("modelAlias", alias)}
        notify={notify}
      />
      <SweepModal
        open={sweepOpen}
        profileId={draft?.id || null}
        profileName={draft?.name ?? null}
        onClose={() => setSweepOpen(false)}
        onStarted={() => setView("benchmarks")}
        notify={notify}
      />

      {config && !config.detected.cudaServer && !config.detected.cpuServer ? (
        <div className="onboarding">
          <FolderCog size={20} />
          <div>
            <strong>llama.cpp not found</strong>
            <span>
              No <code>llama-server</code> was detected under <code>{config.llamaRoot}</code>. LocalLlama runs your
              local llama.cpp build — get set up in a couple of minutes.
            </span>
          </div>
          <button className="primary" onClick={() => setGetStartedTab("llama")}>
            <Package size={16} />
            Setup guide
          </button>
        </div>
      ) : null}

      {/* Both views stay mounted so filters/compare state survive switching;
          the inactive one is display:none. Polling gates on document.hidden. */}
      <div className={`view ${view === "server" ? "" : "view-hidden"}`}>
        <div className="view-content">
          <ServerView
            config={config}
            profiles={profiles}
            selectedId={selectedId}
            draft={draft}
            preview={preview}
            status={status}
            logs={logs}
            busy={busy}
            isDirty={isDirty}
            running={running}
            onSelectProfile={selectProfile}
            onNewProfile={() => draft && setDraft(createProfileFrom(draft, "New Profile"))}
            onDuplicateProfile={() => draft && setDraft(createProfileFrom(draft, `${draft.name} Copy`))}
            updateDraft={updateDraft}
            updateSpec={updateSpec}
            saveDraft={saveDraft}
            revertDraft={revertDraft}
            deleteSelected={deleteSelected}
            startSelected={startSelected}
            stopServer={stopServer}
            onOptimize={() => void openOptimize()}
            restartServer={restartServer}
            onOpenConnect={() => setConnectOpen(true)}
            notify={notify}
          />
        </div>
      </div>

      <div className={`view ${view === "benchmarks" ? "" : "view-hidden"}`}>
        <div className="view-content">
          <BenchmarksView
            draft={draft}
            selectedProfile={selectedProfile}
            isDirty={isDirty}
            runtimeRunning={running}
            busy={busy}
            saveDraft={saveDraft}
            onOptimize={() => void openOptimize()}
            onSaveOptimized={saveOptimizedProfile}
            onMessage={notify}
          />
        </div>
      </div>
    </main>
  );
}
