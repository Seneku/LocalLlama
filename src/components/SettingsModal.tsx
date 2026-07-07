import { FolderCog, X } from "lucide-react";
import { useEffect, useState } from "react";

import { api } from "../api";
import type { AppSettings, RuntimeConfig } from "../shared/types";
import type { Notify } from "./Toasts";
import { TextField } from "./ui";

const emptySettings: AppSettings = {
  llamaRoot: "",
  cudaServerPath: "",
  cpuServerPath: "",
  cudaBenchPath: "",
  cpuBenchPath: "",
  modelsDir: "",
  hfToken: ""
};

interface SettingsModalProps {
  open: boolean;
  onClose(): void;
  onConfigChanged(config: RuntimeConfig): void;
  notify: Notify;
}

export function SettingsModal({ open, onClose, onConfigChanged, notify }: SettingsModalProps) {
  const [settings, setSettings] = useState<AppSettings>(emptySettings);
  const [config, setConfig] = useState<RuntimeConfig | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [picking, setPicking] = useState(false);

  useEffect(() => {
    if (!open) {
      return;
    }
    setLoading(true);
    api
      .settings()
      .then((response) => {
        setSettings(response.settings);
        setConfig(response.config);
      })
      .catch((error) => notify(error.message, "error"))
      .finally(() => setLoading(false));
  }, [open, notify]);

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

  function update<K extends keyof AppSettings>(key: K, value: string) {
    setSettings((current) => ({ ...current, [key]: value }));
  }

  async function browse(key: keyof AppSettings, mode: "folder" | "file", title: string) {
    setPicking(true);
    try {
      const { path } = await api.pickPath(mode, { title });
      if (path) {
        update(key, path);
      }
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), "error");
    } finally {
      setPicking(false);
    }
  }

  async function save() {
    setSaving(true);
    try {
      const response = await api.saveSettings(settings);
      setSettings(response.settings);
      setConfig(response.config);
      onConfigChanged(response.config);
      notify("Settings saved.", "success");
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), "error");
    } finally {
      setSaving(false);
    }
  }

  const binaries: Array<{ key: keyof AppSettings; label: string; resolved: string; detected: boolean }> = config
    ? [
        { key: "cudaServerPath", label: "CUDA server", resolved: config.cudaServerPath, detected: config.detected.cudaServer },
        { key: "cpuServerPath", label: "CPU server", resolved: config.cpuServerPath, detected: config.detected.cpuServer },
        { key: "cudaBenchPath", label: "CUDA bench", resolved: config.cudaBenchPath, detected: config.detected.cudaBench },
        { key: "cpuBenchPath", label: "CPU bench", resolved: config.cpuBenchPath, detected: config.detected.cpuBench }
      ]
    : [];

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" role="dialog" aria-label="Settings" onClick={(event) => event.stopPropagation()}>
        <div className="panel-title">
          <FolderCog size={18} />
          <span>Settings</span>
          <button className="icon-button" title="Close settings" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        {loading ? (
          <div className="empty">Loading settings...</div>
        ) : (
          <>
            <p className="modal-hint">
              Point LocalLlama at your llama.cpp installation. Binary paths are derived from the root — override
              them only if your build lives elsewhere. Empty fields use the defaults shown as placeholders.
            </p>

            <div className="settings-fields">
              <TextField
                label="llama.cpp root"
                value={settings.llamaRoot}
                wide
                placeholder={config?.llamaRoot ?? ""}
                onBrowse={() => browse("llamaRoot", "folder", "Select your llama.cpp folder")}
                browseBusy={picking}
                onChange={(value) => update("llamaRoot", value)}
              />
              {binaries.map(({ key, label, resolved, detected }) => (
                <div key={key} className="settings-path-row">
                  <TextField
                    label={label}
                    value={settings[key]}
                    wide
                    placeholder={resolved}
                    onBrowse={() => browse(key, "file", `Select the ${label} executable`)}
                    browseBusy={picking}
                    onChange={(value) => update(key, value)}
                  />
                  <span className={detected ? "pill ok" : "pill muted"} title={resolved}>
                    {detected ? "found" : "missing"}
                  </span>
                </div>
              ))}

              <div className="settings-divider">Models</div>
              <TextField
                label="Models folder"
                value={settings.modelsDir}
                wide
                placeholder={config?.modelsDir ?? ""}
                onBrowse={() => browse("modelsDir", "folder", "Select your models folder")}
                browseBusy={picking}
                onChange={(value) => update("modelsDir", value)}
              />
              <TextField
                label="Hugging Face token (optional)"
                value={settings.hfToken}
                wide
                type="password"
                placeholder="hf_… — only needed for gated/private models"
                onChange={(value) => update("hfToken", value)}
              />
            </div>

            <div className="modal-actions">
              <button className="primary" onClick={save} disabled={saving}>
                {saving ? "Saving..." : "Save"}
              </button>
              <button onClick={onClose} disabled={saving}>
                Close
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
