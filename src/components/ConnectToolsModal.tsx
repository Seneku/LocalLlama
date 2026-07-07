import { AlertTriangle, ExternalLink, Plug, Wand2, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { buildConnectionInfo, TOOLS, type ToolId } from "../shared/connect";
import type { CommandPreview, LlamaProfile, RuntimeStatus } from "../shared/types";
import type { Notify } from "./Toasts";
import { CopyButton } from "./ui";

interface ConnectToolsModalProps {
  open: boolean;
  status: RuntimeStatus;
  preview: CommandPreview | null;
  draft: LlamaProfile | null;
  onClose(): void;
  onSetAlias(alias: string): void;
  notify: Notify;
}

export function ConnectToolsModal({
  open,
  status,
  preview,
  draft,
  onClose,
  onSetAlias,
  notify
}: ConnectToolsModalProps) {
  const [tool, setTool] = useState<ToolId>("opencode");

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

  const info = useMemo(() => buildConnectionInfo(status, preview, draft), [status, preview, draft]);
  const active = TOOLS.find((entry) => entry.id === tool) ?? TOOLS[0];
  const snippet = active.build(info);

  if (!open) {
    return null;
  }

  function applyAlias() {
    onSetAlias(info.suggestedAlias);
    notify(`Alias set to "${info.suggestedAlias}". Save the profile and Restart to apply it.`, "info");
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal connect" role="dialog" aria-label="Connect to tools" onClick={(event) => event.stopPropagation()}>
        <div className="panel-title">
          <Plug size={18} />
          <span>Connect to tools</span>
          <button className="icon-button" title="Close" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        <p className="modal-hint">
          Point any OpenAI-compatible tool at your running llama.cpp server. Copy the details below, then pick your tool
          for a ready-to-paste config.
          {info.loopbackOnly ? (
            <> The endpoint is loopback-only — set the profile host to <code>0.0.0.0</code> to reach it from another machine.</>
          ) : null}
        </p>

        {!info.running ? (
          <div className="connect-banner">
            <AlertTriangle size={15} />
            <span>The server isn't running yet — start it to make these endpoints live. The values below reflect the current profile.</span>
          </div>
        ) : null}

        <dl className="connect-summary">
          <div>
            <dt>Base URL</dt>
            <dd>
              <code>{info.baseUrl}</code>
              <CopyButton text={info.baseUrl} title="Copy base URL" />
            </dd>
          </div>
          <div>
            <dt>Model</dt>
            <dd>
              <code>{info.modelId}</code>
              <CopyButton text={info.modelId} title="Copy model id" />
            </dd>
          </div>
          <div>
            <dt>API key</dt>
            <dd>
              <code>{info.apiKey}</code>
              <CopyButton text={info.apiKey} title="Copy api key" />
              <small className="dd-note">any non-empty string works</small>
            </dd>
          </div>
        </dl>

        {!info.aliasSet && info.suggestedAlias ? (
          <div className="connect-banner nudge">
            <AlertTriangle size={15} />
            <span>
              This profile has no model alias, so tools will see the full file path as the model id. Set a clean alias:
            </span>
            <button className="ghost" onClick={applyAlias} disabled={!draft}>
              <Wand2 size={14} /> Set alias to “{info.suggestedAlias}”
            </button>
          </div>
        ) : null}

        <div className="log-toolbar connect-tools">
          {TOOLS.map((entry) => (
            <button
              key={entry.id}
              className={`chip ${tool === entry.id ? "active" : ""}`}
              onClick={() => setTool(entry.id)}
            >
              {entry.label}
            </button>
          ))}
        </div>

        <ol className="guide-steps">
          {active.steps.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>

        <div className="panel-title compact">
          {snippet.filename ? <span className="snippet-name">{snippet.filename}</span> : <span>{active.label}</span>}
          <CopyButton text={snippet.code} title="Copy config" />
        </div>
        <pre className="command-preview">{snippet.code}</pre>

        <div className="guide-links">
          <a href={active.docsUrl} target="_blank" rel="noreferrer noopener">
            <ExternalLink size={14} /> {active.label} docs
          </a>
        </div>
      </div>
    </div>
  );
}
