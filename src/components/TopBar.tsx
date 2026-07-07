import { BarChart3, Coffee, Package, Plug, Server, Settings } from "lucide-react";

import type { RuntimeStatus } from "../shared/types";

export type AppView = "server" | "benchmarks";

interface TopBarProps {
  view: AppView;
  status: RuntimeStatus;
  running: boolean;
  uptime: string | null;
  onViewChange(view: AppView): void;
  onOpenGetStarted(): void;
  onOpenConnect(): void;
  onOpenSettings(): void;
}

export function TopBar({
  view,
  status,
  running,
  uptime,
  onViewChange,
  onOpenGetStarted,
  onOpenConnect,
  onOpenSettings
}: TopBarProps) {
  return (
    <header className="topbar">
      <h1>LocalLlama</h1>

      <nav className="view-switcher" role="tablist" aria-label="Main views">
        <button
          role="tab"
          aria-selected={view === "server"}
          className={view === "server" ? "active" : ""}
          onClick={() => onViewChange("server")}
        >
          <Server size={15} />
          Server
        </button>
        <button
          role="tab"
          aria-selected={view === "benchmarks"}
          className={view === "benchmarks" ? "active" : ""}
          onClick={() => onViewChange("benchmarks")}
        >
          <BarChart3 size={15} />
          Benchmarks
        </button>
      </nav>

      <div className="status-strip">
        <span className={`state-chip ${status.state}`}>
          <i className={`runtime-dot ${running ? "live" : ""}`} />
          {status.state}
        </span>
        {status.profileName ? <span className="status-name">{status.profileName}</span> : null}
        {uptime ? <span className="status-uptime">up {uptime}</span> : null}
        {status.health === "unreachable" ? <span className="state-chip unreachable">unreachable</span> : null}
        <button
          className="icon-button"
          title="Get started — install llama.cpp & download models"
          onClick={onOpenGetStarted}
        >
          <Package size={16} />
        </button>
        <a
          className="bmc-button"
          href="https://www.buymeacoffee.com/seneku"
          target="_blank"
          rel="noreferrer noopener"
          title="Support LocalLlama — buy me a coffee"
        >
          <Coffee size={15} />
          <span>Buy me a coffee</span>
        </a>
        <button
          className="icon-button"
          title="Connect to tools — OpenCode, Cline, Continue, Aider…"
          onClick={onOpenConnect}
        >
          <Plug size={16} />
        </button>
        <button className="icon-button" title="Settings" onClick={onOpenSettings}>
          <Settings size={16} />
        </button>
      </div>
    </header>
  );
}
