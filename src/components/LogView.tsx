import { useEffect, useRef, useState } from "react";

import type { LogStream, RuntimeLog } from "../shared/types";

const FILTERS: Array<{ key: LogStream | "all"; label: string }> = [
  { key: "all", label: "All" },
  { key: "stdout", label: "stdout" },
  { key: "stderr", label: "stderr" },
  { key: "system", label: "system" }
];

interface LogViewProps {
  logs: RuntimeLog[];
  height?: number;
  emptyText?: string;
}

export function LogView({ logs, height = 220, emptyText = "No logs yet." }: LogViewProps) {
  const [filter, setFilter] = useState<LogStream | "all">("all");
  const containerRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true);

  const visible = filter === "all" ? logs : logs.filter((entry) => entry.stream === filter);
  const lastId = visible[visible.length - 1]?.id ?? -1;

  useEffect(() => {
    const node = containerRef.current;
    if (node && stickRef.current) {
      node.scrollTop = node.scrollHeight;
    }
  }, [lastId, filter]);

  function handleScroll() {
    const node = containerRef.current;
    if (!node) {
      return;
    }
    stickRef.current = node.scrollHeight - node.scrollTop - node.clientHeight < 32;
  }

  return (
    <div className="log-view">
      <div className="log-toolbar">
        {FILTERS.map(({ key, label }) => {
          const count = key === "all" ? logs.length : logs.filter((entry) => entry.stream === key).length;
          return (
            <button
              key={key}
              className={`chip ${filter === key ? "active" : ""} ${key === "stderr" && count > 0 ? "warn" : ""}`}
              onClick={() => setFilter(key)}
            >
              {label}
              <small>{count}</small>
            </button>
          );
        })}
      </div>
      <div className="logs" style={{ height }} ref={containerRef} onScroll={handleScroll}>
        {visible.length === 0 ? (
          <span className="log-empty">{emptyText}</span>
        ) : (
          visible.map((entry) => (
            <div key={entry.id} className={`log-line ${entry.stream}`}>
              <time>{new Date(entry.time).toLocaleTimeString()}</time>
              <span>{entry.stream}</span>
              <p>{entry.line}</p>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
