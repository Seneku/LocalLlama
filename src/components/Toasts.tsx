import { CheckCircle2, Info, TriangleAlert, X } from "lucide-react";
import { useCallback, useRef, useState } from "react";

export type ToastKind = "success" | "error" | "info";

export interface Toast {
  id: number;
  kind: ToastKind;
  text: string;
}

export type Notify = (text: string, kind?: ToastKind) => void;

export function useToasts() {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const notify = useCallback<Notify>(
    (text, kind = "info") => {
      const id = nextId.current++;
      setToasts((current) => [...current.slice(-4), { id, kind, text }]);
      window.setTimeout(() => dismiss(id), kind === "error" ? 8000 : 3800);
    },
    [dismiss]
  );

  return { toasts, notify, dismiss };
}

const icons = {
  success: CheckCircle2,
  error: TriangleAlert,
  info: Info
} as const;

interface ToastStackProps {
  toasts: Toast[];
  onDismiss(id: number): void;
}

export function ToastStack({ toasts, onDismiss }: ToastStackProps) {
  if (toasts.length === 0) {
    return null;
  }
  return (
    <div className="toast-stack" role="status" aria-live="polite">
      {toasts.map((toast) => {
        const Icon = icons[toast.kind];
        return (
          <div key={toast.id} className={`toast ${toast.kind}`}>
            <Icon size={16} />
            <span>{toast.text}</span>
            <button className="toast-close" title="Dismiss" onClick={() => onDismiss(toast.id)}>
              <X size={14} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
