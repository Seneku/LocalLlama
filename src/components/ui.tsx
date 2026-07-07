import { Check, Copy, FolderOpen } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";

interface TextFieldProps {
  label: string;
  value: string;
  wide?: boolean;
  placeholder?: string;
  type?: "text" | "password";
  /** When set, renders a Browse button that opens a native picker. */
  onBrowse?(): void;
  browseTitle?: string;
  browseBusy?: boolean;
  onChange(value: string): void;
}

export function TextField({
  label,
  value,
  wide = false,
  placeholder,
  type = "text",
  onBrowse,
  browseTitle = "Browse…",
  browseBusy = false,
  onChange
}: TextFieldProps) {
  const input = (
    <input
      type={type}
      value={value}
      placeholder={placeholder}
      spellCheck={false}
      autoComplete={type === "password" ? "off" : undefined}
      onChange={(event) => onChange(event.target.value)}
    />
  );
  return (
    <label className={`field ${wide ? "wide" : ""}`}>
      <span>{label}</span>
      {onBrowse ? (
        <div className="field-with-browse">
          {input}
          <button
            type="button"
            className="icon-button browse-btn"
            title={browseTitle}
            disabled={browseBusy}
            onClick={onBrowse}
          >
            <FolderOpen size={15} />
          </button>
        </div>
      ) : (
        input
      )}
    </label>
  );
}

interface NumberFieldProps {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  onChange(value: number): void;
}

export function NumberField({ label, value, min, max, step, disabled = false, onChange }: NumberFieldProps) {
  const [text, setText] = useState(String(value));
  const [focused, setFocused] = useState(false);

  useEffect(() => {
    if (!focused) {
      setText(String(value));
    }
  }, [value, focused]);

  function handleChange(raw: string) {
    setText(raw);
    const parsed = Number(raw);
    if (raw.trim() !== "" && Number.isFinite(parsed)) {
      onChange(parsed);
    }
  }

  function commit() {
    setFocused(false);
    const parsed = Number(text);
    if (text.trim() === "" || !Number.isFinite(parsed)) {
      setText(String(value));
      return;
    }
    let next = parsed;
    if (min !== undefined) {
      next = Math.max(min, next);
    }
    if (max !== undefined) {
      next = Math.min(max, next);
    }
    onChange(next);
    setText(String(next));
  }

  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        value={text}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onFocus={() => setFocused(true)}
        onBlur={commit}
        onChange={(event) => handleChange(event.target.value)}
      />
    </label>
  );
}

interface SelectFieldProps<T extends string> {
  label: string;
  value: T;
  options: readonly T[];
  labels?: Partial<Record<T, string>>;
  onChange(value: T): void;
}

export function SelectField<T extends string>({ label, value, options, labels = {}, onChange }: SelectFieldProps<T>) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value as T)}>
        {options.map((option) => (
          <option key={option} value={option}>
            {labels[option] ?? option}
          </option>
        ))}
      </select>
    </label>
  );
}

interface ToggleFieldProps {
  label: string;
  checked: boolean;
  hint?: string;
  onChange(value: boolean): void;
}

export function ToggleField({ label, checked, hint, onChange }: ToggleFieldProps) {
  return (
    <label className="toggle-field">
      <span>
        {label}
        {hint ? <small>{hint}</small> : null}
      </span>
      <input
        type="checkbox"
        className="switch"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
    </label>
  );
}

interface CopyButtonProps {
  text: string | null | undefined;
  title: string;
}

export function CopyButton({ text, title }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
      }
    },
    []
  );

  async function copy() {
    if (!text) {
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
      }
      timerRef.current = window.setTimeout(() => setCopied(false), 1600);
    } catch {
      // Clipboard access denied; nothing actionable.
    }
  }

  return (
    <button className={`icon-button ${copied ? "copied" : ""}`} title={copied ? "Copied" : title} onClick={copy} disabled={!text}>
      {copied ? <Check size={16} /> : <Copy size={16} />}
    </button>
  );
}

interface ConfirmButtonProps {
  title: string;
  disabled?: boolean;
  className?: string;
  confirmLabel?: string;
  onConfirm(): void;
  children: ReactNode;
}

export function ConfirmButton({
  title,
  disabled = false,
  className = "",
  confirmLabel = "Confirm?",
  onConfirm,
  children
}: ConfirmButtonProps) {
  const [armed, setArmed] = useState(false);

  useEffect(() => {
    if (!armed) {
      return;
    }
    const timer = window.setTimeout(() => setArmed(false), 3000);
    return () => window.clearTimeout(timer);
  }, [armed]);

  return (
    <button
      className={`${className} ${armed ? "armed" : ""}`.trim()}
      title={armed ? "Click again to confirm" : title}
      disabled={disabled}
      onClick={() => {
        if (armed) {
          setArmed(false);
          onConfirm();
        } else {
          setArmed(true);
        }
      }}
    >
      {armed ? confirmLabel : children}
    </button>
  );
}
