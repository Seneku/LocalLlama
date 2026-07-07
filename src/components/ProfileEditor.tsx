import { Copy, Cpu, RotateCcw, Save, Trash2 } from "lucide-react";
import { useState, type ReactNode } from "react";

import { api } from "../api";
import type { BackendMode, KvCacheType, LlamaProfile, ReasoningMode, SpecType, ThreadsMode } from "../shared/types";
import type { Notify } from "./Toasts";
import { ConfirmButton, NumberField, SelectField, TextField, ToggleField } from "./ui";

interface ProfileEditorProps {
  draft: LlamaProfile | null;
  isDirty: boolean;
  busy: boolean;
  canDelete: boolean;
  updateDraft<K extends keyof LlamaProfile>(key: K, value: LlamaProfile[K]): void;
  updateSpec<K extends keyof LlamaProfile["speculative"]>(key: K, value: LlamaProfile["speculative"][K]): void;
  saveDraft(): Promise<LlamaProfile | null>;
  revertDraft(): void;
  deleteSelected(): Promise<void>;
  onDuplicate(): void;
  notify: Notify;
}

export function ProfileEditor({
  draft,
  isDirty,
  busy,
  canDelete,
  updateDraft,
  updateSpec,
  saveDraft,
  revertDraft,
  deleteSelected,
  onDuplicate,
  notify
}: ProfileEditorProps) {
  const [picking, setPicking] = useState(false);

  async function browseModel(apply: (path: string) => void, title: string) {
    setPicking(true);
    try {
      const { path } = await api.pickPath("file", { title, gguf: true });
      if (path) {
        apply(path);
      }
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), "error");
    } finally {
      setPicking(false);
    }
  }

  if (!draft) {
    return (
      <section className="editor-panel">
        <div className="empty">
          <Cpu size={26} />
          <strong>No profile loaded</strong>
          <span>Select a profile on the left, or create one to get started.</span>
        </div>
      </section>
    );
  }

  return (
    <section className="editor-panel">
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
          <button title="Duplicate profile" onClick={onDuplicate} disabled={busy}>
            <Copy size={16} />
            Duplicate
          </button>
          <ConfirmButton
            className="danger"
            title="Delete profile"
            confirmLabel="Delete?"
            onConfirm={deleteSelected}
            disabled={busy || !canDelete || !draft.id}
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
          onBrowse={() => browseModel((path) => updateDraft("modelPath", path), "Select a GGUF model")}
          browseBusy={picking}
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
              onBrowse={() => browseModel((path) => updateSpec("draftModelPath", path), "Select a draft GGUF model")}
              browseBusy={picking}
              onChange={(value) => updateSpec("draftModelPath", value)}
            />
          </div>
        ) : null}
      </section>
    </section>
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
