import { Copy, Cpu, RotateCcw, Save, Trash2 } from "lucide-react";
import { useState, type ReactNode } from "react";

import { api } from "../api";
import type {
  BackendMode,
  FlashAttentionMode,
  KvCacheType,
  LlamaProfile,
  ReasoningMode,
  SpecType,
  ThreadsMode
} from "../shared/types";
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
          help="The name clients send in the OpenAI `model` field (passed as -a). Set a clean id so tools like OpenCode can target this model."
          onChange={(value) => updateDraft("modelAlias", value)}
        />
        <TextField
          label="Model path"
          value={draft.modelPath}
          wide
          placeholder="E:\Models\model.gguf"
          help="Full path to the .gguf model file this profile launches. Use Browse to pick it."
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
          help="Which llama.cpp build runs the model. Auto uses CUDA if a CUDA build is found, otherwise CPU (Metal on Apple Silicon)."
          onChange={(value) => updateDraft("backendMode", value)}
        />
        <SelectField<ReasoningMode>
          label="Reasoning"
          value={draft.reasoning}
          options={["off", "auto"]}
          help="Whether the model emits its thinking/reasoning trace. Off is faster and terser; Auto lets the model decide."
          onChange={(value) => updateDraft("reasoning", value)}
        />
        <TextField
          label="Host"
          value={draft.host}
          help="Address the server binds to. 127.0.0.1 = this PC only; 0.0.0.0 = reachable from other devices on your network."
          onChange={(value) => updateDraft("host", value)}
        />
        <NumberField
          label="Port"
          value={draft.port}
          min={1}
          max={65535}
          help="TCP port for the OpenAI-compatible API (default 8080)."
          onChange={(value) => updateDraft("port", value)}
        />
        <NumberField
          label="Temperature"
          value={draft.temperature}
          min={0}
          max={2}
          step={0.1}
          help="Default sampling randomness. 0 = deterministic (best for coding); higher = more varied. Clients can override this per request."
          onChange={(value) => updateDraft("temperature", value)}
        />
        <ToggleField
          label="Jinja templates"
          checked={draft.jinja}
          help="Format prompts with the model's built-in chat template. Needed by some models for correct chat and tool-call formatting."
          onChange={(value) => updateDraft("jinja", value)}
        />
      </FormSection>

      <FormSection title="Performance">
        <NumberField
          label="Context size"
          value={draft.contextSize}
          min={256}
          step={256}
          help="Max tokens (prompt + reply) kept in memory. Larger allows longer conversations but uses more VRAM/RAM for the KV cache."
          onChange={(value) => updateDraft("contextSize", value)}
        />
        <NumberField
          label="GPU layers"
          value={draft.gpuLayers}
          min={0}
          max={999}
          help="How many model layers to run on the GPU (999 = all). More on the GPU = faster, until you run out of VRAM. For MoE models pair 999 with 'MoE experts on CPU'."
          onChange={(value) => updateDraft("gpuLayers", value)}
        />
        <ToggleField
          label="Auto-fit VRAM"
          checked={draft.fit}
          help="Lets llama.cpp auto-size unset options (like GPU layers) to fit your VRAM (--fit on). Turn off for full manual control."
          onChange={(value) => updateDraft("fit", value)}
        />
        <NumberField
          label="Fit target (MiB)"
          value={draft.fitTargetMiB}
          min={0}
          step={128}
          disabled={!draft.fit}
          help="Free-VRAM margin --fit leaves on each GPU. Higher is safer (more headroom); lower packs more onto the GPU."
          onChange={(value) => updateDraft("fitTargetMiB", value)}
        />
        <ToggleField
          label="MoE experts on CPU"
          checked={draft.cpuMoe}
          help="For Mixture-of-Experts models: keep the large expert weights in system RAM and run attention on the GPU (--cpu-moe). Lets big MoE models fit a small GPU — usually far faster than partial layer offload."
          onChange={(value) => updateDraft("cpuMoe", value)}
        />
        <NumberField
          label="CPU-expert layers"
          value={draft.nCpuMoe}
          min={0}
          disabled={draft.cpuMoe}
          help="Like 'MoE experts on CPU' but only for the first N layers (--n-cpu-moe); the rest keep their experts on the GPU. Lower N = more on GPU = faster, until VRAM fills."
          onChange={(value) => updateDraft("nCpuMoe", value)}
        />
        <SelectField<ThreadsMode>
          label="Threads"
          value={draft.threadsMode}
          options={["auto", "manual"]}
          help="CPU threads for work that runs on the CPU. Auto picks a sensible default; Manual lets you set the count."
          onChange={(value) => updateDraft("threadsMode", value)}
        />
        <NumberField
          label="Thread count"
          value={draft.threads}
          min={1}
          max={256}
          disabled={draft.threadsMode === "auto"}
          help="Number of CPU threads in Manual mode. Often fastest at your physical core count (not counting hyperthreads)."
          onChange={(value) => updateDraft("threads", value)}
        />
        <NumberField
          label="Parallel slots"
          value={draft.parallelSlots}
          min={1}
          max={64}
          help="Concurrent request slots. Splits the context into N and reserves buffers for N streams — use 1 for a single user; more just wastes VRAM."
          onChange={(value) => updateDraft("parallelSlots", value)}
        />
        <NumberField
          label="Batch size"
          value={draft.batchSize}
          min={0}
          max={16384}
          help="Logical batch size (-b): max tokens queued per iteration. 0 = llama.cpp default (2048). Mostly affects prompt-processing throughput."
          onChange={(value) => updateDraft("batchSize", value)}
        />
        <NumberField
          label="Micro-batch size"
          value={draft.ubatchSize}
          min={0}
          max={8192}
          help="Physical micro-batch size (-ub): tokens per compute pass. 0 = llama.cpp default (512). Smaller cuts compute-buffer VRAM; larger can speed up prompt processing."
          onChange={(value) => updateDraft("ubatchSize", value)}
        />
        <SelectField<FlashAttentionMode>
          label="Flash attention"
          value={draft.flashAttention}
          options={["auto", "on", "off"]}
          help="Fused attention kernel (-fa). Auto lets llama.cpp decide per backend; on usually saves VRAM and speeds up long contexts on modern GPUs."
          onChange={(value) => updateDraft("flashAttention", value)}
        />
        <ToggleField
          label="Mlock"
          checked={draft.mlock}
          help="Locks the model in RAM so the OS can't swap it to disk (--mlock). Prevents paging stalls; needs enough free RAM."
          onChange={(value) => updateDraft("mlock", value)}
        />
        <ToggleField
          label="Memory-map (mmap)"
          checked={draft.mmap}
          help="Maps the model from disk on demand (default, on). Turning it off loads the whole model into RAM (--no-mmap) — pair with Mlock for full residency."
          onChange={(value) => updateDraft("mmap", value)}
        />
        <SelectField<KvCacheType>
          label="KV cache K"
          value={draft.kvCacheK}
          options={["", "f16", "q8_0", "q4_0", "q4_1"]}
          labels={{ "": "default" }}
          help="Quantization of the attention key cache. q8_0 roughly halves KV VRAM vs f16 with little quality loss; q4 saves more but is lossier."
          onChange={(value) => updateDraft("kvCacheK", value)}
        />
        <SelectField<KvCacheType>
          label="KV cache V"
          value={draft.kvCacheV}
          options={["", "f16", "q8_0", "q4_0", "q4_1"]}
          labels={{ "": "default" }}
          help="Quantization of the attention value cache. Match it to KV cache K; q8_0 is a safe VRAM saver."
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
              help="Speculative decoding method. draft-mtp uses the model's built-in MTP heads to guess several tokens ahead and verify them in one pass — faster generation when guesses are usually right."
              onChange={(value) => updateSpec("type", value)}
            />
            <NumberField
              label="Spec n-max"
              value={draft.speculative.draftNMax}
              min={0}
              help="Max tokens drafted per step before the main model verifies them. Higher can speed things up when the draft is accurate, but wastes work when it's wrong."
              onChange={(value) => updateSpec("draftNMax", value)}
            />
            <NumberField
              label="Draft GPU layers"
              value={draft.speculative.draftGpuLayers}
              min={0}
              help="GPU layers for the draft model, separate from the main model's. Only used with an external draft model."
              onChange={(value) => updateSpec("draftGpuLayers", value)}
            />
            <NumberField
              label="Draft p-min"
              value={draft.speculative.draftPMin}
              min={0}
              max={1}
              step={0.05}
              help="Minimum confidence for a drafted token to be proposed (--spec-draft-p-min). Higher = only draft when the model is sure, reducing wasted verification."
              onChange={(value) => updateSpec("draftPMin", value)}
            />
            <SelectField<KvCacheType>
              label="Draft KV cache K"
              value={draft.speculative.draftCacheK}
              options={["", "f16", "q8_0", "q4_0", "q4_1"]}
              labels={{ "": "default" }}
              help="KV cache key quant for the draft model (-ctkd). q8_0 saves VRAM on the draft's cache."
              onChange={(value) => updateSpec("draftCacheK", value)}
            />
            <SelectField<KvCacheType>
              label="Draft KV cache V"
              value={draft.speculative.draftCacheV}
              options={["", "f16", "q8_0", "q4_0", "q4_1"]}
              labels={{ "": "default" }}
              help="KV cache value quant for the draft model (-ctvd). Match it to Draft KV cache K."
              onChange={(value) => updateSpec("draftCacheV", value)}
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
