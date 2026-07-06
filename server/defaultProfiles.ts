import type { LlamaProfile } from "../src/shared/types";

const baseProfile = {
  host: "127.0.0.1",
  port: 8080,
  threadsMode: "auto",
  threads: 0,
  jinja: false,
  mlock: false,
  parallelSlots: 1,
  kvCacheK: "",
  kvCacheV: "",
  speculative: {
    enabled: false,
    type: "none",
    draftNMax: 0,
    draftModelPath: "",
    draftGpuLayers: 0
  }
} satisfies Partial<LlamaProfile>;

// Seeded once for a brand-new install. Intentionally neutral: no model path,
// so the app opens on an empty starter that nudges the user to set their model
// (and, if llama.cpp isn't detected, to open Settings first).
export const defaultProfiles: LlamaProfile[] = [
  {
    ...baseProfile,
    id: "starter",
    name: "Starter",
    description: "Set a model path above to get going. If llama.cpp isn't detected, open Settings first.",
    modelPath: "",
    modelAlias: "",
    backendMode: "auto",
    contextSize: 8192,
    gpuLayers: 999,
    reasoning: "off"
  }
];

// Illustrative configurations used as references and in tests. These are not
// seeded into a user's data; the model paths are placeholders.
export const exampleProfiles: LlamaProfile[] = [
  {
    ...baseProfile,
    id: "gemma4-coding",
    name: "Gemma4 Coding",
    description: "12B coding assistant: full offload, 12k context, reasoning off.",
    modelPath: "C:\\models\\gemma-4-12b-it-Q4_K_M.gguf",
    modelAlias: "",
    backendMode: "auto",
    contextSize: 12288,
    gpuLayers: 999,
    reasoning: "off",
    mlock: true
  },
  {
    ...baseProfile,
    id: "gemma4-general",
    name: "Gemma4 General",
    description: "12B general chat / agent: full offload, 12k context, reasoning auto.",
    modelPath: "C:\\models\\gemma-4-12b-it-Q4_K_M.gguf",
    modelAlias: "",
    backendMode: "auto",
    contextSize: 12288,
    gpuLayers: 999,
    reasoning: "auto",
    mlock: true
  },
  {
    ...baseProfile,
    id: "orinth9b-mtp-coding",
    name: "Orinth9B MTP Coding",
    description: "9B hybrid-SSM coding model: 32k context, q8 KV cache, bundled MTP speculative.",
    modelPath: "C:\\models\\ornith-9b-mtp-Q4_K_M.gguf",
    modelAlias: "",
    backendMode: "auto",
    contextSize: 32768,
    gpuLayers: 99,
    reasoning: "off",
    jinja: true,
    mlock: true,
    kvCacheK: "q8_0",
    kvCacheV: "q8_0",
    speculative: {
      enabled: true,
      type: "draft-mtp",
      draftNMax: 3,
      draftModelPath: "",
      draftGpuLayers: 0
    }
  }
];
