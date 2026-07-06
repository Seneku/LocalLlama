import type { LlamaProfile } from "../src/shared/types";

const GEMMA_MODEL = "E:\\Models\\Gemma4-GGUF\\gemma-4-12B-it-Q4_K_M.gguf";
const ORINTH_MTP_MODEL = "E:\\Models\\Orinth\\ornith-9b-mtp-kl-Q4_K_M.gguf";

const baseProfile = {
  host: "127.0.0.1",
  port: 8080,
  threadsMode: "auto",
  threads: 0,
  jinja: false,
  mlock: true,
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

export const defaultProfiles: LlamaProfile[] = [
  {
    ...baseProfile,
    id: "gemma4-coding",
    name: "Gemma4 Coding",
    description: "Gemma 4 12B coding assistant defaults from start-coding-assistant.cmd.",
    modelPath: GEMMA_MODEL,
    modelAlias: "",
    backendMode: "auto",
    contextSize: 12288,
    gpuLayers: 999,
    reasoning: "off"
  },
  {
    ...baseProfile,
    id: "gemma4-general",
    name: "Gemma4 General",
    description: "Gemma 4 12B general chat / agent defaults from start-general-agent.cmd.",
    modelPath: GEMMA_MODEL,
    modelAlias: "",
    backendMode: "auto",
    contextSize: 12288,
    gpuLayers: 999,
    reasoning: "auto"
  },
  {
    ...baseProfile,
    id: "orinth9b-mtp-coding",
    name: "Orinth9B MTP Coding",
    description: "Orinth 9B MTP coding assistant defaults with 32k context and q8 KV cache.",
    modelPath: ORINTH_MTP_MODEL,
    modelAlias: "",
    backendMode: "auto",
    contextSize: 32768,
    gpuLayers: 99,
    reasoning: "off",
    jinja: true,
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
