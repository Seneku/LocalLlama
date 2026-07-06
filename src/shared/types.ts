export type BackendMode = "auto" | "cuda" | "cpu";
export type ResolvedBackend = "CUDA" | "CPU";
export type ReasoningMode = "off" | "auto";
export type ThreadsMode = "auto" | "manual";
export type KvCacheType = "" | "f16" | "q8_0" | "q4_0" | "q4_1";
export type SpecType = "draft-mtp" | "none";
export type RuntimeState = "stopped" | "starting" | "running" | "exited";
export type LogStream = "stdout" | "stderr" | "system";
export type BenchmarkRunStatus = "running" | "completed" | "failed" | "cancelled";
export type BenchmarkState = "idle" | "running";
export type FlashAttentionMode = "auto" | "on" | "off";
export type EstimateFit = "fits" | "tight" | "over" | "unknown";
export type EstimateConfidence = "low" | "medium" | "high";

export interface SpeculativeSettings {
  enabled: boolean;
  type: SpecType;
  draftNMax: number;
  draftModelPath: string;
  draftGpuLayers: number;
}

export interface LlamaProfile {
  id: string;
  name: string;
  description: string;
  modelPath: string;
  modelAlias: string;
  backendMode: BackendMode;
  host: string;
  port: number;
  contextSize: number;
  threadsMode: ThreadsMode;
  threads: number;
  gpuLayers: number;
  reasoning: ReasoningMode;
  jinja: boolean;
  mlock: boolean;
  parallelSlots: number;
  kvCacheK: KvCacheType;
  kvCacheV: KvCacheType;
  speculative: SpeculativeSettings;
}

export interface RuntimeConfig {
  llamaRoot: string;
  cudaServerPath: string;
  cpuServerPath: string;
  cudaBenchPath: string;
  cpuBenchPath: string;
  dataPath: string;
  defaultThreads: number;
  detected: {
    cudaServer: boolean;
    cpuServer: boolean;
    cudaBench: boolean;
    cpuBench: boolean;
  };
}

export interface CommandPreview {
  executable: string;
  args: string[];
  display: string;
  backend: ResolvedBackend;
  endpoint: string;
  serverExists: boolean;
  modelExists: boolean;
  warnings: string[];
}

export interface RuntimeStatus {
  state: RuntimeState;
  pid: number | null;
  profileId: string | null;
  profileName: string | null;
  startedAt: string | null;
  exitedAt: string | null;
  exitCode: number | null;
  signal: string | null;
  endpoint: string | null;
  health: "unknown" | "ok" | "unreachable";
  command: CommandPreview | null;
}

export interface RuntimeLog {
  id: number;
  time: string;
  stream: LogStream;
  line: string;
}

export interface BenchmarkSettings {
  promptTokens: number;
  generationTokens: number;
  repetitions: number;
  batchSize: number;
  ubatchSize: number;
  noWarmup: boolean;
  flashAttention: FlashAttentionMode;
}

export interface BenchmarkCommandPreview {
  executable: string;
  args: string[];
  display: string;
  backend: ResolvedBackend;
  benchmarkExists: boolean;
  modelExists: boolean;
  warnings: string[];
}

export interface BenchmarkRow {
  test: string;
  promptTokens: number;
  generationTokens: number;
  avgTokensPerSecond: number | null;
  stddevTokensPerSecond: number | null;
  avgMilliseconds: number | null;
  raw: Record<string, unknown>;
}

export interface BenchmarkMetrics {
  promptTokensPerSecond: number | null;
  generationTokensPerSecond: number | null;
  generationMsPerToken: number | null;
  promptStddev: number | null;
  generationStddev: number | null;
  totalSeconds: number | null;
  score: number | null;
}

export interface BenchmarkProfileSnapshot {
  name: string;
  modelPath: string;
  backendMode: BackendMode;
  contextSize: number;
  threadsMode: ThreadsMode;
  threads: number;
  gpuLayers: number;
  kvCacheK: KvCacheType;
  kvCacheV: KvCacheType;
  jinja: boolean;
  reasoning: ReasoningMode;
  parallelSlots: number;
  speculativeEnabled: boolean;
  speculativeType: SpecType;
}

export interface BenchmarkRun {
  id: string;
  profileId: string;
  profileName: string;
  createdAt: string;
  completedAt: string | null;
  status: BenchmarkRunStatus;
  exitCode: number | null;
  signal: string | null;
  backend: ResolvedBackend;
  settings: BenchmarkSettings;
  command: BenchmarkCommandPreview;
  profile: BenchmarkProfileSnapshot;
  rows: BenchmarkRow[];
  metrics: BenchmarkMetrics;
  stdout: string;
  stderr: string;
  error: string | null;
}

export interface BenchmarkStatus {
  state: BenchmarkState;
  activeRunId: string | null;
  profileId: string | null;
  profileName: string | null;
  startedAt: string | null;
  command: BenchmarkCommandPreview | null;
}

export interface GpuInfo {
  name: string;
  totalMiB: number | null;
  usedMiB: number | null;
  freeMiB: number | null;
}

export interface HardwareInfo {
  totalRamMiB: number;
  freeRamMiB: number;
  gpus: GpuInfo[];
}

export interface ModelMetadata {
  architecture: string | null;
  name: string | null;
  parameterSize: string | null;
  fileType: number | null;
  fileSizeMiB: number;
  blockCount: number | null;
  contextLength: number | null;
  embeddingLength: number | null;
  headCount: number | null;
  headCountKv: number | null;
  keyLength: number | null;
  valueLength: number | null;
}

export interface MemoryEstimateBreakdown {
  modelWeightsMiB: number;
  gpuModelWeightsMiB: number;
  cpuModelWeightsMiB: number;
  kvCacheMiB: number;
  computeOverheadMiB: number;
  draftModelMiB: number;
  safetyMarginMiB: number;
}

export interface MemoryEstimate {
  backend: ResolvedBackend;
  fit: EstimateFit;
  confidence: EstimateConfidence;
  totalVramMiB: number | null;
  availableVramMiB: number | null;
  estimatedVramMiB: number;
  estimatedSystemRamMiB: number;
  vramHeadroomMiB: number | null;
  model: ModelMetadata;
  hardware: HardwareInfo;
  breakdown: MemoryEstimateBreakdown;
  assumptions: string[];
  warnings: string[];
}
