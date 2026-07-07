import type {
  AppSettings,
  BenchmarkCommandPreview,
  BenchmarkRun,
  BenchmarkSettings,
  BenchmarkStatus,
  CommandPreview,
  DownloadStatus,
  FavoriteModel,
  LlamaCppRelease,
  LlamaProfile,
  LocalModel,
  MemoryEstimate,
  ModelFilesResponse,
  ModelSearchResult,
  RecommendedResponse,
  RuntimeConfig,
  RuntimeLog,
  RuntimeStatus,
  SettingsResponse
} from "./shared/types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...init?.headers
    }
  });
  const payload = (await response.json()) as unknown;
  if (!response.ok) {
    const message =
      payload && typeof payload === "object" && "error" in payload && typeof payload.error === "string"
        ? payload.error
        : `Request failed: ${response.status}`;
    throw new Error(message);
  }
  return payload as T;
}

export const api = {
  config: () => request<RuntimeConfig>("/api/config"),
  pickPath: (mode: "folder" | "file", opts: { title?: string; gguf?: boolean } = {}) =>
    request<{ path: string | null }>("/api/dialog/pick", {
      method: "POST",
      body: JSON.stringify({ mode, ...opts })
    }),
  settings: () => request<SettingsResponse>("/api/settings"),
  saveSettings: (settings: AppSettings) =>
    request<SettingsResponse>("/api/settings", {
      method: "PUT",
      body: JSON.stringify(settings)
    }),
  profiles: () => request<LlamaProfile[]>("/api/profiles"),
  createProfile: (profile: LlamaProfile) =>
    request<LlamaProfile>("/api/profiles", {
      method: "POST",
      body: JSON.stringify(profile)
    }),
  updateProfile: (profile: LlamaProfile) =>
    request<LlamaProfile>(`/api/profiles/${encodeURIComponent(profile.id)}`, {
      method: "PUT",
      body: JSON.stringify(profile)
    }),
  deleteProfile: (id: string) =>
    request<{ ok: boolean }>(`/api/profiles/${encodeURIComponent(id)}`, {
      method: "DELETE"
    }),
  preview: (profile: LlamaProfile) =>
    request<CommandPreview>("/api/preview", {
      method: "POST",
      body: JSON.stringify({ profile })
    }),
  estimate: (profile: LlamaProfile) =>
    request<MemoryEstimate>("/api/estimate", {
      method: "POST",
      body: JSON.stringify({ profile })
    }),
  start: (profileId: string) =>
    request<RuntimeStatus>("/api/server/start", {
      method: "POST",
      body: JSON.stringify({ profileId })
    }),
  stop: () => request<RuntimeStatus>("/api/server/stop", { method: "POST" }),
  status: () => request<RuntimeStatus>("/api/server/status"),
  logs: () => request<RuntimeLog[]>("/api/server/logs"),
  benchmarks: () => request<BenchmarkRun[]>("/api/benchmarks"),
  benchmarkStatus: () => request<BenchmarkStatus>("/api/benchmarks/status"),
  benchmarkLogs: () => request<RuntimeLog[]>("/api/benchmarks/logs"),
  previewBenchmark: (profile: LlamaProfile, settings: BenchmarkSettings) =>
    request<BenchmarkCommandPreview>("/api/benchmarks/preview", {
      method: "POST",
      body: JSON.stringify({ profile, settings })
    }),
  startBenchmark: (profileId: string, settings: BenchmarkSettings) =>
    request<BenchmarkRun>("/api/benchmarks/start", {
      method: "POST",
      body: JSON.stringify({ profileId, settings })
    }),
  stopBenchmark: () => request<BenchmarkStatus>("/api/benchmarks/stop", { method: "POST" }),
  deleteBenchmark: (id: string) =>
    request<{ ok: boolean }>(`/api/benchmarks/${encodeURIComponent(id)}`, {
      method: "DELETE"
    }),
  llamaCppRelease: () => request<LlamaCppRelease>("/api/llamacpp/latest"),
  searchModels: (query: string, sort = "downloads") =>
    request<ModelSearchResult[]>(`/api/models/search?q=${encodeURIComponent(query)}&sort=${encodeURIComponent(sort)}`),
  recommendedModels: () => request<RecommendedResponse>("/api/models/recommended"),
  modelFiles: (id: string) => request<ModelFilesResponse>(`/api/models/files?id=${encodeURIComponent(id)}`),
  startDownload: (id: string, filename: string) =>
    request<DownloadStatus>("/api/models/download", {
      method: "POST",
      body: JSON.stringify({ id, filename })
    }),
  downloadStatus: () => request<DownloadStatus>("/api/models/download/status"),
  cancelDownload: () => request<DownloadStatus>("/api/models/download/cancel", { method: "POST" }),
  localModels: () => request<LocalModel[]>("/api/models/local"),
  deleteLocalModel: (name: string) =>
    request<{ ok: boolean }>(`/api/models/local/${encodeURIComponent(name)}`, { method: "DELETE" }),
  favorites: () => request<FavoriteModel[]>("/api/models/favorites"),
  addFavorite: (model: ModelSearchResult) =>
    request<FavoriteModel[]>("/api/models/favorites", {
      method: "POST",
      body: JSON.stringify({ model })
    }),
  removeFavorite: (id: string) =>
    request<FavoriteModel[]>(`/api/models/favorites/${encodeURIComponent(id)}`, { method: "DELETE" })
};
