// Turns the live server status + current profile into the connection details and
// ready-to-paste configuration blocks an external tool (OpenCode, Cline/Roo,
// Continue, Aider, or a plain OpenAI client) needs to talk to the running
// llama-server. Pure and DOM-free so it can be unit-tested in isolation; the
// ConnectToolsModal is just a thin renderer over this.
import type { CommandPreview, LlamaProfile, RuntimeStatus } from "./types";

const DEFAULT_ORIGIN = "http://127.0.0.1:8080";

// The server runs unauthenticated, but many tools require a non-empty API-key
// field. Any string works; we hand out a memorable placeholder.
export const PLACEHOLDER_API_KEY = "local";

export type ToolId = "opencode" | "cline" | "continue" | "aider" | "generic";

export interface ConnectionInfo {
  origin: string;
  baseUrl: string;
  modelId: string;
  aliasSet: boolean;
  suggestedAlias: string;
  apiKey: string;
  running: boolean;
  loopbackOnly: boolean;
}

export interface ToolSnippet {
  lang: string;
  filename?: string;
  code: string;
}

export interface ToolDef {
  id: ToolId;
  label: string;
  docsUrl: string;
  steps: string[];
  build(info: ConnectionInfo): ToolSnippet;
}

// Structural (not the full interfaces) so tests can pass small literals.
type StatusLike = Pick<RuntimeStatus, "endpoint" | "state"> | null;
type PreviewLike = Pick<CommandPreview, "endpoint"> | null;
type ProfileLike = Pick<LlamaProfile, "modelAlias" | "modelPath" | "host"> | null;

/** A clean model id from a GGUF path: basename minus directory and ".gguf". */
export function deriveAlias(modelPath: string | undefined): string {
  const base = (modelPath ?? "").split(/[\\/]/).pop() ?? "";
  return base.replace(/\.gguf$/iu, "").trim();
}

function isLoopback(host: string): boolean {
  const h = host.trim().toLowerCase();
  // Empty falls back to the 127.0.0.1 default; 0.0.0.0 binds all interfaces
  // (reachable from the LAN) so it is NOT loopback-only.
  return h === "" || h === "127.0.0.1" || h === "localhost" || h === "::1";
}

export function buildConnectionInfo(
  status: StatusLike,
  preview: PreviewLike,
  draft: ProfileLike
): ConnectionInfo {
  const origin = (status?.endpoint ?? preview?.endpoint ?? DEFAULT_ORIGIN).replace(/\/+$/u, "");
  const baseUrl = /\/v1$/u.test(origin) ? origin : `${origin}/v1`;
  const suggestedAlias = deriveAlias(draft?.modelPath);
  const alias = (draft?.modelAlias ?? "").trim();
  return {
    origin,
    baseUrl,
    modelId: alias || suggestedAlias || "local-model",
    aliasSet: alias.length > 0,
    suggestedAlias,
    apiKey: PLACEHOLDER_API_KEY,
    running: status?.state === "running",
    loopbackOnly: isLoopback(draft?.host ?? "")
  };
}

export const TOOLS: ToolDef[] = [
  {
    id: "opencode",
    label: "OpenCode",
    docsUrl: "https://opencode.ai/docs/providers/",
    steps: [
      "Create or edit opencode.json in your project (or ~/.config/opencode/).",
      "Add the provider block below, then pick the model with /models in OpenCode."
    ],
    build(info) {
      const config = {
        $schema: "https://opencode.ai/config.json",
        provider: {
          localllama: {
            npm: "@ai-sdk/openai-compatible",
            name: "LocalLlama",
            options: { baseURL: info.baseUrl },
            models: { [info.modelId]: { name: info.modelId } }
          }
        },
        model: `localllama/${info.modelId}`
      };
      return { lang: "json", filename: "opencode.json", code: JSON.stringify(config, null, 2) };
    }
  },
  {
    id: "cline",
    label: "Cline / Roo Code",
    docsUrl: "https://docs.cline.bot/provider-config/openai-compatible",
    steps: [
      "Open the extension settings and choose the \"OpenAI Compatible\" provider.",
      "Paste the values below. The API key can be any non-empty string."
    ],
    build(info) {
      const code = [
        "Provider:  OpenAI Compatible",
        `Base URL:  ${info.baseUrl}`,
        `API Key:   ${info.apiKey}`,
        `Model ID:  ${info.modelId}`
      ].join("\n");
      return { lang: "text", code };
    }
  },
  {
    id: "continue",
    label: "Continue.dev",
    docsUrl: "https://docs.continue.dev/customize/model-providers/openai",
    steps: [
      "Open ~/.continue/config.yaml.",
      "Add the model entry below under the top-level models: list."
    ],
    build(info) {
      const code = [
        "models:",
        "  - name: LocalLlama",
        "    provider: openai",
        `    model: "${info.modelId}"`,
        `    apiBase: "${info.baseUrl}"`,
        `    apiKey: "${info.apiKey}"`
      ].join("\n");
      return { lang: "yaml", filename: "config.yaml", code };
    }
  },
  {
    id: "aider",
    label: "Aider",
    docsUrl: "https://aider.chat/docs/llms/openai-compat.html",
    steps: [
      "Set the two environment variables (setx persists them for new terminals).",
      "Open a fresh terminal, then launch aider with the model below."
    ],
    build(info) {
      const code = [
        `setx OPENAI_API_BASE "${info.baseUrl}"`,
        `setx OPENAI_API_KEY "${info.apiKey}"`,
        ":: open a NEW terminal so the variables take effect, then:",
        `aider --model openai/${info.modelId}`
      ].join("\n");
      return { lang: "bash", code };
    }
  },
  {
    id: "generic",
    label: "Generic / curl",
    docsUrl: "https://platform.openai.com/docs/api-reference/chat",
    steps: [
      "Any OpenAI-compatible client works: point base_url at the URL below and send any api key.",
      "Quick smoke test with curl:"
    ],
    build(info) {
      const body = JSON.stringify({
        model: info.modelId,
        messages: [{ role: "user", content: "Hello" }]
      }).replaceAll('"', '\\"');
      const code =
        `curl ${info.baseUrl}/chat/completions ` +
        `-H "Content-Type: application/json" ` +
        `-H "Authorization: Bearer ${info.apiKey}" ` +
        `-d "${body}"`;
      return { lang: "bash", code };
    }
  }
];
