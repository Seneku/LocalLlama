import { describe, expect, test } from "bun:test";

import { buildConnectionInfo, deriveAlias, PLACEHOLDER_API_KEY, TOOLS } from "../src/shared/connect";

function status(endpoint: string | null, state = "running") {
  return { endpoint, state } as never;
}
function profile(partial: { modelAlias?: string; modelPath?: string; host?: string }) {
  return { modelAlias: "", modelPath: "", host: "127.0.0.1", ...partial } as never;
}

describe("deriveAlias", () => {
  test("strips directory and .gguf", () => {
    expect(deriveAlias("D:\\models\\Qwen3-30B-A3B-Q4_K_M.gguf")).toBe("Qwen3-30B-A3B-Q4_K_M");
    expect(deriveAlias("/home/u/gemma-3-12b-it.GGUF")).toBe("gemma-3-12b-it");
    expect(deriveAlias("")).toBe("");
    expect(deriveAlias(undefined)).toBe("");
  });
});

describe("buildConnectionInfo", () => {
  test("appends /v1 to the origin exactly once", () => {
    expect(buildConnectionInfo(status("http://127.0.0.1:8080"), null, profile({})).baseUrl).toBe(
      "http://127.0.0.1:8080/v1"
    );
    // trailing slash is trimmed, not doubled
    expect(buildConnectionInfo(status("http://127.0.0.1:9000/"), null, profile({})).baseUrl).toBe(
      "http://127.0.0.1:9000/v1"
    );
    // already-suffixed origin is left alone
    expect(buildConnectionInfo(status("http://127.0.0.1:8080/v1"), null, profile({})).baseUrl).toBe(
      "http://127.0.0.1:8080/v1"
    );
  });

  test("falls back to preview endpoint, then the default origin", () => {
    expect(buildConnectionInfo(status(null), { endpoint: "http://127.0.0.1:5000" }, profile({})).baseUrl).toBe(
      "http://127.0.0.1:5000/v1"
    );
    expect(buildConnectionInfo(status(null), null, null).baseUrl).toBe("http://127.0.0.1:8080/v1");
  });

  test("model id: alias when set, else filename, else fallback", () => {
    const withAlias = buildConnectionInfo(status(null), null, profile({ modelAlias: "  my-model  " }));
    expect(withAlias.modelId).toBe("my-model");
    expect(withAlias.aliasSet).toBe(true);

    const fromPath = buildConnectionInfo(status(null), null, profile({ modelPath: "x/Llama-3.1-8B-Q4.gguf" }));
    expect(fromPath.modelId).toBe("Llama-3.1-8B-Q4");
    expect(fromPath.aliasSet).toBe(false);
    expect(fromPath.suggestedAlias).toBe("Llama-3.1-8B-Q4");

    expect(buildConnectionInfo(status(null), null, null).modelId).toBe("local-model");
  });

  test("running and loopbackOnly flags", () => {
    expect(buildConnectionInfo(status(null, "running"), null, profile({})).running).toBe(true);
    expect(buildConnectionInfo(status(null, "stopped"), null, profile({})).running).toBe(false);
    expect(buildConnectionInfo(status(null), null, profile({ host: "127.0.0.1" })).loopbackOnly).toBe(true);
    expect(buildConnectionInfo(status(null), null, profile({ host: "localhost" })).loopbackOnly).toBe(true);
    expect(buildConnectionInfo(status(null), null, profile({ host: "0.0.0.0" })).loopbackOnly).toBe(false);
    expect(buildConnectionInfo(status(null), null, profile({ host: "192.168.1.5" })).loopbackOnly).toBe(false);
  });

  test("api key is the shared placeholder", () => {
    expect(buildConnectionInfo(status(null), null, null).apiKey).toBe(PLACEHOLDER_API_KEY);
  });
});

describe("tool snippets", () => {
  const info = buildConnectionInfo(status("http://127.0.0.1:8080"), null, profile({ modelAlias: "my-model" }));
  const byId = Object.fromEntries(TOOLS.map((tool) => [tool.id, tool]));

  test("OpenCode emits valid JSON referencing the model and base URL", () => {
    const { code, filename } = byId.opencode.build(info);
    expect(filename).toBe("opencode.json");
    const parsed = JSON.parse(code);
    expect(parsed.model).toBe("localllama/my-model");
    expect(parsed.provider.localllama.options.baseURL).toBe("http://127.0.0.1:8080/v1");
    expect(parsed.provider.localllama.models["my-model"]).toBeTruthy();
  });

  test("Continue snippet carries provider/apiBase/model", () => {
    const { code } = byId.continue.build(info);
    expect(code).toContain("provider: openai");
    expect(code).toContain('apiBase: "http://127.0.0.1:8080/v1"');
    expect(code).toContain('model: "my-model"');
  });

  test("Aider uses the openai/<model> form and env vars", () => {
    const { code } = byId.aider.build(info);
    expect(code).toContain("aider --model openai/my-model");
    expect(code).toContain('setx OPENAI_API_BASE "http://127.0.0.1:8080/v1"');
  });

  test("Cline field block includes base URL and model id", () => {
    const { code } = byId.cline.build(info);
    expect(code).toContain("http://127.0.0.1:8080/v1");
    expect(code).toContain("my-model");
  });

  test("Generic curl targets <baseUrl>/chat/completions", () => {
    const { code } = byId.generic.build(info);
    expect(code).toContain("http://127.0.0.1:8080/v1/chat/completions");
    expect(code).toContain(`Bearer ${PLACEHOLDER_API_KEY}`);
  });

  test("every tool substitutes the real base URL", () => {
    for (const tool of TOOLS) {
      expect(tool.build(info).code).toContain("http://127.0.0.1:8080/v1");
    }
  });
});
