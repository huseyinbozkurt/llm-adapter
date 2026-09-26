import { describe, expect, it } from "vitest";
import { ConfigurationError, LLMClient, UnsupportedCapabilityError, type LLMConfig } from "../../src/index.js";
import { toMistralToolCallId } from "../../src/providers/mistral/MistralAdapter.js";
import { collect, json, mockFetch, sse, sseBody } from "../helpers/mockFetch.js";

const ok = json({ id: "r", choices: [{ message: { content: "ok" }, finish_reason: "stop" }] });
const user = [{ role: "user" as const, content: "hi" }];

function make(config: LLMConfig, fetch: typeof globalThis.fetch) {
  return new LLMClient({ ...config, fetch });
}

describe("openai-compatible (generic)", () => {
  it("works without an API key and applies headers + query", async () => {
    const m = mockFetch(ok);
    const c = make(
      {
        provider: "openai-compatible",
        baseURL: "http://localhost:11434/v1/",
        model: "llama3.2",
        headers: { "x-team": "a" },
        query: { "api-version": "2025-01-01" },
      },
      m.fetch,
    );
    const res = await c.chat({ messages: user, maxTokens: 10 });
    expect(res.provider).toBe("openai-compatible");
    const req = m.requests[0]!;
    expect(req.url).toBe("http://localhost:11434/v1/chat/completions?api-version=2025-01-01");
    expect(req.headers.authorization).toBeUndefined();
    expect(req.headers["x-team"]).toBe("a");
    expect(m.lastBody()).toMatchObject({ model: "llama3.2", max_tokens: 10 });
  });

  it("validates configuration with helpful errors", () => {
    expect(() => new LLMClient({ provider: "openai-compatible", model: "x" } as unknown as LLMConfig)).toThrow(/baseURL/);
    expect(() => new LLMClient({ provider: "openai", model: "gpt" } as unknown as LLMConfig)).toThrow(ConfigurationError);
    expect(() => new LLMClient({ provider: "openai", apiKey: "k" } as unknown as LLMConfig)).toThrow(/model/);
    expect(() => new LLMClient({ provider: "nope", model: "x" })).toThrow(/Unknown provider "nope"/);
    expect(() => new LLMClient({ provider: "openai-compatible", baseURL: "not a url", model: "x" })).toThrow(/baseURL/);
  });

  it("honours capability overrides and quirks", async () => {
    const m = mockFetch(ok);
    const c = make(
      {
        provider: "openai-compatible",
        baseURL: "http://x/v1",
        model: "qwen",
        name: "vllm",
        capabilities: { vision: false, reasoningControl: true },
        quirks: { maxTokensField: "max_completion_tokens" },
      },
      m.fetch,
    );
    expect(c.supports("vision")).toBe(false);
    await expect(
      c.chat({ messages: [{ role: "user", content: [{ type: "image", url: "https://x/y.png" }] }] }),
    ).rejects.toBeInstanceOf(UnsupportedCapabilityError);
    await c.chat({ messages: user, maxTokens: 5, reasoning: { effort: "high" } });
    expect(m.lastBody()).toMatchObject({ max_completion_tokens: 5, reasoning_effort: "high" });
  });

  it("tolerates servers that omit tool call index/id and send object arguments", async () => {
    const m = mockFetch(
      sse(
        sseBody(
          [
            { choices: [{ delta: { tool_calls: [{ function: { name: "f", arguments: '{"a":1}' } }] } }] },
            { choices: [{ delta: {}, finish_reason: "stop" }] },
          ],
          true,
        ),
      ),
      json({ choices: [{ message: { tool_calls: [{ function: { name: "g", arguments: { b: 2 } } }] }, finish_reason: "stop" }] }),
    );
    const c = make({ provider: "openai-compatible", baseURL: "http://x/v1", model: "m" }, m.fetch);
    const streamed = await c.stream({ messages: user }).finalResponse();
    expect(streamed.toolCalls[0]).toMatchObject({ function: { name: "f", arguments: '{"a":1}' } });
    expect(streamed.toolCalls[0]!.id).toMatch(/^call_/);
    expect(streamed.finishReason).toBe("tool_call");

    const res = await c.chat({ messages: user });
    expect(res.toolCalls[0]!.function.arguments).toBe('{"b":2}');
  });
});

describe("DeepSeek", () => {
  const cfg = { provider: "deepseek" as const, apiKey: "sk-ds-test-00000000", model: "deepseek-reasoner" };

  it("parses reasoning_content and cache-hit usage", async () => {
    const m = mockFetch(
      json({
        choices: [{ message: { content: "4", reasoning_content: "2+2=4" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 10, completion_tokens: 3, total_tokens: 13, prompt_cache_hit_tokens: 8 },
      }),
    );
    const res = await make(cfg, m.fetch).chat({ messages: user });
    expect(m.requests[0]!.url).toBe("https://api.deepseek.com/chat/completions");
    expect(res.reasoning).toBe("2+2=4");
    expect(res.text).toBe("4");
    expect(res.usage).toMatchObject({ inputTokens: 10, cachedInputTokens: 8 });
  });

  it("replays reasoning_content and maps effort", async () => {
    const m = mockFetch(ok);
    await make(cfg, m.fetch).chat({
      messages: [
        { role: "user", content: "q" },
        {
          role: "assistant",
          content: [{ type: "reasoning", text: "because" }],
          toolCalls: [{ id: "c1", type: "function", function: { name: "f", arguments: "{}" } }],
        },
        { role: "tool", toolCallId: "c1", content: "r" },
      ],
      reasoning: { effort: "medium" },
    });
    const body = m.lastBody();
    expect((body.messages as unknown[])[1]).toMatchObject({ role: "assistant", reasoning_content: "because" });
    expect(body.reasoning_effort).toBe("high");

    await make(cfg, m.fetch).chat({ messages: user, reasoning: { effort: "none" } });
    expect(m.lastBody()).toMatchObject({ thinking: { type: "disabled" } });
  });

  it("requires explicit opt-in to emulate json_schema with JSON mode", async () => {
    const m = mockFetch(ok);
    const c = make(cfg, m.fetch);
    const schema = { type: "object", properties: { n: { type: "number" } } };
    await expect(
      c.chat({ messages: user, responseFormat: { type: "json_schema", name: "n", schema } }),
    ).rejects.toThrow(/fallback/);

    await c.chat({
      messages: [{ role: "system", content: "sys" }, ...user],
      responseFormat: { type: "json_schema", name: "n", schema, fallback: "json_mode" },
    });
    const body = m.lastBody();
    expect(body.response_format).toEqual({ type: "json_object" });
    const messages = body.messages as Array<{ role: string; content: string }>;
    expect(messages[0]!.content).toBe("sys");
    expect(messages[1]!.role).toBe("system");
    expect(messages[1]!.content).toContain(JSON.stringify(schema));
  });

  it("streams reasoning deltas before text", async () => {
    const m = mockFetch(
      sse(
        sseBody(
          [
            { choices: [{ delta: { reasoning_content: "think " } }] },
            { choices: [{ delta: { reasoning_content: "more" } }] },
            { choices: [{ delta: { content: "answer" }, finish_reason: "stop" }] },
          ],
          true,
        ),
      ),
    );
    const final = await make(cfg, m.fetch).stream({ messages: user }).finalResponse();
    expect(final.message.content).toEqual([
      { type: "reasoning", text: "think more" },
      { type: "text", text: "answer" },
    ]);
  });
});

describe("Groq", () => {
  it("maps reasoning per model family and reads x_groq usage", async () => {
    const m = mockFetch(
      ok,
      sse(sseBody([{ choices: [{ delta: { content: "x" }, finish_reason: "stop" }], x_groq: { usage: { prompt_tokens: 3, completion_tokens: 1 } } }], true)),
    );
    const c = make({ provider: "groq", apiKey: "gsk_test00000000", model: "openai/gpt-oss-120b" }, m.fetch);
    await c.chat({ messages: user, reasoning: { effort: "max", includeReasoning: true } });
    expect(m.requests[0]!.url).toBe("https://api.groq.com/openai/v1/chat/completions");
    expect(m.lastBody()).toMatchObject({ reasoning_effort: "high", include_reasoning: true });

    const events = await collect(c.stream({ messages: user }));
    expect(events).toContainEqual({ type: "usage", usage: { inputTokens: 3, outputTokens: 1, totalTokens: 4 } });

    const llama = make({ provider: "groq", apiKey: "gsk_test00000000", model: "llama-3.3-70b-versatile" }, m.fetch);
    expect(llama.supports("reasoningControl")).toBe(false);
    expect(llama.supports("vision")).toBe(false);
  });
});

describe("Mistral", () => {
  it("maps required tool choice, remaps foreign tool ids, and omits stream_options", async () => {
    const m = mockFetch(ok, sse(sseBody([{ choices: [{ delta: { content: "x" }, finish_reason: "stop" }] }], true)));
    const c = make({ provider: "mistral", apiKey: "mistral-test-key", model: "mistral-large-latest" }, m.fetch);
    await c.chat({
      messages: [
        { role: "user", content: "q" },
        { role: "assistant", content: "", toolCalls: [{ id: "toolu_01ABCDEF", type: "function", function: { name: "f", arguments: "{}" } }] },
        { role: "tool", toolCallId: "toolu_01ABCDEF", content: "r" },
      ],
      tools: [{ type: "function", function: { name: "f", parameters: { type: "object" } } }],
      toolChoice: "required",
    });
    const body = m.lastBody();
    expect(body.tool_choice).toBe("any");
    const msgs = body.messages as Array<Record<string, unknown>>;
    const mapped = (msgs[1]!.tool_calls as Array<{ id: string }>)[0]!.id;
    expect(mapped).toMatch(/^[A-Za-z0-9]{9}$/);
    expect(msgs[2]!.tool_call_id).toBe(mapped);
    expect(toMistralToolCallId("abcDEF123")).toBe("abcDEF123");

    await collect(c.stream({ messages: user }));
    expect(m.lastBody().stream_options).toBeUndefined();
  });

  it("parses Magistral thinking chunks", async () => {
    const m = mockFetch(
      json({
        choices: [
          {
            message: {
              content: [
                { type: "thinking", thinking: [{ type: "text", text: "hmm" }] },
                { type: "text", text: "answer" },
              ],
            },
            finish_reason: "stop",
          },
        ],
      }),
    );
    const res = await make({ provider: "mistral", apiKey: "k-test-0000", model: "magistral-medium-latest" }, m.fetch).chat({ messages: user });
    expect(res.reasoning).toBe("hmm");
    expect(res.text).toBe("answer");
  });
});

describe("xAI", () => {
  it("applies model-dependent capability rules", async () => {
    const m = mockFetch(ok);
    const grok4 = make({ provider: "xai", apiKey: "xai-test-00000000", model: "grok-4" }, m.fetch);
    expect(grok4.supports("reasoning")).toBe(true);
    expect(grok4.supports("reasoningControl")).toBe(false);
    await expect(grok4.chat({ messages: user, stop: ["x"] })).rejects.toBeInstanceOf(UnsupportedCapabilityError);
    await expect(grok4.chat({ messages: user, reasoning: { effort: "low" } })).rejects.toBeInstanceOf(UnsupportedCapabilityError);

    const mini = make({ provider: "xai", apiKey: "xai-test-00000000", model: "grok-3-mini" }, m.fetch);
    await mini.chat({ messages: user, reasoning: { effort: "max" } });
    expect(m.requests[0]!.url).toBe("https://api.x.ai/v1/chat/completions");
    expect(m.lastBody().reasoning_effort).toBe("xhigh");
  });
});

describe("OpenRouter", () => {
  it("sends attribution headers and the unified reasoning object", async () => {
    const m = mockFetch(ok);
    const c = make(
      { provider: "openrouter", apiKey: "sk-or-test-000000", model: "anthropic/claude-sonnet-5", siteURL: "https://app.example", appName: "Example" },
      m.fetch,
    );
    await c.chat({
      messages: user,
      reasoning: { maxTokens: 2000, includeReasoning: false },
      providerOptions: { openrouter: { body: { provider: { order: ["anthropic"] } } } },
    });
    const req = m.requests[0]!;
    expect(req.headers["http-referer"]).toBe("https://app.example");
    expect(req.headers["x-title"]).toBe("Example");
    expect(m.lastBody()).toMatchObject({ reasoning: { max_tokens: 2000, exclude: true }, provider: { order: ["anthropic"] } });
  });

  it("lists models", async () => {
    const m = mockFetch(json({ data: [{ id: "openai/gpt-5", name: "GPT-5", context_length: 400000, created: 1700000000 }] }));
    const models = await make({ provider: "openrouter", apiKey: "sk-or-test-000000", model: "x" }, m.fetch).listModels();
    expect(models[0]).toMatchObject({ id: "openai/gpt-5", provider: "openrouter", displayName: "GPT-5", contextWindow: 400000 });
    expect(models[0]!.createdAt).toBeInstanceOf(Date);
  });

  it("surfaces in-band stream errors", async () => {
    const m = mockFetch(sse(sseBody([{ choices: [{ delta: { content: "par" } }] }, { error: { code: "rate_limit_error", message: "upstream limited" } }])));
    const c = make({ provider: "openrouter", apiKey: "sk-or-test-000000", model: "x" }, m.fetch);
    await expect(collect(c.stream({ messages: user }))).rejects.toMatchObject({ name: "RateLimitError", provider: "openrouter" });
  });
});
