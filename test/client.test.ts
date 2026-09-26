import { describe, expect, it } from "vitest";
import {
  AbortError,
  ALL_CAPABILITIES,
  BaseProvider,
  createLLM,
  LLMClient,
  LLMRouter,
  ProviderRegistry,
  registerBuiltinProviders,
  StreamAccumulator,
  type ChatRequest,
  type ChatResponse,
  type LLMProvider,
  type ProviderCapabilities,
  type StreamEvent,
} from "../src/index.js";
import { collect, hang, json, mockFetch, sse, sseBody } from "./helpers/mockFetch.js";

const user = [{ role: "user" as const, content: "hi" }];

/** A minimal custom provider implementing the public contract directly. */
class EchoProvider implements LLMProvider {
  readonly name = "echo";
  constructor(readonly defaultModel: string, private readonly caps: Partial<ProviderCapabilities> = {}) {}
  capabilities(): ProviderCapabilities {
    return { ...ALL_CAPABILITIES, ...this.caps };
  }
  async chat(request: ChatRequest): Promise<ChatResponse> {
    const last = request.messages.at(-1);
    const text = `echo:${typeof last?.content === "string" ? last.content : ""}`;
    return {
      provider: this.name,
      model: request.model ?? this.defaultModel,
      message: { role: "assistant", content: [{ type: "text", text }] },
      text,
      toolCalls: [],
      finishReason: "stop",
    };
  }
  async *stream(request: ChatRequest): AsyncGenerator<StreamEvent> {
    const res = await this.chat(request);
    yield { type: "start", model: res.model };
    for (const ch of res.text) yield { type: "text_delta", text: ch };
    yield { type: "finish", reason: "stop" };
  }
}

describe("registry", () => {
  it("registers custom providers without touching the client", async () => {
    const registry = registerBuiltinProviders(new ProviderRegistry());
    registry.register("echo", (c) => new EchoProvider(c.model));
    const client = new LLMClient({ provider: "echo", model: "e1" }, { registry });
    expect((await client.chat({ messages: user })).text).toBe("echo:hi");
    expect(registry.list()).toContain("openai");
    expect(ProviderRegistry.has("echo")).toBe(false); // isolated from the default registry
  });

  it("supports the static default registry", async () => {
    ProviderRegistry.register("echo-static", (c) => new EchoProvider(c.model));
    try {
      const client = createLLM({ provider: "echo-static", model: "m" });
      expect(client.providerName).toBe("echo");
      expect(ProviderRegistry.list()).toContain("anthropic");
    } finally {
      ProviderRegistry.unregister("echo-static");
    }
  });

  it("accepts a provider instance directly", async () => {
    const client = new LLMClient(new EchoProvider("x"));
    const events = await collect(client.stream({ messages: user }));
    expect(events.filter((e) => e.type === "text_delta")).toHaveLength("echo:hi".length);
  });
});

describe("multiple clients", () => {
  it("keeps credentials per instance", async () => {
    const m = mockFetch(json({ choices: [{ message: { content: "x" }, finish_reason: "stop" }] }));
    const a = new LLMClient({ provider: "openai", apiKey: "sk-key-A-00000000", model: "m", fetch: m.fetch });
    const b = new LLMClient({ provider: "openai", apiKey: "sk-key-B-00000000", model: "m", fetch: m.fetch });
    await a.chat({ messages: user });
    await b.chat({ messages: user });
    expect(m.requests.map((r) => r.headers.authorization)).toEqual(["Bearer sk-key-A-00000000", "Bearer sk-key-B-00000000"]);
  });
});

describe("middleware", () => {
  it("wraps chat in registration order and can rewrite the request", async () => {
    const order: string[] = [];
    const client = new LLMClient(new EchoProvider("m"));
    client.use(async (ctx, next) => {
      order.push(`outer:${ctx.provider}:${ctx.model}:${ctx.operation}`);
      const res = await next();
      order.push("outer-done");
      return res;
    });
    client.use({
      name: "rewrite",
      chat: async (ctx, next) => {
        order.push("inner");
        ctx.request = { ...ctx.request, messages: [{ role: "user", content: "rewritten" }] };
        ctx.state.set("seen", true);
        return next();
      },
    });
    const res = await client.chat({ messages: user });
    expect(res.text).toBe("echo:rewritten");
    expect(order).toEqual(["outer:echo:m:chat", "inner", "outer-done"]);
  });

  it("wraps streams via the object form", async () => {
    const client = new LLMClient(new EchoProvider("m"));
    let seen = 0;
    client.use({
      stream: async function* (_ctx, next) {
        for await (const e of next()) {
          if (e.type === "text_delta") seen++;
          yield e;
        }
      },
    });
    await client.stream({ messages: user }).finalResponse();
    expect(seen).toBe("echo:hi".length);
  });

  it("validates capabilities after middleware rewrites", async () => {
    const client = new LLMClient(new EchoProvider("m", { tools: false }));
    client.use((ctx, next) => {
      ctx.request = { ...ctx.request, tools: [{ type: "function", function: { name: "t", parameters: { type: "object" } } }] };
      return next();
    });
    await expect(client.chat({ messages: user })).rejects.toMatchObject({ name: "UnsupportedCapabilityError", capability: "tools" });
  });
});

describe("defaults and capabilities", () => {
  it("merges defaults under per-request values", async () => {
    const m = mockFetch(json({ choices: [{ message: { content: "x" }, finish_reason: "stop" }] }));
    const client = new LLMClient(
      { provider: "openai", apiKey: "sk-test-00000000", model: "gpt-5", fetch: m.fetch },
      { defaults: { temperature: 0.1, maxTokens: 10 } },
    );
    await client.chat({ messages: user, maxTokens: 20 });
    expect(m.lastBody()).toMatchObject({ temperature: 0.1, max_completion_tokens: 20 });
    expect(client.capabilities().reasoning).toBe(true);
    expect(client.supports("vision", "o3-mini")).toBe(false);
  });

  it("throws UnsupportedCapabilityError for streaming when not supported", async () => {
    const client = new LLMClient(new EchoProvider("m", { streaming: false }));
    await expect(collect(client.stream({ messages: user }))).rejects.toMatchObject({ capability: "streaming" });
  });

  it("listModels throws a clear error when the provider cannot list", async () => {
    await expect(new LLMClient(new EchoProvider("m")).listModels()).rejects.toMatchObject({ capability: "listModels" });
  });
});

describe("router", () => {
  it("dispatches by name without routing logic", async () => {
    const m = mockFetch(json({ choices: [{ message: { content: "from-openai" }, finish_reason: "stop" }] }));
    const router = new LLMRouter({
      providers: {
        fast: { provider: "openai", apiKey: "sk-test-00000000", model: "gpt-5-mini", fetch: m.fetch },
        echo: new EchoProvider("e"),
      },
    });
    const calls: string[] = [];
    router.use((ctx, next) => {
      calls.push(ctx.provider);
      return next();
    });
    expect((await router.chat("fast", { messages: user })).text).toBe("from-openai");
    expect((await router.chat("echo", { messages: user })).text).toBe("echo:hi");
    expect(calls).toEqual(["openai", "echo"]);
    expect(router.names()).toEqual(["fast", "echo"]);
    expect(() => router.get("missing" as "fast")).toThrow(/no provider named/);
  });
});

describe("stream cancellation", () => {
  it("aborts a stream mid-flight with AbortError", async () => {
    const controller = new AbortController();
    const m = mockFetch(sse(sseBody([{ choices: [{ delta: { content: "a" } }] }, { choices: [{ delta: { content: "b" } }] }]), [8]));
    const client = new LLMClient({ provider: "openai-compatible", baseURL: "http://x/v1", model: "m", fetch: m.fetch });
    const seen: string[] = [];
    const run = async () => {
      for await (const e of client.stream({ messages: user, signal: controller.signal })) {
        if (e.type === "start") controller.abort();
        seen.push(e.type);
      }
    };
    await expect(run()).rejects.toBeInstanceOf(AbortError);
  });

  it("aborts before the stream opens", async () => {
    const controller = new AbortController();
    const m = mockFetch(hang());
    const client = new LLMClient({ provider: "openai-compatible", baseURL: "http://x/v1", model: "m", fetch: m.fetch, retries: 0 });
    setTimeout(() => controller.abort(), 5);
    await expect(collect(client.stream({ messages: user, signal: controller.signal }))).rejects.toBeInstanceOf(AbortError);
  });

  it("breaking out of the loop stops consumption cleanly", async () => {
    const m = mockFetch(sse(sseBody(Array.from({ length: 50 }, () => ({ choices: [{ delta: { content: "x" } }] })), true)));
    const client = new LLMClient({ provider: "openai-compatible", baseURL: "http://x/v1", model: "m", fetch: m.fetch });
    let n = 0;
    for await (const e of client.stream({ messages: user })) {
      if (e.type === "text_delta" && ++n === 3) break;
    }
    expect(n).toBe(3);
  });
});

describe("StreamAccumulator", () => {
  it("builds a response from events", () => {
    const acc = new StreamAccumulator("p", "m");
    const events: StreamEvent[] = [
      { type: "start", id: "1" },
      { type: "reasoning_delta", text: "r" },
      { type: "text_delta", text: "a" },
      { type: "text_delta", text: "b" },
      { type: "tool_call_start", index: 0, id: "t", name: "f" },
      { type: "tool_call_delta", index: 0, argumentsDelta: '{"x"' },
      { type: "tool_call_delta", index: 0, argumentsDelta: ":1}" },
      { type: "tool_call_end", index: 0 },
      { type: "usage", usage: { inputTokens: 1, outputTokens: 2 } },
      { type: "finish", reason: "tool_call" },
    ];
    events.forEach((e) => acc.add(e));
    const res = acc.result();
    expect(res).toMatchObject({ id: "1", text: "ab", reasoning: "r", finishReason: "tool_call" });
    expect(res.usage).toEqual({ inputTokens: 1, outputTokens: 2, totalTokens: 3 });
    expect(res.toolCalls[0]!.function.arguments).toBe('{"x":1}');
  });
});

describe("BaseProvider for custom HTTP providers", () => {
  it("gives custom adapters retries, errors and provider options", async () => {
    class TinyProvider extends BaseProvider {
      constructor(fetch: typeof globalThis.fetch) {
        super({
          name: "tiny",
          model: "t1",
          baseURL: "https://tiny.example/api",
          headers: { authorization: "Bearer tiny-secret-000" },
          secrets: ["tiny-secret-000"],
          fetch,
          retries: { maxRetries: 1, initialDelayMs: 1 },
          defaultCapabilities: ALL_CAPABILITIES,
        });
      }
      async chat(request: ChatRequest): Promise<ChatResponse> {
        const { data } = await this.http.json({ path: "generate", body: { prompt: "x", ...this.providerOptionsFor(request).body }, ...this.transport(request) });
        const text = String((data as { output: string }).output);
        return { provider: this.name, model: this.modelFor(request), message: { role: "assistant", content: [{ type: "text", text }] }, text, toolCalls: [], finishReason: "stop" };
      }
      async *stream(): AsyncGenerator<StreamEvent> {
        yield { type: "finish", reason: "stop" };
      }
    }
    const m = mockFetch(json({ error: "busy" }, 503), json({ output: "done" }));
    const client = new LLMClient(new TinyProvider(m.fetch));
    const res = await client.chat({ messages: user, providerOptions: { tiny: { body: { flag: true } } } });
    expect(res.text).toBe("done");
    expect(m.requests).toHaveLength(2);
    expect(m.lastBody()).toEqual({ prompt: "x", flag: true });
  });
});
