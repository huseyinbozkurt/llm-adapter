import { describe, expect, it } from "vitest";
import { LLMClient, UnsupportedCapabilityError, type ChatMessage, type StreamEvent, type ToolDefinition } from "../../src/index.js";
import { collect, json, mockFetch, sse, sseBody } from "../helpers/mockFetch.js";

const weatherTool: ToolDefinition = {
  type: "function",
  function: {
    name: "get_weather",
    description: "Weather by city",
    parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
  },
};

function client(fetch: typeof globalThis.fetch, model = "gpt-5") {
  return new LLMClient({ provider: "openai", apiKey: "sk-test-key-000000", model, fetch });
}

describe("OpenAI adapter: request translation", () => {
  it("translates messages, tools, images and options", async () => {
    const m = mockFetch(json({ id: "r", model: "gpt-5", choices: [{ message: { content: "hi" }, finish_reason: "stop" }] }));
    const messages: ChatMessage[] = [
      { role: "system", content: "Be brief." },
      {
        role: "user",
        name: "alice",
        content: [
          { type: "text", text: "What is this?" },
          { type: "image", base64: "AAAA", mimeType: "image/png", detail: "low" },
          { type: "image", url: "https://example.com/a.jpg" },
        ],
      },
      {
        role: "assistant",
        content: [
          { type: "reasoning", text: "thinking..." },
          { type: "text", text: "" },
        ],
        toolCalls: [{ id: "call_1", type: "function", function: { name: "get_weather", arguments: '{"city":"Paris"}' } }],
      },
      { role: "tool", toolCallId: "call_1", content: '{"temp":20}' },
    ];
    await client(m.fetch).chat({
      messages,
      tools: [weatherTool],
      toolChoice: { type: "function", name: "get_weather" },
      parallelToolCalls: false,
      maxTokens: 100,
      temperature: 0.2,
      stop: ["END"],
      reasoning: { effort: "low" },
      metadata: { traceId: "t1" },
    });

    const req = m.requests[0]!;
    expect(req.url).toBe("https://api.openai.com/v1/chat/completions");
    expect(req.headers.authorization).toBe("Bearer sk-test-key-000000");
    expect(req.body).toEqual({
      model: "gpt-5",
      messages: [
        { role: "system", content: "Be brief." },
        {
          role: "user",
          name: "alice",
          content: [
            { type: "text", text: "What is this?" },
            { type: "image_url", image_url: { url: "data:image/png;base64,AAAA", detail: "low" } },
            { type: "image_url", image_url: { url: "https://example.com/a.jpg" } },
          ],
        },
        {
          role: "assistant",
          content: null,
          tool_calls: [{ id: "call_1", type: "function", function: { name: "get_weather", arguments: '{"city":"Paris"}' } }],
        },
        { role: "tool", tool_call_id: "call_1", content: '{"temp":20}' },
      ],
      temperature: 0.2,
      max_completion_tokens: 100,
      stop: ["END"],
      tools: [
        {
          type: "function",
          function: {
            name: "get_weather",
            description: "Weather by city",
            parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
          },
        },
      ],
      tool_choice: { type: "function", function: { name: "get_weather" } },
      parallel_tool_calls: false,
      reasoning_effort: "low",
    });
    // metadata is application-only and never sent.
    expect(JSON.stringify(req.body)).not.toContain("traceId");
  });

  it("maps json_schema structured output natively", async () => {
    const m = mockFetch(json({ choices: [{ message: { content: '{"name":"Ada"}' }, finish_reason: "stop" }] }));
    const res = await client(m.fetch).chat({
      messages: [{ role: "user", content: "who?" }],
      responseFormat: { type: "json_schema", name: "person", schema: { type: "object" }, strict: true },
    });
    expect(m.lastBody().response_format).toEqual({
      type: "json_schema",
      json_schema: { name: "person", schema: { type: "object" }, strict: true },
    });
    expect(JSON.parse(res.text)).toEqual({ name: "Ada" });
  });

  it("merges provider options into the body and headers", async () => {
    const m = mockFetch(json({ choices: [{ message: { content: "x" }, finish_reason: "stop" }] }));
    await client(m.fetch)
      .withProviderOptions({ openai: { body: { seed: 1 } } })
      .chat({
        messages: [{ role: "user", content: "x" }],
        providerOptions: { openai: { body: { service_tier: "flex" }, headers: { "x-extra": "1" } }, anthropic: { body: { top_k: 3 } } },
      });
    expect(m.lastBody()).toMatchObject({ seed: 1, service_tier: "flex" });
    expect(m.lastBody().top_k).toBeUndefined();
    expect(m.requests[0]!.headers["x-extra"]).toBe("1");
  });

  it("rejects reasoning controls on non-reasoning models before sending", async () => {
    const m = mockFetch(json({}));
    await expect(
      client(m.fetch, "gpt-4.1").chat({ messages: [{ role: "user", content: "x" }], reasoning: { effort: "high" } }),
    ).rejects.toBeInstanceOf(UnsupportedCapabilityError);
    expect(m.requests).toHaveLength(0);
  });
});

describe("OpenAI adapter: response normalisation", () => {
  it("normalises tool calls, finish reason and usage", async () => {
    const m = mockFetch(
      json({
        id: "chatcmpl-1",
        model: "gpt-5-2025",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: null,
              tool_calls: [{ id: "call_a", type: "function", function: { name: "get_weather", arguments: '{"city":"Toronto"}' } }],
            },
            finish_reason: "tool_calls",
          },
        ],
        usage: {
          prompt_tokens: 100,
          completion_tokens: 50,
          total_tokens: 150,
          prompt_tokens_details: { cached_tokens: 40 },
          completion_tokens_details: { reasoning_tokens: 30 },
        },
      }),
    );
    const res = await client(m.fetch).chat({ messages: [{ role: "user", content: "weather?" }], tools: [weatherTool] });
    expect(res.provider).toBe("openai");
    expect(res.model).toBe("gpt-5-2025");
    expect(res.finishReason).toBe("tool_call");
    expect(res.rawFinishReason).toBe("tool_calls");
    expect(res.toolCalls).toEqual([
      { id: "call_a", type: "function", function: { name: "get_weather", arguments: '{"city":"Toronto"}' } },
    ]);
    expect(res.message.toolCalls).toEqual(res.toolCalls);
    expect(res.usage).toEqual({ inputTokens: 100, outputTokens: 50, totalTokens: 150, cachedInputTokens: 40, reasoningTokens: 30 });
    expect(res.reasoning).toBeUndefined(); // OpenAI exposes no reasoning text; none is invented
    expect(res.raw).toBeDefined();
  });

  it.each([
    ["stop", "stop"],
    ["length", "length"],
    ["content_filter", "content_filter"],
    ["something_new", "unknown"],
  ])("maps finish_reason %s → %s", async (raw, expected) => {
    const m = mockFetch(json({ choices: [{ message: { content: "x" }, finish_reason: raw }] }));
    const res = await client(m.fetch).chat({ messages: [{ role: "user", content: "x" }] });
    expect(res.finishReason).toBe(expected);
  });

  it("reports structured-output refusals as content_filter", async () => {
    const m = mockFetch(json({ choices: [{ message: { content: null, refusal: "I can't help with that." }, finish_reason: "stop" }] }));
    const res = await client(m.fetch).chat({ messages: [{ role: "user", content: "x" }] });
    expect(res.finishReason).toBe("content_filter");
    expect(res.text).toBe("I can't help with that.");
  });
});

describe("OpenAI adapter: streaming", () => {
  const chunks = [
    { id: "c1", model: "gpt-5", choices: [{ index: 0, delta: { role: "assistant", content: "" } }] },
    { id: "c1", choices: [{ index: 0, delta: { content: "Checking " } }] },
    { id: "c1", choices: [{ index: 0, delta: { content: "weather." } }] },
    {
      id: "c1",
      choices: [
        { index: 0, delta: { tool_calls: [{ index: 0, id: "call_x", type: "function", function: { name: "get_weather", arguments: "" } }] } },
      ],
    },
    { id: "c1", choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '{"city":' } }] } }] },
    { id: "c1", choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '"Toronto"}' } }] } }] },
    {
      id: "c1",
      choices: [
        { index: 0, delta: { tool_calls: [{ index: 1, id: "call_y", function: { name: "get_weather", arguments: '{"city":"Oslo"}' } }] } },
      ],
    },
    { id: "c1", choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] },
    { id: "c1", choices: [], usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 } },
  ];

  it("reconstructs fragmented tool calls and emits normalised events", async () => {
    const m = mockFetch(sse(sseBody(chunks, true)));
    const stream = client(m.fetch).stream({ messages: [{ role: "user", content: "weather?" }], tools: [weatherTool] });
    const events = await collect(stream);

    expect(m.lastBody()).toMatchObject({ stream: true, stream_options: { include_usage: true } });
    expect(events[0]).toEqual({ type: "start", id: "c1", model: "gpt-5" });
    expect(events.filter((e) => e.type === "text_delta").map((e) => (e as { text: string }).text).join("")).toBe("Checking weather.");
    const toolEvents = events.filter((e) => e.type.startsWith("tool_call"));
    expect(toolEvents).toEqual<StreamEvent[]>([
      { type: "tool_call_start", index: 0, id: "call_x", name: "get_weather" },
      { type: "tool_call_delta", index: 0, argumentsDelta: '{"city":' },
      { type: "tool_call_delta", index: 0, argumentsDelta: '"Toronto"}' },
      { type: "tool_call_start", index: 1, id: "call_y", name: "get_weather" },
      { type: "tool_call_delta", index: 1, argumentsDelta: '{"city":"Oslo"}' },
      { type: "tool_call_end", index: 0 },
      { type: "tool_call_end", index: 1 },
    ]);
    expect(events.at(-2)).toEqual({ type: "usage", usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 } });
    expect(events.at(-1)).toEqual({ type: "finish", reason: "tool_call", rawReason: "tool_calls" });

    const final = await stream.finalResponse();
    expect(final.text).toBe("Checking weather.");
    expect(final.toolCalls.map((c) => JSON.parse(c.function.arguments))).toEqual([{ city: "Toronto" }, { city: "Oslo" }]);
    expect(final.finishReason).toBe("tool_call");
  });

  it("finalResponse() drains the stream when not iterated", async () => {
    const m = mockFetch(sse(sseBody(chunks.slice(0, 3), true)));
    const res = await client(m.fetch).stream({ messages: [{ role: "user", content: "x" }] }).finalResponse();
    expect(res.text).toBe("Checking weather.");
  });

  it("textStream() yields only text", async () => {
    const m = mockFetch(sse(sseBody(chunks.slice(0, 3), true)));
    const parts = await collect(client(m.fetch).stream({ messages: [{ role: "user", content: "x" }] }).textStream());
    expect(parts.join("")).toBe("Checking weather.");
  });
});
