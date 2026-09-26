import { describe, expect, it } from "vitest";
import { LLMClient, UnsupportedCapabilityError, type StreamEvent, type ToolDefinition } from "../../src/index.js";
import { collect, json, mockFetch, sse, sseBody } from "../helpers/mockFetch.js";

const tool: ToolDefinition = {
  type: "function",
  function: { name: "get_weather", parameters: { type: "object", properties: { city: { type: "string" } } } },
};

function client(fetch: typeof globalThis.fetch, model = "claude-opus-5") {
  return new LLMClient({ provider: "anthropic", apiKey: "sk-ant-test-000000000", model, fetch });
}

const message = (content: unknown[], stop = "end_turn", usage: Record<string, number> = { input_tokens: 10, output_tokens: 5 }) =>
  json({ id: "msg_1", type: "message", role: "assistant", model: "claude-opus-5", content, stop_reason: stop, usage });

describe("Anthropic adapter: request translation", () => {
  it("hoists system messages, merges tool results, and replays signed thinking", async () => {
    const m = mockFetch(message([{ type: "text", text: "ok" }]));
    await client(m.fetch).chat({
      messages: [
        { role: "system", content: "You are helpful." },
        { role: "user", content: [{ type: "text", text: "Weather?" }, { type: "image", url: "data:image/jpeg;base64,/9j/" }] },
        {
          role: "assistant",
          content: [
            { type: "reasoning", text: "Need weather.", providerData: { anthropic: { signature: "sig123" } } },
            { type: "reasoning", text: "from another provider" },
            { type: "text", text: "Let me check." },
          ],
          toolCalls: [
            { id: "toolu_1", type: "function", function: { name: "get_weather", arguments: '{"city":"Paris"}' } },
            { id: "toolu_2", type: "function", function: { name: "get_weather", arguments: '{"city":"Rome"}' } },
          ],
        },
        { role: "tool", toolCallId: "toolu_1", content: "20C" },
        { role: "tool", toolCallId: "toolu_2", content: "boom", isError: true },
        { role: "user", content: "Thanks" },
      ],
      tools: [tool],
      toolChoice: "required",
      parallelToolCalls: false,
      stop: ["###"],
    });

    const req = m.requests[0]!;
    expect(req.url).toBe("https://api.anthropic.com/v1/messages");
    expect(req.headers["x-api-key"]).toBe("sk-ant-test-000000000");
    expect(req.headers["anthropic-version"]).toBe("2023-06-01");
    expect(req.headers.authorization).toBeUndefined();
    expect(req.body).toEqual({
      model: "claude-opus-5",
      system: [{ type: "text", text: "You are helpful." }],
      max_tokens: 16000,
      stop_sequences: ["###"],
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Weather?" },
            { type: "image", source: { type: "base64", media_type: "image/jpeg", data: "/9j/" } },
          ],
        },
        {
          role: "assistant",
          content: [
            { type: "thinking", thinking: "Need weather.", signature: "sig123" },
            { type: "text", text: "Let me check." },
            { type: "tool_use", id: "toolu_1", name: "get_weather", input: { city: "Paris" } },
            { type: "tool_use", id: "toolu_2", name: "get_weather", input: { city: "Rome" } },
          ],
        },
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "toolu_1", content: "20C" },
            { type: "tool_result", tool_use_id: "toolu_2", content: "boom", is_error: true },
            { type: "text", text: "Thanks" },
          ],
        },
      ],
      tools: [{ name: "get_weather", input_schema: { type: "object", properties: { city: { type: "string" } } } }],
      tool_choice: { type: "any", disable_parallel_tool_use: true },
    });
  });

  it("maps reasoning and structured output into thinking / output_config", async () => {
    const m = mockFetch(message([{ type: "text", text: "{}" }]));
    await client(m.fetch).chat({
      messages: [{ role: "user", content: "x" }],
      reasoning: { effort: "minimal", includeReasoning: true },
      responseFormat: { type: "json_schema", name: "p", schema: { type: "object", additionalProperties: false } },
    });
    expect(m.lastBody()).toMatchObject({
      thinking: { type: "adaptive", display: "summarized" },
      output_config: { effort: "low", format: { type: "json_schema", schema: { type: "object", additionalProperties: false } } },
    });

    const legacy = mockFetch(message([{ type: "text", text: "x" }]));
    await client(legacy.fetch, "claude-haiku-4-5").chat({ messages: [{ role: "user", content: "x" }], reasoning: { maxTokens: 20000 } });
    expect(legacy.lastBody()).toMatchObject({ thinking: { type: "enabled", budget_tokens: 20000 }, max_tokens: 24096 });
  });

  it("rejects JSON mode (Anthropic has no schema-less JSON mode)", async () => {
    const m = mockFetch(message([]));
    await expect(
      client(m.fetch).chat({ messages: [{ role: "user", content: "x" }], responseFormat: { type: "json_object" } }),
    ).rejects.toBeInstanceOf(UnsupportedCapabilityError);
  });
});

describe("Anthropic adapter: response normalisation", () => {
  it("normalises thinking, tool_use, stop reason and cache usage", async () => {
    const m = mockFetch(
      message(
        [
          { type: "thinking", thinking: "Let me think", signature: "sigA" },
          { type: "redacted_thinking", data: "opaque" },
          { type: "text", text: "Checking." },
          { type: "tool_use", id: "toolu_9", name: "get_weather", input: { city: "Toronto" } },
        ],
        "tool_use",
        { input_tokens: 10, cache_read_input_tokens: 90, cache_creation_input_tokens: 5, output_tokens: 20 },
      ),
    );
    const res = await client(m.fetch).chat({ messages: [{ role: "user", content: "x" }], tools: [tool] });
    expect(res.finishReason).toBe("tool_call");
    expect(res.reasoning).toBe("Let me think");
    expect(res.text).toBe("Checking.");
    expect(res.message.content).toEqual([
      { type: "reasoning", text: "Let me think", providerData: { anthropic: { signature: "sigA" } } },
      { type: "reasoning", text: "", providerData: { anthropic: { redactedData: "opaque" } } },
      { type: "text", text: "Checking." },
    ]);
    expect(res.toolCalls).toEqual([
      { id: "toolu_9", type: "function", function: { name: "get_weather", arguments: '{"city":"Toronto"}' } },
    ]);
    expect(res.usage).toEqual({
      inputTokens: 105,
      outputTokens: 20,
      totalTokens: 125,
      cachedInputTokens: 90,
      cacheWriteInputTokens: 5,
    });
  });

  it.each([
    ["end_turn", "stop"],
    ["stop_sequence", "stop"],
    ["max_tokens", "length"],
    ["refusal", "content_filter"],
    ["pause_turn", "unknown"],
  ])("maps stop_reason %s → %s", async (raw, expected) => {
    const m = mockFetch(message([{ type: "text", text: "x" }], raw));
    expect((await client(m.fetch).chat({ messages: [{ role: "user", content: "x" }] })).finishReason).toBe(expected);
  });
});

describe("Anthropic adapter: streaming", () => {
  const events = [
    { event: "message_start", data: { type: "message_start", message: { id: "msg_s", model: "claude-opus-5", usage: { input_tokens: 12, output_tokens: 1 } } } },
    { event: "content_block_start", data: { type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "" } } },
    { event: "content_block_delta", data: { type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "Hmm, " } } },
    { event: "content_block_delta", data: { type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "weather." } } },
    { event: "content_block_delta", data: { type: "content_block_delta", index: 0, delta: { type: "signature_delta", signature: "SIG" } } },
    { event: "content_block_stop", data: { type: "content_block_stop", index: 0 } },
    { event: "ping", data: { type: "ping" } },
    { event: "content_block_start", data: { type: "content_block_start", index: 1, content_block: { type: "text", text: "" } } },
    { event: "content_block_delta", data: { type: "content_block_delta", index: 1, delta: { type: "text_delta", text: "On it." } } },
    { event: "content_block_stop", data: { type: "content_block_stop", index: 1 } },
    { event: "content_block_start", data: { type: "content_block_start", index: 2, content_block: { type: "tool_use", id: "toolu_s", name: "get_weather", input: {} } } },
    { event: "content_block_delta", data: { type: "content_block_delta", index: 2, delta: { type: "input_json_delta", partial_json: "" } } },
    { event: "content_block_delta", data: { type: "content_block_delta", index: 2, delta: { type: "input_json_delta", partial_json: '{"city":' } } },
    { event: "content_block_delta", data: { type: "content_block_delta", index: 2, delta: { type: "input_json_delta", partial_json: '"Toronto"}' } } },
    { event: "content_block_stop", data: { type: "content_block_stop", index: 2 } },
    { event: "message_delta", data: { type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 42 } } },
    { event: "message_stop", data: { type: "message_stop" } },
  ];

  it("normalises the event stream and reconstructs a replayable message", async () => {
    const m = mockFetch(sse(sseBody(events)));
    const stream = client(m.fetch).stream({ messages: [{ role: "user", content: "x" }], tools: [tool] });
    const out = await collect(stream);
    expect(out).toEqual<StreamEvent[]>([
      { type: "start", id: "msg_s", model: "claude-opus-5" },
      { type: "reasoning_delta", text: "Hmm, " },
      { type: "reasoning_delta", text: "weather." },
      { type: "reasoning_end", providerData: { anthropic: { signature: "SIG" } } },
      { type: "text_delta", text: "On it." },
      { type: "tool_call_start", index: 0, id: "toolu_s", name: "get_weather" },
      { type: "tool_call_delta", index: 0, argumentsDelta: '{"city":' },
      { type: "tool_call_delta", index: 0, argumentsDelta: '"Toronto"}' },
      { type: "tool_call_end", index: 0 },
      { type: "usage", usage: { inputTokens: 12, outputTokens: 42, totalTokens: 54 } },
      { type: "finish", reason: "tool_call", rawReason: "tool_use" },
    ]);
    const final = await stream.finalResponse();
    expect(final.message.content[0]).toEqual({ type: "reasoning", text: "Hmm, weather.", providerData: { anthropic: { signature: "SIG" } } });
    expect(JSON.parse(final.toolCalls[0]!.function.arguments)).toEqual({ city: "Toronto" });
  });

  it("surfaces in-stream error events as normalised errors", async () => {
    const m = mockFetch(
      sse(
        sseBody([
          events[0],
          { event: "error", data: { type: "error", error: { type: "overloaded_error", message: "Overloaded" } } },
        ]),
      ),
    );
    const err = await collect(client(m.fetch).stream({ messages: [{ role: "user", content: "x" }] })).catch((e: unknown) => e);
    expect(err).toMatchObject({ name: "ProviderError", statusCode: 529, provider: "anthropic" });
  });
});
