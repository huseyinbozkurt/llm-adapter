import { describe, expect, it } from "vitest";
import { InvalidRequestError, LLMClient, UnsupportedCapabilityError, type ToolDefinition } from "../../src/index.js";
import { collect, json, mockFetch, sse, sseBody } from "../helpers/mockFetch.js";

const tool: ToolDefinition = {
  type: "function",
  function: { name: "get_weather", description: "w", parameters: { type: "object", properties: { city: { type: "string" } } } },
};

const KEY = "AIzaTestKey000000000000000000";
function client(fetch: typeof globalThis.fetch, model = "gemini-3-pro-preview") {
  return new LLMClient({ provider: "gemini", apiKey: KEY, model, fetch });
}

const reply = (parts: unknown[], finishReason = "STOP", usage?: Record<string, number>) =>
  json({
    responseId: "resp1",
    modelVersion: "gemini-3-pro-preview",
    candidates: [{ content: { role: "model", parts }, finishReason }],
    ...(usage ? { usageMetadata: usage } : {}),
  });

describe("Gemini adapter: request translation", () => {
  it("builds contents, systemInstruction, tools and generationConfig", async () => {
    const m = mockFetch(reply([{ text: "ok" }]));
    await client(m.fetch).chat({
      messages: [
        { role: "system", content: "Be brief." },
        { role: "user", content: [{ type: "text", text: "Hi" }, { type: "image", base64: "AAAA", mimeType: "image/png" }] },
        {
          role: "assistant",
          content: "",
          toolCalls: [
            {
              id: "local_1",
              type: "function",
              function: { name: "get_weather", arguments: '{"city":"Paris"}' },
              providerData: { gemini: { nativeId: false, thoughtSignature: "TS1" } },
            },
          ],
        },
        { role: "tool", toolCallId: "local_1", content: '{"temp":20}' },
        { role: "tool", toolCallId: "local_1", name: "get_weather", content: "plain text result" },
      ],
      tools: [tool],
      toolChoice: { type: "function", name: "get_weather" },
      temperature: 0.1,
      maxTokens: 50,
      stop: ["END"],
      responseFormat: { type: "json_schema", name: "x", schema: { type: "object" } },
      reasoning: { effort: "low", includeReasoning: true },
    });

    const req = m.requests[0]!;
    expect(req.url).toBe("https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent");
    expect(req.url).not.toContain(KEY);
    expect(req.headers["x-goog-api-key"]).toBe(KEY);
    expect(req.body).toEqual({
      systemInstruction: { parts: [{ text: "Be brief." }] },
      contents: [
        { role: "user", parts: [{ text: "Hi" }, { inlineData: { mimeType: "image/png", data: "AAAA" } }] },
        { role: "model", parts: [{ functionCall: { name: "get_weather", args: { city: "Paris" } }, thoughtSignature: "TS1" }] },
        {
          role: "user",
          parts: [
            { functionResponse: { name: "get_weather", response: { temp: 20 } } },
            { functionResponse: { name: "get_weather", response: { result: "plain text result" } } },
          ],
        },
      ],
      tools: [{ functionDeclarations: [{ name: "get_weather", description: "w", parametersJsonSchema: tool.function.parameters }] }],
      toolConfig: { functionCallingConfig: { mode: "ANY", allowedFunctionNames: ["get_weather"] } },
      generationConfig: {
        temperature: 0.1,
        maxOutputTokens: 50,
        stopSequences: ["END"],
        responseMimeType: "application/json",
        responseJsonSchema: { type: "object" },
        thinkingConfig: { thinkingLevel: "low", includeThoughts: true },
      },
    });
  });

  it("maps reasoning budgets on Gemini 2.5 and rejects effort-only there", async () => {
    const m = mockFetch(reply([{ text: "ok" }]));
    await client(m.fetch, "gemini-2.5-flash").chat({ messages: [{ role: "user", content: "x" }], reasoning: { maxTokens: 1024 } });
    expect(m.lastBody().generationConfig).toEqual({ thinkingConfig: { thinkingBudget: 1024 } });

    await client(m.fetch, "gemini-2.5-flash").chat({ messages: [{ role: "user", content: "x" }], reasoning: { effort: "none" } });
    expect(m.lastBody().generationConfig).toEqual({ thinkingConfig: { thinkingBudget: 0 } });

    await expect(
      client(m.fetch, "gemini-2.5-flash").chat({ messages: [{ role: "user", content: "x" }], reasoning: { effort: "high" } }),
    ).rejects.toBeInstanceOf(UnsupportedCapabilityError);
  });

  it("requires a tool name it can resolve for tool results", async () => {
    const m = mockFetch(reply([{ text: "ok" }]));
    await expect(
      client(m.fetch).chat({ messages: [{ role: "tool", toolCallId: "unknown", content: "x" }] }),
    ).rejects.toBeInstanceOf(InvalidRequestError);
  });
});

describe("Gemini adapter: response normalisation", () => {
  it("normalises function calls (keeping thought signatures), thoughts and usage", async () => {
    const m = mockFetch(
      reply(
        [
          { text: "Considering...", thought: true },
          { text: "Let me check. " },
          { functionCall: { name: "get_weather", args: { city: "Toronto" } }, thoughtSignature: "SIG-1" },
          { functionCall: { id: "fc-2", name: "get_weather", args: { city: "Oslo" } } },
        ],
        "STOP",
        { promptTokenCount: 100, candidatesTokenCount: 20, thoughtsTokenCount: 30, totalTokenCount: 150, cachedContentTokenCount: 60 },
      ),
    );
    const res = await client(m.fetch).chat({ messages: [{ role: "user", content: "x" }], tools: [tool] });
    expect(res.finishReason).toBe("tool_call");
    expect(res.text).toBe("Let me check. ");
    expect(res.reasoning).toBe("Considering...");
    expect(res.toolCalls).toHaveLength(2);
    expect(res.toolCalls[0]).toMatchObject({
      type: "function",
      function: { name: "get_weather", arguments: '{"city":"Toronto"}' },
      providerData: { gemini: { nativeId: false, thoughtSignature: "SIG-1" } },
    });
    expect(res.toolCalls[0]!.id).toMatch(/^call_/);
    expect(res.toolCalls[1]).toMatchObject({ id: "fc-2", providerData: { gemini: { nativeId: true } } });
    expect(res.usage).toEqual({ inputTokens: 100, outputTokens: 50, totalTokens: 150, cachedInputTokens: 60, reasoningTokens: 30 });

    // Replaying the assistant message sends the signature and the native id back.
    const m2 = mockFetch(reply([{ text: "done" }]));
    await client(m2.fetch).chat({
      messages: [
        { role: "user", content: "x" },
        res.message,
        { role: "tool", toolCallId: res.toolCalls[0]!.id, content: "{}" },
        { role: "tool", toolCallId: "fc-2", content: "{}" },
      ],
      tools: [tool],
    });
    const contents = m2.lastBody().contents as Array<{ parts: unknown[] }>;
    expect(contents[1]!.parts).toContainEqual({
      functionCall: { name: "get_weather", args: { city: "Toronto" } },
      thoughtSignature: "SIG-1",
    });
    expect(contents[1]!.parts).toContainEqual({ functionCall: { name: "get_weather", args: { city: "Oslo" }, id: "fc-2" } });
    expect(contents[2]!.parts).toContainEqual({ functionResponse: { name: "get_weather", id: "fc-2", response: {} } });
  });

  it.each([
    ["STOP", "stop"],
    ["MAX_TOKENS", "length"],
    ["SAFETY", "content_filter"],
    ["RECITATION", "content_filter"],
    ["MALFORMED_FUNCTION_CALL", "error"],
    ["OTHER", "unknown"],
  ])("maps finishReason %s → %s", async (raw, expected) => {
    const m = mockFetch(reply([{ text: "x" }], raw));
    expect((await client(m.fetch).chat({ messages: [{ role: "user", content: "x" }] })).finishReason).toBe(expected);
  });

  it("reports blocked prompts as content_filter", async () => {
    const m = mockFetch(json({ promptFeedback: { blockReason: "SAFETY" } }));
    const res = await client(m.fetch).chat({ messages: [{ role: "user", content: "x" }] });
    expect(res.finishReason).toBe("content_filter");
    expect(res.text).toBe("");
  });
});

describe("Gemini adapter: streaming", () => {
  it("streams text, thoughts and whole function calls", async () => {
    const chunks = [
      { responseId: "r1", modelVersion: "gemini-3-pro-preview", candidates: [{ content: { role: "model", parts: [{ text: "Plan", thought: true }] } }] },
      { candidates: [{ content: { role: "model", parts: [{ text: "Hello " }] } }] },
      { candidates: [{ content: { role: "model", parts: [{ text: "world" }] } }] },
      {
        candidates: [
          {
            content: { role: "model", parts: [{ functionCall: { name: "get_weather", args: { city: "Toronto" } }, thoughtSignature: "S" }] },
            finishReason: "STOP",
          },
        ],
        usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 7, totalTokenCount: 12 },
      },
    ];
    const m = mockFetch(sse(sseBody(chunks)));
    const stream = client(m.fetch).stream({ messages: [{ role: "user", content: "x" }], tools: [tool] });
    const events = await collect(stream);
    expect(m.requests[0]!.url).toContain(":streamGenerateContent?alt=sse");
    expect(events.map((e) => e.type)).toEqual([
      "start",
      "reasoning_delta",
      "reasoning_end",
      "text_delta",
      "text_delta",
      "tool_call_start",
      "tool_call_delta",
      "tool_call_end",
      "usage",
      "finish",
    ]);
    const final = await stream.finalResponse();
    expect(final.text).toBe("Hello world");
    expect(final.reasoning).toBe("Plan");
    expect(final.finishReason).toBe("tool_call");
    expect(final.toolCalls[0]!.providerData).toEqual({ gemini: { nativeId: false, thoughtSignature: "S" } });
    expect(final.usage).toEqual({ inputTokens: 5, outputTokens: 7, totalTokens: 12 });
  });
});
