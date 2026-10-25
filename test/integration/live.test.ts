/**
 * Live API tests. Each provider's block runs only when its key is set, e.g.
 *
 *   OPENAI_API_KEY=... ANTHROPIC_API_KEY=... npm run test:integration
 *
 * Override models with <PROVIDER>_MODEL (e.g. OPENAI_MODEL=gpt-5-mini).
 * These make real, billed requests with small token limits.
 */
import { describe, expect, it } from "vitest";
import { LLMClient, type ChatMessage, type LLMConfig, type ToolDefinition } from "../../src/index.js";

interface LiveTarget {
  provider: string;
  keyVar: string;
  defaultModel: string;
  extra?: Partial<LLMConfig>;
}

const targets: LiveTarget[] = [
  { provider: "openai", keyVar: "OPENAI_API_KEY", defaultModel: "gpt-5-mini" },
  { provider: "anthropic", keyVar: "ANTHROPIC_API_KEY", defaultModel: "claude-haiku-4-5" },
  { provider: "deepseek", keyVar: "DEEPSEEK_API_KEY", defaultModel: "deepseek-chat" },
  { provider: "gemini", keyVar: "GEMINI_API_KEY", defaultModel: "gemini-2.5-flash" },
  { provider: "groq", keyVar: "GROQ_API_KEY", defaultModel: "openai/gpt-oss-20b" },
  { provider: "mistral", keyVar: "MISTRAL_API_KEY", defaultModel: "mistral-small-latest" },
  { provider: "xai", keyVar: "XAI_API_KEY", defaultModel: "grok-3-mini" },
  { provider: "openrouter", keyVar: "OPENROUTER_API_KEY", defaultModel: "openai/gpt-5-mini" },
];

const weather: ToolDefinition = {
  type: "function",
  function: {
    name: "get_weather",
    description: "Get the current weather for a city",
    parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"], additionalProperties: false },
  },
};

for (const t of targets) {
  const apiKey = process.env[t.keyVar];
  const model = process.env[`${t.provider.toUpperCase()}_MODEL`] ?? t.defaultModel;

  describe.skipIf(!apiKey)(`live: ${t.provider} (${model})`, () => {
    // Built lazily: describe bodies run even when the block is skipped.
    let cached: LLMClient | undefined;
    const llm = () =>
      (cached ??= new LLMClient({ provider: t.provider, apiKey, model, timeout: 120_000, ...t.extra } as LLMConfig));

    it("chat", async () => {
      const res = await llm().chat({
        messages: [
          { role: "system", content: "Reply with exactly one word." },
          { role: "user", content: "Say hello." },
        ],
        maxTokens: 2000,
      });
      expect(res.provider).toBe(t.provider);
      expect(res.text.length).toBeGreaterThan(0);
      expect(res.usage?.inputTokens).toBeGreaterThan(0);
    });

    it("stream", async () => {
      const stream = llm().stream({ messages: [{ role: "user", content: "Count from 1 to 5." }], maxTokens: 2000 });
      let deltas = 0;
      for await (const e of stream) if (e.type === "text_delta") deltas++;
      const final = await stream.finalResponse();
      expect(deltas).toBeGreaterThan(0);
      expect(final.text).toMatch(/1[\s\S]*5/);
    });

    it("tool call round trip", async () => {
      const messages: ChatMessage[] = [
        { role: "user", content: "What's the weather in Toronto? Use the tool." },
      ];
      const first = await llm().chat({ messages, tools: [weather], maxTokens: 2000 });
      expect(first.toolCalls.length).toBeGreaterThan(0);
      const call = first.toolCalls[0]!;
      expect(JSON.parse(call.function.arguments)).toHaveProperty("city");
      messages.push(first.message, { role: "tool", toolCallId: call.id, content: JSON.stringify({ tempC: 21, sky: "clear" }) });
      const second = await llm().chat({ messages, tools: [weather], maxTokens: 2000 });
      expect(second.text).toMatch(/21/);
    });
  });
}
