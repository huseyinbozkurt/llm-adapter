import { ALL_CAPABILITIES, type ModelCapabilityRule } from "../../core/capabilities.js";
import { requireString, validateBaseConfig, type GroqConfig } from "../../core/config.js";
import type { ChatRequest, TokenUsage } from "../../core/types.js";
import { getRecord, type JsonRecord } from "../../utils/json.js";
import { commonSettings } from "../base.js";
import { OpenAICompatibleAdapter } from "../openai-compatible/OpenAICompatibleAdapter.js";

export const GROQ_MODEL_RULES: readonly ModelCapabilityRule[] = [
  { match: /llama-4/, capabilities: { vision: true, structuredOutput: true } },
  { match: /gpt-oss|kimi-k2/, capabilities: { structuredOutput: true } },
  { match: /gpt-oss|qwen3|qwq|deepseek-r1/, capabilities: { reasoning: true } },
  { match: /gpt-oss|qwen3/, capabilities: { reasoningControl: true } },
];

/**
 * Groq (OpenAI-compatible).
 *
 * Differences handled here:
 * - Reasoning controls differ by model family: GPT-OSS takes
 *   `reasoning_effort` low/medium/high and `include_reasoning`; Qwen 3 takes
 *   `reasoning_effort` none/default and `reasoning_format`.
 * - Reasoning text arrives in the `reasoning` field.
 * - Streaming usage may arrive under `x_groq.usage`.
 * - Vision and json_schema are available on a subset of models only.
 */
export class GroqAdapter extends OpenAICompatibleAdapter {
  static fromConfig(config: GroqConfig): GroqAdapter {
    validateBaseConfig(config);
    const apiKey = requireString(config, "apiKey");
    return new GroqAdapter({
      ...commonSettings(config),
      name: "groq",
      baseURL: config.baseURL ?? "https://api.groq.com/openai/v1",
      apiKey,
      headers: config.headers,
      quirks: { maxTokensField: "max_completion_tokens", streamUsage: true },
      defaultCapabilities: {
        ...ALL_CAPABILITIES,
        vision: false,
        structuredOutput: false,
        reasoning: false,
        reasoningControl: false,
      },
      capabilityRules: GROQ_MODEL_RULES,
    });
  }

  protected override applyReasoning(body: JsonRecord, request: ChatRequest, model: string): void {
    const r = request.reasoning;
    if (!r) return;
    const gptOss = /gpt-oss/.test(model);
    if (r.effort !== undefined) {
      if (gptOss) {
        const e = r.effort;
        body.reasoning_effort = e === "none" || e === "minimal" || e === "low" ? "low" : e === "medium" ? "medium" : "high";
      } else {
        body.reasoning_effort = r.effort === "none" ? "none" : "default";
      }
    } else if (r.maxTokens !== undefined) {
      super.applyReasoning(body, request, model); // throws: Groq has no budget control
    }
    // Only reasoning models accept these fields.
    if (r.includeReasoning !== undefined && this.capabilities(model).reasoning) {
      if (gptOss) body.include_reasoning = r.includeReasoning;
      else body.reasoning_format = r.includeReasoning ? "parsed" : "hidden";
    }
  }

  protected override parseUsage(obj: unknown): TokenUsage | undefined {
    return super.parseUsage(obj) ?? super.parseUsage(getRecord(obj, "x_groq"));
  }
}
