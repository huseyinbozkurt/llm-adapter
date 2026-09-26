import { ALL_CAPABILITIES } from "../../core/capabilities.js";
import { requireString, validateBaseConfig, type DeepSeekConfig } from "../../core/config.js";
import type { ChatRequest, ReasoningEffort, ReasoningPart } from "../../core/types.js";
import type { JsonRecord } from "../../utils/json.js";
import { commonSettings } from "../base.js";
import { OpenAICompatibleAdapter } from "../openai-compatible/OpenAICompatibleAdapter.js";

/**
 * DeepSeek (OpenAI-compatible).
 *
 * Differences handled here:
 * - Reasoning text arrives in `reasoning_content` (message and stream delta).
 * - When a conversation uses tools in thinking mode, DeepSeek requires the
 *   prior assistant turns' `reasoning_content` to be sent back, so reasoning
 *   parts of assistant messages are replayed in that field.
 * - Reasoning effort accepts `low` / `high` / `max`; `none` disables thinking
 *   via `thinking: { type: "disabled" }`.
 * - Only JSON mode (`json_object`) is available; `json_schema` requires
 *   `fallback: "json_mode"`.
 * - No image input.
 * - Usage reports cache hits as `prompt_cache_hit_tokens`.
 */
export class DeepSeekAdapter extends OpenAICompatibleAdapter {
  static fromConfig(config: DeepSeekConfig): DeepSeekAdapter {
    validateBaseConfig(config);
    const apiKey = requireString(config, "apiKey");
    return new DeepSeekAdapter({
      ...commonSettings(config),
      name: "deepseek",
      baseURL: config.baseURL ?? "https://api.deepseek.com",
      apiKey,
      headers: config.headers,
      quirks: { maxTokensField: "max_tokens", streamUsage: true },
      defaultCapabilities: {
        ...ALL_CAPABILITIES,
        vision: false,
        structuredOutput: false,
        jsonMode: true,
        reasoning: true,
        reasoningControl: true,
      },
    });
  }

  protected override applyReasoning(body: JsonRecord, request: ChatRequest, model: string): void {
    if (request.reasoning?.effort === "none") {
      body.thinking = { type: "disabled" };
      return;
    }
    super.applyReasoning(body, request, model);
  }

  protected override mapEffort(effort: ReasoningEffort): string {
    switch (effort) {
      case "none":
      case "minimal":
      case "low":
        return "low";
      case "medium":
      case "high":
        return "high";
      case "xhigh":
      case "max":
        return "max";
    }
  }

  protected override assistantReasoningFields(parts: ReasoningPart[]): JsonRecord | undefined {
    const text = parts.map((p) => p.text).join("");
    return text ? { reasoning_content: text } : undefined;
  }
}
