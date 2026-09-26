import { ALL_CAPABILITIES, type ModelCapabilityRule } from "../../core/capabilities.js";
import { requireString, validateBaseConfig, type XAIConfig } from "../../core/config.js";
import { UnsupportedCapabilityError } from "../../core/errors.js";
import type { ReasoningEffort } from "../../core/types.js";
import { commonSettings } from "../base.js";
import { OpenAICompatibleAdapter } from "../openai-compatible/OpenAICompatibleAdapter.js";

/**
 * Grok model rules. Reasoning models reject `stop`, and only some of them
 * accept `reasoning_effort` (others return an error when it is present).
 */
export const XAI_MODEL_RULES: readonly ModelCapabilityRule[] = [
  { match: /^grok-3(?!-mini)/, capabilities: { vision: false } },
  { match: /^grok-(3-mini|4)/, capabilities: { reasoning: true, stopSequences: false } },
  { match: /^grok-(3-mini|4\.(?:[5-9]|\d{2,}))/, capabilities: { reasoningControl: true } },
  { match: /non-reasoning/, capabilities: { reasoning: false, reasoningControl: false, stopSequences: true } },
];

/**
 * xAI Grok via its OpenAI-compatible Chat Completions endpoint.
 *
 * Differences handled here:
 * - `reasoning_effort` only on models that accept it (see rules); `max` maps
 *   to `xhigh`, `minimal` to `low`; reasoning cannot be disabled by effort.
 * - Reasoning text (when exposed) arrives in `reasoning_content`.
 */
export class XAIAdapter extends OpenAICompatibleAdapter {
  static fromConfig(config: XAIConfig): XAIAdapter {
    validateBaseConfig(config);
    const apiKey = requireString(config, "apiKey");
    return new XAIAdapter({
      ...commonSettings(config),
      name: "xai",
      baseURL: config.baseURL ?? "https://api.x.ai/v1",
      apiKey,
      headers: config.headers,
      quirks: { maxTokensField: "max_tokens", streamUsage: true },
      defaultCapabilities: { ...ALL_CAPABILITIES, reasoning: false, reasoningControl: false },
      capabilityRules: XAI_MODEL_RULES,
    });
  }

  protected override mapEffort(effort: ReasoningEffort): string {
    switch (effort) {
      case "none":
        throw new UnsupportedCapabilityError("reasoningControl", 'xai: reasoning effort "none" is not supported.', {
          provider: "xai",
        });
      case "minimal":
        return "low";
      case "max":
        return "xhigh";
      default:
        return effort;
    }
  }
}
