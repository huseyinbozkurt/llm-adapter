import { ALL_CAPABILITIES, type ModelCapabilityRule } from "../../core/capabilities.js";
import { requireString, validateBaseConfig, type OpenAIConfig } from "../../core/config.js";
import type { ReasoningEffort } from "../../core/types.js";
import { commonSettings } from "../base.js";
import { OpenAICompatibleAdapter } from "../openai-compatible/OpenAICompatibleAdapter.js";

/**
 * Per-model capability rules for OpenAI. Later rules override earlier ones.
 * Unknown models get the provider defaults; override via `config.capabilities`.
 */
export const OPENAI_MODEL_RULES: readonly ModelCapabilityRule[] = [
  { match: /^(o\d|gpt-5)/, capabilities: { reasoning: true, reasoningControl: true } },
  { match: /^gpt-5.*-chat/, capabilities: { reasoning: false, reasoningControl: false } },
  { match: /^o[13]-mini/, capabilities: { vision: false } },
  { match: /^o1-(mini|preview)/, capabilities: { tools: false, toolChoice: false, structuredOutput: false, systemMessages: false } },
  { match: /^gpt-3\.5/, capabilities: { vision: false, structuredOutput: false } },
];

/**
 * OpenAI via the Chat Completions API.
 *
 * Differences from the generic compatible adapter:
 * - `max_completion_tokens` instead of the deprecated `max_tokens` (required by reasoning models).
 * - OpenAI never returns reasoning text through Chat Completions; only
 *   `usage.reasoningTokens` is reported. No reasoning content is invented.
 * - Organization / project headers.
 */
export class OpenAIAdapter extends OpenAICompatibleAdapter {
  static fromConfig(config: OpenAIConfig): OpenAIAdapter {
    validateBaseConfig(config);
    const apiKey = requireString(config, "apiKey");
    return new OpenAIAdapter({
      ...commonSettings(config),
      name: "openai",
      baseURL: config.baseURL ?? "https://api.openai.com/v1",
      apiKey,
      headers: {
        ...(config.organization ? { "openai-organization": config.organization } : {}),
        ...(config.project ? { "openai-project": config.project } : {}),
        ...config.headers,
      },
      quirks: { maxTokensField: "max_completion_tokens", streamUsage: true },
      defaultCapabilities: { ...ALL_CAPABILITIES, reasoning: false, reasoningControl: false },
      capabilityRules: OPENAI_MODEL_RULES,
    });
  }

  protected override supportsMessageName(): boolean {
    return true;
  }

  /** OpenAI's highest documented level is `xhigh`; `max` maps to it. Other values pass through. */
  protected override mapEffort(effort: ReasoningEffort): string {
    return effort === "max" ? "xhigh" : effort;
  }
}
