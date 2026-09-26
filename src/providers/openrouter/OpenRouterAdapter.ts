import { ALL_CAPABILITIES } from "../../core/capabilities.js";
import { requireString, validateBaseConfig, type OpenRouterConfig } from "../../core/config.js";
import type { ChatRequest } from "../../core/types.js";
import { compact, type JsonRecord } from "../../utils/json.js";
import { commonSettings } from "../base.js";
import { OpenAICompatibleAdapter } from "../openai-compatible/OpenAICompatibleAdapter.js";

/**
 * OpenRouter (OpenAI-compatible gateway to many upstream models).
 *
 * Differences handled here:
 * - Unified `reasoning: { effort, max_tokens, exclude }` object instead of
 *   `reasoning_effort`; OpenRouter translates it per upstream model.
 * - Reasoning text arrives in the `reasoning` field.
 * - Optional attribution headers (`HTTP-Referer`, `X-Title`).
 *
 * Capabilities depend entirely on the routed model, so defaults are
 * permissive and OpenRouter itself rejects unsupported combinations.
 * Routing preferences (`provider`, `models`, `transforms`, ...) go through
 * `providerOptions.openrouter.body`.
 */
export class OpenRouterAdapter extends OpenAICompatibleAdapter {
  static fromConfig(config: OpenRouterConfig): OpenRouterAdapter {
    validateBaseConfig(config);
    const apiKey = requireString(config, "apiKey");
    return new OpenRouterAdapter({
      ...commonSettings(config),
      name: "openrouter",
      baseURL: config.baseURL ?? "https://openrouter.ai/api/v1",
      apiKey,
      headers: {
        ...(config.siteURL ? { "http-referer": config.siteURL } : {}),
        ...(config.appName ? { "x-title": config.appName } : {}),
        ...config.headers,
      },
      quirks: { maxTokensField: "max_tokens", streamUsage: true },
      defaultCapabilities: { ...ALL_CAPABILITIES, reasoning: true, reasoningControl: true },
    });
  }

  protected override applyReasoning(body: JsonRecord, request: ChatRequest): void {
    const r = request.reasoning;
    if (!r) return;
    const reasoning = compact({
      // OpenRouter's scale tops out at "xhigh".
      effort: r.effort === "max" ? "xhigh" : r.effort,
      // OpenRouter accepts either effort or max_tokens; effort wins when both are given.
      max_tokens: r.effort === undefined ? r.maxTokens : undefined,
      exclude: r.includeReasoning === false ? true : undefined,
    });
    if (Object.keys(reasoning).length > 0) body.reasoning = reasoning;
  }
}
