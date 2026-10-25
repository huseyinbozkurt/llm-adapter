import type { ChatRequest } from "./types.js";
import { UnsupportedCapabilityError } from "./errors.js";

export interface ProviderCapabilities {
  streaming: boolean;
  tools: boolean;
  /** Tool choice other than "auto" (required / none / specific function). */
  toolChoice: boolean;
  vision: boolean;
  /** Native schema-constrained output (`json_schema`). */
  structuredOutput: boolean;
  /** JSON mode without a schema (`json_object`). */
  jsonMode: boolean;
  /** The model performs reasoning (it may or may not expose the text). */
  reasoning: boolean;
  /** The API accepts reasoning controls (effort and/or budget). */
  reasoningControl: boolean;
  systemMessages: boolean;
  stopSequences: boolean;
  listModels: boolean;
}

export type Capability = keyof ProviderCapabilities;

/** A per-model override applied on top of a provider's defaults. */
export interface ModelCapabilityRule {
  match: RegExp;
  capabilities: Partial<ProviderCapabilities>;
}

export const ALL_CAPABILITIES: ProviderCapabilities = {
  streaming: true,
  tools: true,
  toolChoice: true,
  vision: true,
  structuredOutput: true,
  jsonMode: true,
  reasoning: false,
  reasoningControl: false,
  systemMessages: true,
  stopSequences: true,
  listModels: true,
};

/**
 * Resolves capabilities for a model: defaults, then every matching rule in
 * order (later rules win), then explicit user overrides.
 */
export function resolveCapabilities(
  defaults: ProviderCapabilities,
  rules: readonly ModelCapabilityRule[],
  model: string,
  overrides?: Partial<ProviderCapabilities>,
): ProviderCapabilities {
  let caps: ProviderCapabilities = { ...defaults };
  for (const rule of rules) {
    if (rule.match.test(model)) caps = { ...caps, ...rule.capabilities };
  }
  return overrides ? { ...caps, ...overrides } : caps;
}

/** True when any message contains an image part. */
export function hasImages(request: ChatRequest): boolean {
  return request.messages.some(
    (m) => Array.isArray(m.content) && m.content.some((p) => p.type === "image"),
  );
}

/**
 * Throws UnsupportedCapabilityError when a request uses a feature the
 * provider/model cannot honour. Called by LLMClient before any network I/O.
 *
 * `json_schema` with `fallback: "json_mode"` is accepted when JSON mode is
 * available; LLMClient rewrites the request in that case.
 */
export function assertRequestSupported(
  provider: string,
  model: string,
  caps: ProviderCapabilities,
  request: ChatRequest,
  operation: "chat" | "stream",
): void {
  const fail = (capability: Capability | string, what: string): never => {
    throw new UnsupportedCapabilityError(capability, `${provider} (model "${model}") does not support ${what}.`, {
      provider,
    });
  };

  if (operation === "stream" && !caps.streaming) fail("streaming", "streaming");
  if (request.tools && request.tools.length > 0 && !caps.tools) fail("tools", "tool calling");
  if (request.toolChoice && request.toolChoice !== "auto" && !caps.toolChoice) {
    fail("toolChoice", `toolChoice ${JSON.stringify(request.toolChoice)}`);
  }
  if (!caps.systemMessages && request.messages.some((m) => m.role === "system")) fail("systemMessages", "system messages");
  if (request.stop && request.stop.length > 0 && !caps.stopSequences) fail("stopSequences", "stop sequences");
  if (!caps.vision && hasImages(request)) fail("vision", "image inputs");

  const format = request.responseFormat;
  if (format?.type === "json_object" && !caps.jsonMode) fail("jsonMode", "JSON mode (responseFormat json_object)");
  if (format?.type === "json_schema" && !caps.structuredOutput) {
    if (format.fallback !== "json_mode") {
      fail(
        "structuredOutput",
        'native structured output (json_schema). Set responseFormat.fallback = "json_mode" to accept unvalidated JSON-mode output',
      );
    }
    if (!caps.jsonMode) fail("jsonMode", "structured output or JSON mode");
  }

  const r = request.reasoning;
  if (r && (r.effort !== undefined || r.maxTokens !== undefined) && !caps.reasoningControl) {
    fail("reasoningControl", "reasoning controls (reasoning.effort / reasoning.maxTokens)");
  }
}
