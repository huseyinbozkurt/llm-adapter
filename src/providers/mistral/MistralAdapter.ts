import { ALL_CAPABILITIES, type ModelCapabilityRule } from "../../core/capabilities.js";
import { requireString, validateBaseConfig, type MistralConfig } from "../../core/config.js";
import { getArray, getString, type JsonRecord } from "../../utils/json.js";
import { commonSettings } from "../base.js";
import { OpenAICompatibleAdapter } from "../openai-compatible/OpenAICompatibleAdapter.js";

export const MISTRAL_MODEL_RULES: readonly ModelCapabilityRule[] = [
  { match: /codestral|devstral|ministral|mistral-embed|open-mistral-nemo/, capabilities: { vision: false } },
  { match: /magistral/, capabilities: { reasoning: true } },
];

const MISTRAL_ID = /^[A-Za-z0-9]{9}$/;
const ID_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

/** Deterministically maps any id to 9 alphanumerics (FNV-1a based). */
export function toMistralToolCallId(id: string): string {
  if (MISTRAL_ID.test(id)) return id;
  let h1 = 0x811c9dc5;
  let h2 = 0x01000193;
  for (let i = 0; i < id.length; i++) {
    const c = id.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 0x01000193) >>> 0;
    h2 = Math.imul(h2 ^ c, 0x5bd1e995) >>> 0;
  }
  let out = "";
  for (let i = 0; i < 9; i++) {
    const source = i % 2 === 0 ? h1 : h2;
    out += ID_ALPHABET[(source >>> ((i * 3) % 24)) % ID_ALPHABET.length];
    h1 = Math.imul(h1, 31) >>> 0;
  }
  return out;
}

/**
 * Mistral (OpenAI-compatible with deviations).
 *
 * Differences handled here:
 * - `toolChoice: "required"` is sent as `"any"`.
 * - Tool call ids must be exactly 9 alphanumerics; ids from other providers
 *   (e.g. when switching providers mid-conversation) are mapped
 *   deterministically so assistant tool calls and tool results still match.
 * - Reasoning models (Magistral) return `content` as an array of chunks with
 *   `{ type: "thinking", thinking: [...] }` entries.
 * - `stream_options` is not sent; Mistral includes usage in the final chunk.
 * - No normalised reasoning controls; use `providerOptions.mistral.body` for
 *   Mistral-specific parameters such as `prompt_mode`.
 */
export class MistralAdapter extends OpenAICompatibleAdapter {
  static fromConfig(config: MistralConfig): MistralAdapter {
    validateBaseConfig(config);
    const apiKey = requireString(config, "apiKey");
    return new MistralAdapter({
      ...commonSettings(config),
      name: "mistral",
      baseURL: config.baseURL ?? "https://api.mistral.ai/v1",
      apiKey,
      headers: config.headers,
      quirks: { maxTokensField: "max_tokens", streamUsage: false, requiredToolChoice: "any" },
      defaultCapabilities: { ...ALL_CAPABILITIES, reasoning: false, reasoningControl: false },
      capabilityRules: MISTRAL_MODEL_RULES,
    });
  }

  protected override mapToolCallId(id: string): string {
    return toMistralToolCallId(id);
  }

  protected override deltaContent(obj: JsonRecord): { text?: string; reasoning?: string } {
    if (!Array.isArray(obj.content)) return super.deltaContent(obj);
    let text = "";
    let reasoning = "";
    for (const chunk of obj.content) {
      const type = getString(chunk, "type");
      if (type === "text") text += getString(chunk, "text") ?? "";
      else if (type === "thinking") {
        for (const inner of getArray(chunk, "thinking")) {
          if (getString(inner, "type") === "text") reasoning += getString(inner, "text") ?? "";
        }
      }
    }
    return { text: text || undefined, reasoning: reasoning || undefined };
  }
}
