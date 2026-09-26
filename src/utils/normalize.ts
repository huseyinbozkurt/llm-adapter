import type {
  ContentPart,
  FinishReason,
  ImagePart,
  MessageContent,
  ProviderData,
  ReasoningPart,
  TextPart,
  TokenUsage,
  ToolCall,
} from "../core/types.js";
import { InvalidRequestError } from "../core/errors.js";

export function toParts(content: MessageContent | undefined): ContentPart[] {
  if (content === undefined) return [];
  if (typeof content === "string") return content === "" ? [] : [{ type: "text", text: content }];
  return content;
}

export function textOf(content: MessageContent | undefined): string {
  if (content === undefined) return "";
  if (typeof content === "string") return content;
  return content
    .filter((p): p is TextPart => p.type === "text")
    .map((p) => p.text)
    .join("");
}

export function reasoningOf(parts: readonly ContentPart[]): string | undefined {
  const reasoning = parts.filter((p): p is ReasoningPart => p.type === "reasoning");
  if (reasoning.length === 0) return undefined;
  const text = reasoning.map((p) => p.text).join("");
  return text === "" ? undefined : text;
}

/** Appends text to the last text part, or starts a new one. */
export function appendText(parts: ContentPart[], text: string): void {
  const last = parts[parts.length - 1];
  if (last?.type === "text") last.text += text;
  else parts.push({ type: "text", text });
}

export function appendReasoning(parts: ContentPart[], text: string, providerData?: ProviderData): void {
  const last = parts[parts.length - 1];
  if (last?.type === "reasoning" && !last.providerData) {
    last.text += text;
    if (providerData) last.providerData = providerData;
  } else {
    parts.push(providerData ? { type: "reasoning", text, providerData } : { type: "reasoning", text });
  }
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

export interface ResolvedImage {
  kind: "base64" | "url";
  /** Base64 data (kind=base64) or the URL (kind=url). */
  value: string;
  mimeType?: string;
}

const DATA_URL = /^data:([^;,]+)?(?:;[^,]*)?;base64,(.*)$/s;

/** Normalises an ImagePart: data URLs become base64; validates exactly one source. */
export function resolveImage(part: ImagePart, provider: string): ResolvedImage {
  if (part.base64 && part.url) {
    throw new InvalidRequestError(`${provider}: image part must set either "url" or "base64", not both.`, { provider });
  }
  if (part.base64) {
    if (!part.mimeType) {
      throw new InvalidRequestError(`${provider}: image part with "base64" requires "mimeType".`, { provider });
    }
    return { kind: "base64", value: part.base64, mimeType: part.mimeType };
  }
  if (part.url) {
    const m = DATA_URL.exec(part.url);
    if (m) return { kind: "base64", value: m[2] ?? "", mimeType: m[1] ?? part.mimeType };
    return { kind: "url", value: part.url, mimeType: part.mimeType };
  }
  throw new InvalidRequestError(`${provider}: image part needs "url" or "base64".`, { provider });
}

export function imageToDataURL(img: ResolvedImage): string {
  return img.kind === "url" ? img.value : `data:${img.mimeType ?? "application/octet-stream"};base64,${img.value}`;
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

let idCounter = 0;
/** Generates a locally-unique id for tool calls whose provider does not supply one. */
export function generateId(prefix: string): string {
  idCounter = (idCounter + 1) % Number.MAX_SAFE_INTEGER;
  return `${prefix}_${Date.now().toString(36)}${idCounter.toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

// ---------------------------------------------------------------------------
// Finish reasons & usage
// ---------------------------------------------------------------------------

/** Normalises OpenAI-style finish_reason values (also used by compatible providers). */
export function normalizeOpenAIFinishReason(reason: string | null | undefined): FinishReason {
  switch (reason) {
    case "stop":
    case "end_turn":
    case "eos":
      return "stop";
    case "length":
    case "max_tokens":
    case "model_length":
      return "length";
    case "tool_calls":
    case "function_call":
    case "tool_call":
      return "tool_call";
    case "content_filter":
      return "content_filter";
    case "error":
      return "error";
    default:
      return "unknown";
  }
}

/** Fills totalTokens when missing and drops undefined fields. */
export function finalizeUsage(usage: TokenUsage): TokenUsage | undefined {
  const out: TokenUsage = {};
  for (const [k, v] of Object.entries(usage) as [keyof TokenUsage, number | undefined][]) {
    if (typeof v === "number") out[k] = v;
  }
  if (out.totalTokens === undefined && out.inputTokens !== undefined && out.outputTokens !== undefined) {
    out.totalTokens = out.inputTokens + out.outputTokens;
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

export function finishReasonFor(toolCalls: readonly ToolCall[], reason: FinishReason): FinishReason {
  // Some providers report "stop" even when the turn ended in tool calls.
  return toolCalls.length > 0 && reason === "stop" ? "tool_call" : reason;
}
