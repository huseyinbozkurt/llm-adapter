/**
 * Unified request ↔ Anthropic Messages API.
 *
 * Notable differences from the unified model:
 * - System messages are a top-level `system` field, not messages. All system
 *   messages are hoisted there in order.
 * - Tool results are `tool_result` blocks inside a *user* message; consecutive
 *   tool messages (and a following user message) are merged into one turn.
 * - Tool call arguments are a JSON object (`input`), not a string.
 * - Thinking blocks carry a `signature` that must be replayed verbatim; only
 *   reasoning parts that originated from Anthropic are sent back.
 * - `max_tokens` is mandatory.
 */
import type {
  AssistantMessage,
  ChatMessage,
  ChatRequest,
  ContentPart,
  FinishReason,
  TokenUsage,
  ToolCall,
  ToolChoice,
  UserContentPart,
} from "../../core/types.js";
import { InvalidRequestError, ProviderError } from "../../core/errors.js";
import { getNumber, getRecord, getString, isRecord, type JsonRecord } from "../../utils/json.js";
import { finalizeUsage, resolveImage, textOf } from "../../utils/normalize.js";

export const PROVIDER = "anthropic";

function imageBlock(part: Extract<UserContentPart, { type: "image" }>): JsonRecord {
  const img = resolveImage(part, PROVIDER);
  return img.kind === "base64"
    ? { type: "image", source: { type: "base64", media_type: img.mimeType, data: img.value } }
    : { type: "image", source: { type: "url", url: img.value } };
}

function userBlocks(content: string | UserContentPart[]): JsonRecord[] {
  if (typeof content === "string") return content === "" ? [] : [{ type: "text", text: content }];
  return content.map((p) => (p.type === "text" ? { type: "text", text: p.text } : imageBlock(p)));
}

function parseArguments(call: ToolCall): unknown {
  if (call.function.arguments.trim() === "") return {};
  try {
    return JSON.parse(call.function.arguments) as unknown;
  } catch (cause) {
    throw new InvalidRequestError(
      `${PROVIDER}: tool call "${call.function.name}" (${call.id}) has arguments that are not valid JSON; Anthropic requires an object.`,
      { provider: PROVIDER, cause },
    );
  }
}

function assistantBlocks(m: AssistantMessage): JsonRecord[] {
  const blocks: JsonRecord[] = [];
  const parts: ContentPart[] = typeof m.content === "string" ? [{ type: "text", text: m.content }] : m.content;
  for (const p of parts) {
    if (p.type === "text") {
      if (p.text !== "") blocks.push({ type: "text", text: p.text });
    } else if (p.type === "reasoning") {
      const data = p.providerData?.[PROVIDER];
      const signature = getString(data, "signature");
      const redacted = getString(data, "redactedData");
      if (redacted !== undefined) blocks.push({ type: "redacted_thinking", data: redacted });
      else if (signature !== undefined) blocks.push({ type: "thinking", thinking: p.text, signature });
      // Reasoning from other providers has no signature and cannot be replayed.
    } else {
      throw new InvalidRequestError(`${PROVIDER}: assistant messages cannot contain images.`, { provider: PROVIDER });
    }
  }
  for (const call of m.toolCalls ?? []) {
    blocks.push({ type: "tool_use", id: call.id, name: call.function.name, input: parseArguments(call) });
  }
  return blocks;
}

export function toAnthropicMessages(messages: readonly ChatMessage[]): { system?: JsonRecord[]; messages: JsonRecord[] } {
  const system: JsonRecord[] = [];
  const out: { role: "user" | "assistant"; content: JsonRecord[] }[] = [];

  const push = (role: "user" | "assistant", blocks: JsonRecord[]) => {
    const last = out[out.length - 1];
    // Anthropic requires alternating roles: merge consecutive same-role turns.
    if (last && last.role === role) last.content.push(...blocks);
    else out.push({ role, content: blocks });
  };

  for (const m of messages) {
    switch (m.role) {
      case "system": {
        const text = textOf(m.content);
        if (text !== "") system.push({ type: "text", text });
        break;
      }
      case "user":
        push("user", userBlocks(m.content));
        break;
      case "assistant":
        push("assistant", assistantBlocks(m));
        break;
      case "tool":
        push("user", [
          {
            type: "tool_result",
            tool_use_id: m.toolCallId,
            content: typeof m.content === "string" ? m.content : userBlocks(m.content),
            ...(m.isError ? { is_error: true } : {}),
          },
        ]);
        break;
    }
  }
  return { ...(system.length > 0 ? { system } : {}), messages: out };
}

export function toAnthropicToolChoice(choice: ToolChoice | undefined, parallel: boolean | undefined): JsonRecord | undefined {
  let tc: JsonRecord | undefined;
  if (choice === "auto") tc = { type: "auto" };
  else if (choice === "required") tc = { type: "any" };
  else if (choice === "none") tc = { type: "none" };
  else if (choice) tc = { type: "tool", name: choice.name };
  if (parallel === false) tc = { ...(tc ?? { type: "auto" }), disable_parallel_tool_use: true };
  return tc;
}

export function toAnthropicTools(request: ChatRequest): JsonRecord[] | undefined {
  if (!request.tools || request.tools.length === 0) return undefined;
  return request.tools.map((t) => ({
    name: t.function.name,
    ...(t.function.description ? { description: t.function.description } : {}),
    input_schema: t.function.parameters,
    ...(t.function.strict !== undefined ? { strict: t.function.strict } : {}),
  }));
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

export function normalizeAnthropicStopReason(reason: string | undefined): FinishReason {
  switch (reason) {
    case "end_turn":
    case "stop_sequence":
      return "stop";
    case "max_tokens":
    case "model_context_window_exceeded":
      return "length";
    case "tool_use":
      return "tool_call";
    case "refusal":
      return "content_filter";
    default:
      // e.g. "pause_turn" (server tools paused a long turn) has no portable equivalent.
      return "unknown";
  }
}

/** Anthropic reports uncached, cache-read and cache-write input separately; totals follow the unified convention. */
export function parseAnthropicUsage(u: unknown): TokenUsage | undefined {
  if (!isRecord(u)) return undefined;
  const uncached = getNumber(u, "input_tokens");
  const cacheRead = getNumber(u, "cache_read_input_tokens");
  const cacheWrite = getNumber(u, "cache_creation_input_tokens");
  const input = uncached === undefined ? undefined : uncached + (cacheRead ?? 0) + (cacheWrite ?? 0);
  return finalizeUsage({
    inputTokens: input,
    outputTokens: getNumber(u, "output_tokens"),
    cachedInputTokens: cacheRead,
    cacheWriteInputTokens: cacheWrite,
  });
}

export function parseAnthropicContent(blocks: unknown[]): { content: ContentPart[]; toolCalls: ToolCall[] } {
  const content: ContentPart[] = [];
  const toolCalls: ToolCall[] = [];
  for (const b of blocks) {
    switch (getString(b, "type")) {
      case "text": {
        const text = getString(b, "text") ?? "";
        const last = content[content.length - 1];
        if (last?.type === "text") last.text += text;
        else content.push({ type: "text", text });
        break;
      }
      case "thinking":
        content.push({
          type: "reasoning",
          text: getString(b, "thinking") ?? "",
          providerData: { [PROVIDER]: { signature: getString(b, "signature") ?? "" } },
        });
        break;
      case "redacted_thinking":
        content.push({ type: "reasoning", text: "", providerData: { [PROVIDER]: { redactedData: getString(b, "data") ?? "" } } });
        break;
      case "tool_use":
        toolCalls.push({
          id: getString(b, "id") ?? "",
          type: "function",
          function: { name: getString(b, "name") ?? "", arguments: JSON.stringify(getRecord(b, "input") ?? {}) },
        });
        break;
      default:
        // Server-tool blocks (web search results, etc.) are not normalised; see `response.raw`.
        break;
    }
  }
  return { content, toolCalls };
}

export function assertMessage(data: unknown): JsonRecord {
  if (!isRecord(data) || !Array.isArray(data.content)) {
    throw new ProviderError(`${PROVIDER}: unexpected response body`, { provider: PROVIDER, cause: data });
  }
  return data;
}
