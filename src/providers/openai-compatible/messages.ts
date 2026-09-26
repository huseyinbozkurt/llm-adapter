/**
 * Unified messages/tools → OpenAI Chat Completions wire format.
 */
import type {
  AssistantMessage,
  ChatMessage,
  ReasoningPart,
  ResponseFormat,
  ToolChoice,
  ToolDefinition,
  UserContentPart,
} from "../../core/types.js";
import { UnsupportedCapabilityError } from "../../core/errors.js";
import { imageToDataURL, resolveImage, textOf } from "../../utils/normalize.js";
import type { JsonRecord } from "../../utils/json.js";

export interface MessageTranslationHooks {
  provider: string;
  /** Extra fields for an assistant message built from its reasoning parts (e.g. DeepSeek `reasoning_content`). */
  assistantReasoningFields?(parts: ReasoningPart[]): JsonRecord | undefined;
  /** Rewrites tool call ids (Mistral requires a specific id format). */
  mapToolCallId?(id: string): string;
  /** Whether `name` is forwarded on user/assistant messages. */
  supportsName?: boolean;
}

function userContent(parts: string | UserContentPart[], provider: string): string | JsonRecord[] {
  if (typeof parts === "string") return parts;
  // Plain text-only content is sent as a string for maximum compatibility.
  if (parts.every((p) => p.type === "text")) return parts.map((p) => p.text).join("");
  return parts.map((p): JsonRecord => {
    if (p.type === "text") return { type: "text", text: p.text };
    const img = resolveImage(p, provider);
    return { type: "image_url", image_url: { url: imageToDataURL(img), ...(p.detail ? { detail: p.detail } : {}) } };
  });
}

function assistantMessage(m: AssistantMessage, hooks: MessageTranslationHooks): JsonRecord {
  const text = textOf(m.content);
  const out: JsonRecord = { role: "assistant" };
  const mapId = hooks.mapToolCallId ?? ((id: string) => id);
  if (m.toolCalls && m.toolCalls.length > 0) {
    out.tool_calls = m.toolCalls.map((c) => ({
      id: mapId(c.id),
      type: "function",
      function: { name: c.function.name, arguments: c.function.arguments },
    }));
    // OpenAI expects null (not "") content alongside tool calls when there is no text.
    out.content = text === "" ? null : text;
  } else {
    out.content = text;
  }
  if (typeof m.content !== "string" && hooks.assistantReasoningFields) {
    const reasoning = m.content.filter((p): p is ReasoningPart => p.type === "reasoning");
    if (reasoning.length > 0) Object.assign(out, hooks.assistantReasoningFields(reasoning));
  }
  if (m.name && hooks.supportsName) out.name = m.name;
  return out;
}

export function toOpenAIMessages(messages: readonly ChatMessage[], hooks: MessageTranslationHooks): JsonRecord[] {
  const mapId = hooks.mapToolCallId ?? ((id: string) => id);
  return messages.map((m): JsonRecord => {
    switch (m.role) {
      case "system":
        return { role: "system", content: textOf(m.content) };
      case "user":
        return {
          role: "user",
          content: userContent(m.content, hooks.provider),
          ...(m.name && hooks.supportsName ? { name: m.name } : {}),
        };
      case "assistant":
        return assistantMessage(m, hooks);
      case "tool": {
        if (typeof m.content !== "string" && m.content.some((p) => p.type === "image")) {
          // Chat Completions tool messages only carry text.
          throw new UnsupportedCapabilityError("vision", `${hooks.provider}: images in tool results are not supported.`, {
            provider: hooks.provider,
          });
        }
        return { role: "tool", tool_call_id: mapId(m.toolCallId), content: textOf(m.content) };
      }
    }
  });
}

export function toOpenAITools(tools: readonly ToolDefinition[]): JsonRecord[] {
  return tools.map((t) => ({
    type: "function",
    function: {
      name: t.function.name,
      ...(t.function.description ? { description: t.function.description } : {}),
      parameters: t.function.parameters,
      ...(t.function.strict !== undefined ? { strict: t.function.strict } : {}),
    },
  }));
}

export function toOpenAIToolChoice(choice: ToolChoice, required: "required" | "any"): unknown {
  if (choice === "required") return required;
  if (choice === "auto" || choice === "none") return choice;
  return { type: "function", function: { name: choice.name } };
}

export function toOpenAIResponseFormat(format: ResponseFormat): JsonRecord | undefined {
  switch (format.type) {
    case "text":
      return undefined;
    case "json_object":
      return { type: "json_object" };
    case "json_schema":
      return {
        type: "json_schema",
        json_schema: {
          name: format.name,
          ...(format.description ? { description: format.description } : {}),
          schema: format.schema,
          ...(format.strict !== undefined ? { strict: format.strict } : {}),
        },
      };
  }
}
