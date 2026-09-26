/**
 * Unified request ↔ Gemini `generateContent` API.
 *
 * Notable differences from the unified model:
 * - Roles are `user` / `model`; system messages become `systemInstruction`.
 * - Tool calls are `functionCall` parts with an object `args`; results are
 *   `functionResponse` parts keyed by function *name* (plus `id` when Gemini
 *   issued one) and must be a JSON object.
 * - Gemini may omit tool call ids; the adapter generates local ids so the
 *   unified API always has one, and only sends ids back that Gemini issued.
 * - Gemini 3 attaches `thoughtSignature` to function-call parts; it MUST be
 *   replayed with the call in later turns, so it is kept in
 *   `ToolCall.providerData.gemini`.
 * - Thought summaries are parts flagged `thought: true`.
 */
import type {
  AssistantMessage,
  ChatMessage,
  ContentPart,
  FinishReason,
  TokenUsage,
  ToolCall,
  ToolChoice,
  ToolDefinition,
  UserContentPart,
} from "../../core/types.js";
import { InvalidRequestError } from "../../core/errors.js";
import { asArray, getNumber, getRecord, getString, isRecord, tryParseJson, type JsonRecord } from "../../utils/json.js";
import { finalizeUsage, generateId, resolveImage, textOf } from "../../utils/normalize.js";

export const PROVIDER = "gemini";

function imagePart(part: Extract<UserContentPart, { type: "image" }>): JsonRecord {
  const img = resolveImage(part, PROVIDER);
  if (img.kind === "base64") return { inlineData: { mimeType: img.mimeType, data: img.value } };
  if (!img.mimeType) {
    throw new InvalidRequestError(`${PROVIDER}: image URLs require "mimeType" (sent as fileData).`, { provider: PROVIDER });
  }
  // fileData accepts Gemini Files API URIs and other URIs the API supports.
  return { fileData: { mimeType: img.mimeType, fileUri: img.value } };
}

function userParts(content: string | UserContentPart[]): JsonRecord[] {
  if (typeof content === "string") return [{ text: content }];
  return content.map((p) => (p.type === "text" ? { text: p.text } : imagePart(p)));
}

function geminiData(call: { providerData?: Record<string, Record<string, unknown>> }): Record<string, unknown> | undefined {
  return call.providerData?.[PROVIDER];
}

function modelParts(m: AssistantMessage): JsonRecord[] {
  const parts: JsonRecord[] = [];
  const content: ContentPart[] = typeof m.content === "string" ? [{ type: "text", text: m.content }] : m.content;
  for (const p of content) {
    if (p.type === "text") {
      if (p.text !== "") parts.push({ text: p.text });
    } else if (p.type === "reasoning") {
      // Only Gemini-originated thoughts carrying a signature are replayed.
      const sig = getString(p.providerData?.[PROVIDER], "thoughtSignature");
      if (sig) parts.push({ text: p.text, thought: true, thoughtSignature: sig });
    }
  }
  for (const call of m.toolCalls ?? []) {
    const data = geminiData(call);
    const args = call.function.arguments.trim() === "" ? {} : tryParseJson(call.function.arguments);
    if (!isRecord(args)) {
      throw new InvalidRequestError(
        `${PROVIDER}: tool call "${call.function.name}" (${call.id}) arguments must be a JSON object.`,
        { provider: PROVIDER },
      );
    }
    parts.push({
      functionCall: { name: call.function.name, args, ...(data?.nativeId === true ? { id: call.id } : {}) },
      ...(typeof data?.thoughtSignature === "string" ? { thoughtSignature: data.thoughtSignature } : {}),
    });
  }
  return parts;
}

function functionResponseBody(content: string, isError: boolean | undefined): JsonRecord {
  const parsed = tryParseJson(content);
  if (isError) return { error: parsed ?? content };
  if (isRecord(parsed)) return parsed;
  return { result: parsed ?? content };
}

export function toGeminiContents(messages: readonly ChatMessage[]): { systemInstruction?: JsonRecord; contents: JsonRecord[] } {
  const system: string[] = [];
  const contents: { role: "user" | "model"; parts: JsonRecord[] }[] = [];
  const calls = new Map<string, ToolCall>();
  for (const m of messages) if (m.role === "assistant") for (const c of m.toolCalls ?? []) calls.set(c.id, c);

  const push = (role: "user" | "model", parts: JsonRecord[]) => {
    const last = contents[contents.length - 1];
    // Parallel function responses must share one turn; merging same-role turns covers that.
    if (last && last.role === role) last.parts.push(...parts);
    else contents.push({ role, parts });
  };

  for (const m of messages) {
    switch (m.role) {
      case "system":
        system.push(textOf(m.content));
        break;
      case "user":
        push("user", userParts(m.content));
        break;
      case "assistant": {
        const parts = modelParts(m);
        if (parts.length > 0) push("model", parts);
        break;
      }
      case "tool": {
        const call = calls.get(m.toolCallId);
        const name = m.name ?? call?.function.name;
        if (!name) {
          throw new InvalidRequestError(
            `${PROVIDER}: tool result "${m.toolCallId}" has no matching tool call in the history; set "name" on the tool message.`,
            { provider: PROVIDER },
          );
        }
        const nativeId = call && geminiData(call)?.nativeId === true;
        push("user", [
          {
            functionResponse: {
              name,
              ...(nativeId ? { id: m.toolCallId } : {}),
              response: functionResponseBody(textOf(m.content), m.isError),
            },
          },
        ]);
        break;
      }
    }
  }
  const text = system.filter((s) => s !== "").join("\n\n");
  return { ...(text ? { systemInstruction: { parts: [{ text }] } } : {}), contents };
}

export function toGeminiTools(tools: readonly ToolDefinition[]): JsonRecord[] {
  return [
    {
      functionDeclarations: tools.map((t) => ({
        name: t.function.name,
        ...(t.function.description ? { description: t.function.description } : {}),
        // parametersJsonSchema accepts standard JSON Schema (vs. the OpenAPI subset of `parameters`).
        parametersJsonSchema: t.function.parameters,
      })),
    },
  ];
}

export function toGeminiToolConfig(choice: ToolChoice | undefined): JsonRecord | undefined {
  if (!choice) return undefined;
  if (choice === "auto") return { functionCallingConfig: { mode: "AUTO" } };
  if (choice === "none") return { functionCallingConfig: { mode: "NONE" } };
  if (choice === "required") return { functionCallingConfig: { mode: "ANY" } };
  return { functionCallingConfig: { mode: "ANY", allowedFunctionNames: [choice.name] } };
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

export function normalizeGeminiFinishReason(reason: string | undefined): FinishReason {
  switch (reason) {
    case "STOP":
      return "stop";
    case "MAX_TOKENS":
      return "length";
    case "SAFETY":
    case "RECITATION":
    case "BLOCKLIST":
    case "PROHIBITED_CONTENT":
    case "SPII":
    case "IMAGE_SAFETY":
      return "content_filter";
    case "MALFORMED_FUNCTION_CALL":
    case "UNEXPECTED_TOOL_CALL":
    case "TOO_MANY_TOOL_CALLS":
      return "error";
    default:
      return "unknown";
  }
}

/** Gemini reports thoughts separately from candidates; outputTokens includes both (unified convention). */
export function parseGeminiUsage(meta: unknown): TokenUsage | undefined {
  if (!isRecord(meta)) return undefined;
  const candidates = getNumber(meta, "candidatesTokenCount");
  const thoughts = getNumber(meta, "thoughtsTokenCount");
  const output = candidates === undefined && thoughts === undefined ? undefined : (candidates ?? 0) + (thoughts ?? 0);
  return finalizeUsage({
    inputTokens: getNumber(meta, "promptTokenCount"),
    outputTokens: output,
    totalTokens: getNumber(meta, "totalTokenCount"),
    cachedInputTokens: getNumber(meta, "cachedContentTokenCount"),
    reasoningTokens: thoughts,
  });
}

export interface ParsedPart {
  kind: "text" | "thought" | "call" | "other";
  text?: string;
  signature?: string;
  call?: ToolCall;
}

export function parseGeminiPart(part: unknown): ParsedPart {
  if (!isRecord(part)) return { kind: "other" };
  const signature = getString(part, "thoughtSignature");
  const fc = getRecord(part, "functionCall");
  if (fc) {
    const nativeId = getString(fc, "id");
    const data: Record<string, unknown> = { nativeId: nativeId !== undefined };
    if (signature) data.thoughtSignature = signature;
    return {
      kind: "call",
      call: {
        id: nativeId ?? generateId("call"),
        type: "function",
        function: { name: getString(fc, "name") ?? "", arguments: JSON.stringify(getRecord(fc, "args") ?? {}) },
        providerData: { [PROVIDER]: data },
      },
    };
  }
  const text = getString(part, "text");
  if (part.thought === true) return { kind: "thought", text: text ?? "", signature };
  if (text !== undefined) return { kind: "text", text, signature };
  return { kind: "other", signature };
}

export function firstCandidate(data: unknown): JsonRecord | undefined {
  const c = asArray(isRecord(data) ? data.candidates : undefined)[0];
  return isRecord(c) ? c : undefined;
}

/** Strips a leading "models/" so both "gemini-x" and "models/gemini-x" work. */
export function modelPath(model: string): string {
  return `models/${model.replace(/^models\//, "")}`;
}
