/**
 * Gemini `streamGenerateContent?alt=sse` chunks → normalised StreamEvents.
 *
 * Each SSE chunk is a partial GenerateContentResponse: text parts are
 * incremental, but function calls always arrive whole in a single part, so a
 * call is emitted as start + one delta + end. `usageMetadata` is cumulative;
 * the last value wins.
 */
import type { StreamEvent, TokenUsage } from "../../core/types.js";
import { errorFromStreamPayload } from "../../core/errors.js";
import { asArray, getRecord, getString, isRecord } from "../../utils/json.js";
import { firstCandidate, normalizeGeminiFinishReason, parseGeminiPart, parseGeminiUsage, PROVIDER } from "./translate.js";

export class GeminiStreamProcessor {
  private started = false;
  private toolIndex = 0;
  private inThought = false;
  private finishRaw: string | undefined;
  private blockReason: string | undefined;
  private usage: TokenUsage | undefined;

  process(chunk: unknown): StreamEvent[] {
    if (!isRecord(chunk)) return [];
    const err = getRecord(chunk, "error");
    if (err) throw errorFromStreamPayload(PROVIDER, getString(err, "status"), getString(err, "message"));

    const events: StreamEvent[] = [];
    if (!this.started) {
      this.started = true;
      events.push({ type: "start", id: getString(chunk, "responseId"), model: getString(chunk, "modelVersion") });
    }

    this.blockReason = getString(getRecord(chunk, "promptFeedback"), "blockReason") ?? this.blockReason;
    const candidate = firstCandidate(chunk);
    for (const raw of asArray(getRecord(candidate, "content")?.parts)) {
      const part = parseGeminiPart(raw);
      if (part.kind === "thought") {
        if (part.text) events.push({ type: "reasoning_delta", text: part.text });
        this.inThought = true;
        if (part.signature) events.push(this.closeThought(part.signature));
        continue;
      }
      if (this.inThought) events.push(this.closeThought());
      if (part.kind === "text" && part.text) events.push({ type: "text_delta", text: part.text });
      else if (part.kind === "call" && part.call) {
        const index = this.toolIndex++;
        events.push(
          { type: "tool_call_start", index, id: part.call.id, name: part.call.function.name, providerData: part.call.providerData },
          { type: "tool_call_delta", index, argumentsDelta: part.call.function.arguments },
          { type: "tool_call_end", index },
        );
      }
    }
    this.finishRaw = getString(candidate, "finishReason") ?? this.finishRaw;
    this.usage = parseGeminiUsage(chunk.usageMetadata) ?? this.usage;
    return events;
  }

  finish(): StreamEvent[] {
    const events: StreamEvent[] = [];
    if (this.inThought) events.push(this.closeThought());
    if (this.usage) events.push({ type: "usage", usage: this.usage });
    if (this.blockReason && !this.finishRaw) {
      events.push({ type: "finish", reason: "content_filter", rawReason: this.blockReason });
    } else {
      let reason = normalizeGeminiFinishReason(this.finishRaw);
      // Gemini reports STOP when a turn ends in function calls.
      if (reason === "stop" && this.toolIndex > 0) reason = "tool_call";
      events.push({ type: "finish", reason, ...(this.finishRaw ? { rawReason: this.finishRaw } : {}) });
    }
    return events;
  }

  private closeThought(signature?: string): StreamEvent {
    this.inThought = false;
    return signature
      ? { type: "reasoning_end", providerData: { [PROVIDER]: { thoughtSignature: signature } } }
      : { type: "reasoning_end" };
  }
}
