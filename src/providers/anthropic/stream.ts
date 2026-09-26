/**
 * Anthropic SSE events → normalised StreamEvents.
 *
 * Anthropic streams typed content blocks (`content_block_start/delta/stop`)
 * addressed by block index. Tool calls get a separate, dense normalised index
 * (0, 1, 2, ...) regardless of how many text/thinking blocks precede them.
 */
import type { StreamEvent } from "../../core/types.js";
import { errorFromStreamPayload } from "../../core/errors.js";
import { getRecord, getString, isRecord } from "../../utils/json.js";
import { normalizeAnthropicStopReason, parseAnthropicUsage, PROVIDER } from "./translate.js";

type BlockKind = "text" | "thinking" | "tool" | "other";

export class AnthropicStreamProcessor {
  private readonly blocks = new Map<number, { kind: BlockKind; toolIndex?: number; signature?: string; hasArgs?: boolean }>();
  private nextToolIndex = 0;
  private stopReason: string | undefined;
  private finished = false;

  process(eventName: string | undefined, payload: unknown): StreamEvent[] {
    if (!isRecord(payload)) return [];
    const type = getString(payload, "type") ?? eventName;
    switch (type) {
      case "message_start": {
        const message = getRecord(payload, "message");
        this.mergeUsage(message?.usage);
        return [{ type: "start", id: getString(message, "id"), model: getString(message, "model") }];
      }
      case "content_block_start":
        return this.blockStart(payload);
      case "content_block_delta":
        return this.blockDelta(payload);
      case "content_block_stop":
        return this.blockStop(payload);
      case "message_delta": {
        const delta = getRecord(payload, "delta");
        this.stopReason = getString(delta, "stop_reason") ?? this.stopReason;
        this.mergeUsage(payload.usage);
        return [];
      }
      case "message_stop":
        return this.finish();
      case "error": {
        const err = getRecord(payload, "error");
        throw errorFromStreamPayload(PROVIDER, getString(err, "type"), getString(err, "message"));
      }
      default:
        return []; // "ping" and future event types
    }
  }

  /** Emits usage + finish exactly once (on message_stop, or at end of stream). */
  finish(): StreamEvent[] {
    if (this.finished) return [];
    this.finished = true;
    const events: StreamEvent[] = [];
    const usage = parseAnthropicUsage(this.rawUsage);
    if (usage) events.push({ type: "usage", usage });
    events.push({
      type: "finish",
      reason: normalizeAnthropicStopReason(this.stopReason),
      ...(this.stopReason ? { rawReason: this.stopReason } : {}),
    });
    return events;
  }

  // Anthropic sends cumulative counts; later values replace earlier ones.
  private rawUsage: Record<string, unknown> = {};
  private mergeUsage(u: unknown): void {
    if (!isRecord(u)) return;
    for (const [k, v] of Object.entries(u)) if (typeof v === "number") this.rawUsage[k] = v;
  }

  private blockStart(payload: Record<string, unknown>): StreamEvent[] {
    const index = typeof payload.index === "number" ? payload.index : -1;
    const block = getRecord(payload, "content_block");
    switch (getString(block, "type")) {
      case "text": {
        this.blocks.set(index, { kind: "text" });
        const text = getString(block, "text");
        return text ? [{ type: "text_delta", text }] : [];
      }
      case "thinking": {
        this.blocks.set(index, { kind: "thinking" });
        const text = getString(block, "thinking");
        return text ? [{ type: "reasoning_delta", text }] : [];
      }
      case "redacted_thinking":
        this.blocks.set(index, { kind: "other" });
        return [{ type: "reasoning_end", providerData: { [PROVIDER]: { redactedData: getString(block, "data") ?? "" } } }];
      case "tool_use": {
        const toolIndex = this.nextToolIndex++;
        this.blocks.set(index, { kind: "tool", toolIndex });
        return [{ type: "tool_call_start", index: toolIndex, id: getString(block, "id") ?? "", name: getString(block, "name") ?? "" }];
      }
      default:
        // server_tool_use / *_tool_result blocks are not normalised.
        this.blocks.set(index, { kind: "other" });
        return [];
    }
  }

  private blockDelta(payload: Record<string, unknown>): StreamEvent[] {
    const index = typeof payload.index === "number" ? payload.index : -1;
    const block = this.blocks.get(index);
    const delta = getRecord(payload, "delta");
    switch (getString(delta, "type")) {
      case "text_delta": {
        const text = getString(delta, "text");
        return text ? [{ type: "text_delta", text }] : [];
      }
      case "thinking_delta": {
        const text = getString(delta, "thinking");
        return text ? [{ type: "reasoning_delta", text }] : [];
      }
      case "signature_delta":
        if (block) block.signature = (block.signature ?? "") + (getString(delta, "signature") ?? "");
        return [];
      case "input_json_delta": {
        const partial = getString(delta, "partial_json");
        if (!partial || block?.kind !== "tool" || block.toolIndex === undefined) return [];
        block.hasArgs = true;
        return [{ type: "tool_call_delta", index: block.toolIndex, argumentsDelta: partial }];
      }
      default:
        return []; // citations_delta etc.
    }
  }

  private blockStop(payload: Record<string, unknown>): StreamEvent[] {
    const index = typeof payload.index === "number" ? payload.index : -1;
    const block = this.blocks.get(index);
    if (!block) return [];
    if (block.kind === "tool" && block.toolIndex !== undefined) {
      // A tool with no parameters streams no input deltas; match the non-streaming "{}".
      const events: StreamEvent[] = block.hasArgs ? [] : [{ type: "tool_call_delta", index: block.toolIndex, argumentsDelta: "{}" }];
      events.push({ type: "tool_call_end", index: block.toolIndex });
      return events;
    }
    if (block.kind === "thinking") {
      return [{ type: "reasoning_end", providerData: { [PROVIDER]: { signature: block.signature ?? "" } } }];
    }
    return [];
  }
}
