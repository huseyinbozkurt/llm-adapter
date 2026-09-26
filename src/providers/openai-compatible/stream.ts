/**
 * Chat Completions streaming chunks → normalised StreamEvents.
 *
 * Tool calls arrive as fragments keyed by `index`: the first fragment usually
 * carries `id` and `function.name`, later ones append to
 * `function.arguments`. Some compatible servers send the whole call in one
 * fragment, omit `index`, or omit `id`; all of these are handled here.
 */
import type { FinishReason, StreamEvent, TokenUsage } from "../../core/types.js";
import { errorFromStreamPayload } from "../../core/errors.js";
import { asArray, getNumber, getRecord, getString, isRecord, type JsonRecord } from "../../utils/json.js";
import { generateId } from "../../utils/normalize.js";

export interface ChunkHooks {
  provider: string;
  /** Text and reasoning carried by a `delta` object. */
  deltaContent(delta: JsonRecord): { text?: string; reasoning?: string };
  usage(chunk: JsonRecord): TokenUsage | undefined;
  finishReason(raw: string): FinishReason;
}

interface PendingToolCall {
  index: number;
  id?: string;
  name?: string;
  started: boolean;
  bufferedArgs: string;
}

export class OpenAIStreamProcessor {
  private started = false;
  private readonly calls = new Map<string, PendingToolCall>();
  private nextIndex = 0;
  private finishRaw: string | undefined;
  private usage: TokenUsage | undefined;

  constructor(private readonly hooks: ChunkHooks) {}

  process(chunk: unknown): StreamEvent[] {
    if (!isRecord(chunk)) return [];
    const events: StreamEvent[] = [];

    const err = getRecord(chunk, "error");
    if (err) {
      // e.g. OpenRouter reports upstream failures as an in-band error chunk.
      throw errorFromStreamPayload(this.hooks.provider, getString(err, "type") ?? getString(err, "code"), getString(err, "message"));
    }

    if (!this.started) {
      this.started = true;
      events.push({ type: "start", id: getString(chunk, "id"), model: getString(chunk, "model") });
    }

    for (const choice of asArray(chunk.choices)) {
      if (!isRecord(choice)) continue;
      // Only the first choice is surfaced (n > 1 is not part of the unified API).
      if ((getNumber(choice, "index") ?? 0) !== 0) continue;
      const delta = getRecord(choice, "delta");
      if (delta) {
        const { text, reasoning } = this.hooks.deltaContent(delta);
        if (reasoning) events.push({ type: "reasoning_delta", text: reasoning });
        if (text) events.push({ type: "text_delta", text });
        for (const tc of asArray(delta.tool_calls)) events.push(...this.toolCallFragment(tc));
      }
      const finish = getString(choice, "finish_reason");
      if (finish) this.finishRaw = finish;
    }

    const usage = this.hooks.usage(chunk);
    if (usage) this.usage = usage;
    return events;
  }

  /** Flushes open tool calls, usage and the finish event. Call once at end of stream. */
  finish(): StreamEvent[] {
    const events: StreamEvent[] = [];
    for (const call of this.sortedCalls()) {
      if (!call.started) events.push(...this.startCall(call));
      events.push({ type: "tool_call_end", index: call.index });
    }
    if (this.usage) events.push({ type: "usage", usage: this.usage });
    let reason: FinishReason = this.finishRaw ? this.hooks.finishReason(this.finishRaw) : "unknown";
    if (reason === "stop" && this.calls.size > 0) reason = "tool_call";
    events.push({ type: "finish", reason, ...(this.finishRaw ? { rawReason: this.finishRaw } : {}) });
    return events;
  }

  private sortedCalls(): PendingToolCall[] {
    return [...this.calls.values()].sort((a, b) => a.index - b.index);
  }

  private toolCallFragment(raw: unknown): StreamEvent[] {
    if (!isRecord(raw)) return [];
    const fn = getRecord(raw, "function");
    const wireIndex = getNumber(raw, "index");
    const id = getString(raw, "id");
    // Key by wire index when present; otherwise by id; otherwise a new call.
    const key = wireIndex !== undefined ? `i${wireIndex}` : id !== undefined ? `id:${id}` : `n${this.nextIndex}`;
    let call = this.calls.get(key);
    if (!call) {
      call = { index: this.nextIndex++, started: false, bufferedArgs: "" };
      this.calls.set(key, call);
    }
    if (id && !call.id) call.id = id;
    const name = fn ? getString(fn, "name") : undefined;
    if (name && !call.name) call.name = name;

    const events: StreamEvent[] = [];
    if (!call.started && call.name) events.push(...this.startCall(call));
    const args = fn ? getString(fn, "arguments") : undefined;
    if (args) {
      if (call.started) events.push({ type: "tool_call_delta", index: call.index, argumentsDelta: args });
      else call.bufferedArgs += args;
    }
    return events;
  }

  private startCall(call: PendingToolCall): StreamEvent[] {
    call.started = true;
    call.id ??= generateId("call");
    const events: StreamEvent[] = [{ type: "tool_call_start", index: call.index, id: call.id, name: call.name ?? "" }];
    if (call.bufferedArgs) {
      events.push({ type: "tool_call_delta", index: call.index, argumentsDelta: call.bufferedArgs });
      call.bufferedArgs = "";
    }
    return events;
  }
}
