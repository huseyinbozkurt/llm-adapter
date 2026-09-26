import type { ChatResponse, ContentPart, FinishReason, StreamEvent, TokenUsage, ToolCall } from "../core/types.js";
import { appendReasoning, appendText, finalizeUsage, reasoningOf, textOf } from "./normalize.js";

/**
 * Folds normalised stream events into a ChatResponse. Used by ChatStream to
 * offer `finalResponse()`, and handy for custom providers and tests.
 */
export class StreamAccumulator {
  private readonly parts: ContentPart[] = [];
  private readonly toolCalls = new Map<number, ToolCall>();
  private reasoningOpen = false;
  private id: string | undefined;
  private model: string | undefined;
  private usage: TokenUsage = {};
  private finishReason: FinishReason = "unknown";
  private rawFinishReason: string | undefined;
  private finished = false;

  constructor(
    private readonly provider: string,
    private readonly defaultModel: string,
  ) {}

  add(event: StreamEvent): void {
    switch (event.type) {
      case "start":
        this.id = event.id ?? this.id;
        this.model = event.model ?? this.model;
        break;
      case "text_delta":
        this.reasoningOpen = false;
        appendText(this.parts, event.text);
        break;
      case "reasoning_delta": {
        const last = this.parts[this.parts.length - 1];
        if (this.reasoningOpen && last?.type === "reasoning") last.text += event.text;
        else this.parts.push({ type: "reasoning", text: event.text });
        this.reasoningOpen = true;
        break;
      }
      case "reasoning_end": {
        const last = this.parts[this.parts.length - 1];
        if (this.reasoningOpen && last?.type === "reasoning") {
          if (event.providerData) last.providerData = event.providerData;
        } else if (event.providerData) {
          // Opaque block with no streamed text (e.g. redacted thinking).
          appendReasoning(this.parts, "", event.providerData);
        }
        this.reasoningOpen = false;
        break;
      }
      case "tool_call_start":
        this.reasoningOpen = false;
        this.toolCalls.set(event.index, {
          id: event.id,
          type: "function",
          function: { name: event.name, arguments: "" },
          ...(event.providerData ? { providerData: event.providerData } : {}),
        });
        break;
      case "tool_call_delta": {
        const call = this.toolCalls.get(event.index);
        if (call) call.function.arguments += event.argumentsDelta;
        break;
      }
      case "tool_call_end":
        break;
      case "usage":
        // Later usage events refine earlier ones (providers report incrementally).
        this.usage = { ...this.usage, ...event.usage };
        break;
      case "finish":
        this.finishReason = event.reason;
        this.rawFinishReason = event.rawReason;
        this.finished = true;
        break;
    }
  }

  get isFinished(): boolean {
    return this.finished;
  }

  result(raw?: unknown): ChatResponse {
    const toolCalls = [...this.toolCalls.entries()].sort(([a], [b]) => a - b).map(([, c]) => c);
    const content = this.parts.map((p) => ({ ...p }));
    const reasoning = reasoningOf(content);
    return {
      id: this.id,
      provider: this.provider,
      model: this.model ?? this.defaultModel,
      message: {
        role: "assistant",
        content,
        ...(toolCalls.length > 0 ? { toolCalls } : {}),
      },
      text: textOf(content),
      ...(reasoning !== undefined ? { reasoning } : {}),
      toolCalls,
      finishReason: this.finishReason,
      rawFinishReason: this.rawFinishReason,
      usage: finalizeUsage(this.usage),
      raw,
    };
  }
}
