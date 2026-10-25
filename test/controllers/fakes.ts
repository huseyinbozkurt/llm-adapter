import { ALL_CAPABILITIES, type ChatRequest, type ChatResponse, type LLMProvider, type ProviderCapabilities, type StreamEvent, type TokenUsage } from "../../src/index.js";

export type Behaviour = { error: unknown } | { text: string; usage?: TokenUsage };

/**
 * Scriptable provider: each call consumes the next behaviour (the last one
 * repeats). Streams can fail before the first event or after `failAfter` events.
 */
export class FakeProvider implements LLMProvider {
  calls: ChatRequest[] = [];
  private i = 0;
  constructor(
    readonly name: string,
    readonly defaultModel: string,
    private readonly behaviours: Behaviour[],
    private readonly caps: Partial<ProviderCapabilities> = {},
    private readonly streamOpts: { failAfter?: number; delayMs?: number } = {},
  ) {}

  capabilities(): ProviderCapabilities {
    return { ...ALL_CAPABILITIES, ...this.caps };
  }

  private next(request: ChatRequest): Behaviour {
    this.calls.push(request);
    return this.behaviours[Math.min(this.i++, this.behaviours.length - 1)]!;
  }

  async chat(request: ChatRequest): Promise<ChatResponse> {
    const b = this.next(request);
    if (this.streamOpts.delayMs) await new Promise((r) => setTimeout(r, this.streamOpts.delayMs));
    if ("error" in b) throw b.error;
    return {
      provider: this.name,
      model: this.defaultModel,
      message: { role: "assistant", content: [{ type: "text", text: b.text }] },
      text: b.text,
      toolCalls: [],
      finishReason: "stop",
      usage: b.usage,
    };
  }

  async *stream(request: ChatRequest): AsyncGenerator<StreamEvent> {
    const b = this.next(request);
    if (this.streamOpts.delayMs) await new Promise((r) => setTimeout(r, this.streamOpts.delayMs));
    if ("error" in b && this.streamOpts.failAfter === undefined) throw b.error;
    yield { type: "start", model: this.defaultModel };
    const text = "error" in b ? "partial" : b.text;
    let n = 1;
    for (const ch of text) {
      if ("error" in b && n++ >= (this.streamOpts.failAfter ?? 0)) throw b.error;
      yield { type: "text_delta", text: ch };
    }
    if (!("error" in b) && b.usage) yield { type: "usage", usage: b.usage };
    yield { type: "finish", reason: "stop" };
  }
}
