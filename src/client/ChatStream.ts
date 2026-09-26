import type { ChatResponse, StreamEvent } from "../core/types.js";
import { LLMError } from "../core/errors.js";
import { StreamAccumulator } from "../utils/streaming.js";

/**
 * The value returned by `LLMClient.stream()`.
 *
 * - Iterate it (once) for normalised events.
 * - `finalResponse()` resolves to the assembled ChatResponse; if the stream
 *   has not been iterated it is consumed internally.
 * - `textStream()` yields only text deltas.
 *
 * The request starts lazily, on first iteration or `finalResponse()`.
 */
export class ChatStream implements AsyncIterable<StreamEvent> {
  private readonly accumulator: StreamAccumulator;
  private consumed = false;
  private done: Promise<ChatResponse>;
  private resolveDone!: (r: ChatResponse) => void;
  private rejectDone!: (e: unknown) => void;

  constructor(
    private readonly source: () => AsyncIterable<StreamEvent>,
    provider: string,
    model: string,
  ) {
    this.accumulator = new StreamAccumulator(provider, model);
    this.done = new Promise<ChatResponse>((resolve, reject) => {
      this.resolveDone = resolve;
      this.rejectDone = reject;
    });
    // Avoid unhandled-rejection noise when only iteration is used.
    this.done.catch(() => undefined);
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<StreamEvent> {
    if (this.consumed) throw new LLMError("A ChatStream can only be iterated once; use finalResponse() for the result.");
    this.consumed = true;
    let completed = false;
    try {
      for await (const event of this.source()) {
        this.accumulator.add(event);
        yield event;
      }
      completed = true;
      this.resolveDone(this.accumulator.result());
    } catch (error) {
      this.rejectDone(error);
      throw error;
    } finally {
      if (!completed) this.rejectDone(new LLMError("Stream was not consumed to completion."));
    }
  }

  async finalResponse(): Promise<ChatResponse> {
    if (!this.consumed) {
      for await (const _ of this) {
        // drain
      }
    }
    return this.done;
  }

  async *textStream(): AsyncGenerator<string> {
    for await (const event of this) if (event.type === "text_delta") yield event.text;
  }
}
