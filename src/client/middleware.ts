import type { ProviderCapabilities } from "../core/capabilities.js";
import type { ChatRequest, ChatResponse, StreamEvent } from "../core/types.js";

/**
 * Per-call context passed through the middleware chain.
 *
 * Middleware may replace `ctx.request` (e.g. to inject a system prompt)
 * before calling `next()`; the provider receives whatever `ctx.request` holds
 * when the innermost `next()` runs. Capability checks happen there too, so
 * rewritten requests are validated.
 */
export interface MiddlewareContext {
  readonly operation: "chat" | "stream";
  readonly provider: string;
  /** Model the request will use (request.model or the client default). */
  readonly model: string;
  request: ChatRequest;
  readonly capabilities: ProviderCapabilities;
  /** Scratch space shared by all middleware for this one call. */
  readonly state: Map<string, unknown>;
}

export type ChatNext = () => Promise<ChatResponse>;
export type StreamNext = () => AsyncIterable<StreamEvent>;

export type ChatMiddleware = (ctx: MiddlewareContext, next: ChatNext) => Promise<ChatResponse>;
export type StreamMiddleware = (ctx: MiddlewareContext, next: StreamNext) => AsyncIterable<StreamEvent>;

/**
 * A plain function wraps `chat()` calls. Use the object form to also (or
 * only) wrap streams:
 *
 * ```ts
 * client.use({
 *   name: "timing",
 *   chat: async (ctx, next) => { ... return next(); },
 *   stream: async function* (ctx, next) { yield* next(); },
 * });
 * ```
 */
export type Middleware =
  | ChatMiddleware
  | {
      name?: string;
      chat?: ChatMiddleware;
      stream?: StreamMiddleware;
    };

export function chatMiddlewareOf(m: Middleware): ChatMiddleware | undefined {
  return typeof m === "function" ? m : m.chat;
}

export function streamMiddlewareOf(m: Middleware): StreamMiddleware | undefined {
  return typeof m === "function" ? undefined : m.stream;
}

/** Composes middleware so the first registered is the outermost. */
export function composeChat(middleware: readonly ChatMiddleware[], ctx: MiddlewareContext, final: ChatNext): Promise<ChatResponse> {
  const dispatch = (i: number): Promise<ChatResponse> => {
    const mw = middleware[i];
    if (!mw) return final();
    let called = false;
    return mw(ctx, () => {
      if (called) return Promise.reject(new Error("next() called multiple times in chat middleware"));
      called = true;
      return dispatch(i + 1);
    });
  };
  return dispatch(0);
}

export function composeStream(middleware: readonly StreamMiddleware[], ctx: MiddlewareContext, final: StreamNext): AsyncIterable<StreamEvent> {
  const dispatch = (i: number): AsyncIterable<StreamEvent> => {
    const mw = middleware[i];
    return mw ? mw(ctx, () => dispatch(i + 1)) : final();
  };
  return dispatch(0);
}
