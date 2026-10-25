import type { ProviderCapabilities } from "../core/capabilities.js";
import {
  AbortError,
  AuthenticationError,
  BudgetExceededError,
  ContextLengthExceededError,
  FallbackExhaustedError,
  ModelNotFoundError,
  NetworkError,
  ProviderError,
  RateLimitError,
  TimeoutError,
  UnsupportedCapabilityError,
} from "../core/errors.js";
import type { ChatRequest, ChatResponse, StreamEvent } from "../core/types.js";
import type { LLMProvider } from "../providers/base.js";
import type { ProviderRegistry } from "../registry/ProviderRegistry.js";
import {
  assertNoModelOverride,
  canServe,
  forTarget,
  toNamedClients,
  unionCapabilities,
  type NamedClient,
  type Target,
} from "./targets.js";

export interface FallbackAttempt {
  target: string;
  error: unknown;
  /** Index of the target that failed. */
  index: number;
  operation: "chat" | "stream";
}

export interface FallbackOptions {
  /** Targets tried in order. Array entries are named `provider:model`; use a record for custom names. */
  targets: Record<string, Target> | Target[];
  /** Decides whether an error moves on to the next target. See `defaultShouldFallback`. */
  shouldFallback?: (error: unknown, attempt: FallbackAttempt) => boolean;
  /** Skip targets whose capabilities cannot serve the request. Default true. */
  skipIncapable?: boolean;
  /** Observability hook, called before moving to the next target. */
  onFallback?: (attempt: FallbackAttempt) => void;
  registry?: ProviderRegistry;
  /** Name reported by this provider. Default "fallback". */
  name?: string;
}

/**
 * Falls back on failures another target might not share: rate limits and
 * quota, outages, network/timeouts, bad credentials or unknown models on
 * that target, context length (a larger-context target may fit), capability
 * gaps, and local budgets. It never falls back on caller aborts, content
 * policy rejections or generally invalid requests, which would fail
 * everywhere.
 */
export function defaultShouldFallback(error: unknown): boolean {
  if (error instanceof AbortError) return false;
  return (
    error instanceof RateLimitError ||
    error instanceof ProviderError ||
    error instanceof NetworkError ||
    error instanceof TimeoutError ||
    error instanceof AuthenticationError ||
    error instanceof ModelNotFoundError ||
    error instanceof ContextLengthExceededError ||
    error instanceof UnsupportedCapabilityError ||
    error instanceof BudgetExceededError
  );
}

/**
 * An LLMProvider that tries targets in order until one succeeds.
 *
 * Streams fall back only if a target fails *before* its first event; once
 * output has been delivered, a later failure is rethrown (the partial output
 * cannot be un-sent). The response's `provider`/`model` identify the target
 * that actually served it.
 *
 * ```ts
 * const llm = new LLMClient(new FallbackProvider({ targets: [primaryConfig, backupConfig] }));
 * ```
 */
export class FallbackProvider implements LLMProvider {
  readonly name: string;
  readonly defaultModel: string;
  readonly targets: readonly NamedClient[];
  private readonly options: FallbackOptions;

  constructor(options: FallbackOptions) {
    this.options = options;
    this.targets = toNamedClients(options.targets, options.registry);
    this.name = options.name ?? "fallback";
    this.defaultModel = this.targets[0]!.client.model;
  }

  capabilities(): ProviderCapabilities {
    return unionCapabilities(this.targets);
  }

  async chat(request: ChatRequest): Promise<ChatResponse> {
    assertNoModelOverride(this.name, this.defaultModel, request);
    const errors: { target: string; error: unknown }[] = [];
    for (const [index, { name, client }] of this.targets.entries()) {
      if (!this.eligible(client, request, "chat", name, errors)) continue;
      try {
        return await client.chat(forTarget(request));
      } catch (error) {
        this.handleFailure(error, { target: name, error, index, operation: "chat" }, errors);
      }
    }
    throw new FallbackExhaustedError(errors, { provider: this.name });
  }

  async *stream(request: ChatRequest): AsyncGenerator<StreamEvent> {
    assertNoModelOverride(this.name, this.defaultModel, request);
    const errors: { target: string; error: unknown }[] = [];
    for (const [index, { name, client }] of this.targets.entries()) {
      if (!this.eligible(client, request, "stream", name, errors)) continue;
      const iterator = client.stream(forTarget(request))[Symbol.asyncIterator]();
      let first: IteratorResult<StreamEvent>;
      try {
        first = await iterator.next();
      } catch (error) {
        this.handleFailure(error, { target: name, error, index, operation: "stream" }, errors);
        continue;
      }
      // Committed to this target from here on: later errors are rethrown.
      if (first.done) return;
      let completed = false;
      try {
        yield first.value;
        for (let r = await iterator.next(); !r.done; r = await iterator.next()) yield r.value;
        completed = true;
      } finally {
        // Consumer stopped early: close the target stream (and its connection).
        if (!completed) await iterator.return?.(undefined);
      }
      return;
    }
    throw new FallbackExhaustedError(errors, { provider: this.name });
  }

  private eligible(
    client: NamedClient["client"],
    request: ChatRequest,
    operation: "chat" | "stream",
    name: string,
    errors: { target: string; error: unknown }[],
  ): boolean {
    if (this.options.skipIncapable === false || canServe(client, request, operation)) return true;
    errors.push({
      target: name,
      error: new UnsupportedCapabilityError("request", `${name}: cannot serve this request (capabilities)`, {
        provider: client.providerName,
      }),
    });
    return false;
  }

  private handleFailure(error: unknown, attempt: FallbackAttempt, errors: { target: string; error: unknown }[]): void {
    const decide = this.options.shouldFallback ?? defaultShouldFallback;
    // Aborts always stop the chain, whatever the custom policy says.
    if (error instanceof AbortError || !decide(error, attempt)) throw error;
    errors.push({ target: attempt.target, error });
    this.options.onFallback?.(attempt);
  }
}
