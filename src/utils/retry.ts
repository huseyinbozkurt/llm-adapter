import type { RetryOptions } from "../core/config.js";
import { AbortError, LLMError } from "../core/errors.js";

export type ResolvedRetryOptions = Required<Omit<RetryOptions, "shouldRetry">> & Pick<RetryOptions, "shouldRetry">;

/**
 * Parses `Retry-After` (seconds or HTTP date) and the non-standard
 * `retry-after-ms` header. Returns milliseconds, or undefined.
 */
export function parseRetryAfter(headers: Headers, now: number = Date.now()): number | undefined {
  const ms = headers.get("retry-after-ms");
  if (ms !== null) {
    const n = Number(ms);
    if (Number.isFinite(n) && n >= 0) return n;
  }
  const value = headers.get("retry-after");
  if (value === null) return undefined;
  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) return seconds * 1000;
  const date = Date.parse(value);
  if (!Number.isNaN(date)) return Math.max(0, date - now);
  return undefined;
}

/** Exponential backoff with optional full jitter; Retry-After wins when present. */
export function computeDelay(
  attempt: number,
  options: ResolvedRetryOptions,
  retryAfterMs?: number,
  random: () => number = Math.random,
): number {
  if (retryAfterMs !== undefined) return Math.min(retryAfterMs, options.maxDelayMs);
  const exp = Math.min(options.initialDelayMs * options.multiplier ** attempt, options.maxDelayMs);
  return options.jitter ? Math.floor(random() * exp) : exp;
}

export function isRetryable(error: unknown, attempt: number, options: ResolvedRetryOptions): boolean {
  if (error instanceof AbortError) return false;
  if (options.shouldRetry) return options.shouldRetry(error, attempt);
  return error instanceof LLMError && error.retryable;
}

/** Resolves after `ms`, or rejects with AbortError when the signal fires. */
export function sleep(ms: number, signal?: AbortSignal, provider?: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new AbortError(`${provider ?? "request"}: aborted`, { provider, cause: signal.reason }));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new AbortError(`${provider ?? "request"}: aborted`, { provider, cause: signal?.reason }));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/**
 * Runs `fn` with retries. `fn` receives the zero-based attempt number.
 * Only errors judged retryable are retried; the last error is rethrown.
 */
export async function withRetry<T>(
  fn: (attempt: number) => Promise<T>,
  options: ResolvedRetryOptions,
  context: { signal?: AbortSignal; provider?: string; onRetry?: (error: unknown, attempt: number, delayMs: number) => void } = {},
): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await fn(attempt);
    } catch (error) {
      if (attempt >= options.maxRetries || !isRetryable(error, attempt, options)) throw error;
      const retryAfter = error instanceof LLMError ? error.retryAfterMs : undefined;
      const delay = computeDelay(attempt, options, retryAfter);
      context.onRetry?.(error, attempt, delay);
      await sleep(delay, context.signal, context.provider);
    }
  }
}
