import type { Middleware, MiddlewareContext } from "../client/middleware.js";
import { AbortError, ConfigurationError, RateLimitError } from "../core/errors.js";
import { sleep } from "../utils/retry.js";

export interface RateLimiterOptions {
  /** At most `limit` requests start per `intervalMs` (sliding window). */
  requests?: { limit: number; intervalMs: number };
  /** At most this many requests in flight (streams count until they end). */
  maxConcurrent?: number;
  /** Fail with RateLimitError instead of waiting longer than this (ms). Default: wait indefinitely. */
  maxWaitMs?: number;
}

/**
 * Client-side limiter: a sliding-window request limit plus a concurrency
 * cap, with FIFO fairness. Waiting honours the request's AbortSignal.
 */
export class RateLimiter {
  private readonly starts: number[] = [];
  private active = 0;
  private queue: Promise<void> = Promise.resolve();
  private releaseWaiters: Array<() => void> = [];

  constructor(private readonly options: RateLimiterOptions) {
    const r = options.requests;
    if (r && !(r.limit >= 1 && r.intervalMs > 0)) {
      throw new ConfigurationError("RateLimiter: requests.limit must be >= 1 and requests.intervalMs > 0.");
    }
    if (options.maxConcurrent !== undefined && !(options.maxConcurrent >= 1)) {
      throw new ConfigurationError("RateLimiter: maxConcurrent must be >= 1.");
    }
  }

  /** Current in-flight count (for metrics). */
  get inFlight(): number {
    return this.active;
  }

  /**
   * Waits for a slot and returns its release function. Callers must call it
   * exactly once when the request finishes.
   */
  async acquire(signal?: AbortSignal, label = "rate limiter"): Promise<() => void> {
    const deadline = this.options.maxWaitMs === undefined ? Infinity : Date.now() + this.options.maxWaitMs;
    // Serialise acquisition so waiters are served in arrival order.
    const turn = this.queue.then(() => this.waitForSlot(signal, deadline, label));
    this.queue = turn.then(
      () => undefined,
      () => undefined,
    );
    await turn;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.active--;
      const waiters = this.releaseWaiters;
      this.releaseWaiters = [];
      waiters.forEach((w) => w());
    };
  }

  private async waitForSlot(signal: AbortSignal | undefined, deadline: number, label: string): Promise<void> {
    for (;;) {
      if (signal?.aborted) throw aborted(label, signal);
      const now = Date.now();
      const r = this.options.requests;
      if (r) while (this.starts.length > 0 && this.starts[0]! <= now - r.intervalMs) this.starts.shift();

      const windowWait = r && this.starts.length >= r.limit ? this.starts[0]! + r.intervalMs - now : 0;
      const concurrencyFull = this.options.maxConcurrent !== undefined && this.active >= this.options.maxConcurrent;
      if (windowWait === 0 && !concurrencyFull) {
        this.active++;
        if (r) this.starts.push(now);
        return;
      }

      const remaining = deadline - now;
      // Fail fast when the window cannot open before the deadline.
      if (remaining <= 0 || windowWait > remaining) throw this.exhausted(label, windowWait);
      if (concurrencyFull) {
        // Wake on a release, or when the window opens / the deadline passes.
        const cap = Math.min(remaining, windowWait > 0 ? windowWait : Infinity);
        await this.waitForRelease(signal, Number.isFinite(cap) ? cap : undefined, label);
      } else {
        await sleep(Math.max(1, windowWait), signal, label);
      }
    }
  }

  private waitForRelease(signal: AbortSignal | undefined, timeoutMs: number | undefined, label: string): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      let timer: ReturnType<typeof setTimeout> | undefined;
      const cleanup = () => {
        if (timer) clearTimeout(timer);
        signal?.removeEventListener("abort", onAbort);
        this.releaseWaiters = this.releaseWaiters.filter((w) => w !== done);
      };
      const done = () => {
        cleanup();
        resolve();
      };
      const onAbort = () => {
        cleanup();
        reject(aborted(label, signal));
      };
      this.releaseWaiters.push(done);
      if (timeoutMs !== undefined) timer = setTimeout(done, Math.max(1, timeoutMs));
      signal?.addEventListener("abort", onAbort, { once: true });
    });
  }

  private exhausted(label: string, windowWait: number): RateLimitError {
    return new RateLimitError(`${label}: local rate limit reached (waited longer than maxWaitMs).`, {
      provider: label,
      code: "local_rate_limit",
      retryable: true,
      retryAfterMs: Number.isFinite(windowWait) ? Math.max(0, windowWait) : undefined,
    });
  }
}

export interface RateLimitMiddlewareOptions extends RateLimiterOptions {
  /**
   * Partition key, e.g. `(ctx) => String(ctx.request.metadata?.userId)`.
   * Each key gets its own limiter. Default: one limiter for all calls.
   */
  key?: (ctx: MiddlewareContext) => string;
}

/**
 * Middleware that applies RateLimiter to chat and stream calls. Install it
 * on a shared client (or reuse one instance across clients) to share limits.
 */
export function rateLimit(options: RateLimitMiddlewareOptions): Middleware & { limiterFor(key: string): RateLimiter } {
  const limiters = new Map<string, RateLimiter>();
  const limiterFor = (key: string): RateLimiter => {
    let l = limiters.get(key);
    if (!l) {
      l = new RateLimiter(options);
      limiters.set(key, l);
    }
    return l;
  };
  // Validate eagerly so misconfiguration fails at setup, not on first request.
  limiterFor("");
  const keyOf = (ctx: MiddlewareContext) => (options.key ? options.key(ctx) : "");

  return {
    name: "rateLimit",
    limiterFor,
    chat: async (ctx, next) => {
      const release = await limiterFor(keyOf(ctx)).acquire(ctx.request.signal, ctx.provider);
      try {
        return await next();
      } finally {
        release();
      }
    },
    stream: async function* (ctx, next) {
      const release = await limiterFor(keyOf(ctx)).acquire(ctx.request.signal, ctx.provider);
      try {
        yield* next();
      } finally {
        release();
      }
    },
  };
}

function aborted(label: string, signal: AbortSignal | undefined): AbortError {
  return new AbortError(`${label}: aborted while waiting for the rate limiter`, { provider: label, cause: signal?.reason });
}
