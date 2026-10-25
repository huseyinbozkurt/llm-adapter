import type { Middleware, MiddlewareContext } from "../client/middleware.js";
import { BudgetExceededError, ConfigurationError } from "../core/errors.js";
import type { StreamEvent, TokenUsage } from "../core/types.js";

/** Per-million-token rates in any currency you choose. No prices ship with the library. */
export interface ModelPrice {
  inputPerMillion: number;
  outputPerMillion: number;
  /** Rate for cache-read input tokens. Defaults to `inputPerMillion`. */
  cachedInputPerMillion?: number;
  /** Rate for cache-write input tokens (Anthropic). Defaults to `inputPerMillion`. */
  cacheWriteInputPerMillion?: number;
}

export type PriceLookup = (provider: string, model: string) => ModelPrice | undefined;

/** Cost of one call. Reasoning tokens are billed as output (they are included in outputTokens). */
export function computeCost(usage: TokenUsage, price: ModelPrice): number {
  const cached = usage.cachedInputTokens ?? 0;
  const written = usage.cacheWriteInputTokens ?? 0;
  const plainInput = Math.max(0, (usage.inputTokens ?? 0) - cached - written);
  return (
    (plainInput * price.inputPerMillion +
      cached * (price.cachedInputPerMillion ?? price.inputPerMillion) +
      written * (price.cacheWriteInputPerMillion ?? price.inputPerMillion) +
      (usage.outputTokens ?? 0) * price.outputPerMillion) /
    1_000_000
  );
}

export interface BudgetLimits {
  maxRequests?: number;
  maxInputTokens?: number;
  maxOutputTokens?: number;
  maxTotalTokens?: number;
  /** Requires `pricing`. */
  maxCost?: number;
}

export interface BudgetSpend {
  requests: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  cost: number;
  /** Requests whose provider reported no usage (not counted in token totals). */
  unmeteredRequests: number;
  /** Start of the current window (epoch ms). */
  windowStart: number;
}

export interface UsageRecord {
  key: string;
  provider: string;
  model: string;
  usage?: TokenUsage;
  cost?: number;
}

export interface BudgetOptions {
  limits: BudgetLimits;
  pricing?: PriceLookup;
  /** Fixed window length; spend resets when it elapses. Default: never resets. */
  windowMs?: number;
  /** Partition key, e.g. per tenant: `(ctx) => String(ctx.request.metadata?.tenantId)`. */
  key?: (ctx: MiddlewareContext) => string;
  /** Called after every metered call (analytics, persistence). */
  onUsage?: (record: UsageRecord) => void;
  /** Clock override for tests. */
  now?: () => number;
}

/**
 * Tracks token / cost spend and blocks new requests once a limit is reached.
 *
 * Limits are checked before each request against spend *so far*, so
 * concurrent in-flight requests can overshoot a limit by their own usage
 * (soft limit). Pricing uses the client's provider/model; when wrapping a
 * FallbackProvider or RoutingProvider, attach the budget middleware to the
 * target clients (one controller can be shared by many clients).
 */
export class BudgetController {
  private readonly spends = new Map<string, BudgetSpend>();
  private readonly now: () => number;

  constructor(private readonly options: BudgetOptions) {
    if (options.limits.maxCost !== undefined && !options.pricing) {
      throw new ConfigurationError("BudgetController: limits.maxCost requires a pricing lookup.");
    }
    for (const [name, value] of Object.entries(options.limits)) {
      if (value !== undefined && !(typeof value === "number" && value >= 0)) {
        throw new ConfigurationError(`BudgetController: limits.${name} must be a non-negative number.`);
      }
    }
    this.now = options.now ?? Date.now;
  }

  spend(key = ""): BudgetSpend {
    return { ...this.current(key) };
  }

  /** Remaining allowance per configured limit (never negative). */
  remaining(key = ""): Partial<Record<keyof BudgetLimits, number>> {
    const s = this.current(key);
    const used: Record<keyof BudgetLimits, number> = {
      maxRequests: s.requests,
      maxInputTokens: s.inputTokens,
      maxOutputTokens: s.outputTokens,
      maxTotalTokens: s.totalTokens,
      maxCost: s.cost,
    };
    const out: Partial<Record<keyof BudgetLimits, number>> = {};
    for (const [name, limit] of Object.entries(this.options.limits) as [keyof BudgetLimits, number | undefined][]) {
      if (limit !== undefined) out[name] = Math.max(0, limit - used[name]);
    }
    return out;
  }

  reset(key?: string): void {
    if (key === undefined) this.spends.clear();
    else this.spends.delete(key);
  }

  /** Throws BudgetExceededError if any limit is already reached for `key`. */
  check(key: string, provider: string, model: string): void {
    if (this.options.limits.maxCost !== undefined && !this.options.pricing?.(provider, model)) {
      throw new ConfigurationError(`BudgetController: no price for ${provider}/${model}; cannot enforce maxCost.`, { provider });
    }
    const exceeded = Object.entries(this.remaining(key)).find(([, left]) => left <= 0);
    if (exceeded) {
      const [limit] = exceeded;
      throw new BudgetExceededError(limit, `Budget limit ${limit} reached${key ? ` for "${key}"` : ""}.`, { provider });
    }
  }

  /** Records one completed call. Used by the middleware; callable directly for custom flows. */
  record(key: string, provider: string, model: string, usage: TokenUsage | undefined): UsageRecord {
    const s = this.current(key);
    s.requests++;
    let cost: number | undefined;
    if (usage) {
      const input = usage.inputTokens ?? 0;
      const output = usage.outputTokens ?? 0;
      s.inputTokens += input;
      s.outputTokens += output;
      s.totalTokens += usage.totalTokens ?? input + output;
      const price = this.options.pricing?.(provider, model);
      if (price) {
        cost = computeCost(usage, price);
        s.cost += cost;
      }
    } else {
      s.unmeteredRequests++;
    }
    const record: UsageRecord = { key, provider, model, usage, cost };
    this.options.onUsage?.(record);
    return record;
  }

  middleware(): Middleware {
    const self = this;
    const keyOf = (ctx: MiddlewareContext) => (this.options.key ? this.options.key(ctx) : "");
    return {
      name: "budget",
      chat: async (ctx, next) => {
        const key = keyOf(ctx);
        self.check(key, ctx.provider, ctx.model);
        const result = await next();
        self.record(key, ctx.provider, ctx.model, result.usage);
        return result;
      },
      stream: async function* (ctx: MiddlewareContext, next: () => AsyncIterable<StreamEvent>) {
        const key = keyOf(ctx);
        self.check(key, ctx.provider, ctx.model);
        let usage: TokenUsage | undefined;
        let started = false;
        try {
          for await (const event of next()) {
            started = true;
            if (event.type === "usage") usage = { ...usage, ...event.usage };
            yield event;
          }
        } finally {
          // Partial streams still consumed tokens; record whatever was reported.
          if (started) self.record(key, ctx.provider, ctx.model, usage);
        }
      },
    };
  }

  private current(key: string): BudgetSpend {
    const now = this.now();
    let s = this.spends.get(key);
    if (!s || (this.options.windowMs !== undefined && now - s.windowStart >= this.options.windowMs)) {
      s = { requests: 0, inputTokens: 0, outputTokens: 0, totalTokens: 0, cost: 0, unmeteredRequests: 0, windowStart: now };
      this.spends.set(key, s);
    }
    return s;
  }
}

/** Convenience: `client.use(budget({...}))` when you don't need the controller handle. */
export function budget(options: BudgetOptions): Middleware {
  return new BudgetController(options).middleware();
}
