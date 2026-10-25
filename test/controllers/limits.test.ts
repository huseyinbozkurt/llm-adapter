import { describe, expect, it } from "vitest";
import {
  AbortError,
  BudgetController,
  BudgetExceededError,
  ConfigurationError,
  FallbackProvider,
  LLMClient,
  RateLimitError,
  RateLimiter,
  computeCost,
  rateLimit,
  type ChatRequest,
} from "../../src/index.js";
import { FakeProvider } from "./fakes.js";

const req: ChatRequest = { messages: [{ role: "user", content: "hi" }] };
const usage = { inputTokens: 100, outputTokens: 50, totalTokens: 150 };

describe("RateLimiter", () => {
  it("caps concurrency and serves waiters in order", async () => {
    const limiter = new RateLimiter({ maxConcurrent: 2 });
    const order: number[] = [];
    let peak = 0;
    await Promise.all(
      [0, 1, 2, 3, 4].map(async (i) => {
        const release = await limiter.acquire();
        order.push(i);
        peak = Math.max(peak, limiter.inFlight);
        await new Promise((r) => setTimeout(r, 5));
        release();
      }),
    );
    expect(peak).toBe(2);
    expect(order).toEqual([0, 1, 2, 3, 4]);
  });

  it("enforces a sliding request window", async () => {
    const limiter = new RateLimiter({ requests: { limit: 2, intervalMs: 60 } });
    const started = Date.now();
    for (let i = 0; i < 3; i++) (await limiter.acquire())();
    expect(Date.now() - started).toBeGreaterThanOrEqual(55);
  });

  it("fails fast with RateLimitError beyond maxWaitMs", async () => {
    const limiter = new RateLimiter({ requests: { limit: 1, intervalMs: 10_000 }, maxWaitMs: 20 });
    (await limiter.acquire())();
    const err = await limiter.acquire().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(RateLimitError);
    expect((err as RateLimitError).code).toBe("local_rate_limit");
    expect((err as RateLimitError).retryAfterMs).toBeGreaterThan(9000);

    const busy = new RateLimiter({ maxConcurrent: 1, maxWaitMs: 20 });
    await busy.acquire(); // never released
    await expect(busy.acquire()).rejects.toBeInstanceOf(RateLimitError);
  });

  it("stops waiting when the caller aborts", async () => {
    const limiter = new RateLimiter({ maxConcurrent: 1 });
    await limiter.acquire();
    const controller = new AbortController();
    const pending = limiter.acquire(controller.signal);
    setTimeout(() => controller.abort(), 5);
    await expect(pending).rejects.toBeInstanceOf(AbortError);
  });

  it("validates options", () => {
    expect(() => new RateLimiter({ maxConcurrent: 0 })).toThrow(ConfigurationError);
    expect(() => rateLimit({ requests: { limit: 0, intervalMs: 1 } })).toThrow(ConfigurationError);
  });
});

describe("rateLimit middleware", () => {
  it("limits chat and holds stream slots until the stream ends", async () => {
    const provider = new FakeProvider("p", "m", [{ text: "abc" }], {}, { delayMs: 5 });
    const mw = rateLimit({ maxConcurrent: 1 });
    const llm = new LLMClient(provider).use(mw);
    const limiter = mw.limiterFor("");

    const stream = llm.stream(req)[Symbol.asyncIterator]();
    await stream.next();
    expect(limiter.inFlight).toBe(1);
    await stream.return?.(undefined);
    expect(limiter.inFlight).toBe(0);

    await Promise.all([llm.chat(req), llm.chat(req), llm.chat(req)]);
    expect(limiter.inFlight).toBe(0);
  });

  it("partitions by key", async () => {
    const mw = rateLimit({ maxConcurrent: 1, maxWaitMs: 10, key: (ctx) => String(ctx.request.metadata?.user) });
    const llm = new LLMClient(new FakeProvider("p", "m", [{ text: "x" }], {}, { delayMs: 30 })).use(mw);
    const results = await Promise.allSettled([
      llm.chat({ ...req, metadata: { user: "a" } }),
      llm.chat({ ...req, metadata: { user: "b" } }),
      llm.chat({ ...req, metadata: { user: "a" } }),
    ]);
    expect(results.map((r) => r.status)).toEqual(["fulfilled", "fulfilled", "rejected"]);
  });

  it("a local limit triggers fallback to the next target", async () => {
    const limitedClient = new LLMClient(new FakeProvider("a", "a1", [{ text: "a" }], {}, { delayMs: 30 })).use(
      rateLimit({ maxConcurrent: 1, maxWaitMs: 5 }),
    );
    const llm = new LLMClient(new FallbackProvider({ targets: [limitedClient, new FakeProvider("b", "b1", [{ text: "b" }])] }));
    const [first, second] = await Promise.all([llm.chat(req), llm.chat(req)]);
    expect([first.text, second.text].sort()).toEqual(["a", "b"]);
  });
});

describe("BudgetController", () => {
  const pricing = (provider: string, model: string) =>
    provider === "p" && model === "m" ? { inputPerMillion: 2, outputPerMillion: 10, cachedInputPerMillion: 0.5 } : undefined;

  it("computes cost from usage and rates", () => {
    expect(computeCost({ inputTokens: 1_000_000, cachedInputTokens: 400_000, outputTokens: 100_000 }, pricing("p", "m")!)).toBeCloseTo(
      0.6 * 2 + 0.4 * 0.5 + 0.1 * 10,
    );
  });

  it("tracks spend and blocks once a token limit is reached", async () => {
    const records: unknown[] = [];
    const ctl = new BudgetController({ limits: { maxTotalTokens: 300 }, onUsage: (r) => records.push(r) });
    const llm = new LLMClient(new FakeProvider("p", "m", [{ text: "x", usage }])).use(ctl.middleware());
    await llm.chat(req);
    await llm.chat(req);
    expect(ctl.spend()).toMatchObject({ requests: 2, totalTokens: 300, inputTokens: 200, outputTokens: 100 });
    expect(ctl.remaining()).toEqual({ maxTotalTokens: 0 });
    await expect(llm.chat(req)).rejects.toBeInstanceOf(BudgetExceededError);
    expect(records).toHaveLength(2);
  });

  it("enforces cost limits and refuses unpriced models before sending", async () => {
    expect(() => new BudgetController({ limits: { maxCost: 1 } })).toThrow(ConfigurationError);

    const ctl = new BudgetController({ limits: { maxCost: 0.001 }, pricing });
    const priced = new FakeProvider("p", "m", [{ text: "x", usage }]);
    const llm = new LLMClient(priced).use(ctl.middleware());
    await llm.chat(req); // 100*2/1e6 + 50*10/1e6 = 0.0007
    expect(ctl.spend().cost).toBeCloseTo(0.0007);
    await llm.chat(req);
    await expect(llm.chat(req)).rejects.toMatchObject({ name: "BudgetExceededError", limit: "maxCost" });

    const unpriced = new FakeProvider("p", "other", [{ text: "x", usage }]);
    await expect(new LLMClient(unpriced).use(ctl.middleware()).chat(req)).rejects.toBeInstanceOf(ConfigurationError);
    expect(unpriced.calls).toHaveLength(0);
  });

  it("meters streams (including partially consumed ones) and unmetered calls", async () => {
    const ctl = new BudgetController({ limits: { maxRequests: 10 } });
    const llm = new LLMClient(new FakeProvider("p", "m", [{ text: "abc", usage }])).use(ctl.middleware());
    await llm.stream(req).finalResponse();
    for await (const e of llm.stream(req)) if (e.type === "text_delta") break;
    expect(ctl.spend()).toMatchObject({ requests: 2, totalTokens: 150, unmeteredRequests: 1 });
  });

  it("partitions by key and resets per window", async () => {
    let now = 0;
    const ctl = new BudgetController({
      limits: { maxRequests: 1 },
      windowMs: 1000,
      key: (ctx) => String(ctx.request.metadata?.tenant),
      now: () => now,
    });
    const llm = new LLMClient(new FakeProvider("p", "m", [{ text: "x", usage }])).use(ctl.middleware());
    await llm.chat({ ...req, metadata: { tenant: "a" } });
    await llm.chat({ ...req, metadata: { tenant: "b" } });
    await expect(llm.chat({ ...req, metadata: { tenant: "a" } })).rejects.toBeInstanceOf(BudgetExceededError);
    now = 1000;
    await expect(llm.chat({ ...req, metadata: { tenant: "a" } })).resolves.toBeDefined();
  });
});
