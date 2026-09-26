import { describe, expect, it } from "vitest";
import {
  AbortError,
  AuthenticationError,
  ContentFilterError,
  ContextLengthExceededError,
  errorFromHttp,
  InvalidRequestError,
  LLMError,
  ModelNotFoundError,
  NetworkError,
  ProviderError,
  RateLimitError,
  redact,
  TimeoutError,
} from "../../src/core/errors.js";
import { computeDelay, parseRetryAfter, withRetry } from "../../src/utils/retry.js";
import { normalizeRetryOptions } from "../../src/core/config.js";
import { LLMClient } from "../../src/index.js";
import { hang, json, mockFetch } from "../helpers/mockFetch.js";

const msgs = [{ role: "user" as const, content: "hi" }];
const okBody = { id: "x", model: "m", choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }] };

describe("errorFromHttp", () => {
  const cases: Array<[number, string | undefined, string, new (...a: never[]) => LLMError, boolean]> = [
    [401, undefined, "bad key", AuthenticationError, false],
    [403, undefined, "forbidden", AuthenticationError, false],
    [429, "rate_limit_exceeded", "slow down", RateLimitError, true],
    [429, "insufficient_quota", "You exceeded your current quota", RateLimitError, false],
    [429, "RESOURCE_EXHAUSTED", "You exceeded your current quota. Quota exceeded for metric: requests per minute", RateLimitError, true],
    [400, "context_length_exceeded", "This model's maximum context length is 8192 tokens", ContextLengthExceededError, false],
    [400, "invalid_request_error", "prompt is too long: 250000 tokens > 200000 maximum", ContextLengthExceededError, false],
    [400, "content_policy_violation", "flagged by our safety system", ContentFilterError, false],
    [404, "model_not_found", "The model `gpt-9` does not exist", ModelNotFoundError, false],
    [400, undefined, "temperature must be <= 2", InvalidRequestError, false],
    [500, undefined, "oops", ProviderError, true],
    [502, undefined, "bad gateway", ProviderError, true],
    [503, undefined, "unavailable", ProviderError, true],
    [504, undefined, "timeout", ProviderError, true],
    [529, "overloaded_error", "Overloaded", ProviderError, true],
  ];
  it.each(cases)("maps %i/%s", (status, code, message, cls, retryable) => {
    const err = errorFromHttp({ provider: "p", status, code, message });
    expect(err).toBeInstanceOf(cls);
    expect(err.retryable).toBe(retryable);
    expect(err.statusCode).toBe(status);
    expect(err.provider).toBe("p");
  });
});

describe("redact", () => {
  it("removes configured secrets and key-like strings", () => {
    const out = redact("key sk-abcdefghijklmnop and my-secret-value and Bearer abc.def.ghi123 AIzaSyA1234567890123456789012", [
      "my-secret-value",
    ]);
    expect(out).not.toContain("sk-abcdefghijklmnop");
    expect(out).not.toContain("my-secret-value");
    expect(out).not.toContain("abc.def.ghi123");
    expect(out).not.toContain("AIzaSy");
  });
});

describe("retry helpers", () => {
  it("parses Retry-After seconds, dates and retry-after-ms", () => {
    expect(parseRetryAfter(new Headers({ "retry-after": "3" }))).toBe(3000);
    expect(parseRetryAfter(new Headers({ "retry-after-ms": "250" }))).toBe(250);
    const now = Date.parse("2026-01-01T00:00:00Z");
    expect(parseRetryAfter(new Headers({ "retry-after": "Thu, 01 Jan 2026 00:00:05 GMT" }), now)).toBe(5000);
    expect(parseRetryAfter(new Headers())).toBeUndefined();
  });

  it("computes exponential backoff with jitter and caps", () => {
    const opts = normalizeRetryOptions({ initialDelayMs: 100, maxDelayMs: 1000, jitter: false });
    expect([0, 1, 2, 3, 4].map((a) => computeDelay(a, opts))).toEqual([100, 200, 400, 800, 1000]);
    const jittered = normalizeRetryOptions({ initialDelayMs: 100, jitter: true });
    expect(computeDelay(2, jittered, undefined, () => 0.5)).toBe(200);
    expect(computeDelay(0, opts, 5000)).toBe(1000); // Retry-After capped by maxDelayMs
  });

  it("retries only retryable errors", async () => {
    const opts = normalizeRetryOptions({ maxRetries: 3, initialDelayMs: 1, jitter: false });
    let calls = 0;
    await expect(
      withRetry(async () => {
        calls++;
        throw new InvalidRequestError("bad", { retryable: false });
      }, opts),
    ).rejects.toBeInstanceOf(InvalidRequestError);
    expect(calls).toBe(1);

    calls = 0;
    const result = await withRetry(async () => {
      if (++calls < 3) throw new ProviderError("flaky", { retryable: true });
      return "ok";
    }, opts);
    expect(result).toBe("ok");
    expect(calls).toBe(3);
  });
});

describe("HTTP behaviour through the client", () => {
  const base = { provider: "openai-compatible" as const, baseURL: "http://localhost:1234/v1", model: "m" };

  it("retries 429/5xx with Retry-After, then succeeds", async () => {
    const m = mockFetch(
      json({ error: { message: "slow" } }, 429, { "retry-after-ms": "5" }),
      json({ error: { message: "down" } }, 503),
      json(okBody),
    );
    const client = new LLMClient({ ...base, fetch: m.fetch, retries: { maxRetries: 3, initialDelayMs: 1 } });
    const res = await client.chat({ messages: msgs });
    expect(res.text).toBe("ok");
    expect(m.requests).toHaveLength(3);
  });

  it("does not retry authentication or invalid-request errors", async () => {
    for (const status of [401, 400]) {
      const m = mockFetch(json({ error: { message: "nope" } }, status), json(okBody));
      const client = new LLMClient({ ...base, fetch: m.fetch, retries: 3 });
      await expect(client.chat({ messages: msgs })).rejects.toBeInstanceOf(LLMError);
      expect(m.requests).toHaveLength(1);
    }
  });

  it("gives up after maxRetries", async () => {
    const m = mockFetch(json({ error: { message: "down" } }, 500));
    const client = new LLMClient({ ...base, fetch: m.fetch, retries: { maxRetries: 2, initialDelayMs: 1 } });
    await expect(client.chat({ messages: msgs })).rejects.toBeInstanceOf(ProviderError);
    expect(m.requests).toHaveLength(3);
  });

  it("wraps network failures as retryable NetworkError", async () => {
    let calls = 0;
    const failing = (async () => {
      calls++;
      throw new TypeError("fetch failed", { cause: Object.assign(new Error("reset"), { code: "ECONNRESET" }) });
    }) as typeof fetch;
    const client = new LLMClient({ ...base, fetch: failing, retries: { maxRetries: 1, initialDelayMs: 1 } });
    const err = await client.chat({ messages: msgs }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(NetworkError);
    expect((err as NetworkError).code).toBe("ECONNRESET");
    expect(calls).toBe(2);
  });

  it("distinguishes caller abort from timeout", async () => {
    const aborted = mockFetch(hang());
    const client = new LLMClient({ ...base, fetch: aborted.fetch, retries: 0 });
    const controller = new AbortController();
    const pending = client.chat({ messages: msgs, signal: controller.signal });
    setTimeout(() => controller.abort(), 10);
    await expect(pending).rejects.toBeInstanceOf(AbortError);

    const timed = mockFetch(hang());
    const slow = new LLMClient({ ...base, fetch: timed.fetch, retries: 0, timeout: 20 });
    const err = await slow.chat({ messages: msgs }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(TimeoutError);
    expect(err).not.toBeInstanceOf(AbortError);
  });

  it("never retries after an abort, even when retries remain", async () => {
    const m = mockFetch(hang());
    const client = new LLMClient({ ...base, fetch: m.fetch, retries: 5 });
    const controller = new AbortController();
    controller.abort();
    await expect(client.chat({ messages: msgs, signal: controller.signal })).rejects.toBeInstanceOf(AbortError);
    expect(m.requests.length).toBeLessThanOrEqual(1);
  });

  it("keeps API keys out of error messages", async () => {
    const key = "sk-test-SUPERSECRETKEY123456";
    const m = mockFetch(json({ error: { message: `Incorrect API key provided: ${key}` } }, 401));
    const client = new LLMClient({ provider: "openai", apiKey: key, model: "gpt-x", fetch: m.fetch });
    const err = (await client.chat({ messages: msgs }).catch((e: unknown) => e)) as LLMError;
    expect(err).toBeInstanceOf(AuthenticationError);
    expect(err.message).not.toContain(key);
    expect(JSON.stringify(err)).not.toContain(key);
    expect(JSON.stringify((err as { cause?: unknown }).cause)).not.toContain(key);
  });
});
