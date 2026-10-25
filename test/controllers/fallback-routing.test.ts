import { describe, expect, it } from "vitest";
import {
  AbortError,
  AuthenticationError,
  ContentFilterError,
  FallbackExhaustedError,
  FallbackProvider,
  InvalidRequestError,
  LLMClient,
  ProviderError,
  RateLimitError,
  RoutingProvider,
  estimateInputTokens,
  hasImages,
  type ChatRequest,
} from "../../src/index.js";
import { collect } from "../helpers/mockFetch.js";
import { FakeProvider } from "./fakes.js";

const req: ChatRequest = { messages: [{ role: "user", content: "hi" }] };
const withImage: ChatRequest = { messages: [{ role: "user", content: [{ type: "image", url: "https://x/y.png" }] }] };

describe("FallbackProvider", () => {
  it("falls through retryable / target-specific failures in order", async () => {
    const a = new FakeProvider("a", "a1", [{ error: new RateLimitError("a: slow", { retryable: false }) }]);
    const b = new FakeProvider("b", "b1", [{ error: new ProviderError("b: down", { retryable: true }) }]);
    const c = new FakeProvider("c", "c1", [{ text: "from c" }]);
    const seen: string[] = [];
    const llm = new LLMClient(new FallbackProvider({ targets: { a, b, c }, onFallback: (x) => seen.push(`${x.target}:${x.index}`) }));
    const res = await llm.chat(req);
    expect(res.text).toBe("from c");
    expect(res.provider).toBe("c");
    expect(seen).toEqual(["a:0", "b:1"]);
  });

  it("does not fall back on content-policy or generic invalid requests", async () => {
    for (const error of [new ContentFilterError("nope"), new InvalidRequestError("bad param")]) {
      const b = new FakeProvider("b", "b1", [{ text: "unused" }]);
      const llm = new LLMClient(new FallbackProvider({ targets: [new FakeProvider("a", "a1", [{ error }]), b] }));
      await expect(llm.chat(req)).rejects.toBe(error);
      expect(b.calls).toHaveLength(0);
    }
  });

  it("never falls back on caller aborts, even with a permissive policy", async () => {
    const abort = new AbortError("aborted");
    const b = new FakeProvider("b", "b1", [{ text: "unused" }]);
    const llm = new LLMClient(
      new FallbackProvider({ targets: [new FakeProvider("a", "a1", [{ error: abort }]), b], shouldFallback: () => true }),
    );
    await expect(llm.chat(req)).rejects.toBe(abort);
    expect(b.calls).toHaveLength(0);
  });

  it("skips targets that cannot serve the request", async () => {
    const noVision = new FakeProvider("a", "a1", [{ text: "wrong" }], { vision: false });
    const vision = new FakeProvider("b", "b1", [{ text: "saw it" }]);
    const llm = new LLMClient(new FallbackProvider({ targets: [noVision, vision] }));
    expect((await llm.chat(withImage)).text).toBe("saw it");
    expect(noVision.calls).toHaveLength(0);
  });

  it("reports every attempt when all targets fail", async () => {
    const llm = new LLMClient(
      new FallbackProvider({
        targets: [
          new FakeProvider("a", "a1", [{ error: new AuthenticationError("a: key") }]),
          new FakeProvider("b", "b1", [{ error: new ProviderError("b: 503", { retryable: true }) }]),
        ],
      }),
    );
    const err = (await llm.chat(req).catch((e: unknown) => e)) as FallbackExhaustedError;
    expect(err).toBeInstanceOf(FallbackExhaustedError);
    expect(err.errors.map((e) => e.target)).toEqual(["a:a1", "b:b1"]);
    expect(err.retryable).toBe(true);
  });

  it("rejects per-request model overrides instead of ignoring them", async () => {
    const llm = new LLMClient(new FallbackProvider({ targets: [new FakeProvider("a", "a1", [{ text: "x" }])] }));
    await expect(llm.chat({ ...req, model: "other" })).rejects.toBeInstanceOf(InvalidRequestError);
  });

  it("streams: falls back before the first event, commits after it", async () => {
    const failsEarly = new FakeProvider("a", "a1", [{ error: new ProviderError("down", { retryable: true }) }]);
    const ok = new FakeProvider("b", "b1", [{ text: "hey" }]);
    const llm = new LLMClient(new FallbackProvider({ targets: [failsEarly, ok] }));
    expect((await llm.stream(req).finalResponse()).text).toBe("hey");

    const failsMidway = new FakeProvider("a", "a1", [{ error: new ProviderError("dropped", { retryable: true }) }], {}, { failAfter: 2 });
    const backup = new FakeProvider("b", "b1", [{ text: "unused" }]);
    const committed = new LLMClient(new FallbackProvider({ targets: [failsMidway, backup] }));
    const seen: string[] = [];
    const run = async () => {
      for await (const e of committed.stream(req)) if (e.type === "text_delta") seen.push(e.text);
    };
    await expect(run()).rejects.toBeInstanceOf(ProviderError);
    expect(seen.join("")).toBe("p");
    expect(backup.calls).toHaveLength(0);
  });

  it("exposes the union of target capabilities", () => {
    const fb = new FallbackProvider({
      targets: [new FakeProvider("a", "a1", [], { vision: false }), new FakeProvider("b", "b1", [], { tools: false })],
    });
    expect(fb.capabilities()).toMatchObject({ vision: true, tools: true });
    expect(fb.defaultModel).toBe("a1");
  });
});

describe("RoutingProvider", () => {
  const make = () => {
    const fast = new FakeProvider("fast", "f1", [{ text: "fast" }], { vision: false });
    const vision = new FakeProvider("vision", "v1", [{ text: "vision" }]);
    const smart = new FakeProvider("smart", "s1", [{ text: "smart" }]);
    return { fast, vision, smart };
  };

  it("routes by first matching capable rule, else default", async () => {
    const t = make();
    const decisions: string[] = [];
    const router = new RoutingProvider({
      targets: t,
      rules: [
        { name: "images", when: ({ request }) => hasImages(request), use: "vision" },
        { name: "premium", when: ({ request }) => request.metadata?.tier === "premium", use: "smart" },
        { name: "long", when: ({ request }) => estimateInputTokens(request) > 1000, use: "smart" },
      ],
      default: "fast",
      onRoute: (d) => decisions.push(`${d.target}/${d.rule ?? "default"}`),
    });
    const llm = new LLMClient(router);
    expect((await llm.chat(req)).text).toBe("fast");
    expect((await llm.chat(withImage)).text).toBe("vision");
    expect((await llm.chat({ ...req, metadata: { tier: "premium" } })).text).toBe("smart");
    expect((await llm.chat({ messages: [{ role: "user", content: "x".repeat(8000) }] })).text).toBe("smart");
    expect(await collect(llm.stream(req).textStream())).toEqual(["f", "a", "s", "t"]);
    expect(decisions).toEqual(["fast/default", "vision/images", "smart/premium", "smart/long", "fast/default"]);
    expect(router.select(withImage)).toBe("vision");
  });

  it("skips a matching rule whose target is incapable", async () => {
    const t = make();
    const router = new RoutingProvider({
      targets: t,
      rules: [{ when: () => true, use: "fast" }],
      default: "vision",
    });
    expect(router.select(withImage)).toBe("vision");
  });

  it("validates target names at construction", () => {
    const t = make();
    expect(() => new RoutingProvider({ targets: t, default: "nope" as "fast" })).toThrow(/unknown target/);
  });

  it("composes with FallbackProvider", async () => {
    const primary = new FakeProvider("p", "p1", [{ error: new ProviderError("down", { retryable: true }) }]);
    const backup = new FakeProvider("b", "b1", [{ text: "backup" }]);
    const cheap = new FakeProvider("c", "c1", [{ text: "cheap" }]);
    const llm = new LLMClient(
      new RoutingProvider({
        targets: { resilient: new FallbackProvider({ targets: [primary, backup] }), cheap },
        rules: [{ when: ({ request }) => request.metadata?.important === true, use: "resilient" }],
        default: "cheap",
      }),
    );
    expect((await llm.chat({ ...req, metadata: { important: true } })).text).toBe("backup");
    expect((await llm.chat(req)).text).toBe("cheap");
  });
});
