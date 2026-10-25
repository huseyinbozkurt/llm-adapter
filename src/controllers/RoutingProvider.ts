import { hasImages, type ProviderCapabilities } from "../core/capabilities.js";
import { ConfigurationError } from "../core/errors.js";
import type { ChatRequest, ChatResponse, StreamEvent } from "../core/types.js";
import type { LLMProvider } from "../providers/base.js";
import type { ProviderRegistry } from "../registry/ProviderRegistry.js";
import { assertNoModelOverride, canServe, forTarget, toNamedClients, unionCapabilities, type NamedClient, type Target } from "./targets.js";

export interface RouteContext {
  operation: "chat" | "stream";
  request: ChatRequest;
}

export interface RoutingRule<K extends string> {
  /** Label used in `onRoute` / debugging. */
  name?: string;
  when: (ctx: RouteContext) => boolean;
  use: K;
}

export interface RoutingOptions<K extends string> {
  targets: Record<K, Target>;
  /** Evaluated in order; the first matching rule whose target can serve the request wins. */
  rules?: RoutingRule<K>[];
  /** Used when no rule selects a capable target. */
  default: K;
  /** Skip rule targets that cannot serve the request (capabilities). Default true. */
  requireCapable?: boolean;
  onRoute?: (decision: { target: K; rule?: string; operation: "chat" | "stream" }) => void;
  registry?: ProviderRegistry;
  /** Name reported by this provider. Default "router". */
  name?: string;
}

/**
 * An LLMProvider that picks one target per request using explicit rules.
 * Rules only see the request; there is no hidden policy. Compose with
 * FallbackProvider (either way round) for resilience.
 *
 * ```ts
 * const llm = new LLMClient(new RoutingProvider({
 *   targets: { fast: groqConfig, vision: openaiConfig, smart: anthropicConfig },
 *   rules: [
 *     { when: ({ request }) => hasImages(request), use: "vision" },
 *     { when: ({ request }) => request.metadata?.tier === "premium", use: "smart" },
 *   ],
 *   default: "fast",
 * }));
 * ```
 */
export class RoutingProvider<K extends string = string> implements LLMProvider {
  readonly name: string;
  readonly defaultModel: string;
  private readonly clients: Map<K, NamedClient>;
  private readonly options: RoutingOptions<K>;

  constructor(options: RoutingOptions<K>) {
    this.options = options;
    this.name = options.name ?? "router";
    const named = toNamedClients(options.targets, options.registry);
    this.clients = new Map(named.map((n) => [n.name as K, n]));
    for (const key of [options.default, ...(options.rules ?? []).map((r) => r.use)]) {
      if (!this.clients.has(key)) throw new ConfigurationError(`RoutingProvider: unknown target "${key}".`);
    }
    this.defaultModel = this.clients.get(options.default)!.client.model;
  }

  capabilities(): ProviderCapabilities {
    return unionCapabilities([...this.clients.values()]);
  }

  /** The target a request would be routed to. */
  select(request: ChatRequest, operation: "chat" | "stream" = "chat"): K {
    return this.decide(request, operation).target;
  }

  async chat(request: ChatRequest): Promise<ChatResponse> {
    assertNoModelOverride(this.name, this.defaultModel, request);
    const { target } = this.route(request, "chat");
    return this.clients.get(target)!.client.chat(forTarget(request));
  }

  async *stream(request: ChatRequest): AsyncGenerator<StreamEvent> {
    assertNoModelOverride(this.name, this.defaultModel, request);
    const { target } = this.route(request, "stream");
    yield* this.clients.get(target)!.client.stream(forTarget(request));
  }

  private route(request: ChatRequest, operation: "chat" | "stream"): { target: K; rule?: string } {
    const decision = this.decide(request, operation);
    this.options.onRoute?.({ ...decision, operation });
    return decision;
  }

  private decide(request: ChatRequest, operation: "chat" | "stream"): { target: K; rule?: string } {
    const ctx: RouteContext = { operation, request };
    const requireCapable = this.options.requireCapable !== false;
    for (const [i, rule] of (this.options.rules ?? []).entries()) {
      if (!rule.when(ctx)) continue;
      const client = this.clients.get(rule.use)!.client;
      if (requireCapable && !canServe(client, request, operation)) continue;
      return { target: rule.use, rule: rule.name ?? `rule-${i}` };
    }
    // The default is used even if incapable: the target client then raises a clear UnsupportedCapabilityError.
    return { target: this.options.default };
  }
}

// ---------------------------------------------------------------------------
// Helpers for writing rules
// ---------------------------------------------------------------------------

export { hasImages };

export function hasTools(request: ChatRequest): boolean {
  return (request.tools?.length ?? 0) > 0;
}

/**
 * Rough input size in tokens (characters / 4, images counted as 1000).
 * Only for coarse routing such as "long prompts → large-context model";
 * it is not a tokenizer.
 */
export function estimateInputTokens(request: ChatRequest): number {
  let chars = 0;
  let images = 0;
  for (const m of request.messages) {
    if (typeof m.content === "string") chars += m.content.length;
    else for (const p of m.content) {
      if (p.type === "image") images++;
      else chars += p.text.length;
    }
    if (m.role === "assistant") for (const c of m.toolCalls ?? []) chars += c.function.arguments.length + c.function.name.length;
  }
  for (const t of request.tools ?? []) chars += JSON.stringify(t).length;
  return Math.ceil(chars / 4) + images * 1000;
}
