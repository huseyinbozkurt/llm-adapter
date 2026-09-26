import "../registry/builtins.js";
import { assertRequestSupported, type Capability, type ProviderCapabilities } from "../core/capabilities.js";
import type { LLMConfig } from "../core/config.js";
import { UnsupportedCapabilityError } from "../core/errors.js";
import type { ChatMessage, ChatRequest, ChatResponse, ModelInfo, ProviderOptions, StreamEvent } from "../core/types.js";
import type { LLMProvider } from "../providers/base.js";
import { ProviderRegistry } from "../registry/ProviderRegistry.js";
import { ChatStream } from "./ChatStream.js";
import {
  chatMiddlewareOf,
  composeChat,
  composeStream,
  streamMiddlewareOf,
  type Middleware,
  type MiddlewareContext,
} from "./middleware.js";

/** Request fields a client can default. `messages` and `signal` are always per-call. */
export type RequestDefaults = Partial<Omit<ChatRequest, "messages" | "signal">>;

export interface LLMClientOptions {
  /** Registry to resolve `config.provider` against. Defaults to the global registry. */
  registry?: ProviderRegistry;
  middleware?: Middleware[];
  /** Defaults merged under every request (per-request values win). */
  defaults?: RequestDefaults;
}

function isProvider(value: unknown): value is LLMProvider {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as LLMProvider).chat === "function" &&
    typeof (value as LLMProvider).stream === "function" &&
    typeof (value as LLMProvider).capabilities === "function"
  );
}

function mergeProviderOptions(a: ProviderOptions | undefined, b: ProviderOptions | undefined): ProviderOptions | undefined {
  if (!a) return b;
  if (!b) return a;
  const out: ProviderOptions = { ...a };
  for (const [name, opts] of Object.entries(b)) {
    const prev = out[name];
    out[name] = prev
      ? { body: { ...prev.body, ...opts?.body }, headers: { ...prev.headers, ...opts?.headers } }
      : opts;
  }
  return out;
}

/**
 * Rewrites `json_schema` → JSON mode + schema instruction when the caller
 * opted into `fallback: "json_mode"` and the model lacks native support.
 */
function applyStructuredOutputFallback(request: ChatRequest, caps: ProviderCapabilities): ChatRequest {
  const format = request.responseFormat;
  if (format?.type !== "json_schema" || caps.structuredOutput || format.fallback !== "json_mode" || !caps.jsonMode) {
    return request;
  }
  const instruction: ChatMessage = {
    role: "system",
    content:
      `Respond with a single JSON object only, no prose. It must conform to this JSON Schema (name: "${format.name}"):\n` +
      JSON.stringify(format.schema),
  };
  const firstNonSystem = request.messages.findIndex((m) => m.role !== "system");
  const at = firstNonSystem === -1 ? request.messages.length : firstNonSystem;
  const messages = [...request.messages.slice(0, at), instruction, ...request.messages.slice(at)];
  return { ...request, messages, responseFormat: { type: "json_object" } };
}

/**
 * Provider-independent entry point. Holds one provider instance (and so one
 * set of credentials); create as many clients as you need.
 */
export class LLMClient {
  readonly provider: LLMProvider;
  private readonly middleware: Middleware[];
  private readonly defaults: RequestDefaults;

  constructor(config: LLMConfig | LLMProvider, options: LLMClientOptions = {}) {
    this.provider = isProvider(config) ? config : (options.registry ?? ProviderRegistry.default).create(config);
    this.middleware = [...(options.middleware ?? [])];
    this.defaults = options.defaults ?? {};
  }

  /** Provider name, e.g. "anthropic". */
  get providerName(): string {
    return this.provider.name;
  }

  /** Default model for this client. */
  get model(): string {
    return this.defaults.model ?? this.provider.defaultModel;
  }

  /** Adds middleware (outermost first, in registration order). Returns `this`. */
  use(middleware: Middleware): this {
    this.middleware.push(middleware);
    return this;
  }

  /** A new client sharing this provider and middleware, with extra defaults. */
  withDefaults(defaults: RequestDefaults): LLMClient {
    return new LLMClient(this.provider, {
      middleware: this.middleware,
      defaults: {
        ...this.defaults,
        ...defaults,
        providerOptions: mergeProviderOptions(this.defaults.providerOptions, defaults.providerOptions),
      },
    });
  }

  /** Shorthand for `withDefaults({ providerOptions })`. */
  withProviderOptions(providerOptions: ProviderOptions): LLMClient {
    return this.withDefaults({ providerOptions });
  }

  capabilities(model: string = this.model): ProviderCapabilities {
    return this.provider.capabilities(model);
  }

  supports(capability: Capability, model?: string): boolean {
    return this.capabilities(model)[capability];
  }

  async listModels(options: { signal?: AbortSignal } = {}): Promise<ModelInfo[]> {
    if (!this.provider.listModels) {
      throw new UnsupportedCapabilityError("listModels", `${this.provider.name} does not support listing models.`, {
        provider: this.provider.name,
      });
    }
    return this.provider.listModels(options);
  }

  async chat(request: ChatRequest): Promise<ChatResponse> {
    const ctx = this.context("chat", request);
    const chain = this.middleware.map(chatMiddlewareOf).filter((m) => m !== undefined);
    return composeChat(chain, ctx, () => this.provider.chat(this.prepare(ctx.request, "chat")));
  }

  /**
   * Streams normalised events. The request is sent when iteration starts.
   * Break out of the loop or abort the signal to cancel.
   */
  stream(request: ChatRequest): ChatStream {
    const ctx = this.context("stream", request);
    const chain = this.middleware.map(streamMiddlewareOf).filter((m) => m !== undefined);
    const provider = this.provider;
    const prepare = (r: ChatRequest) => this.prepare(r, "stream");
    const final = (): AsyncIterable<StreamEvent> => ({
      [Symbol.asyncIterator]: () => provider.stream(prepare(ctx.request))[Symbol.asyncIterator](),
    });
    return new ChatStream(() => composeStream(chain, ctx, final), provider.name, ctx.model);
  }

  // -------------------------------------------------------------------------

  private withRequestDefaults(request: ChatRequest): ChatRequest {
    return {
      ...this.defaults,
      ...request,
      providerOptions: mergeProviderOptions(this.defaults.providerOptions, request.providerOptions),
    };
  }

  private context(operation: "chat" | "stream", request: ChatRequest): MiddlewareContext {
    const merged = this.withRequestDefaults(request);
    const model = merged.model ?? this.provider.defaultModel;
    return {
      operation,
      provider: this.provider.name,
      model,
      request: merged,
      capabilities: this.provider.capabilities(model),
      state: new Map(),
    };
  }

  /** Validates against capabilities and applies portable rewrites. Runs after middleware. */
  private prepare(request: ChatRequest, operation: "chat" | "stream"): ChatRequest {
    const model = request.model ?? this.provider.defaultModel;
    const caps = this.provider.capabilities(model);
    assertRequestSupported(this.provider.name, model, caps, request, operation);
    return applyStructuredOutputFallback(request, caps);
  }
}

/** Functional alias for `new LLMClient(config, options)`. */
export function createLLM(config: LLMConfig | LLMProvider, options?: LLMClientOptions): LLMClient {
  return new LLMClient(config, options);
}
