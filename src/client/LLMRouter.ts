import type { LLMConfig } from "../core/config.js";
import { ConfigurationError } from "../core/errors.js";
import type { ChatRequest, ChatResponse } from "../core/types.js";
import type { LLMProvider } from "../providers/base.js";
import type { ProviderRegistry } from "../registry/ProviderRegistry.js";
import type { ChatStream } from "./ChatStream.js";
import { LLMClient, type LLMClientOptions } from "./LLMClient.js";
import type { Middleware } from "./middleware.js";

export interface LLMRouterOptions<K extends string> {
  /** Named targets: a config, a provider instance, or an existing client. */
  providers: Record<K, LLMConfig | LLMProvider | LLMClient>;
  registry?: ProviderRegistry;
  /** Middleware applied to every client created by the router. */
  middleware?: Middleware[];
}

/**
 * Holds several named clients and dispatches by name. It makes no routing
 * decisions of its own; selection policies (cost, latency, fallback) can be
 * layered on top later.
 */
export class LLMRouter<K extends string = string> {
  private readonly clients = new Map<K, LLMClient>();

  constructor(options: LLMRouterOptions<K>) {
    const clientOptions: LLMClientOptions = { registry: options.registry, middleware: options.middleware };
    for (const [name, target] of Object.entries(options.providers) as [K, LLMConfig | LLMProvider | LLMClient][]) {
      const client = target instanceof LLMClient ? target : new LLMClient(target, clientOptions);
      if (target instanceof LLMClient) for (const m of options.middleware ?? []) client.use(m);
      this.clients.set(name, client);
    }
  }

  names(): K[] {
    return [...this.clients.keys()];
  }

  has(name: string): name is K {
    return this.clients.has(name as K);
  }

  get(name: K): LLMClient {
    const client = this.clients.get(name);
    if (!client) {
      throw new ConfigurationError(`LLMRouter: no provider named "${name}". Known: ${this.names().join(", ")}.`);
    }
    return client;
  }

  chat(name: K, request: ChatRequest): Promise<ChatResponse> {
    return this.get(name).chat(request);
  }

  stream(name: K, request: ChatRequest): ChatStream {
    return this.get(name).stream(request);
  }

  /** Adds middleware to every client in the router. */
  use(middleware: Middleware): this {
    for (const client of this.clients.values()) client.use(middleware);
    return this;
  }
}
