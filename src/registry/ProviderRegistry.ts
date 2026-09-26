import type { LLMConfig, ProviderConfigMap } from "../core/config.js";
import { ConfigurationError } from "../core/errors.js";
import type { LLMProvider } from "../providers/base.js";

export type ProviderFactory<C = LLMConfig> = (config: C) => LLMProvider;

/**
 * Maps provider names to factories. Holds factories only — never
 * configuration or credentials.
 *
 * A process-wide default registry (with all built-in providers) backs the
 * static methods; create separate instances for isolation (e.g. in tests)
 * and pass them to `LLMClient` via `options.registry`.
 */
export class ProviderRegistry {
  private readonly factories = new Map<string, ProviderFactory<never>>();

  register<K extends keyof ProviderConfigMap>(name: K, factory: ProviderFactory<ProviderConfigMap[K]>): this;
  register<C extends LLMConfig = LLMConfig>(name: string, factory: ProviderFactory<C>): this;
  register(name: string, factory: ProviderFactory<never>): this {
    if (!name) throw new ConfigurationError("Provider name must be a non-empty string.");
    this.factories.set(name, factory);
    return this;
  }

  unregister(name: string): boolean {
    return this.factories.delete(name);
  }

  has(name: string): boolean {
    return this.factories.has(name);
  }

  list(): string[] {
    return [...this.factories.keys()].sort();
  }

  create(config: LLMConfig): LLMProvider {
    const factory = this.factories.get(config?.provider) as ProviderFactory<LLMConfig> | undefined;
    if (!factory) {
      throw new ConfigurationError(
        `Unknown provider "${String(config?.provider)}". Registered providers: ${this.list().join(", ") || "(none)"}.`,
      );
    }
    return factory(config);
  }

  // Static conveniences over the default registry. -------------------------

  private static defaultInstance: ProviderRegistry | undefined;
  private static defaultInitializer: ((registry: ProviderRegistry) => void) | undefined;

  /** @internal Wires built-in providers into the default registry lazily. */
  static setDefaultInitializer(init: (registry: ProviderRegistry) => void): void {
    ProviderRegistry.defaultInitializer = init;
  }

  static get default(): ProviderRegistry {
    if (!ProviderRegistry.defaultInstance) {
      ProviderRegistry.defaultInstance = new ProviderRegistry();
      ProviderRegistry.defaultInitializer?.(ProviderRegistry.defaultInstance);
    }
    return ProviderRegistry.defaultInstance;
  }

  static register<K extends keyof ProviderConfigMap>(name: K, factory: ProviderFactory<ProviderConfigMap[K]>): ProviderRegistry;
  static register<C extends LLMConfig = LLMConfig>(name: string, factory: ProviderFactory<C>): ProviderRegistry;
  static register(name: string, factory: ProviderFactory<never>): ProviderRegistry {
    return ProviderRegistry.default.register(name, factory as ProviderFactory<LLMConfig>);
  }

  static has(name: string): boolean {
    return ProviderRegistry.default.has(name);
  }

  static list(): string[] {
    return ProviderRegistry.default.list();
  }

  static unregister(name: string): boolean {
    return ProviderRegistry.default.unregister(name);
  }
}
