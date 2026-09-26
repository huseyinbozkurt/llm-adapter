import {
  resolveCapabilities,
  type ModelCapabilityRule,
  type ProviderCapabilities,
} from "../core/capabilities.js";
import { normalizeRetryOptions, type BaseProviderConfig, type Logger, type RetryOptions } from "../core/config.js";
import type { ChatRequest, ChatResponse, ModelInfo, ProviderOptions, StreamEvent } from "../core/types.js";
import { HttpClient } from "../utils/http.js";

/**
 * The contract every provider implements. This is all LLMClient knows about.
 */
export interface LLMProvider {
  /** Provider name, used in responses, errors and `providerOptions` lookup. */
  readonly name: string;
  /** Model used when a request does not specify one. */
  readonly defaultModel: string;

  chat(request: ChatRequest): Promise<ChatResponse>;
  stream(request: ChatRequest): AsyncIterable<StreamEvent>;
  /** Capabilities for a model (defaults to `defaultModel`). */
  capabilities(model?: string): ProviderCapabilities;
  listModels?(options?: { signal?: AbortSignal }): Promise<ModelInfo[]>;
}

/** Everything BaseProvider needs; built by each adapter from its public config. */
export interface BaseProviderSettings {
  name: string;
  model: string;
  baseURL: string;
  /** Includes authentication headers. Never logged. */
  headers: Record<string, string>;
  query?: Record<string, string>;
  /** Secrets to scrub from error messages (API keys). */
  secrets: Array<string | undefined>;
  timeout?: number;
  retries?: number | RetryOptions;
  fetch?: typeof globalThis.fetch;
  logger?: Logger;
  providerOptions?: ProviderOptions;
  capabilityOverrides?: Partial<ProviderCapabilities>;
  defaultCapabilities: ProviderCapabilities;
  capabilityRules?: readonly ModelCapabilityRule[];
  parseError?: (body: unknown) => { message?: string; code?: string };
}

export const DEFAULT_TIMEOUT_MS = 60_000;

/** Picks the fields every adapter forwards unchanged from its public config. */
export function commonSettings(config: BaseProviderConfig): Pick<
  BaseProviderSettings,
  "model" | "timeout" | "retries" | "fetch" | "logger" | "providerOptions" | "capabilityOverrides"
> {
  return {
    model: config.model,
    timeout: config.timeout,
    retries: config.retries,
    fetch: config.fetch,
    logger: config.logger,
    providerOptions: config.providerOptions,
    capabilityOverrides: config.capabilities,
  };
}

/**
 * Optional convenience base class: HTTP client, capability resolution and
 * provider-options plumbing. Adapters are free to implement LLMProvider
 * directly instead.
 */
export abstract class BaseProvider implements LLMProvider {
  readonly name: string;
  readonly defaultModel: string;
  protected readonly http: HttpClient;
  protected readonly settings: BaseProviderSettings;

  protected constructor(settings: BaseProviderSettings) {
    this.settings = settings;
    this.name = settings.name;
    this.defaultModel = settings.model;
    this.http = new HttpClient({
      provider: settings.name,
      baseURL: settings.baseURL,
      headers: settings.headers,
      query: settings.query,
      timeoutMs: settings.timeout ?? DEFAULT_TIMEOUT_MS,
      retry: normalizeRetryOptions(settings.retries),
      fetch: settings.fetch,
      logger: settings.logger,
      secrets: settings.secrets.filter((s): s is string => typeof s === "string" && s !== ""),
      parseError: settings.parseError,
    });
  }

  abstract chat(request: ChatRequest): Promise<ChatResponse>;
  abstract stream(request: ChatRequest): AsyncIterable<StreamEvent>;

  capabilities(model: string = this.defaultModel): ProviderCapabilities {
    return resolveCapabilities(
      this.settings.defaultCapabilities,
      this.settings.capabilityRules ?? [],
      model,
      this.settings.capabilityOverrides,
    );
  }

  protected modelFor(request: ChatRequest): string {
    return request.model ?? this.defaultModel;
  }

  /** Config-level and request-level options for this provider, request winning. */
  protected providerOptionsFor(request: ChatRequest): { body: Record<string, unknown>; headers: Record<string, string> } {
    const fromConfig = this.settings.providerOptions?.[this.name];
    const fromRequest = request.providerOptions?.[this.name];
    return {
      body: { ...fromConfig?.body, ...fromRequest?.body },
      headers: { ...fromConfig?.headers, ...fromRequest?.headers },
    };
  }

  /** Transport options shared by chat and stream calls. */
  protected transport(request: ChatRequest): { signal?: AbortSignal; timeoutMs?: number; headers: Record<string, string> } {
    return {
      signal: request.signal,
      timeoutMs: request.timeout,
      headers: this.providerOptionsFor(request).headers,
    };
  }
}
