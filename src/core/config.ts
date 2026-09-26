import type { ProviderCapabilities } from "./capabilities.js";
import type { ProviderOptions } from "./types.js";
import { ConfigurationError } from "./errors.js";

export interface RetryOptions {
  /** Retries after the first attempt. Default 2. */
  maxRetries?: number;
  /** Base delay for exponential backoff (ms). Default 500. */
  initialDelayMs?: number;
  /** Upper bound for any single delay, including Retry-After (ms). Default 30 000. */
  maxDelayMs?: number;
  /** Backoff multiplier. Default 2. */
  multiplier?: number;
  /** Full jitter when true (default); deterministic backoff when false. */
  jitter?: boolean;
  /** Final say on whether an error is retried. Defaults to `error.retryable`. */
  shouldRetry?: (error: unknown, attempt: number) => boolean;
}

export interface Logger {
  debug(message: string, meta?: Record<string, unknown>): void;
  warn?(message: string, meta?: Record<string, unknown>): void;
}

/** Options shared by every provider. */
export interface BaseProviderConfig {
  provider: string;
  model: string;
  apiKey?: string;
  baseURL?: string;
  /** Extra headers on every request. Never logged. */
  headers?: Record<string, string>;
  /** Request timeout in ms (also used as the idle timeout between stream chunks). Default 60 000. */
  timeout?: number;
  /** Retry count, or full retry options. Default 2 retries. */
  retries?: number | RetryOptions;
  /** Default `max_tokens` for providers that require one (Anthropic). */
  defaultMaxTokens?: number;
  /** Override detected capabilities, e.g. for a custom OpenAI-compatible server. */
  capabilities?: Partial<ProviderCapabilities>;
  /** Default provider-specific options merged under per-request ones. */
  providerOptions?: ProviderOptions;
  /** Custom fetch (proxies, testing). Defaults to global fetch. */
  fetch?: typeof globalThis.fetch;
  /** Debug logger. Secrets and auth headers are never passed to it. */
  logger?: Logger;
}

/** Quirk switches for OpenAI-compatible servers that deviate from OpenAI. */
export interface OpenAICompatibleQuirks {
  /** Which field carries the output token limit. Default `max_tokens`. */
  maxTokensField?: "max_tokens" | "max_completion_tokens";
  /** Send `stream_options: { include_usage: true }`. Default true. */
  streamUsage?: boolean;
  /** Map `toolChoice: "required"` to this wire value. Default "required". */
  requiredToolChoice?: "required" | "any";
}

export interface OpenAICompatibleConfig extends BaseProviderConfig {
  provider: "openai-compatible";
  baseURL: string;
  /** Optional: many local servers (Ollama, LM Studio) need none. */
  apiKey?: string;
  /** Query parameters appended to every request URL (e.g. Azure `api-version`). */
  query?: Record<string, string>;
  /** Name reported in responses/errors. Default "openai-compatible". */
  name?: string;
  quirks?: OpenAICompatibleQuirks;
}

export interface OpenAIConfig extends BaseProviderConfig {
  provider: "openai";
  apiKey: string;
  organization?: string;
  project?: string;
}

export interface AnthropicConfig extends BaseProviderConfig {
  provider: "anthropic";
  apiKey: string;
  /** `anthropic-version` header. Default "2023-06-01". */
  apiVersion?: string;
  /** Values for the `anthropic-beta` header. */
  betas?: string[];
}

export interface GeminiConfig extends BaseProviderConfig {
  provider: "gemini";
  apiKey: string;
  /** API version path segment. Default "v1beta". */
  apiVersion?: string;
}

export interface DeepSeekConfig extends BaseProviderConfig {
  provider: "deepseek";
  apiKey: string;
}

export interface OpenRouterConfig extends BaseProviderConfig {
  provider: "openrouter";
  apiKey: string;
  /** Sent as `HTTP-Referer` for OpenRouter app attribution. */
  siteURL?: string;
  /** Sent as `X-Title` for OpenRouter app attribution. */
  appName?: string;
}

export interface GroqConfig extends BaseProviderConfig {
  provider: "groq";
  apiKey: string;
}

export interface MistralConfig extends BaseProviderConfig {
  provider: "mistral";
  apiKey: string;
}

export interface XAIConfig extends BaseProviderConfig {
  provider: "xai";
  apiKey: string;
}

/**
 * Built-in provider configs, keyed by provider name. Custom providers can add
 * typed configs via declaration merging:
 *
 * ```ts
 * declare module "llm-adapter" {
 *   interface ProviderConfigMap { "my-provider": MyProviderConfig }
 * }
 * ```
 */
export interface ProviderConfigMap {
  openai: OpenAIConfig;
  anthropic: AnthropicConfig;
  gemini: GeminiConfig;
  deepseek: DeepSeekConfig;
  openrouter: OpenRouterConfig;
  groq: GroqConfig;
  mistral: MistralConfig;
  xai: XAIConfig;
  "openai-compatible": OpenAICompatibleConfig;
}

export type KnownProviderConfig = ProviderConfigMap[keyof ProviderConfigMap];

/** Any config. Known provider names are type-checked against their config. */
export type LLMConfig = KnownProviderConfig | (BaseProviderConfig & { provider: string; [key: string]: unknown });

// ---------------------------------------------------------------------------
// Validation helpers (used by provider factories)
// ---------------------------------------------------------------------------

export function requireString(config: BaseProviderConfig, field: string, hint?: string): string {
  const value = (config as unknown as Record<string, unknown>)[field];
  if (typeof value !== "string" || value.trim() === "") {
    throw new ConfigurationError(
      `${config.provider}: missing required config field "${field}"${hint ? ` (${hint})` : ""}.`,
      { provider: config.provider },
    );
  }
  return value;
}

export function validateBaseConfig(config: BaseProviderConfig): void {
  if (!config || typeof config !== "object") throw new ConfigurationError("LLM config must be an object.");
  if (typeof config.provider !== "string" || config.provider === "") {
    throw new ConfigurationError('LLM config is missing "provider".');
  }
  requireString(config, "model", "the default model id, e.g. from your provider's model list");
  if (config.timeout !== undefined && !(Number.isFinite(config.timeout) && config.timeout > 0)) {
    throw new ConfigurationError(`${config.provider}: "timeout" must be a positive number of milliseconds.`, {
      provider: config.provider,
    });
  }
  const retries = typeof config.retries === "number" ? config.retries : config.retries?.maxRetries;
  if (retries !== undefined && !(Number.isInteger(retries) && retries >= 0)) {
    throw new ConfigurationError(`${config.provider}: "retries" must be a non-negative integer.`, {
      provider: config.provider,
    });
  }
  if (config.baseURL !== undefined) {
    try {
      new URL(config.baseURL);
    } catch {
      throw new ConfigurationError(`${config.provider}: "baseURL" is not a valid URL.`, { provider: config.provider });
    }
  }
}

export function normalizeRetryOptions(retries: number | RetryOptions | undefined): Required<Omit<RetryOptions, "shouldRetry">> &
  Pick<RetryOptions, "shouldRetry"> {
  const opts: RetryOptions = typeof retries === "number" ? { maxRetries: retries } : (retries ?? {});
  return {
    maxRetries: opts.maxRetries ?? 2,
    initialDelayMs: opts.initialDelayMs ?? 500,
    maxDelayMs: opts.maxDelayMs ?? 30_000,
    multiplier: opts.multiplier ?? 2,
    jitter: opts.jitter ?? true,
    shouldRetry: opts.shouldRetry,
  };
}
