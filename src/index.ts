// Client
export { LLMClient, createLLM, type LLMClientOptions, type RequestDefaults } from "./client/LLMClient.js";
export { LLMRouter, type LLMRouterOptions } from "./client/LLMRouter.js";
export { ChatStream } from "./client/ChatStream.js";
export type {
  Middleware,
  ChatMiddleware,
  StreamMiddleware,
  MiddlewareContext,
  ChatNext,
  StreamNext,
} from "./client/middleware.js";

// Core types
export type * from "./core/types.js";
export type * from "./core/config.js";
export {
  ALL_CAPABILITIES,
  assertRequestSupported,
  resolveCapabilities,
  type Capability,
  type ModelCapabilityRule,
  type ProviderCapabilities,
} from "./core/capabilities.js";
export {
  LLMError,
  AuthenticationError,
  RateLimitError,
  TimeoutError,
  AbortError,
  InvalidRequestError,
  ContextLengthExceededError,
  ContentFilterError,
  ProviderError,
  ModelNotFoundError,
  UnsupportedCapabilityError,
  NetworkError,
  ConfigurationError,
  BudgetExceededError,
  FallbackExhaustedError,
  isLLMError,
  type LLMErrorOptions,
} from "./core/errors.js";

// Providers
export { BaseProvider, type LLMProvider, type BaseProviderSettings, commonSettings } from "./providers/base.js";
export {
  OpenAICompatibleAdapter,
  OPENAI_COMPATIBLE_DEFAULT_CAPABILITIES,
  type OpenAICompatibleSettings,
} from "./providers/openai-compatible/OpenAICompatibleAdapter.js";
export { OpenAIAdapter } from "./providers/openai/OpenAIAdapter.js";
export { AnthropicAdapter } from "./providers/anthropic/AnthropicAdapter.js";
export { GeminiAdapter } from "./providers/gemini/GeminiAdapter.js";
export { DeepSeekAdapter } from "./providers/deepseek/DeepSeekAdapter.js";
export { OpenRouterAdapter } from "./providers/openrouter/OpenRouterAdapter.js";
export { GroqAdapter } from "./providers/groq/GroqAdapter.js";
export { MistralAdapter } from "./providers/mistral/MistralAdapter.js";
export { XAIAdapter } from "./providers/xai/XAIAdapter.js";

// Request controllers
export {
  FallbackProvider,
  defaultShouldFallback,
  type FallbackOptions,
  type FallbackAttempt,
} from "./controllers/FallbackProvider.js";
export {
  RoutingProvider,
  hasImages,
  hasTools,
  estimateInputTokens,
  type RoutingOptions,
  type RoutingRule,
  type RouteContext,
} from "./controllers/RoutingProvider.js";
export { RateLimiter, rateLimit, type RateLimiterOptions, type RateLimitMiddlewareOptions } from "./controllers/rateLimit.js";
export {
  BudgetController,
  budget,
  computeCost,
  type BudgetOptions,
  type BudgetLimits,
  type BudgetSpend,
  type ModelPrice,
  type PriceLookup,
  type UsageRecord,
} from "./controllers/budget.js";
export type { Target } from "./controllers/targets.js";

// Registry
export { ProviderRegistry, type ProviderFactory } from "./registry/ProviderRegistry.js";
export { registerBuiltinProviders } from "./registry/builtins.js";

// Utilities useful to custom provider authors
export { StreamAccumulator } from "./utils/streaming.js";
export { parseSSE, type SSEMessage } from "./utils/sse.js";
export { HttpClient, type HttpClientOptions } from "./utils/http.js";
export { withRetry, parseRetryAfter, computeDelay } from "./utils/retry.js";
