import type {
  AnthropicConfig,
  DeepSeekConfig,
  GeminiConfig,
  GroqConfig,
  MistralConfig,
  OpenAICompatibleConfig,
  OpenAIConfig,
  OpenRouterConfig,
  XAIConfig,
} from "../core/config.js";
import { AnthropicAdapter } from "../providers/anthropic/AnthropicAdapter.js";
import { DeepSeekAdapter } from "../providers/deepseek/DeepSeekAdapter.js";
import { GeminiAdapter } from "../providers/gemini/GeminiAdapter.js";
import { GroqAdapter } from "../providers/groq/GroqAdapter.js";
import { MistralAdapter } from "../providers/mistral/MistralAdapter.js";
import { OpenAICompatibleAdapter } from "../providers/openai-compatible/OpenAICompatibleAdapter.js";
import { OpenAIAdapter } from "../providers/openai/OpenAIAdapter.js";
import { OpenRouterAdapter } from "../providers/openrouter/OpenRouterAdapter.js";
import { XAIAdapter } from "../providers/xai/XAIAdapter.js";
import { ProviderRegistry } from "./ProviderRegistry.js";

/** Registers every built-in provider on a registry. */
export function registerBuiltinProviders(registry: ProviderRegistry): ProviderRegistry {
  return registry
    .register("openai", (c: OpenAIConfig) => OpenAIAdapter.fromConfig(c))
    .register("anthropic", (c: AnthropicConfig) => AnthropicAdapter.fromConfig(c))
    .register("gemini", (c: GeminiConfig) => GeminiAdapter.fromConfig(c))
    .register("deepseek", (c: DeepSeekConfig) => DeepSeekAdapter.fromConfig(c))
    .register("openrouter", (c: OpenRouterConfig) => OpenRouterAdapter.fromConfig(c))
    .register("groq", (c: GroqConfig) => GroqAdapter.fromConfig(c))
    .register("mistral", (c: MistralConfig) => MistralAdapter.fromConfig(c))
    .register("xai", (c: XAIConfig) => XAIAdapter.fromConfig(c))
    .register("openai-compatible", (c: OpenAICompatibleConfig) => OpenAICompatibleAdapter.fromCompatibleConfig(c));
}

ProviderRegistry.setDefaultInitializer(registerBuiltinProviders);
