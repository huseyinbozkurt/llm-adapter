/**
 * Shared plumbing for controllers that dispatch to several targets
 * (FallbackProvider, RoutingProvider).
 */
import { LLMClient } from "../client/LLMClient.js";
import { ALL_CAPABILITIES, assertRequestSupported, type ProviderCapabilities } from "../core/capabilities.js";
import type { LLMConfig } from "../core/config.js";
import { ConfigurationError, InvalidRequestError } from "../core/errors.js";
import type { ChatRequest } from "../core/types.js";
import type { LLMProvider } from "../providers/base.js";
import type { ProviderRegistry } from "../registry/ProviderRegistry.js";

/** Anything a controller can dispatch to. Clients keep their own middleware. */
export type Target = LLMClient | LLMProvider | LLMConfig;

export interface NamedClient {
  name: string;
  client: LLMClient;
}

export function toClient(target: Target, registry?: ProviderRegistry): LLMClient {
  return target instanceof LLMClient ? target : new LLMClient(target, { registry });
}

export function toNamedClients(targets: Record<string, Target> | Target[], registry?: ProviderRegistry): NamedClient[] {
  const entries: [string, Target][] = Array.isArray(targets)
    ? targets.map((t, i) => [defaultName(t, i), t])
    : Object.entries(targets);
  if (entries.length === 0) throw new ConfigurationError("At least one target is required.");
  return entries.map(([name, t]) => ({ name, client: toClient(t, registry) }));
}

function defaultName(target: Target, index: number): string {
  const client = target instanceof LLMClient ? target : undefined;
  if (client) return `${client.providerName}:${client.model}`;
  if ("defaultModel" in target && typeof target.defaultModel === "string") return `${target.name}:${target.defaultModel}`;
  if ("provider" in target && typeof target.provider === "string") return `${target.provider}:${String(target.model)}`;
  return `target-${index}`;
}

/** True when the target's current model can honour the request. */
export function canServe(client: LLMClient, request: ChatRequest, operation: "chat" | "stream"): boolean {
  try {
    assertRequestSupported(client.providerName, client.model, client.capabilities(), request, operation);
    return true;
  } catch {
    return false;
  }
}

/** Capabilities of a multi-target provider: a feature is available if any target has it. */
export function unionCapabilities(clients: readonly NamedClient[]): ProviderCapabilities {
  const out = Object.fromEntries(Object.keys(ALL_CAPABILITIES).map((k) => [k, false])) as unknown as ProviderCapabilities;
  for (const { client } of clients) {
    const caps = client.capabilities();
    for (const key of Object.keys(out) as (keyof ProviderCapabilities)[]) out[key] ||= caps[key];
  }
  return out;
}

/**
 * Model ids are provider-specific, so a per-request `model` cannot be applied
 * across targets. Rather than silently ignoring it, reject it.
 */
export function assertNoModelOverride(provider: string, defaultModel: string, request: ChatRequest): void {
  if (request.model !== undefined && request.model !== defaultModel) {
    throw new InvalidRequestError(
      `${provider}: "model" cannot be set per request on a multi-target provider; configure models on its targets.`,
      { provider },
    );
  }
}

/** Removes the (virtual) model so each target uses its own. */
export function forTarget(request: ChatRequest): ChatRequest {
  if (request.model === undefined) return request;
  const { model: _model, ...rest } = request;
  return rest;
}
