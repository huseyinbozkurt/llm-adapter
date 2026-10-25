# Architecture

## Overview

```
Application
   │  LLMClient / LLMRouter          ← portable API, middleware, capability checks
   ▼
Middleware chain                     ← logging, metrics, caching, fallback (user-supplied)
   ▼
prepare(): capability assertion + portable rewrites (e.g. json_schema → JSON mode fallback)
   ▼
LLMProvider (adapter)                ← translation only: unified ⇄ native wire format
   ▼
HttpClient                           ← fetch, timeouts, AbortSignal, retries, error mapping, redaction
```

- **`LLMProvider`** (`src/providers/base.ts`) is the only contract the client knows: `chat`, `stream`, `capabilities(model)`, optional `listModels`, plus `name` / `defaultModel`.
- **`BaseProvider`** is an optional helper base class that wires up `HttpClient`, capability resolution and `providerOptions`.
- **`OpenAICompatibleAdapter`** implements the Chat Completions dialect once. OpenAI, DeepSeek, OpenRouter, Groq, Mistral and xAI subclass it and override small, named hooks (`applyReasoning`, `mapEffort`, `deltaContent`, `parseUsage`, `assistantReasoningFields`, `mapToolCallId`, quirks).
- **Anthropic** and **Gemini** have native adapters: their message models differ too much to share the compatible code.
- **`ProviderRegistry`** maps names → factories. `LLMClient` never switches on provider names. The registry stores factories only, never credentials.

## Provider differences that shaped the design

| Concern | OpenAI-style | Anthropic | Gemini |
|---|---|---|---|
| System prompt | message | top-level `system` | `systemInstruction` |
| Tool args | JSON **string** | JSON **object** (`input`) | JSON **object** (`args`) |
| Tool results | `role: "tool"` message | `tool_result` block in a *user* turn | `functionResponse` part, keyed by **name** |
| Tool call ids | always | always | often absent → generated locally |
| Replay requirements | none | thinking `signature` | `thoughtSignature` on function calls (Gemini 3) |
| Streaming | SSE `choices[].delta`, tool args fragmented by `index` | typed SSE block events | SSE of partial responses; function calls arrive whole |
| Reasoning text | not exposed (OpenAI); `reasoning_content` / `reasoning` (others) | `thinking` blocks | `thought: true` parts |
| Reasoning control | `reasoning_effort` | `output_config.effort` + `thinking` (adaptive / budget) | `thinkingLevel` (3.x) / `thinkingBudget` (2.5) |
| Structured output | `response_format.json_schema` | `output_config.format` | `responseJsonSchema` |
| JSON mode | `json_object` | none | `responseMimeType` |
| Usage | cached tokens included in prompt | cache read/write reported separately | thoughts reported separately |

These are why:

- `ToolCall.function.arguments` stays a **string** (OpenAI shape; lossless for fragments/invalid JSON). Object-based providers are converted at the edge.
- Assistant content is `ContentPart[]` with a `reasoning` part type, and parts / tool calls carry opaque `providerData` so a `ChatResponse.message` can be pushed straight back into `messages` and replay correctly (Anthropic signatures, Gemini thought signatures, DeepSeek `reasoning_content`).
- `TokenUsage` has one documented convention (input includes cached; output includes reasoning) and every adapter converts into it.

## Deliberately *not* normalised

- **Server-side / built-in tools** (web search, code execution, citations): exposed only through `raw` and `providerOptions`.
- **Prompt caching controls** (`cache_control`, cached contents): `providerOptions`.
- **Multiple choices (`n > 1`)**, logprobs, seeds, penalties, top-k: `providerOptions.<name>.body`.
- **Exact effort levels**: a portable scale is mapped to the *nearest* supported level; each adapter documents its mapping. Unsupported controls throw rather than being dropped.
- **Refusal detail categories**: surfaced as `finishReason: "content_filter"` plus `rawFinishReason`.
- **Model metadata / pricing**: capability rules are pattern-based and overridable; pricing is intentionally absent (belongs in a future model registry fed by `usage`).

## Capability model

`capabilities(model)` = provider defaults → matching per-model rules → user overrides (`config.capabilities`). `LLMClient` checks every request against it *after* middleware and *before* network I/O, throwing `UnsupportedCapabilityError` instead of silently dropping features. JSON-schema emulation via JSON mode happens only with explicit `fallback: "json_mode"`.

## Dependencies

Zero runtime dependencies. Every provider speaks HTTPS+JSON+SSE, which Node ≥ 20's `fetch`, `ReadableStream`, `TextDecoder` and `AbortController` cover. Using official SDKs would force installs per provider, duplicate retry/timeout logic we need to normalise anyway, and leak SDK error types. Dev dependencies: TypeScript, Vitest, `@types/node`.

## Errors, retries, timeouts, cancellation

- `HttpClient` maps status + provider error body to the `LLMError` hierarchy (`AuthenticationError`, `RateLimitError`, `ContextLengthExceededError`, …) with `provider`, `statusCode`, `retryable`, `code`, `requestId`, `cause`.
- Retries: exponential backoff with full jitter, `Retry-After` / `retry-after-ms` respected (capped), only for `retryable` errors (429 except quota exhaustion, 408, 409, 5xx, transient network codes). Streams retry only while opening.
- Each attempt gets its own controller combining the caller's signal and a timer; the timer resets on every stream chunk (idle timeout). Afterwards we know which fired: `AbortError` (caller) vs `TimeoutError` (retryable).
- Secrets: auth headers are never logged; logger receives only method/path/status/timing. Error text and error `cause` bodies are passed through `redact()` (configured keys + key-shaped strings). Gemini auth uses a header, never a URL parameter.

## Request controllers

Controllers sit above adapters and reuse the two existing extension points, so none of them needs changes to `LLMClient`:

| Controller | Kind | Why this kind |
|---|---|---|
| `FallbackProvider` | `LLMProvider` | It must call *different* targets, which middleware (bound to one provider) cannot do. |
| `RoutingProvider` | `LLMProvider` | Same reason. It nests with `FallbackProvider` in either direction. |
| `rateLimit` / `RateLimiter` | middleware | It gates a call without changing where the call goes. |
| `BudgetController` | middleware | It meters `usage` and blocks new calls; it doesn't change where calls go. |

Design notes:

- **Targets are `LLMClient`s.** Each target keeps its own middleware, validation and structured-output fallback. A multi-target provider reports the *union* of its targets' capabilities, and skips any target that can't serve a given request.
- **No per-request `model`.** Model ids are provider-specific, so a multi-target provider rejects a per-request `model` rather than ignoring it.
- **Streams commit on the first event.** A stream falls back only before its first event, because output already delivered cannot be taken back.
- **Budgets use caller-supplied rates.** No prices ship with the library, and spend is checked before each call, which makes limits soft under concurrency. Rate limiting and budgets both accept a `key(ctx)` function for per-tenant limits.
