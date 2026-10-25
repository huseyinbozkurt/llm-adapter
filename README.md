# llm-adapter

One TypeScript API for many LLM providers. You switch providers by changing config, not code.

- **Native adapters:** OpenAI, Anthropic (Claude), Google Gemini, DeepSeek, OpenRouter, Groq, Mistral, xAI (Grok)
- **Any OpenAI-compatible endpoint:** Ollama, LM Studio, vLLM, Together, Fireworks, your own gateway
- Streaming, tool calling, structured output, images, reasoning controls, capability detection, middleware, retries, timeouts, cancellation
- Request controllers: fallback, rule-based routing, rate limiting, and token/cost budgets
- **No runtime dependencies.** It uses Node's built-in `fetch`, so you don't install any provider SDKs.

Requires Node.js ≥ 20.3. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design.

---

## 1. Installation

This is a private package. It is not published to npm. You install it straight from its private GitHub repository over SSH. npm clones the repository and builds `dist/` during the install, because the `prepare` script runs `npm run build`.

### Prerequisites

- **Node.js 20.3 or later.** Loading it with `require()` instead of `import` needs Node 20.19+ or 22.12+.
- **An SSH key with read access to the repository.** Check it with:

```bash
ssh -T git@github.com
```

  It should greet you by your GitHub username.

### Install a released version (recommended)

Install a version tag, so every install gets exactly the same code:

```bash
npm install git+ssh://git@github.com/huseyinbozkurt/llm-adapter.git#v0.1.0
```

npm records it in your app's `package.json` in GitHub shorthand. npm still fetches it over SSH:

```json
"dependencies": {
  "llm-adapter": "github:huseyinbozkurt/llm-adapter#v0.1.0"
}
```

The package name is still `llm-adapter`:

```ts
import { LLMClient } from "llm-adapter";
```

You can also install a branch (`#main`) or a commit (`#6c2c98a`) instead of a tag. Your `package-lock.json` records the exact commit either way.

### Upgrade to a newer version

Change the tag in your app's `package.json` (for example `#v0.1.0` → `#v0.2.0`), then reinstall:

```bash
npm install
```

### TypeScript projects

Type definitions are included. Set `"moduleResolution"` to `"NodeNext"`, `"Node16"` or `"Bundler"` in your app's `tsconfig.json`. The old `"node"` setting won't find them.

### CI and deployment servers

The machine that runs `npm install` needs SSH access to the repository too. For CI, add a read-only [deploy key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys) to the repository and load its private key on the CI machine. In GitHub Actions you can do this with `webfactory/ssh-agent`.

### Releasing a new version (maintainer)

Run these from this repository's `main` branch. Tests must pass before you tag.

```bash
npm test
```

```bash
npm version minor
```

```bash
git push --follow-tags
```

`npm version` (`patch`, `minor` or `major`) updates `package.json`, commits, and creates the `vX.Y.Z` tag. Pushing the tag publishes the release. Apps then install it by putting that tag in their `package.json`.

Nothing is ever sent to the public npm registry. `"private": true` in `package.json` makes `npm publish` refuse to run.

### Local development without GitHub

To try changes in another app before pushing, install from the local folder:

```bash
npm install /path/to/llm-adapter
```

This links the folder, so run `npm run build` here after each change.

## 2. Basic usage

```ts
import { LLMClient } from "llm-adapter";

const llm = new LLMClient({
  provider: "anthropic",
  apiKey: process.env.ANTHROPIC_API_KEY!,
  model: "claude-sonnet-5",
});

const response = await llm.chat({
  messages: [
    { role: "system", content: "You are helpful." },
    { role: "user", content: "Hello" },
  ],
});

console.log(response.text);          // convenience: concatenated text
console.log(response.finishReason);  // "stop" | "length" | "tool_call" | "content_filter" | "error" | "unknown"
console.log(response.usage);         // { inputTokens, outputTokens, totalTokens, cachedInputTokens?, reasoningTokens? }
console.log(response.raw);           // untouched provider payload (not portable)
```

`createLLM(config)` does the same thing as `new LLMClient(config)`.

To continue a conversation, push `response.message` back into `messages`. It is a valid assistant message. It keeps any opaque data the provider needs to see again, such as Anthropic thinking signatures or Gemini thought signatures.

```ts
messages.push(response.message, { role: "user", content: "And then?" });
```

## 3. Provider configuration

Only the config changes between providers:

```ts
new LLMClient({ provider: "openai",     apiKey: process.env.OPENAI_API_KEY!,     model: "gpt-5" });
new LLMClient({ provider: "anthropic",  apiKey: process.env.ANTHROPIC_API_KEY!,  model: "claude-opus-5" });
new LLMClient({ provider: "gemini",     apiKey: process.env.GEMINI_API_KEY!,     model: "gemini-2.5-flash" });
new LLMClient({ provider: "deepseek",   apiKey: process.env.DEEPSEEK_API_KEY!,   model: "deepseek-chat" });
new LLMClient({ provider: "openrouter", apiKey: process.env.OPENROUTER_API_KEY!, model: "anthropic/claude-sonnet-5",
                siteURL: "https://myapp.example", appName: "My App" });
new LLMClient({ provider: "groq",       apiKey: process.env.GROQ_API_KEY!,       model: "openai/gpt-oss-120b" });
new LLMClient({ provider: "mistral",    apiKey: process.env.MISTRAL_API_KEY!,    model: "mistral-large-latest" });
new LLMClient({ provider: "xai",        apiKey: process.env.XAI_API_KEY!,        model: "grok-4" });
```

Options shared by all providers:

| Option | Default | Notes |
|---|---|---|
| `model` | required | Any string. There is no hard-coded list of models. |
| `apiKey` | required* | *Optional for `openai-compatible`. |
| `baseURL` | provider default | Point at a proxy or regional endpoint. |
| `timeout` | `60_000` | Milliseconds. Also used as the idle timeout between stream chunks. |
| `retries` | `2` | A number, or `{ maxRetries, initialDelayMs, maxDelayMs, multiplier, jitter, shouldRetry }`. |
| `headers` | — | Extra headers. They are never logged. |
| `capabilities` | detected | Override what the library thinks the model supports. |
| `providerOptions` | — | Default provider-specific options. See [Escape hatch](#provider-specific-escape-hatch). |
| `defaultMaxTokens` | — | Used by providers that require `max_tokens` (Anthropic). |
| `fetch` | global `fetch` | Use it for proxies, instrumentation or tests. |
| `logger` | — | `{ debug(msg, meta) }`. It only receives method, path, status, attempt and duration. |

Provider-specific options: `organization` / `project` (OpenAI), `apiVersion` / `betas` (Anthropic), `apiVersion` (Gemini), `siteURL` / `appName` (OpenRouter).

The library checks your config and throws `ConfigurationError` with a clear message if something is wrong:

```
openai: missing required config field "apiKey".
```

### Several providers at once

Each client holds its own credentials, so clients never share state. `LLMRouter` gives you named clients. It does not make routing decisions for you.

```ts
import { LLMRouter } from "llm-adapter";

const router = new LLMRouter({
  providers: {
    fast:  { provider: "groq",      apiKey: process.env.GROQ_API_KEY!,      model: "openai/gpt-oss-20b" },
    smart: { provider: "anthropic", apiKey: process.env.ANTHROPIC_API_KEY!, model: "claude-opus-5" },
    cheap: { provider: "deepseek",  apiKey: process.env.DEEPSEEK_API_KEY!,  model: "deepseek-chat" },
  },
});

await router.chat("smart", { messages });
router.stream("fast", { messages });
router.get("cheap").capabilities();
```

## 4. Streaming

```ts
const stream = llm.stream({ messages });

for await (const event of stream) {
  switch (event.type) {
    case "text_delta":      process.stdout.write(event.text); break;
    case "reasoning_delta": /* only when the provider exposes reasoning */ break;
    case "tool_call_start": console.log("calling", event.name); break;
    case "usage":           console.log(event.usage); break;
    case "finish":          console.log("\n", event.reason); break;
  }
}

const full = await stream.finalResponse(); // assembled ChatResponse, including tool calls
```

Event types: `start`, `text_delta`, `reasoning_delta`, `reasoning_end`, `tool_call_start`, `tool_call_delta`, `tool_call_end`, `usage`, `finish`. The request is sent when you start iterating. A `ChatStream` can only be iterated once.

- Want only the final result? Call `await llm.stream({ messages }).finalResponse()`.
- Want only text? Use `for await (const t of llm.stream({ messages }).textStream())`.

The adapters hide how each provider streams: OpenAI SSE chunks, Anthropic typed block events, and Gemini partial responses all become the same events. Tool-call arguments that arrive in fragments (`{"city":` then `"Toronto"}`) are put back together for you, with a stable, dense `index` per call.

## 5. Tool calls

```ts
import type { ChatMessage, ToolDefinition } from "llm-adapter";

const tools: ToolDefinition[] = [{
  type: "function",
  function: {
    name: "get_weather",
    description: "Current weather for a city",
    parameters: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
      additionalProperties: false,
    },
  },
}];

const messages: ChatMessage[] = [{ role: "user", content: "Weather in Toronto?" }];

let res = await llm.chat({ messages, tools, toolChoice: "auto" });

while (res.toolCalls.length > 0) {
  messages.push(res.message);
  for (const call of res.toolCalls) {
    const args = JSON.parse(call.function.arguments); // always a JSON string
    const result = await getWeather(args.city);
    messages.push({ role: "tool", toolCallId: call.id, content: JSON.stringify(result) });
  }
  res = await llm.chat({ messages, tools });
}
console.log(res.text);
```

- `toolChoice` can be `"auto"`, `"none"`, `"required"`, or `{ type: "function", name }`.
- `parallelToolCalls: false` asks for at most one call per turn.
- Tool messages accept `isError: true`. Anthropic receives it as `is_error`, and Gemini receives it as an `error` payload.
- Every call gets an `id`, even from providers that don't send one (Gemini). Only ids that came from the provider are sent back to it.

## 6. Structured output

```ts
const res = await llm.chat({
  messages: [{ role: "user", content: "Extract: Ada Lovelace, born 1815" }],
  responseFormat: {
    type: "json_schema",
    name: "person",
    schema: {
      type: "object",
      properties: { name: { type: "string" }, born: { type: "integer" } },
      required: ["name", "born"],
      additionalProperties: false,
    },
    strict: true,
  },
});
const person = JSON.parse(res.text);
```

Where the provider supports schemas natively, the library uses that: OpenAI and compatible providers (`response_format`), Anthropic (`output_config.format`), and Gemini (`responseJsonSchema`).

Some providers can't enforce a schema, for example DeepSeek, or Groq models that lack native support. There the request throws `UnsupportedCapabilityError`. If you accept JSON that is valid but not checked against the schema, opt in explicitly:

```ts
responseFormat: { type: "json_schema", name: "person", schema, fallback: "json_mode" }
```

With `fallback: "json_mode"`, the library turns on the provider's JSON mode and adds the schema as a system instruction.

`{ type: "json_object" }` asks for JSON mode without a schema. Anthropic has no JSON mode, so this is unsupported there.

## 7. Images

```ts
await llm.chat({
  messages: [{
    role: "user",
    content: [
      { type: "text", text: "What's in these?" },
      { type: "image", url: "https://example.com/cat.jpg" },
      { type: "image", base64: fs.readFileSync("dog.png").toString("base64"), mimeType: "image/png" },
      { type: "image", url: "data:image/jpeg;base64,/9j/4AAQ..." },
    ],
  }],
});
```

Each image is converted to the provider's own format: `image_url` for OpenAI-style APIs, `source` for Anthropic, and `inlineData` / `fileData` for Gemini. Gemini needs `mimeType` for URL images. If the model has no vision support, the request throws `UnsupportedCapabilityError` before anything is sent.

## Reasoning models

```ts
const res = await llm.chat({
  messages,
  reasoning: { effort: "high", includeReasoning: true }, // or { maxTokens: 8000 }
});
res.reasoning;              // text, only if the provider exposed any
res.usage?.reasoningTokens; // when reported
```

| Provider | `effort` | `maxTokens` | Reasoning text |
|---|---|---|---|
| OpenAI | `reasoning_effort` | ✗ | never exposed via Chat Completions |
| Anthropic | `output_config.effort` + adaptive thinking | `budget_tokens` (older models) | `thinking` blocks |
| Gemini 3 / 2.5 | `thinkingLevel` / only `"none"`→budget 0 | `thinkingBudget` | thought summaries |
| DeepSeek | `low`/`high`/`max`; `none` disables | ✗ | `reasoning_content` |
| OpenRouter | `reasoning.effort` | `reasoning.max_tokens` | `reasoning` |
| Groq | per model family | ✗ | `reasoning` |
| xAI | on supported Grok models | ✗ | `reasoning_content` |

Each adapter maps an effort level to the nearest level the provider accepts. A control the provider or model can't honour throws `UnsupportedCapabilityError`. The library never makes up reasoning text.

## Capabilities

```ts
llm.capabilities();                  // for the default model
llm.capabilities("o3-mini");         // capabilities can differ per model
llm.supports("tools");               // boolean
```

The capability keys are: `streaming`, `tools`, `toolChoice`, `vision`, `structuredOutput`, `jsonMode`, `reasoning`, `reasoningControl`, `systemMessages`, `stopSequences`, `listModels`.

Detection comes from each provider's defaults plus per-model pattern rules. If a value is wrong for your model or server, override it with the `capabilities` config option.

`await llm.listModels()` returns `ModelInfo[]` (`id`, `displayName`, `contextWindow`, …) where the provider supports listing.

## 8. Custom OpenAI-compatible endpoints

```ts
// Ollama
new LLMClient({ provider: "openai-compatible", baseURL: "http://localhost:11434/v1", model: "llama3.2" });

// LM Studio
new LLMClient({ provider: "openai-compatible", baseURL: "http://localhost:1234/v1", model: "qwen2.5-7b-instruct" });

// vLLM / Together / Fireworks / a corporate gateway
new LLMClient({
  provider: "openai-compatible",
  name: "together",                       // shows up in responses and errors
  baseURL: "https://api.together.xyz/v1",
  apiKey: process.env.TOGETHER_API_KEY,
  model: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
  headers: { "x-team": "search" },
  query: { "api-version": "2025-01-01" }, // appended to every URL
  capabilities: { vision: false, reasoning: true, reasoningControl: true },
  quirks: {
    maxTokensField: "max_completion_tokens", // default "max_tokens"
    streamUsage: false,                      // don't send stream_options
    requiredToolChoice: "any",               // how toolChoice "required" is sent
  },
});
```

## 9. Custom providers

To add a provider, implement `LLMProvider` and register it. You don't need to change anything in the client.

```ts
import {
  BaseProvider, LLMClient, ProviderRegistry, ALL_CAPABILITIES,
  type ChatRequest, type ChatResponse, type StreamEvent, type BaseProviderConfig,
} from "llm-adapter";

interface AcmeConfig extends BaseProviderConfig { provider: "acme"; apiKey: string }

class AcmeAdapter extends BaseProvider {
  constructor(config: AcmeConfig) {
    super({
      name: "acme",
      model: config.model,
      baseURL: "https://api.acme.ai/v1",
      headers: { authorization: `Bearer ${config.apiKey}` },
      secrets: [config.apiKey],            // scrubbed from error messages
      timeout: config.timeout,
      retries: config.retries,
      defaultCapabilities: { ...ALL_CAPABILITIES, vision: false },
    });
  }

  async chat(request: ChatRequest): Promise<ChatResponse> {
    // this.http gives you timeouts, retries, AbortSignal and normalised errors
    const { data } = await this.http.json({ path: "generate", body: { /* translate */ }, ...this.transport(request) });
    /* translate `data` into a ChatResponse */
  }

  async *stream(request: ChatRequest): AsyncGenerator<StreamEvent> {
    /* use this.http.stream() + parseSSE(), then yield normalised events */
  }
}

ProviderRegistry.register("acme", (config) => new AcmeAdapter(config as AcmeConfig));
const llm = new LLMClient({ provider: "acme", apiKey: "...", model: "acme-1" });
```

For an OpenAI-like API, extend `OpenAICompatibleAdapter` instead. Then override only the hooks you need (`applyReasoning`, `mapEffort`, `deltaContent`, `parseUsage`, `assistantReasoningFields`, `mapToolCallId`, `finalizeBody`). This is how the DeepSeek, Groq, Mistral, xAI and OpenRouter adapters are built.

For isolated registries (tests, multi-tenant setups), use `new ProviderRegistry()` and `registerBuiltinProviders(registry)`, then pass it in with `new LLMClient(config, { registry })`. You can also pass any `LLMProvider` instance straight to `new LLMClient(provider)`.

### Middleware

```ts
llm.use(async (ctx, next) => {
  const started = Date.now();
  const result = await next();
  console.log({ provider: ctx.provider, model: ctx.model, ms: Date.now() - started, usage: result.usage });
  return result;
});

// The object form can also wrap streams
llm.use({
  stream: async function* (ctx, next) {
    for await (const event of next()) {
      if (event.type === "usage") record(ctx.model, event.usage);
      yield event;
    }
  },
});
```

Middleware may replace `ctx.request` before calling `next()`. The capability check runs after all middleware, so a rewritten request is still validated. Typical uses for middleware are logging, tracing, cost tracking (from `usage`), caching, rate limiting and fallback.

### Provider-specific escape hatch

```ts
await llm.chat({
  messages,
  providerOptions: {
    anthropic: { body: { top_k: 20 }, headers: { "anthropic-beta": "some-beta" } },
    openai:    { body: { service_tier: "flex", seed: 7 } },
    openrouter:{ body: { provider: { order: ["anthropic"] } } },
  },
});

// or set them as defaults for a client:
const tuned = llm.withProviderOptions({ openai: { body: { seed: 7 } } });
```

The library only reads the entry that matches the active provider. It merges that entry's `body` into the native request last, so it can override anything. This code is not portable, by design.

## Request controllers

The library has four controllers. Each one is either a provider or a piece of middleware, so they combine with each other and with your own middleware.

### Fallback

`FallbackProvider` tries its targets in order until one succeeds.

```ts
import { FallbackProvider, LLMClient } from "llm-adapter";

const llm = new LLMClient(new FallbackProvider({
  targets: {
    primary: { provider: "anthropic", apiKey: process.env.ANTHROPIC_API_KEY!, model: "claude-opus-5" },
    backup:  { provider: "openai",    apiKey: process.env.OPENAI_API_KEY!,    model: "gpt-5" },
  },
  onFallback: ({ target, error }) => console.warn(`${target} failed`, error),
}));

const res = await llm.chat({ messages });
res.provider; // the target that actually answered
```

- **What triggers a fallback.** Errors another target might not share: rate limits or quota, 5xx, network errors, timeouts, bad credentials or an unknown model on that target, context length, capability gaps, and local budgets. You can change this with `shouldFallback`.
- **What never triggers one.** Caller aborts, content-policy rejections and generally invalid requests, because those would fail on every target.
- **Capabilities.** Targets whose capabilities can't serve the request are skipped.
- **Streams.** A stream falls back only if it fails before its first event. After output has started, the error is rethrown.
- **All targets fail.** You get a `FallbackExhaustedError` whose `.errors` lists every attempt.
- **Models.** A per-request `model` is rejected, because model ids differ between providers. Set the model on each target instead.

### Routing

`RoutingProvider` picks one target per request using rules you write. It never makes decisions on its own.

```ts
import { RoutingProvider, hasImages, estimateInputTokens } from "llm-adapter";

const llm = new LLMClient(new RoutingProvider({
  targets: { fast: groqConfig, vision: openaiConfig, smart: anthropicConfig },
  rules: [
    { name: "images",  when: ({ request }) => hasImages(request), use: "vision" },
    { name: "premium", when: ({ request }) => request.metadata?.tier === "premium", use: "smart" },
    { name: "long",    when: ({ request }) => estimateInputTokens(request) > 50_000, use: "smart" },
  ],
  default: "fast",
}));
```

- The first rule that matches and whose target can serve the request wins. If none does, the `default` target is used.
- A target can itself be a `FallbackProvider`, for example `{ resilient: new FallbackProvider({...}) }`.
- `estimateInputTokens` is a rough character-based estimate. Use it for coarse routing only.

### Rate limiting

```ts
import { rateLimit } from "llm-adapter";

llm.use(rateLimit({
  requests: { limit: 60, intervalMs: 60_000 }, // sliding window
  maxConcurrent: 4,                           // a stream holds its slot until it ends
  maxWaitMs: 10_000,                          // after this, throw RateLimitError (code "local_rate_limit")
  key: (ctx) => String(ctx.request.metadata?.userId), // optional: a separate limit per key
}));
```

- Waiting is first-come, first-served, and it stops if the request's `AbortSignal` fires.
- To share one limit across several clients, reuse the same middleware instance on each of them.
- Inside a `FallbackProvider`, a target that hits its local limit makes the provider fall back to the next target.

### Budgets

```ts
import { BudgetController } from "llm-adapter";

const budget = new BudgetController({
  limits: { maxTotalTokens: 2_000_000, maxCost: 25 },
  // Your own rates, per million tokens. No prices ship with the library.
  pricing: (provider, model) => myPriceTable[`${provider}/${model}`],
  windowMs: 24 * 60 * 60 * 1000,                  // reset daily
  key: (ctx) => String(ctx.request.metadata?.tenantId),
  onUsage: (record) => metrics.record(record),
});

llm.use(budget.middleware());
budget.spend("tenant-42");     // { requests, inputTokens, outputTokens, totalTokens, cost, ... }
budget.remaining("tenant-42"); // { maxTotalTokens, maxCost }
```

- **When a limit is reached**, the next request throws `BudgetExceededError` before anything is sent.
- **Soft limits.** Requests already in flight can push the total past a limit.
- **Unpriced models.** If `maxCost` is set and your `pricing` function has no price for a model, the request is refused before it is sent.
- **Streams.** Streams are metered from their `usage` event, including streams you stop reading early. Calls whose provider reports no usage are counted under `unmeteredRequests`.
- **With fallback or routing.** Pricing looks up the client's own provider and model. When you use fallback or routing, attach the budget middleware to the target clients. One controller can be shared by all of them.

## 10. Error handling

```ts
import {
  LLMError, AuthenticationError, RateLimitError, TimeoutError, AbortError,
  InvalidRequestError, ContextLengthExceededError, ContentFilterError,
  ModelNotFoundError, ProviderError, NetworkError, UnsupportedCapabilityError,
  BudgetExceededError, FallbackExhaustedError,
} from "llm-adapter";

try {
  await llm.chat({ messages });
} catch (err) {
  if (err instanceof ContextLengthExceededError) { /* trim history */ }
  else if (err instanceof RateLimitError) { console.log(err.retryAfterMs, err.retryable); }
  else if (err instanceof UnsupportedCapabilityError) { console.log(err.capability); }
  else if (err instanceof LLMError) {
    console.log(err.provider, err.statusCode, err.code, err.requestId, err.retryable, err.cause);
  }
}
```

- **Retries.** Retryable failures are retried automatically: 429 (except quota exhaustion), 408, 409, 5xx, and transient network errors. The library uses exponential backoff with jitter and respects `Retry-After`. It never retries authentication errors, invalid requests, content-policy rejections, context-length errors or aborts. Streams are only retried before the first event arrives.
- **Secrets.** API keys never show up in error messages, in `cause`, or in logs.

## 11. Cancellation and timeouts

```ts
const controller = new AbortController();
setTimeout(() => controller.abort(), 5_000);

try {
  for await (const e of llm.stream({ messages, signal: controller.signal })) { /* ... */ }
} catch (err) {
  if (err instanceof AbortError) { /* the caller cancelled */ }
  if (err instanceof TimeoutError) { /* the configured timeout elapsed */ }
}

await llm.chat({ messages, timeout: 10_000 }); // per-request timeout
```

Breaking out of a `for await` loop also closes the connection.

## 12. Testing

```bash
npm test                  # unit + mocked-provider tests (no API keys needed)
npm run typecheck
npm run test:integration  # live tests; each provider runs only if its key is set
```

The integration tests read these keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, `XAI_API_KEY`, `OPENROUTER_API_KEY`. To pick a model, set `<PROVIDER>_MODEL`, for example `OPENAI_MODEL=gpt-5-mini`.

To test your own code without the network, pass a mock `fetch` in the config, or give the client an `LLMProvider` stub:

```ts
const llm = new LLMClient({ provider: "openai", apiKey: "test", model: "gpt-5", fetch: myMockFetch });
```

## Usage conventions

Every adapter reports token usage the same way:

- `inputTokens` includes cached tokens.
- `outputTokens` includes reasoning tokens.
- `cachedInputTokens`, `cacheWriteInputTokens` and `reasoningTokens` are subsets of those totals.

This makes it possible to compute cost later from a separate pricing table. Prices are deliberately not in the adapters.
