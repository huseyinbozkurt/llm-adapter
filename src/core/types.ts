/**
 * Provider-independent request / response / stream types.
 *
 * Nothing in this file references a specific provider's wire format. Adapters
 * translate to and from these shapes.
 */

// ---------------------------------------------------------------------------
// Content parts
// ---------------------------------------------------------------------------

export interface TextPart {
  type: "text";
  text: string;
}

/**
 * An image input. Provide exactly one of `url` or `base64`.
 * `url` may be an http(s) URL or a `data:` URL.
 */
export interface ImagePart {
  type: "image";
  url?: string;
  base64?: string;
  /** Required with `base64`; recommended for URLs on providers that need it (Gemini). */
  mimeType?: string;
  /** Resolution hint. Only honoured by providers that support it (OpenAI-style APIs). */
  detail?: "auto" | "low" | "high";
}

/**
 * Reasoning ("thinking") produced by the model.
 *
 * Only returned when the provider actually exposes reasoning text or an opaque
 * reasoning block; the library never synthesises it. `text` may be an empty
 * string when the provider returns an opaque/redacted block.
 *
 * `providerData` holds opaque values (e.g. Anthropic thinking signatures) that
 * must be sent back unchanged when the assistant message is replayed in a
 * later request to the *same* provider. Other providers ignore reasoning parts
 * in input.
 */
export interface ReasoningPart {
  type: "reasoning";
  text: string;
  providerData?: ProviderData;
}

export type ContentPart = TextPart | ImagePart | ReasoningPart;

/** Parts allowed in user-authored content. */
export type UserContentPart = TextPart | ImagePart;

export type MessageContent = string | ContentPart[];

/**
 * Opaque provider-owned values keyed by provider name. Applications should
 * treat this as a black box and simply keep it attached to the message.
 */
export type ProviderData = Record<string, Record<string, unknown>>;

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

export type Role = "system" | "user" | "assistant" | "tool";

export interface SystemMessage {
  role: "system";
  content: string | TextPart[];
}

export interface UserMessage {
  role: "user";
  content: string | UserContentPart[];
  /** Participant name. Only forwarded to providers that support it. */
  name?: string;
}

export interface AssistantMessage {
  role: "assistant";
  content: MessageContent;
  toolCalls?: ToolCall[];
  name?: string;
}

export interface ToolMessage {
  role: "tool";
  /** The `ToolCall.id` this message answers. */
  toolCallId: string;
  content: string | UserContentPart[];
  /** Tool name. Needed by providers that key tool results by name (Gemini); inferred from history when omitted. */
  name?: string;
  /** Marks the result as an error. Mapped where supported (Anthropic `is_error`). */
  isError?: boolean;
}

export type ChatMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage;

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

export interface JsonSchemaObject {
  type: "object";
  properties?: Record<string, unknown>;
  required?: string[];
  additionalProperties?: boolean;
  [keyword: string]: unknown;
}

export interface ToolDefinition {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters: JsonSchemaObject;
    /** Request strict schema adherence where the provider supports it. */
    strict?: boolean;
  };
}

export interface ToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    /** JSON-encoded arguments exactly as produced by the model (may be invalid JSON). */
    arguments: string;
  };
  /** Opaque values that must round-trip with the call (e.g. Gemini thought signatures). */
  providerData?: ProviderData;
}

export type ToolChoice = "auto" | "none" | "required" | { type: "function"; name: string };

// ---------------------------------------------------------------------------
// Structured output
// ---------------------------------------------------------------------------

export type ResponseFormat =
  | { type: "text" }
  | { type: "json_object" }
  | {
      type: "json_schema";
      name: string;
      description?: string;
      schema: Record<string, unknown>;
      strict?: boolean;
      /**
       * What to do when the provider/model has no native schema-constrained
       * output. Default `"error"` throws UnsupportedCapabilityError.
       * `"json_mode"` uses the provider's JSON mode and adds the schema as a
       * system instruction; the output is valid JSON but NOT guaranteed to
       * match the schema.
       */
      fallback?: "error" | "json_mode";
    };

// ---------------------------------------------------------------------------
// Reasoning
// ---------------------------------------------------------------------------

/**
 * Portable effort scale. Adapters map to the nearest level the provider
 * accepts (documented per adapter); `none` asks the provider to disable
 * reasoning where that is possible.
 */
export type ReasoningEffort = "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

export interface ReasoningOptions {
  effort?: ReasoningEffort;
  /** Token budget for reasoning, on providers with budget-style controls. */
  maxTokens?: number;
  /**
   * Ask the provider to return readable reasoning (summaries / thoughts) when
   * it can. Providers that never expose reasoning simply return none.
   */
  includeReasoning?: boolean;
}

// ---------------------------------------------------------------------------
// Provider-specific escape hatch
// ---------------------------------------------------------------------------

export interface ProviderSpecificOptions {
  /** Merged (shallowly, last) into the native request body. Not portable. */
  body?: Record<string, unknown>;
  /** Extra HTTP headers for this request. */
  headers?: Record<string, string>;
}

/** Keyed by provider name, e.g. `{ anthropic: { body: { top_k: 20 } } }`. Only the active provider's entry is used. */
export type ProviderOptions = Partial<Record<string, ProviderSpecificOptions>>;

// ---------------------------------------------------------------------------
// Request / response
// ---------------------------------------------------------------------------

export interface ChatRequest {
  messages: ChatMessage[];
  /** Overrides the client's default model for this request. */
  model?: string;
  temperature?: number;
  topP?: number;
  maxTokens?: number;
  stop?: string[];
  tools?: ToolDefinition[];
  toolChoice?: ToolChoice;
  /** Allow multiple tool calls in one turn. Omitted = provider default. */
  parallelToolCalls?: boolean;
  responseFormat?: ResponseFormat;
  reasoning?: ReasoningOptions;
  /** Free-form application metadata. Visible to middleware; never sent to providers. */
  metadata?: Record<string, unknown>;
  providerOptions?: ProviderOptions;
  signal?: AbortSignal;
  /** Per-request timeout override (ms). */
  timeout?: number;
}

export type FinishReason = "stop" | "length" | "tool_call" | "content_filter" | "error" | "unknown";

/**
 * Normalised token usage. Conventions (applied by every adapter):
 * - `inputTokens` is the full prompt size INCLUDING cached tokens.
 * - `outputTokens` INCLUDES reasoning tokens.
 * - Breakdown fields are subsets of the totals above and only set when reported.
 */
export interface TokenUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  /** Prompt tokens served from cache. */
  cachedInputTokens?: number;
  /** Prompt tokens written to cache (Anthropic). */
  cacheWriteInputTokens?: number;
  reasoningTokens?: number;
}

export interface ChatResponse {
  id?: string;
  provider: string;
  model: string;
  /** Can be pushed back into `messages` as-is to continue the conversation. */
  message: AssistantMessage & { content: ContentPart[] };
  /** Concatenated text parts, for convenience. */
  text: string;
  /** Concatenated reasoning text, when the provider exposed any. */
  reasoning?: string;
  toolCalls: ToolCall[];
  finishReason: FinishReason;
  /** The provider's own finish/stop reason string, before normalisation. */
  rawFinishReason?: string;
  usage?: TokenUsage;
  /** The untouched provider response body. Not portable. */
  raw?: unknown;
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

export type StreamEvent =
  | { type: "start"; id?: string; model?: string }
  | { type: "text_delta"; text: string }
  | { type: "reasoning_delta"; text: string }
  /** Closes the current reasoning block; carries opaque replay data when the provider sends any. */
  | { type: "reasoning_end"; providerData?: ProviderData }
  | { type: "tool_call_start"; index: number; id: string; name: string; providerData?: ProviderData }
  | { type: "tool_call_delta"; index: number; argumentsDelta: string }
  | { type: "tool_call_end"; index: number }
  | { type: "usage"; usage: TokenUsage }
  | { type: "finish"; reason: FinishReason; rawReason?: string };

export type StreamEventType = StreamEvent["type"];

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

export interface ModelInfo {
  id: string;
  provider: string;
  displayName?: string;
  contextWindow?: number;
  maxOutputTokens?: number;
  createdAt?: Date;
  raw?: unknown;
}
