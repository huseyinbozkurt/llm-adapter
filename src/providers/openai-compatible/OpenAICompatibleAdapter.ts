import { ALL_CAPABILITIES, type ModelCapabilityRule, type ProviderCapabilities } from "../../core/capabilities.js";
import { requireString, validateBaseConfig, type OpenAICompatibleConfig, type OpenAICompatibleQuirks } from "../../core/config.js";
import { ProviderError, UnsupportedCapabilityError } from "../../core/errors.js";
import type {
  ChatRequest,
  ChatResponse,
  ContentPart,
  FinishReason,
  ModelInfo,
  ReasoningEffort,
  ReasoningPart,
  StreamEvent,
  TokenUsage,
  ToolCall,
} from "../../core/types.js";
import { asArray, compact, getArray, getNumber, getRecord, getString, isRecord, tryParseJson, type JsonRecord } from "../../utils/json.js";
import {
  finalizeUsage,
  finishReasonFor,
  generateId,
  normalizeOpenAIFinishReason,
  reasoningOf,
  textOf,
} from "../../utils/normalize.js";
import { parseSSE } from "../../utils/sse.js";
import { BaseProvider, commonSettings, type BaseProviderSettings } from "../base.js";
import { toOpenAIMessages, toOpenAIResponseFormat, toOpenAITools, toOpenAIToolChoice } from "./messages.js";
import { OpenAIStreamProcessor } from "./stream.js";

/** Settings for the adapter and its subclasses (derived from their public configs). */
export interface OpenAICompatibleSettings extends Omit<BaseProviderSettings, "headers" | "secrets"> {
  apiKey?: string;
  headers?: Record<string, string>;
  quirks?: OpenAICompatibleQuirks;
}

/**
 * Conservative defaults for an arbitrary endpoint: the common Chat Completions
 * surface is assumed, reasoning controls are not. Override per server via
 * `config.capabilities`.
 */
export const OPENAI_COMPATIBLE_DEFAULT_CAPABILITIES: ProviderCapabilities = {
  ...ALL_CAPABILITIES,
  reasoning: false,
  reasoningControl: false,
};

/**
 * Generic adapter for any OpenAI Chat Completions–compatible API.
 *
 * Provider-specific adapters extend this class and override the protected
 * hooks below instead of duplicating request/response translation.
 */
export class OpenAICompatibleAdapter extends BaseProvider {
  protected readonly quirks: Required<OpenAICompatibleQuirks>;

  constructor(settings: OpenAICompatibleSettings) {
    const headers: Record<string, string> = { accept: "application/json", ...settings.headers };
    if (settings.apiKey) headers.authorization = `Bearer ${settings.apiKey}`;
    super({ ...settings, headers, secrets: [settings.apiKey] });
    this.quirks = {
      maxTokensField: settings.quirks?.maxTokensField ?? "max_tokens",
      streamUsage: settings.quirks?.streamUsage ?? true,
      requiredToolChoice: settings.quirks?.requiredToolChoice ?? "required",
    };
  }

  /** Factory used by the registry for `provider: "openai-compatible"`. */
  static fromCompatibleConfig(config: OpenAICompatibleConfig): OpenAICompatibleAdapter {
    validateBaseConfig(config);
    requireString(config, "baseURL", 'e.g. "http://localhost:11434/v1" for Ollama');
    return new OpenAICompatibleAdapter({
      ...commonSettings(config),
      name: config.name ?? "openai-compatible",
      baseURL: config.baseURL,
      apiKey: config.apiKey,
      headers: config.headers,
      query: config.query,
      quirks: config.quirks,
      defaultCapabilities: OPENAI_COMPATIBLE_DEFAULT_CAPABILITIES,
    });
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  async chat(request: ChatRequest): Promise<ChatResponse> {
    const model = this.modelFor(request);
    const body = this.buildBody(request, model, false);
    const { data } = await this.http.json({ path: this.chatPath(), body, ...this.transport(request) });
    return this.parseResponse(data, model);
  }

  async *stream(request: ChatRequest): AsyncGenerator<StreamEvent> {
    const model = this.modelFor(request);
    const body = this.buildBody(request, model, true);
    const { body: bytes } = await this.http.stream({ path: this.chatPath(), body, ...this.transport(request) });
    const processor = new OpenAIStreamProcessor({
      provider: this.name,
      deltaContent: (delta) => this.deltaContent(delta),
      usage: (chunk) => this.parseUsage(chunk),
      finishReason: (raw) => this.normalizeFinishReason(raw),
    });
    for await (const message of parseSSE(bytes)) {
      if (message.data === "[DONE]") break;
      const parsed = tryParseJson(message.data);
      if (parsed === undefined) continue; // tolerate non-JSON keep-alives
      yield* processor.process(parsed);
    }
    yield* processor.finish();
  }

  async listModels(options: { signal?: AbortSignal } = {}): Promise<ModelInfo[]> {
    const { data } = await this.http.json({ path: "models", signal: options.signal });
    return getArray(data, "data").flatMap((m): ModelInfo[] => {
      const id = getString(m, "id");
      if (!id) return [];
      const created = getNumber(m, "created");
      return [
        compact({
          id,
          provider: this.name,
          displayName: getString(m, "name"),
          contextWindow: getNumber(m, "context_length") ?? getNumber(m, "context_window") ?? getNumber(m, "max_context_length"),
          createdAt: created !== undefined ? new Date(created * 1000) : undefined,
          raw: m,
        }),
      ];
    });
  }

  // -------------------------------------------------------------------------
  // Hooks for subclasses
  // -------------------------------------------------------------------------

  protected chatPath(): string {
    return "chat/completions";
  }

  /** Maps unified reasoning options onto the body. Default: OpenAI-style `reasoning_effort`. */
  protected applyReasoning(body: JsonRecord, request: ChatRequest, model: string): void {
    const r = request.reasoning;
    if (!r) return;
    if (r.effort !== undefined) body.reasoning_effort = this.mapEffort(r.effort, model);
    else if (r.maxTokens !== undefined) {
      throw new UnsupportedCapabilityError(
        "reasoningControl",
        `${this.name}: reasoning.maxTokens is not supported; use reasoning.effort.`,
        { provider: this.name },
      );
    }
  }

  /** Maps the portable effort scale to the provider's accepted values. Default: pass through. */
  protected mapEffort(effort: ReasoningEffort, _model: string): string {
    return effort;
  }

  /** Extra fields on replayed assistant messages built from reasoning parts. Default: none. */
  protected assistantReasoningFields(_parts: ReasoningPart[]): JsonRecord | undefined {
    return undefined;
  }

  protected mapToolCallId(id: string): string {
    return id;
  }

  protected supportsMessageName(): boolean {
    return false;
  }

  /** Final chance to adjust the body (after all common fields, before provider options). */
  protected finalizeBody(_body: JsonRecord, _request: ChatRequest, _model: string): void {}

  /** Reads text + reasoning from a non-streaming `message` or a streaming `delta`. */
  protected deltaContent(obj: JsonRecord): { text?: string; reasoning?: string } {
    // Servers disagree on the reasoning field: DeepSeek / xAI / vLLM use
    // `reasoning_content`, OpenRouter / Groq / Ollama use `reasoning`.
    const reasoning = getString(obj, "reasoning_content") ?? getString(obj, "reasoning");
    const content = obj.content;
    let text: string | undefined;
    if (typeof content === "string") text = content;
    else if (Array.isArray(content)) {
      text = content.map((p) => (getString(p, "type") === "text" ? (getString(p, "text") ?? "") : "")).join("");
    }
    return { text: text || undefined, reasoning: reasoning || undefined };
  }

  protected parseUsage(obj: unknown): TokenUsage | undefined {
    const u = getRecord(obj, "usage");
    if (!u) return undefined;
    const input = getNumber(u, "prompt_tokens");
    const output = getNumber(u, "completion_tokens");
    return finalizeUsage({
      inputTokens: input,
      outputTokens: output,
      totalTokens: getNumber(u, "total_tokens"),
      // OpenAI: prompt_tokens_details.cached_tokens; DeepSeek: prompt_cache_hit_tokens.
      cachedInputTokens: getNumber(getRecord(u, "prompt_tokens_details"), "cached_tokens") ?? getNumber(u, "prompt_cache_hit_tokens"),
      reasoningTokens: getNumber(getRecord(u, "completion_tokens_details"), "reasoning_tokens"),
    });
  }

  protected normalizeFinishReason(raw: string): FinishReason {
    return normalizeOpenAIFinishReason(raw);
  }

  // -------------------------------------------------------------------------
  // Translation
  // -------------------------------------------------------------------------

  protected buildBody(request: ChatRequest, model: string, stream: boolean): JsonRecord {
    const body: JsonRecord = {
      model,
      messages: toOpenAIMessages(request.messages, {
        provider: this.name,
        assistantReasoningFields: (parts) => this.assistantReasoningFields(parts),
        mapToolCallId: (id) => this.mapToolCallId(id),
        supportsName: this.supportsMessageName(),
      }),
    };
    if (request.temperature !== undefined) body.temperature = request.temperature;
    if (request.topP !== undefined) body.top_p = request.topP;
    if (request.maxTokens !== undefined) body[this.quirks.maxTokensField] = request.maxTokens;
    if (request.stop && request.stop.length > 0) body.stop = request.stop;
    if (request.tools && request.tools.length > 0) {
      body.tools = toOpenAITools(request.tools);
      if (request.toolChoice) body.tool_choice = toOpenAIToolChoice(request.toolChoice, this.quirks.requiredToolChoice);
      if (request.parallelToolCalls !== undefined) body.parallel_tool_calls = request.parallelToolCalls;
    }
    if (request.responseFormat) {
      const rf = toOpenAIResponseFormat(request.responseFormat);
      if (rf) body.response_format = rf;
    }
    this.applyReasoning(body, request, model);
    if (stream) {
      body.stream = true;
      if (this.quirks.streamUsage) body.stream_options = { include_usage: true };
    }
    this.finalizeBody(body, request, model);
    return { ...body, ...this.providerOptionsFor(request).body };
  }

  protected parseResponse(data: unknown, model: string): ChatResponse {
    if (!isRecord(data)) throw new ProviderError(`${this.name}: unexpected response body`, { provider: this.name, cause: data });
    const choice = asArray(data.choices)[0];
    const message = getRecord(choice, "message");
    if (!message) {
      throw new ProviderError(`${this.name}: response contained no choices`, { provider: this.name, cause: data });
    }

    const { text, reasoning } = this.deltaContent(message);
    const content: ContentPart[] = [];
    if (reasoning) content.push({ type: "reasoning", text: reasoning });
    if (text) content.push({ type: "text", text });

    const refusal = getString(message, "refusal");
    if (refusal && !text) content.push({ type: "text", text: refusal });

    const toolCalls: ToolCall[] = asArray(message.tool_calls).flatMap((tc): ToolCall[] => {
      const fn = getRecord(tc, "function");
      const name = getString(fn, "name");
      if (!name) return [];
      const args = fn?.arguments;
      return [
        {
          id: getString(tc, "id") ?? generateId("call"),
          type: "function",
          // Some compatible servers return arguments as an object rather than a JSON string.
          function: { name, arguments: typeof args === "string" ? args : JSON.stringify(args ?? {}) },
        },
      ];
    });

    const rawFinish = getString(choice, "finish_reason");
    let finishReason = finishReasonFor(toolCalls, rawFinish ? this.normalizeFinishReason(rawFinish) : "unknown");
    // OpenAI structured outputs report a model refusal in `message.refusal` with finish_reason "stop".
    if (refusal && !text) finishReason = "content_filter";

    const reasoningText = reasoningOf(content);
    return {
      id: getString(data, "id"),
      provider: this.name,
      model: getString(data, "model") ?? model,
      message: { role: "assistant", content, ...(toolCalls.length > 0 ? { toolCalls } : {}) },
      text: textOf(content),
      ...(reasoningText !== undefined ? { reasoning: reasoningText } : {}),
      toolCalls,
      finishReason,
      rawFinishReason: rawFinish,
      usage: this.parseUsage(data),
      raw: data,
    };
  }
}

/** Shorthand used by subclasses to declare model rules. */
export type CapabilityRules = readonly ModelCapabilityRule[];
