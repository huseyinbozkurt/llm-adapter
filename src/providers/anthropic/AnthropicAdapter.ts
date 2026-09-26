import { ALL_CAPABILITIES, type ModelCapabilityRule } from "../../core/capabilities.js";
import { requireString, validateBaseConfig, type AnthropicConfig } from "../../core/config.js";
import type { ChatRequest, ChatResponse, ModelInfo, ReasoningEffort, StreamEvent } from "../../core/types.js";
import { compact, getArray, getNumber, getString, isRecord, tryParseJson, type JsonRecord } from "../../utils/json.js";
import { finishReasonFor, reasoningOf, textOf } from "../../utils/normalize.js";
import { parseSSE } from "../../utils/sse.js";
import { BaseProvider, commonSettings } from "../base.js";
import { AnthropicStreamProcessor } from "./stream.js";
import {
  assertMessage,
  normalizeAnthropicStopReason,
  parseAnthropicContent,
  parseAnthropicUsage,
  PROVIDER,
  toAnthropicMessages,
  toAnthropicToolChoice,
  toAnthropicTools,
} from "./translate.js";

/**
 * Models that predate adaptive thinking: they take
 * `thinking: { type: "enabled", budget_tokens }` and reject `adaptive`.
 */
const BUDGET_THINKING_MODELS = /^claude-(3-7-sonnet|sonnet-4-(5|\d{8})|opus-4-(1|5|\d{8})|haiku-4-5)/;
const CLAUDE_3 = /^claude-3-(?!7)/;

export const ANTHROPIC_MODEL_RULES: readonly ModelCapabilityRule[] = [
  { match: /^claude-3-(?!7)/, capabilities: { reasoning: false, reasoningControl: false, structuredOutput: false } },
  { match: /^claude-(sonnet|opus)-4-\d{8}/, capabilities: { structuredOutput: false } },
];

/**
 * Anthropic Messages API (native, via fetch).
 *
 * Reasoning mapping:
 * - `reasoning.maxTokens` → `thinking: { type: "enabled", budget_tokens }`
 *   (older models only; current models reject budgets with a 400).
 * - `reasoning.effort` → `output_config.effort` plus adaptive thinking on
 *   models that support it. `minimal` maps to `low`; `none` disables thinking.
 * - `includeReasoning: true` → `thinking.display: "summarized"` on current
 *   models (they return empty thinking text by default).
 *
 * Anthropic has no schema-less JSON mode, so `json_object` is unsupported;
 * `json_schema` uses native `output_config.format`.
 */
export class AnthropicAdapter extends BaseProvider {
  private readonly defaultMaxTokens: number | undefined;

  constructor(config: AnthropicConfig) {
    validateBaseConfig(config);
    const apiKey = requireString(config, "apiKey");
    super({
      ...commonSettings(config),
      name: PROVIDER,
      baseURL: config.baseURL ?? "https://api.anthropic.com/v1",
      headers: {
        "x-api-key": apiKey,
        "anthropic-version": config.apiVersion ?? "2023-06-01",
        ...(config.betas && config.betas.length > 0 ? { "anthropic-beta": config.betas.join(",") } : {}),
        ...config.headers,
      },
      secrets: [apiKey],
      defaultCapabilities: { ...ALL_CAPABILITIES, jsonMode: false, reasoning: true, reasoningControl: true },
      capabilityRules: ANTHROPIC_MODEL_RULES,
    });
    this.defaultMaxTokens = config.defaultMaxTokens;
  }

  static fromConfig(config: AnthropicConfig): AnthropicAdapter {
    return new AnthropicAdapter(config);
  }

  async chat(request: ChatRequest): Promise<ChatResponse> {
    const model = this.modelFor(request);
    const { data } = await this.http.json({ path: "messages", body: this.buildBody(request, model, false), ...this.transport(request) });
    const msg = assertMessage(data);
    const { content, toolCalls } = parseAnthropicContent(msg.content as unknown[]);
    const stopReason = getString(msg, "stop_reason");
    const reasoning = reasoningOf(content);
    return {
      id: getString(msg, "id"),
      provider: this.name,
      model: getString(msg, "model") ?? model,
      message: { role: "assistant", content, ...(toolCalls.length > 0 ? { toolCalls } : {}) },
      text: textOf(content),
      ...(reasoning !== undefined ? { reasoning } : {}),
      toolCalls,
      finishReason: finishReasonFor(toolCalls, normalizeAnthropicStopReason(stopReason)),
      rawFinishReason: stopReason,
      usage: parseAnthropicUsage(msg.usage),
      raw: data,
    };
  }

  async *stream(request: ChatRequest): AsyncGenerator<StreamEvent> {
    const model = this.modelFor(request);
    const { body } = await this.http.stream({ path: "messages", body: this.buildBody(request, model, true), ...this.transport(request) });
    const processor = new AnthropicStreamProcessor();
    for await (const message of parseSSE(body)) {
      const payload = tryParseJson(message.data);
      if (payload === undefined) continue;
      yield* processor.process(message.event, payload);
    }
    yield* processor.finish();
  }

  async listModels(options: { signal?: AbortSignal } = {}): Promise<ModelInfo[]> {
    const models: ModelInfo[] = [];
    let afterId: string | undefined;
    // Paginated with `has_more` / `last_id`; bounded to avoid runaway loops.
    for (let page = 0; page < 50; page++) {
      const { data } = await this.http.json({
        path: "models",
        query: afterId ? { limit: "1000", after_id: afterId } : { limit: "1000" },
        signal: options.signal,
      });
      for (const m of getArray(data, "data")) {
        const id = getString(m, "id");
        if (!id) continue;
        const created = getString(m, "created_at");
        models.push(
          compact({
            id,
            provider: this.name,
            displayName: getString(m, "display_name"),
            contextWindow: getNumber(m, "max_input_tokens"),
            maxOutputTokens: getNumber(m, "max_tokens"),
            createdAt: created ? new Date(created) : undefined,
            raw: m,
          }),
        );
      }
      if (!isRecord(data) || data.has_more !== true) break;
      afterId = getString(data, "last_id");
      if (!afterId) break;
    }
    return models;
  }

  // -------------------------------------------------------------------------

  private buildBody(request: ChatRequest, model: string, stream: boolean): JsonRecord {
    const { system, messages } = toAnthropicMessages(request.messages);
    const body: JsonRecord = { model, messages };
    if (system) body.system = system;

    const budget = request.reasoning?.maxTokens;
    let maxTokens = request.maxTokens ?? this.defaultMaxTokens ?? (CLAUDE_3.test(model) ? 4096 : 16_000);
    // budget_tokens must be below max_tokens; only adjust when the caller left max_tokens unset.
    if (budget !== undefined && request.maxTokens === undefined && maxTokens <= budget) maxTokens = budget + 4096;
    body.max_tokens = maxTokens;

    if (request.temperature !== undefined) body.temperature = request.temperature;
    if (request.topP !== undefined) body.top_p = request.topP;
    if (request.stop && request.stop.length > 0) body.stop_sequences = request.stop;

    const tools = toAnthropicTools(request);
    if (tools) {
      body.tools = tools;
      const choice = toAnthropicToolChoice(request.toolChoice, request.parallelToolCalls);
      if (choice) body.tool_choice = choice;
    }

    const outputConfig: JsonRecord = {};
    if (request.responseFormat?.type === "json_schema") {
      outputConfig.format = { type: "json_schema", schema: request.responseFormat.schema };
    }
    this.applyReasoning(body, outputConfig, request, model);
    if (Object.keys(outputConfig).length > 0) body.output_config = outputConfig;

    if (stream) body.stream = true;
    return { ...body, ...this.providerOptionsFor(request).body };
  }

  private applyReasoning(body: JsonRecord, outputConfig: JsonRecord, request: ChatRequest, model: string): void {
    const r = request.reasoning;
    if (!r) return;
    const legacy = BUDGET_THINKING_MODELS.test(model);

    if (r.effort === "none") {
      body.thinking = { type: "disabled" };
      return;
    }
    if (r.maxTokens !== undefined) {
      body.thinking = { type: "enabled", budget_tokens: r.maxTokens };
    } else if (!legacy && (r.effort !== undefined || r.includeReasoning)) {
      body.thinking = { type: "adaptive", ...(r.includeReasoning ? { display: "summarized" } : {}) };
    }
    if (r.effort !== undefined) outputConfig.effort = mapEffort(r.effort);
  }
}

function mapEffort(effort: Exclude<ReasoningEffort, "none">): string {
  return effort === "minimal" ? "low" : effort;
}
