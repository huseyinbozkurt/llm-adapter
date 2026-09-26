import { ALL_CAPABILITIES, type ModelCapabilityRule } from "../../core/capabilities.js";
import { requireString, validateBaseConfig, type GeminiConfig } from "../../core/config.js";
import { ProviderError, UnsupportedCapabilityError } from "../../core/errors.js";
import type { ChatRequest, ChatResponse, ContentPart, ModelInfo, ReasoningEffort, StreamEvent, ToolCall } from "../../core/types.js";
import { asArray, compact, getArray, getNumber, getRecord, getString, isRecord, tryParseJson, type JsonRecord } from "../../utils/json.js";
import { appendText, finishReasonFor, reasoningOf, textOf } from "../../utils/normalize.js";
import { parseSSE } from "../../utils/sse.js";
import { BaseProvider, commonSettings } from "../base.js";
import { GeminiStreamProcessor } from "./stream.js";
import {
  firstCandidate,
  modelPath,
  normalizeGeminiFinishReason,
  parseGeminiPart,
  parseGeminiUsage,
  PROVIDER,
  toGeminiContents,
  toGeminiToolConfig,
  toGeminiTools,
} from "./translate.js";

/** Gemini 1.x / 2.x: budget-style thinking control only (no thinkingLevel). */
const BUDGET_ONLY = /^(models\/)?gemini-[12]\./;

export const GEMINI_MODEL_RULES: readonly ModelCapabilityRule[] = [
  { match: /^(models\/)?gemini-(1\.|2\.0)/, capabilities: { reasoning: false, reasoningControl: false } },
];

/**
 * Google Gemini via the native `generateContent` REST API.
 *
 * Authentication uses the `x-goog-api-key` header (never a query parameter,
 * so the key cannot leak through logged URLs).
 *
 * Reasoning mapping:
 * - Gemini 3+: `effort` → `thinkingConfig.thinkingLevel` (none→minimal,
 *   xhigh/max→high); Gemini 3 cannot fully disable thinking.
 * - Gemini 2.5: only `maxTokens` → `thinkingBudget`; `effort: "none"` →
 *   `thinkingBudget: 0`; other efforts are unsupported (set maxTokens).
 * - `includeReasoning` → `thinkingConfig.includeThoughts`.
 */
export class GeminiAdapter extends BaseProvider {
  constructor(config: GeminiConfig) {
    validateBaseConfig(config);
    const apiKey = requireString(config, "apiKey");
    const version = config.apiVersion ?? "v1beta";
    super({
      ...commonSettings(config),
      name: PROVIDER,
      baseURL: config.baseURL ?? `https://generativelanguage.googleapis.com/${version}`,
      headers: { "x-goog-api-key": apiKey, ...config.headers },
      secrets: [apiKey],
      defaultCapabilities: { ...ALL_CAPABILITIES, reasoning: true, reasoningControl: true },
      capabilityRules: GEMINI_MODEL_RULES,
    });
  }

  static fromConfig(config: GeminiConfig): GeminiAdapter {
    return new GeminiAdapter(config);
  }

  async chat(request: ChatRequest): Promise<ChatResponse> {
    const model = this.modelFor(request);
    const { data } = await this.http.json({
      path: `${modelPath(model)}:generateContent`,
      body: this.buildBody(request, model),
      ...this.transport(request),
    });
    if (!isRecord(data)) throw new ProviderError(`${PROVIDER}: unexpected response body`, { provider: PROVIDER, cause: data });

    const candidate = firstCandidate(data);
    const content: ContentPart[] = [];
    const toolCalls: ToolCall[] = [];
    for (const raw of asArray(getRecord(candidate, "content")?.parts)) {
      const part = parseGeminiPart(raw);
      if (part.kind === "text" && part.text) appendText(content, part.text);
      else if (part.kind === "thought") {
        content.push({
          type: "reasoning",
          text: part.text ?? "",
          ...(part.signature ? { providerData: { [PROVIDER]: { thoughtSignature: part.signature } } } : {}),
        });
      } else if (part.kind === "call" && part.call) toolCalls.push(part.call);
    }

    const blockReason = getString(getRecord(data, "promptFeedback"), "blockReason");
    const rawFinish = getString(candidate, "finishReason") ?? blockReason;
    const finishReason = candidate ? normalizeGeminiFinishReason(rawFinish) : blockReason ? "content_filter" : "unknown";
    const reasoning = reasoningOf(content);
    return {
      id: getString(data, "responseId"),
      provider: this.name,
      model: getString(data, "modelVersion") ?? model,
      message: { role: "assistant", content, ...(toolCalls.length > 0 ? { toolCalls } : {}) },
      text: textOf(content),
      ...(reasoning !== undefined ? { reasoning } : {}),
      toolCalls,
      finishReason: finishReasonFor(toolCalls, finishReason),
      rawFinishReason: rawFinish,
      usage: parseGeminiUsage(data.usageMetadata),
      raw: data,
    };
  }

  async *stream(request: ChatRequest): AsyncGenerator<StreamEvent> {
    const model = this.modelFor(request);
    const { body } = await this.http.stream({
      path: `${modelPath(model)}:streamGenerateContent`,
      query: { alt: "sse" },
      body: this.buildBody(request, model),
      ...this.transport(request),
    });
    const processor = new GeminiStreamProcessor();
    for await (const message of parseSSE(body)) {
      const chunk = tryParseJson(message.data);
      if (chunk !== undefined) yield* processor.process(chunk);
    }
    yield* processor.finish();
  }

  async listModels(options: { signal?: AbortSignal } = {}): Promise<ModelInfo[]> {
    const models: ModelInfo[] = [];
    let pageToken: string | undefined;
    for (let page = 0; page < 50; page++) {
      const { data } = await this.http.json({
        path: "models",
        query: pageToken ? { pageSize: "1000", pageToken } : { pageSize: "1000" },
        signal: options.signal,
      });
      for (const m of getArray(data, "models")) {
        const name = getString(m, "name");
        if (!name) continue;
        models.push(
          compact({
            id: name.replace(/^models\//, ""),
            provider: this.name,
            displayName: getString(m, "displayName"),
            contextWindow: getNumber(m, "inputTokenLimit"),
            maxOutputTokens: getNumber(m, "outputTokenLimit"),
            raw: m,
          }),
        );
      }
      pageToken = getString(data, "nextPageToken");
      if (!pageToken) break;
    }
    return models;
  }

  // -------------------------------------------------------------------------

  private buildBody(request: ChatRequest, model: string): JsonRecord {
    const { systemInstruction, contents } = toGeminiContents(request.messages);
    const body: JsonRecord = { contents };
    if (systemInstruction) body.systemInstruction = systemInstruction;
    if (request.tools && request.tools.length > 0) {
      body.tools = toGeminiTools(request.tools);
      const toolConfig = toGeminiToolConfig(request.toolChoice);
      if (toolConfig) body.toolConfig = toolConfig;
    }

    const gen: JsonRecord = compact({
      temperature: request.temperature,
      topP: request.topP,
      maxOutputTokens: request.maxTokens,
      stopSequences: request.stop && request.stop.length > 0 ? request.stop : undefined,
    });
    const format = request.responseFormat;
    if (format?.type === "json_object") gen.responseMimeType = "application/json";
    else if (format?.type === "json_schema") {
      gen.responseMimeType = "application/json";
      gen.responseJsonSchema = format.schema;
    }
    const thinking = this.thinkingConfig(request, model);
    if (thinking) gen.thinkingConfig = thinking;
    if (Object.keys(gen).length > 0) body.generationConfig = gen;

    return { ...body, ...this.providerOptionsFor(request).body };
  }

  private thinkingConfig(request: ChatRequest, model: string): JsonRecord | undefined {
    const r = request.reasoning;
    if (!r) return undefined;
    const cfg: JsonRecord = {};
    if (r.maxTokens !== undefined) cfg.thinkingBudget = r.maxTokens;
    if (r.effort !== undefined) {
      if (BUDGET_ONLY.test(model)) {
        if (r.effort === "none") cfg.thinkingBudget = 0;
        else if (r.maxTokens === undefined) {
          throw new UnsupportedCapabilityError(
            "reasoningControl",
            `${PROVIDER}: model "${model}" has no reasoning effort levels; use reasoning.maxTokens (thinking budget).`,
            { provider: PROVIDER },
          );
        }
      } else if (r.maxTokens === undefined) {
        // thinkingLevel and thinkingBudget are mutually exclusive; an explicit budget wins.
        cfg.thinkingLevel = mapThinkingLevel(r.effort);
      }
    }
    if (r.includeReasoning !== undefined) cfg.includeThoughts = r.includeReasoning;
    return Object.keys(cfg).length > 0 ? cfg : undefined;
  }
}

function mapThinkingLevel(effort: ReasoningEffort): string {
  switch (effort) {
    case "none":
    case "minimal":
      return "minimal";
    case "low":
      return "low";
    case "medium":
      return "medium";
    default:
      return "high";
  }
}
