import type { Logger } from "../core/config.js";
import { AbortError, errorFromHttp, LLMError, networkError, redact, TimeoutError } from "../core/errors.js";
import { getRecord, getString, isRecord, tryParseJson } from "./json.js";
import { parseRetryAfter, withRetry, type ResolvedRetryOptions } from "./retry.js";

export interface HttpClientOptions {
  provider: string;
  baseURL: string;
  /** Static headers, including auth. Never logged or put in errors. */
  headers: Record<string, string>;
  query?: Record<string, string>;
  timeoutMs: number;
  retry: ResolvedRetryOptions;
  fetch?: typeof globalThis.fetch;
  logger?: Logger;
  /** Values scrubbed from any error text. */
  secrets: string[];
  /** Extracts message/code from a provider error body. */
  parseError?: (body: unknown) => { message?: string; code?: string };
}

export interface RequestOptions {
  method?: "GET" | "POST";
  path: string;
  body?: unknown;
  query?: Record<string, string>;
  headers?: Record<string, string>;
  signal?: AbortSignal;
  timeoutMs?: number;
}

export interface JsonResult {
  data: unknown;
  headers: Headers;
  status: number;
}

export interface StreamResult {
  headers: Headers;
  status: number;
  /** Body chunks. Enforces the idle timeout and the caller's AbortSignal. */
  body: AsyncIterable<Uint8Array>;
}

/** Default extractor that understands the OpenAI, Anthropic and Google error envelopes. */
export function defaultParseError(body: unknown): { message?: string; code?: string } {
  if (!isRecord(body)) return {};
  const err = getRecord(body, "error");
  if (err) {
    return {
      message: getString(err, "message"),
      code: getString(err, "code") ?? getString(err, "type") ?? getString(err, "status"),
    };
  }
  const errString = getString(body, "error");
  return {
    message: getString(body, "message") ?? errString ?? getString(body, "detail"),
    code: getString(body, "code") ?? getString(body, "type"),
  };
}

const MAX_ERROR_BODY = 16_384;

/**
 * Thin fetch wrapper: URL building, timeouts, cancellation, retry, and
 * error normalisation. Knows nothing about any provider's payloads.
 */
export class HttpClient {
  constructor(private readonly options: HttpClientOptions) {}

  get provider(): string {
    return this.options.provider;
  }

  async json(req: RequestOptions): Promise<JsonResult> {
    return withRetry(
      async (attempt) => {
        const attemptCtl = this.attemptController(req);
        try {
          const res = await this.send(req, attemptCtl, attempt);
          const text = await res.text();
          return { data: text === "" ? undefined : tryParseJson(text), headers: res.headers, status: res.status };
        } catch (error) {
          throw this.mapThrown(error, attemptCtl);
        } finally {
          attemptCtl.dispose();
        }
      },
      this.options.retry,
      { signal: req.signal, provider: this.provider, onRetry: this.logRetry },
    );
  }

  /**
   * Opens a streaming request. Retries apply only to opening the stream
   * (connection failures, 429/5xx before any data); once the body starts,
   * errors propagate to the consumer.
   */
  async stream(req: RequestOptions): Promise<StreamResult> {
    return withRetry(
      async (attempt) => {
        const attemptCtl = this.attemptController(req);
        let res: Response;
        try {
          res = await this.send(req, attemptCtl, attempt);
        } catch (error) {
          attemptCtl.dispose();
          throw this.mapThrown(error, attemptCtl);
        }
        if (!res.body) {
          attemptCtl.dispose();
          throw errorFromHttp({ provider: this.provider, status: res.status, message: "response has no body" });
        }
        return { headers: res.headers, status: res.status, body: this.iterateBody(res.body, attemptCtl) };
      },
      this.options.retry,
      { signal: req.signal, provider: this.provider, onRetry: this.logRetry },
    );
  }

  private readonly logRetry = (error: unknown, attempt: number, delayMs: number): void => {
    this.options.logger?.debug("llm.retry", {
      provider: this.provider,
      attempt: attempt + 1,
      delayMs,
      error: error instanceof LLMError ? error.name : "Error",
      statusCode: error instanceof LLMError ? error.statusCode : undefined,
    });
  };

  private buildURL(path: string, query?: Record<string, string>): string {
    const base = this.options.baseURL.replace(/\/+$/, "");
    const url = new URL(`${base}/${path.replace(/^\/+/, "")}`);
    for (const [k, v] of Object.entries({ ...this.options.query, ...query })) url.searchParams.set(k, v);
    return url.toString();
  }

  private async send(req: RequestOptions, ctl: AttemptController, attempt: number): Promise<Response> {
    if (req.signal?.aborted) {
      throw new AbortError(`${this.provider}: request aborted`, { provider: this.provider, cause: req.signal.reason });
    }
    const fetchImpl = this.options.fetch ?? globalThis.fetch;
    const method = req.method ?? (req.body === undefined ? "GET" : "POST");
    const headers: Record<string, string> = { ...this.options.headers, ...req.headers };
    if (req.body !== undefined) headers["content-type"] = "application/json";
    const started = Date.now();
    const res = await fetchImpl(this.buildURL(req.path, req.query), {
      method,
      headers,
      body: req.body === undefined ? undefined : JSON.stringify(req.body),
      signal: ctl.signal,
    });
    // Only non-sensitive fields are logged: never headers, bodies or query strings.
    this.options.logger?.debug("llm.http", {
      provider: this.provider,
      method,
      path: req.path,
      status: res.status,
      attempt: attempt + 1,
      durationMs: Date.now() - started,
    });
    if (!res.ok) throw await this.httpError(res);
    return res;
  }

  private async httpError(res: Response): Promise<LLMError> {
    let text = "";
    try {
      text = (await res.text()).slice(0, MAX_ERROR_BODY);
    } catch {
      // Body unreadable; fall back to the status line.
    }
    const parsed = tryParseJson(text);
    const { message, code } = (this.options.parseError ?? defaultParseError)(parsed);
    const fallback = text && parsed === undefined ? text.slice(0, 500) : undefined;
    const raw = message ?? fallback ?? `HTTP ${res.status} ${res.statusText}`.trim();
    return errorFromHttp({
      provider: this.provider,
      status: res.status,
      message: redact(raw, this.options.secrets),
      code,
      requestId: res.headers.get("x-request-id") ?? res.headers.get("request-id") ?? undefined,
      retryAfterMs: parseRetryAfter(res.headers),
      // The raw body is kept for debugging, but only after secret redaction.
      cause: { status: res.status, body: tryParseJson(redact(text, this.options.secrets)) ?? redact(text, this.options.secrets) },
    });
  }

  private mapThrown(error: unknown, ctl: AttemptController): unknown {
    if (error instanceof LLMError) return error;
    if (ctl.userAborted()) {
      return new AbortError(`${this.provider}: request aborted`, { provider: this.provider, cause: error });
    }
    if (ctl.timedOut()) {
      return new TimeoutError(`${this.provider}: request timed out after ${ctl.timeoutMs}ms`, {
        provider: this.provider,
        retryable: true,
        cause: error,
      });
    }
    return networkError(this.provider, error, this.options.secrets);
  }

  private attemptController(req: RequestOptions): AttemptController {
    return new AttemptController(req.signal, req.timeoutMs ?? this.options.timeoutMs);
  }

  private async *iterateBody(body: ReadableStream<Uint8Array>, ctl: AttemptController): AsyncGenerator<Uint8Array> {
    const reader = body.getReader();
    try {
      for (;;) {
        ctl.resetTimer(); // idle timeout between chunks
        let chunk: Awaited<ReturnType<typeof reader.read>>;
        try {
          chunk = await reader.read();
        } catch (error) {
          throw this.mapThrown(error, ctl);
        }
        if (chunk.done) return;
        yield chunk.value;
      }
    } finally {
      ctl.dispose();
      // Cancels the underlying connection if the consumer stopped early.
      reader.cancel().catch(() => undefined);
    }
  }
}

/** One attempt's abort wiring: caller signal + timeout, distinguishable afterwards. */
class AttemptController {
  private readonly controller = new AbortController();
  private timer: ReturnType<typeof setTimeout> | undefined;
  private didTimeOut = false;
  private readonly onUserAbort = () => this.controller.abort(this.userSignal?.reason);

  constructor(
    private readonly userSignal: AbortSignal | undefined,
    readonly timeoutMs: number,
  ) {
    if (userSignal?.aborted) this.controller.abort(userSignal.reason);
    else userSignal?.addEventListener("abort", this.onUserAbort, { once: true });
    this.resetTimer();
  }

  get signal(): AbortSignal {
    return this.controller.signal;
  }

  resetTimer(): void {
    if (this.timer) clearTimeout(this.timer);
    if (this.controller.signal.aborted) return;
    this.timer = setTimeout(() => {
      this.didTimeOut = true;
      this.controller.abort(new Error("timeout"));
    }, this.timeoutMs);
  }

  userAborted(): boolean {
    return this.userSignal?.aborted === true;
  }

  timedOut(): boolean {
    return this.didTimeOut && !this.userAborted();
  }

  dispose(): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = undefined;
    this.userSignal?.removeEventListener("abort", this.onUserAbort);
  }
}
