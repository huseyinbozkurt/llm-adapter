/**
 * Normalised error hierarchy. Adapters never throw raw fetch / provider
 * errors; the original is always preserved in `cause`.
 *
 * Error messages must never contain secrets. Anything derived from a provider
 * response body passes through `redact()` before it is used in a message.
 */

export interface LLMErrorOptions {
  provider?: string;
  statusCode?: number;
  retryable?: boolean;
  /** Provider error code/type, e.g. `rate_limit_error`, `insufficient_quota`. */
  code?: string;
  requestId?: string;
  retryAfterMs?: number;
  cause?: unknown;
}

export class LLMError extends Error {
  readonly provider: string | undefined;
  readonly statusCode: number | undefined;
  readonly retryable: boolean;
  readonly code: string | undefined;
  readonly requestId: string | undefined;
  readonly retryAfterMs: number | undefined;

  constructor(message: string, options: LLMErrorOptions = {}) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.name = new.target.name;
    this.provider = options.provider;
    this.statusCode = options.statusCode;
    this.retryable = options.retryable ?? false;
    this.code = options.code;
    this.requestId = options.requestId;
    this.retryAfterMs = options.retryAfterMs;
  }

  toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      message: this.message,
      provider: this.provider,
      statusCode: this.statusCode,
      retryable: this.retryable,
      code: this.code,
      requestId: this.requestId,
    };
  }
}

export class AuthenticationError extends LLMError {}
export class RateLimitError extends LLMError {}
export class InvalidRequestError extends LLMError {}
export class ContextLengthExceededError extends InvalidRequestError {}
/** The provider refused the input on content-policy grounds. Never retried. */
export class ContentFilterError extends InvalidRequestError {}
export class ModelNotFoundError extends LLMError {}
export class ProviderError extends LLMError {}
export class NetworkError extends LLMError {}
/** The configured timeout elapsed. Distinct from AbortError (caller cancelled). */
export class TimeoutError extends LLMError {}
/** The caller's AbortSignal fired. Never retried. */
export class AbortError extends LLMError {}
export class ConfigurationError extends LLMError {}

/** A local budget (tokens / cost) is exhausted. Raised before any request is sent. */
export class BudgetExceededError extends LLMError {
  readonly limit: string;
  constructor(limit: string, message: string, options: LLMErrorOptions = {}) {
    super(message, { code: "budget_exceeded", ...options, retryable: false });
    this.limit = limit;
  }
}

/** Every target of a FallbackProvider failed. `errors` lists each attempt in order. */
export class FallbackExhaustedError extends LLMError {
  readonly errors: ReadonlyArray<{ target: string; error: unknown }>;
  constructor(errors: ReadonlyArray<{ target: string; error: unknown }>, options: LLMErrorOptions = {}) {
    const summary = errors
      .map((e) => `${e.target}: ${e.error instanceof Error ? e.error.name : "Error"}`)
      .join(", ");
    super(`All fallback targets failed (${summary}).`, {
      ...options,
      retryable: errors.some((e) => e.error instanceof LLMError && e.error.retryable),
      cause: errors[errors.length - 1]?.error,
    });
    this.errors = errors;
  }
}

export class UnsupportedCapabilityError extends LLMError {
  readonly capability: string;
  constructor(capability: string, message: string, options: LLMErrorOptions = {}) {
    super(message, options);
    this.capability = capability;
  }
}

// ---------------------------------------------------------------------------
// Secret redaction
// ---------------------------------------------------------------------------

const KEY_PATTERNS: RegExp[] = [
  /\bsk-[A-Za-z0-9_\-*]{8,}/g, // OpenAI / DeepSeek / OpenRouter / Anthropic style
  /\bxai-[A-Za-z0-9_\-*]{8,}/g,
  /\bgsk_[A-Za-z0-9_\-*]{8,}/g, // Groq
  /\bAIza[0-9A-Za-z_\-]{20,}/g, // Google
  /\b[Bb]earer\s+[A-Za-z0-9._\-~+/=]{8,}/g,
];

/** Removes known secrets and anything that looks like an API key. */
export function redact(text: string, secrets: ReadonlyArray<string | undefined> = []): string {
  let out = text;
  for (const secret of secrets) {
    if (secret && secret.length >= 4) out = out.split(secret).join("[REDACTED]");
  }
  for (const pattern of KEY_PATTERNS) out = out.replace(pattern, "[REDACTED]");
  return out;
}

// ---------------------------------------------------------------------------
// HTTP status → error mapping
// ---------------------------------------------------------------------------

export interface HttpErrorDetails {
  provider: string;
  status: number;
  /** Already-redacted provider message, if one could be extracted. */
  message?: string;
  code?: string;
  requestId?: string;
  retryAfterMs?: number;
  cause?: unknown;
}

const CONTEXT_LENGTH_HINTS =
  /context[_ ]length|context window|maximum context|prompt is too long|too many tokens|exceeds the maximum (number of )?tokens|input is too long|token limit/i;
const CONTENT_FILTER_HINTS = /content[_ ]?(policy|filter|management)|safety system|flagged|moderation/i;
const MODEL_HINTS = /model/i;

/** Maps an HTTP failure to the normalised error class. */
export function errorFromHttp(d: HttpErrorDetails): LLMError {
  const text = d.message ?? `HTTP ${d.status}`;
  const message = `${d.provider}: ${text}`;
  const base = {
    provider: d.provider,
    statusCode: d.status,
    code: d.code,
    requestId: d.requestId,
    retryAfterMs: d.retryAfterMs,
    cause: d.cause,
  };
  const hint = `${d.code ?? ""} ${text}`;

  if (d.status === 401 || d.status === 403) return new AuthenticationError(message, { ...base, retryable: false });
  if (d.status === 429) {
    // Billing exhaustion is reported as 429 by some providers (OpenAI
    // `insufficient_quota`) and will not resolve by waiting. Plain "quota"
    // wording is NOT treated as exhaustion: Gemini uses it for per-minute
    // limits. A Retry-After header always means "retry later".
    const exhausted =
      d.retryAfterMs === undefined && /insufficient_quota|credit balance|out of credits|billing hard limit/i.test(hint);
    return new RateLimitError(message, { ...base, retryable: !exhausted });
  }
  if (d.status === 408) return new TimeoutError(message, { ...base, retryable: true });
  if (d.status === 404) {
    if (MODEL_HINTS.test(hint)) return new ModelNotFoundError(message, { ...base, retryable: false });
    return new InvalidRequestError(message, { ...base, retryable: false });
  }
  if (d.status === 400 || d.status === 413 || d.status === 422) {
    if (CONTEXT_LENGTH_HINTS.test(hint)) return new ContextLengthExceededError(message, { ...base, retryable: false });
    if (CONTENT_FILTER_HINTS.test(hint)) return new ContentFilterError(message, { ...base, retryable: false });
    if (/model_not_found|model not found|does not exist|unknown model|invalid model/i.test(hint)) {
      return new ModelNotFoundError(message, { ...base, retryable: false });
    }
    return new InvalidRequestError(message, { ...base, retryable: false });
  }
  if (d.status === 409) return new ProviderError(message, { ...base, retryable: true });
  if (d.status >= 500) return new ProviderError(message, { ...base, retryable: true });
  return new ProviderError(message, { ...base, retryable: false });
}

/** Maps an error payload that arrived inside a stream (no HTTP status of its own). */
export function errorFromStreamPayload(provider: string, type: string | undefined, message: string | undefined): LLMError {
  const statusByType: Record<string, number> = {
    invalid_request_error: 400,
    authentication_error: 401,
    permission_error: 403,
    not_found_error: 404,
    request_too_large: 413,
    rate_limit_error: 429,
    api_error: 500,
    overloaded_error: 529,
  };
  const status = (type ? statusByType[type] : undefined) ?? 500;
  const err = errorFromHttp({ provider, status, message: message ?? type ?? "stream error", code: type });
  // Errors inside an already-started stream are not retried by the client.
  return err;
}

const RETRYABLE_NETWORK_CODES = new Set([
  "ECONNRESET",
  "ECONNREFUSED",
  "ETIMEDOUT",
  "EPIPE",
  "EAI_AGAIN",
  "ENETUNREACH",
  "EHOSTUNREACH",
  "UND_ERR_SOCKET",
  "UND_ERR_CONNECT_TIMEOUT",
  "UND_ERR_HEADERS_TIMEOUT",
  "UND_ERR_BODY_TIMEOUT",
  "UND_ERR_CLOSED",
]);

function networkCode(error: unknown): string | undefined {
  let current: unknown = error;
  for (let depth = 0; depth < 4 && current && typeof current === "object"; depth++) {
    const code = (current as { code?: unknown }).code;
    if (typeof code === "string") return code;
    current = (current as { cause?: unknown }).cause;
  }
  return undefined;
}

/** Wraps a low-level fetch failure (DNS, socket reset, ...). */
export function networkError(provider: string, error: unknown, secrets: ReadonlyArray<string | undefined>): NetworkError {
  const code = networkCode(error);
  const raw = error instanceof Error ? error.message : String(error);
  return new NetworkError(`${provider}: network error${code ? ` (${code})` : ""}: ${redact(raw, secrets)}`, {
    provider,
    code,
    retryable: code === undefined || RETRYABLE_NETWORK_CODES.has(code),
    cause: error,
  });
}

export function isLLMError(error: unknown): error is LLMError {
  return error instanceof LLMError;
}
