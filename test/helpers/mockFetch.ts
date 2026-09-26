/**
 * Scriptable fetch mock. Each call consumes the next scripted responder and
 * records the request (URL, method, headers, parsed JSON body).
 */

export interface RecordedRequest {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: unknown;
  signal: AbortSignal | undefined;
}

export type Responder = (req: RecordedRequest) => Response | Promise<Response>;

export interface MockFetch {
  fetch: typeof globalThis.fetch;
  requests: RecordedRequest[];
  /** Body of the most recent request. */
  lastBody(): Record<string, unknown>;
}

export function mockFetch(...responders: Responder[]): MockFetch {
  const requests: RecordedRequest[] = [];
  let i = 0;
  const fetchImpl = async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const headers: Record<string, string> = {};
    new Headers(init?.headers).forEach((v, k) => (headers[k] = v));
    const req: RecordedRequest = {
      url: String(input),
      method: init?.method ?? "GET",
      headers,
      body: typeof init?.body === "string" ? JSON.parse(init.body) : undefined,
      signal: init?.signal ?? undefined,
    };
    requests.push(req);
    const responder = responders[Math.min(i++, responders.length - 1)];
    if (!responder) throw new Error("mockFetch: no responder");
    return responder(req);
  };
  return {
    fetch: fetchImpl as typeof globalThis.fetch,
    requests,
    lastBody: () => requests[requests.length - 1]?.body as Record<string, unknown>,
  };
}

export function json(body: unknown, status = 200, headers: Record<string, string> = {}): Responder {
  return () => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });
}

/** Streams `text` split into chunks at the given byte sizes (cycled), to exercise fragmentation. */
export function sse(text: string, chunkSizes: number[] = [7, 3, 19, 1, 11]): Responder {
  return (req) => {
    const bytes = new TextEncoder().encode(text);
    let offset = 0;
    let n = 0;
    const stream = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (req.signal?.aborted) {
          controller.error(req.signal.reason);
          return;
        }
        if (offset >= bytes.length) {
          controller.close();
          return;
        }
        const size = chunkSizes[n++ % chunkSizes.length] ?? 8;
        controller.enqueue(bytes.slice(offset, offset + size));
        offset += size;
      },
    });
    return new Response(stream, { status: 200, headers: { "content-type": "text/event-stream" } });
  };
}

/** Builds an SSE body from data payloads (optionally with event names). */
export function sseBody(events: Array<unknown | { event: string; data: unknown }>, done = false): string {
  const lines = events.map((e) => {
    if (e && typeof e === "object" && "event" in e && "data" in e) {
      const ev = e as { event: string; data: unknown };
      return `event: ${ev.event}\ndata: ${JSON.stringify(ev.data)}\n\n`;
    }
    return `data: ${JSON.stringify(e)}\n\n`;
  });
  return lines.join("") + (done ? "data: [DONE]\n\n" : "");
}

/** A response that never resolves until the request is aborted. */
export function hang(): Responder {
  return (req) =>
    new Promise<Response>((_, reject) => {
      if (req.signal?.aborted) return reject(req.signal.reason);
      req.signal?.addEventListener("abort", () => reject(req.signal?.reason ?? new Error("aborted")), { once: true });
    });
}

export async function collect<T>(iter: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const x of iter) out.push(x);
  return out;
}
