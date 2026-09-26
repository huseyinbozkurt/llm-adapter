/**
 * Server-Sent Events parser (WHATWG spec subset) over a byte stream.
 *
 * Handles events split across arbitrary chunk boundaries, multi-byte UTF-8
 * characters split across chunks, CRLF / CR / LF line endings, multi-line
 * `data:` fields and comment lines.
 */

export interface SSEMessage {
  event?: string;
  data: string;
  id?: string;
}

export async function* parseSSE(body: AsyncIterable<Uint8Array>): AsyncGenerator<SSEMessage> {
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let event: string | undefined;
  let data: string[] = [];
  let id: string | undefined;
  let sawField = false;

  const dispatch = (): SSEMessage | undefined => {
    const hadField = sawField;
    const msg: SSEMessage | undefined = data.length > 0 ? { event, data: data.join("\n"), id } : undefined;
    event = undefined;
    data = [];
    sawField = false;
    return hadField ? msg : undefined;
  };

  const processLine = (line: string): SSEMessage | undefined => {
    if (line === "") return dispatch();
    if (line.startsWith(":")) return undefined; // comment / keep-alive
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    sawField = true;
    if (field === "data") data.push(value);
    else if (field === "event") event = value;
    else if (field === "id") id = value;
    return undefined;
  };

  function* drain(final: boolean): Generator<SSEMessage> {
    for (;;) {
      const match = /\r\n|\r|\n/.exec(buffer);
      if (!match) break;
      // A trailing lone "\r" might be the first half of "\r\n": wait for more input.
      if (!final && match[0] === "\r" && match.index === buffer.length - 1) break;
      const line = buffer.slice(0, match.index);
      buffer = buffer.slice(match.index + match[0].length);
      const msg = processLine(line);
      if (msg) yield msg;
    }
  }

  for await (const chunk of body) {
    buffer += decoder.decode(chunk, { stream: true });
    yield* drain(false);
  }
  buffer += decoder.decode();
  yield* drain(true);
  if (buffer !== "") {
    const msg = processLine(buffer);
    buffer = "";
    if (msg) yield msg;
  }
  // Stream ended without a trailing blank line: dispatch what we have.
  const last = dispatch();
  if (last) yield last;
}
