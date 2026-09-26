import { describe, expect, it } from "vitest";
import { parseSSE } from "../../src/utils/sse.js";
import { collect } from "../helpers/mockFetch.js";

async function* chunks(text: string, sizes: number[]): AsyncGenerator<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  let offset = 0;
  let i = 0;
  while (offset < bytes.length) {
    const size = sizes[i++ % sizes.length] ?? 1;
    yield bytes.slice(offset, offset + size);
    offset += size;
  }
}

describe("parseSSE", () => {
  it("parses events regardless of chunk boundaries", async () => {
    const text = 'event: a\ndata: {"x":1}\n\n: keep-alive\n\ndata: line1\ndata: line2\n\n';
    for (const sizes of [[1], [2], [3, 5], [1000]]) {
      const events = await collect(parseSSE(chunks(text, sizes)));
      expect(events).toEqual([
        { event: "a", data: '{"x":1}', id: undefined },
        { event: undefined, data: "line1\nline2", id: undefined },
      ]);
    }
  });

  it("handles CRLF and CR line endings split across chunks", async () => {
    const text = "data: one\r\n\r\ndata: two\r\rdata: three\n\n";
    const events = await collect(parseSSE(chunks(text, [1])));
    expect(events.map((e) => e.data)).toEqual(["one", "two", "three"]);
  });

  it("does not corrupt multi-byte UTF-8 split across chunks", async () => {
    const text = 'data: {"t":"héllo 🌍 世界"}\n\n';
    const events = await collect(parseSSE(chunks(text, [1])));
    expect(JSON.parse(events[0]!.data)).toEqual({ t: "héllo 🌍 世界" });
  });

  it("dispatches a final event without a trailing blank line", async () => {
    const events = await collect(parseSSE(chunks("data: last", [4])));
    expect(events.map((e) => e.data)).toEqual(["last"]);
  });
});
