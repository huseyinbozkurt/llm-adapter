/**
 * Small guards for reading untyped provider payloads without `any`.
 */

export type JsonRecord = Record<string, unknown>;

export function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function asRecord(value: unknown): JsonRecord | undefined {
  return isRecord(value) ? value : undefined;
}

export function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

export function getString(obj: unknown, key: string): string | undefined {
  if (!isRecord(obj)) return undefined;
  const v = obj[key];
  return typeof v === "string" ? v : undefined;
}

export function getNumber(obj: unknown, key: string): number | undefined {
  if (!isRecord(obj)) return undefined;
  const v = obj[key];
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

export function getRecord(obj: unknown, key: string): JsonRecord | undefined {
  return isRecord(obj) ? asRecord(obj[key]) : undefined;
}

export function getArray(obj: unknown, key: string): unknown[] {
  return isRecord(obj) ? asArray(obj[key]) : [];
}

/** JSON.parse that returns undefined instead of throwing. */
export function tryParseJson(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return undefined;
  }
}

/** Removes keys whose value is undefined (keeps request bodies minimal). */
export function compact<T extends JsonRecord>(obj: T): T {
  const out: JsonRecord = {};
  for (const [k, v] of Object.entries(obj)) if (v !== undefined) out[k] = v;
  return out as T;
}
