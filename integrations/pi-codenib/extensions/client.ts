// SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
//
// SPDX-License-Identifier: Apache-2.0

const DEFAULT_API_BASE = "http://127.0.0.1:8765";
const MAX_RESPONSE_BYTES = 2 * 1024 * 1024;
const MAX_QUERY_BYTES = 4096;
const DEFAULT_TIMEOUT_MS = 20_000;

export interface CodeNibClientOptions {
  apiBase?: string;
  repoId?: string;
  allowRemote?: boolean;
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
}

export interface VisualSearchResult {
  artifact_path: string;
  score: number;
  caption: string;
  role_hint: string;
  mime_type: string;
  source_paths: string[];
  symbols: string[];
  entry_sha256: string;
}

export interface VisualSearchResponse {
  query: string;
  result_count: number;
  provider: string;
  model: string;
  model_revision: string;
  dimensions: number;
  index_sha256: string;
  results: VisualSearchResult[];
}

export class CodeNibClient {
  readonly apiBase: string;
  readonly repoId: string;
  private readonly fetchImpl: typeof fetch;
  private readonly timeoutMs: number;

  constructor(options: CodeNibClientOptions = {}) {
    this.apiBase = validateApiBase(options.apiBase ?? DEFAULT_API_BASE, Boolean(options.allowRemote));
    this.repoId = validateRepoId(options.repoId ?? "", false);
    this.fetchImpl = options.fetchImpl ?? fetch;
    this.timeoutMs = validateTimeout(options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  }

  async status(signal?: AbortSignal): Promise<Record<string, unknown>> {
    const health = await this.request("/api/health", signal);
    return {
      api_base: this.apiBase,
      repo_id: this.repoId,
      repository_configured: Boolean(this.repoId),
      health,
    };
  }

  async searchVisuals(
    query: string,
    limit = 6,
    signal?: AbortSignal,
  ): Promise<VisualSearchResponse> {
    const normalizedQuery = validateQuery(query);
    if (!Number.isInteger(limit) || limit < 1 || limit > 12) {
      throw new Error("visual search limit must be an integer from 1 to 12");
    }
    const parameters = new URLSearchParams({ q: normalizedQuery, limit: String(limit) });
    const value = await this.request(
      `${this.repositoryPath()}/wiki-multimodal/search?${parameters}`,
      signal,
    );
    return validateVisualSearch(value);
  }

  async visualEvidence(signal?: AbortSignal): Promise<Record<string, unknown>> {
    return asObject(
      await this.request(
        `${this.repositoryPath()}/wiki-multimodal`,
        signal,
      ),
      "visual evidence response",
    );
  }

  async architecture(signal?: AbortSignal): Promise<Record<string, unknown>> {
    return asObject(
      await this.request(
        `${this.repositoryPath()}/wiki-archify-overview`,
        signal,
      ),
      "architecture response",
    );
  }

  async videos(signal?: AbortSignal): Promise<Record<string, unknown>> {
    return asObject(
      await this.request(
        `${this.repositoryPath()}/wiki-storyboard-videos`,
        signal,
      ),
      "storyboard video response",
    );
  }

  private repositoryPath(): string {
    return `/api/repos/${encodeURIComponent(validateRepoId(this.repoId, true))}`;
  }

  private async request(path: string, parentSignal?: AbortSignal): Promise<unknown> {
    const timeout = AbortSignal.timeout(this.timeoutMs);
    const signal = parentSignal ? AbortSignal.any([parentSignal, timeout]) : timeout;
    let response: Response;
    try {
      response = await this.fetchImpl(`${this.apiBase}${path}`, {
        headers: { Accept: "application/json" },
        redirect: "error",
        signal,
      });
    } catch (error) {
      if (signal.aborted) throw new Error("CodeNib request was cancelled or timed out");
      throw new Error(`CodeNib runtime request failed: ${errorMessage(error)}`);
    }
    const declaredLength = Number(response.headers.get("content-length") ?? "0");
    if (Number.isFinite(declaredLength) && declaredLength > MAX_RESPONSE_BYTES) {
      throw new Error("CodeNib response exceeds the byte limit");
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.byteLength > MAX_RESPONSE_BYTES) {
      throw new Error("CodeNib response exceeds the byte limit");
    }
    const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    if (!response.ok) {
      let detail = text;
      try {
        const parsed = JSON.parse(text) as { detail?: unknown };
        if (typeof parsed.detail === "string") detail = parsed.detail;
      } catch {
        // The bounded plain-text response still carries useful diagnostics.
      }
      throw new Error(`CodeNib runtime returned ${response.status}: ${detail.slice(0, 500)}`);
    }
    try {
      return JSON.parse(text) as unknown;
    } catch {
      throw new Error("CodeNib runtime returned invalid JSON");
    }
  }
}

export function clientFromEnvironment(
  environment: NodeJS.ProcessEnv = process.env,
  fetchImpl?: typeof fetch,
): CodeNibClient {
  return new CodeNibClient({
    apiBase: environment.CODENIB_API_BASE,
    repoId: environment.CODENIB_REPO_ID,
    allowRemote: environment.CODENIB_ALLOW_REMOTE === "1",
    timeoutMs: environment.CODENIB_TIMEOUT_MS
      ? Number(environment.CODENIB_TIMEOUT_MS)
      : undefined,
    fetchImpl,
  });
}

function validateApiBase(value: string, allowRemote: boolean): string {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error("CODENIB_API_BASE must be a valid URL");
  }
  if (url.username || url.password || url.search || url.hash || url.pathname !== "/") {
    throw new Error("CODENIB_API_BASE must contain only scheme, host, and optional port");
  }
  const local = ["127.0.0.1", "localhost", "::1"].includes(url.hostname);
  if (!local && !allowRemote) {
    throw new Error("remote CodeNib runtimes require CODENIB_ALLOW_REMOTE=1");
  }
  if ((local && url.protocol !== "http:" && url.protocol !== "https:") ||
      (!local && url.protocol !== "https:")) {
    throw new Error("remote CodeNib runtimes require HTTPS");
  }
  return url.origin;
}

function validateRepoId(value: string, required: boolean): string {
  const repoId = value.trim();
  if (!repoId && !required) return "";
  if (!repoId || Buffer.byteLength(repoId, "utf8") > 512 || /[\x00-\x1f\x7f]/.test(repoId)) {
    throw new Error("CODENIB_REPO_ID must identify one indexed repository");
  }
  return repoId;
}

function validateTimeout(value: number): number {
  if (!Number.isInteger(value) || value < 100 || value > 120_000) {
    throw new Error("CODENIB_TIMEOUT_MS must be an integer from 100 to 120000");
  }
  return value;
}

function validateQuery(value: string): string {
  const query = value.trim();
  if (!query || Buffer.byteLength(query, "utf8") > MAX_QUERY_BYTES) {
    throw new Error("visual search query is empty or too long");
  }
  return query;
}

function validateVisualSearch(value: unknown): VisualSearchResponse {
  const response = asObject(value, "visual search response");
  if (!Array.isArray(response.results) || response.results.length > 12) {
    throw new Error("CodeNib visual search results are invalid");
  }
  for (const result of response.results) {
    const item = asObject(result, "visual search result");
    if (typeof item.artifact_path !== "string" || typeof item.score !== "number") {
      throw new Error("CodeNib visual search result fields are invalid");
    }
  }
  return response as unknown as VisualSearchResponse;
}

function asObject(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`CodeNib ${label} is not an object`);
  }
  return value as Record<string, unknown>;
}

function errorMessage(value: unknown): string {
  return value instanceof Error ? value.message : String(value);
}
