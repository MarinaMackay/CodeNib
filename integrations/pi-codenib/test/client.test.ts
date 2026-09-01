// SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from "vitest";
import { CodeNibClient, clientFromEnvironment } from "../extensions/client.js";

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("CodeNibClient", () => {
  it("queries the bounded visual semantic endpoint", async () => {
    const fetchImpl = vi.fn(async (_input: string | URL | Request, _init?: RequestInit) =>
      jsonResponse({
        query: "request flow",
        result_count: 1,
        provider: "local",
        model: "local/hash-visual-embedding-v1",
        model_revision: "",
        dimensions: 64,
        index_sha256: "a".repeat(64),
        results: [
          {
            artifact_path: "docs/flow.svg",
            score: 0.9,
            caption: "Request flow",
            role_hint: "diagram",
            mime_type: "image/svg+xml",
            source_paths: ["src/app.py"],
            symbols: ["route"],
            entry_sha256: "b".repeat(64),
          },
        ],
      }),
    );
    const client = new CodeNibClient({ repoId: "owner/repo", fetchImpl });

    const response = await client.searchVisuals(" request flow ", 4);

    expect(response.results[0]?.artifact_path).toBe("docs/flow.svg");
    expect(fetchImpl.mock.calls[0]?.[0]).toContain(
      "/api/repos/owner%2Frepo/wiki-multimodal/search?q=request+flow&limit=4",
    );
    expect(fetchImpl.mock.calls[0]?.[1]).toMatchObject({ redirect: "error" });
  });

  it("fails closed for remote HTTP and unapproved remote hosts", () => {
    expect(() =>
      new CodeNibClient({ apiBase: "https://example.com", repoId: "repo" }),
    ).toThrow("CODENIB_ALLOW_REMOTE=1");
    expect(() =>
      new CodeNibClient({
        apiBase: "http://example.com",
        repoId: "repo",
        allowRemote: true,
      }),
    ).toThrow("require HTTPS");
  });

  it("bounds errors and rejects invalid result shapes", async () => {
    const failed = new CodeNibClient({
      repoId: "repo",
      fetchImpl: vi.fn(async () => jsonResponse({ detail: "missing index" }, 404)),
    });
    await expect(failed.searchVisuals("diagram")).rejects.toThrow(
      "CodeNib runtime returned 404: missing index",
    );

    const malformed = new CodeNibClient({
      repoId: "repo",
      fetchImpl: vi.fn(async () => jsonResponse({ results: "not-an-array" })),
    });
    await expect(malformed.searchVisuals("diagram")).rejects.toThrow(
      "visual search results are invalid",
    );
  });

  it("reads explicit environment configuration", () => {
    const client = clientFromEnvironment({
      CODENIB_API_BASE: "https://codenib.example",
      CODENIB_REPO_ID: "owner/repo",
      CODENIB_ALLOW_REMOTE: "1",
      CODENIB_TIMEOUT_MS: "5000",
    });

    expect(client.apiBase).toBe("https://codenib.example");
    expect(client.repoId).toBe("owner/repo");
  });

  it("loads without repository config but fails only repository tool calls", async () => {
    const client = new CodeNibClient({
      fetchImpl: vi.fn(async () => jsonResponse({ status: "ok" })),
    });

    await expect(client.status()).resolves.toMatchObject({
      repository_configured: false,
    });
    await expect(client.searchVisuals("architecture")).rejects.toThrow(
      "CODENIB_REPO_ID",
    );
  });
});
