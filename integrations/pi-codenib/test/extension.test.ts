// SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
//
// SPDX-License-Identifier: Apache-2.0

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { describe, expect, it, vi } from "vitest";
import type { CodeNibClient } from "../extensions/client.js";
import { registerCodeNibTools } from "../extensions/index.js";

describe("Pi extension", () => {
  it("registers four read-only tools and forwards cancellation", async () => {
    const definitions: Array<Record<string, unknown>> = [];
    const pi = {
      registerTool(definition: Record<string, unknown>) {
        definitions.push(definition);
      },
    } as unknown as ExtensionAPI;
    const searchVisuals = vi.fn(async () => ({ results: [] }));
    const client = {
      searchVisuals,
      visualEvidence: vi.fn(async () => ({ bundle_sha256: "a".repeat(64) })),
      architecture: vi.fn(async () => ({ diagram_type: "architecture" })),
      videos: vi.fn(async () => ({ video_count: 1 })),
    } as unknown as CodeNibClient;

    registerCodeNibTools(pi, client);

    expect(definitions.map((definition) => definition.name)).toEqual([
      "codenib_visual_search",
      "codenib_visual_evidence",
      "codenib_architecture",
      "codenib_storyboard_videos",
    ]);
    const controller = new AbortController();
    const search = definitions[0] as {
      execute: (...args: unknown[]) => Promise<{ content: Array<{ text: string }> }>;
    };
    const result = await search.execute("call", { query: "flow", limit: 3 }, controller.signal);
    expect(searchVisuals).toHaveBeenCalledWith("flow", 3, controller.signal);
    expect(JSON.parse(result.content[0]!.text)).toEqual({ results: [] });
  });
});
