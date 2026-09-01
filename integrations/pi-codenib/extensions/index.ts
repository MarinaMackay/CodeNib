// SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
//
// SPDX-License-Identifier: Apache-2.0

import { Type } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { clientFromEnvironment, type CodeNibClient } from "./client.js";

function textResult(value: unknown) {
  return {
    content: [{ type: "text" as const, text: JSON.stringify(value, null, 2) }],
    details: value,
  };
}

export function registerCodeNibTools(pi: ExtensionAPI, client: CodeNibClient): void {
  pi.registerTool({
    name: "codenib_visual_search",
    label: "CodeNib visual search",
    description:
      "Search repository images and diagrams in CodeNib's shared visual semantic space.",
    promptSnippet: "Search source-grounded repository visuals by natural language",
    promptGuidelines: [
      "Use codenib_visual_search when a repository diagram, screenshot, or image may explain the question.",
      "Treat returned source paths and symbols as evidence links; do not invent visual facts absent from the result.",
    ],
    parameters: Type.Object({
      query: Type.String({ description: "Natural-language visual search query" }),
      limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 12 })),
    }),
    async execute(_toolCallId, params, signal) {
      return textResult(await client.searchVisuals(params.query, params.limit ?? 6, signal));
    },
  });

  pi.registerTool({
    name: "codenib_visual_evidence",
    label: "CodeNib visual evidence",
    description:
      "Read persisted visual facts, relations, and source-code bindings for the indexed repository.",
    promptSnippet: "Inspect repository visual facts and source grounding",
    parameters: Type.Object({}),
    async execute(_toolCallId, _params, signal) {
      return textResult(await client.visualEvidence(signal));
    },
  });

  pi.registerTool({
    name: "codenib_architecture",
    label: "CodeNib architecture",
    description:
      "Read the validated Archify architecture IR and its revision-pinned source citations.",
    promptSnippet: "Inspect CodeNib's typed repository architecture overview",
    parameters: Type.Object({}),
    async execute(_toolCallId, _params, signal) {
      return textResult(await client.architecture(signal));
    },
  });

  pi.registerTool({
    name: "codenib_storyboard_videos",
    label: "CodeNib storyboard videos",
    description:
      "List authenticated MP4 walkthroughs compiled from validated visual storyboards.",
    promptSnippet: "Inspect rendered source-grounded code walkthrough videos",
    parameters: Type.Object({}),
    async execute(_toolCallId, _params, signal) {
      return textResult(await client.videos(signal));
    },
  });
}

export default function codenibExtension(pi: ExtensionAPI): void {
  const client = clientFromEnvironment();
  registerCodeNibTools(pi, client);
  pi.registerCommand("codenib-status", {
    description: "Check the configured CodeNib runtime and repository",
    handler: async (_args, ctx) => {
      try {
        const status = await client.status();
        const repository = status.repository_configured
          ? String(status.repo_id)
          : "runtime reachable; set CODENIB_REPO_ID to enable tools";
        ctx.ui.notify(`CodeNib ready: ${repository}`, "info");
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        ctx.ui.notify(`CodeNib unavailable: ${message}`, "error");
      }
    },
  });
}
