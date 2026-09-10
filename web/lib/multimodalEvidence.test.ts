// SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import {
  isHighConfidenceVisualBinding,
  materializedWikiMediaSlots,
  wikiMediaCitationCount,
  type WikiMediaSlot,
  type WikiVisualCodeBinding,
} from "./api";

const slot = (overrides: Partial<WikiMediaSlot> = {}): WikiMediaSlot => ({
  id: "architecture",
  kind: "diagram",
  placement: "lead",
  title: "Architecture",
  purpose: "Explain the repository architecture.",
  source_citations: ["src/app.py", "src/graph.py"],
  prompt: "Show the validated architecture.",
  human_prior: { editable: true, notes: [] },
  ...overrides,
});

const binding = (score: number): WikiVisualCodeBinding => ({
  artifact_path: "docs/architecture.png",
  entity_name: "Router",
  source_path: "src/router.py",
  symbol: "Router",
  kind: "class",
  line: 42,
  score,
  evidence: "source-grounded binding",
});

describe("multimodal visual evidence helpers", () => {
  it("only exposes media slots with a materialized asset URI", () => {
    const visible = materializedWikiMediaSlots([
      slot(),
      slot({ asset: { slot_id: "architecture", kind: "diagram", uri: "  /media/a.svg  ", mime_type: "image/svg+xml", model: "model", provider: "local", prompt: "p", source_citations: ["src/app.py"] } }),
      slot({ asset: { slot_id: "video", kind: "video", uri: "", mime_type: "video/mp4", model: "model", provider: "local", prompt: "p", source_citations: [] } }),
    ]);

    expect(visible).toHaveLength(1);
    expect(visible[0].asset?.uri).toBe("  /media/a.svg  ");
  });

  it("deduplicates visual source citations and prefers asset provenance", () => {
    expect(
      wikiMediaCitationCount(
        slot({
          source_citations: ["src/app.py", "src/graph.py"],
          asset: {
            slot_id: "architecture",
            kind: "diagram",
            uri: "/media/a.svg",
            mime_type: "image/svg+xml",
            model: "model",
            provider: "local",
            prompt: "p",
            source_citations: ["src/app.py", "src/app.py", "src/router.py"],
          },
        }),
      ),
    ).toBe(2);
  });

  it("uses the intended high-confidence binding threshold", () => {
    expect(isHighConfidenceVisualBinding(binding(0.8))).toBe(true);
    expect(isHighConfidenceVisualBinding(binding(0.7999))).toBe(false);
    expect(isHighConfidenceVisualBinding(binding(Number.NaN))).toBe(false);
    expect(isHighConfidenceVisualBinding(binding(Number.POSITIVE_INFINITY))).toBe(false);
  });
});
