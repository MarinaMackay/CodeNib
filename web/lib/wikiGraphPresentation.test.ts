// SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
//
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { wikiGraphPresentation } from "./wikiGraphPresentation";

describe("wikiGraphPresentation", () => {
  it("opens the indexed architecture map on the overview page", () => {
    expect(wikiGraphPresentation("overview", true)).toEqual({
      defaultOpen: true,
      title: "Repository architecture",
      description: "Source-linked components and dependencies from the indexed graph",
    });
  });

  it("does not claim an overview map when graph data is unavailable", () => {
    expect(wikiGraphPresentation("overview", false).defaultOpen).toBe(false);
  });

  it("keeps maps on detail pages collapsed by default", () => {
    expect(wikiGraphPresentation("architecture", true)).toEqual({
      defaultOpen: false,
      title: "Subsystem map",
      description: null,
    });
  });
});
