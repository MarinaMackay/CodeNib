// SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
//
// SPDX-License-Identifier: Apache-2.0

export interface WikiGraphPresentation {
  defaultOpen: boolean;
  title: string;
  description: string | null;
}
export function wikiGraphPresentation(
  pageId: string,
  available: boolean,
): WikiGraphPresentation {
  if (pageId === "overview") {
    return {
      defaultOpen: available,
      title: "Repository architecture",
      description: "Source-linked components and dependencies from the indexed graph",
    };
  }
  return {
    defaultOpen: false,
    title: "Subsystem map",
    description: null,
  };
}
