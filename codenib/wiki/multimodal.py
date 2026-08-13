# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic multimodal planning hooks for wiki pages.

The first multimodal wiki layer plans source-grounded media opportunities
without calling an image, video, or diagram model. Later generators can consume
these slots and write concrete assets while preserving the same page contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal

MediaKind = Literal["diagram", "image", "storyboard", "video"]
MediaPlacement = Literal["lead", "section", "aside", "appendix"]

_MAX_SOURCE_CITATIONS = 6


@dataclass(frozen=True)
class WikiMediaSlot:
    """A planned multimodal asset for one wiki page."""

    id: str
    kind: MediaKind
    placement: MediaPlacement
    title: str
    purpose: str
    source_citations: tuple[str, ...] = ()
    prompt: str = ""
    human_prior: dict[str, Any] = field(
        default_factory=lambda: {"editable": True, "notes": []}
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_citations"] = list(self.source_citations)
        return data


def _citation_files(citations: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    files = []
    seen = set()
    for citation in citations or ():
        file = str(citation.get("file") or "").strip()
        if not file or file in seen:
            continue
        seen.add(file)
        files.append(file)
        if len(files) >= _MAX_SOURCE_CITATIONS:
            break
    return tuple(files)


def plan_media_slots(
    *,
    page_id: str,
    title: str,
    citations: Iterable[dict[str, Any]] = (),
    diagram: str = "",
    relations: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Return deterministic media slots for a wiki page.

    These slots intentionally describe *what* should be rendered, not *how* to
    call a provider. That keeps the wiki API stable while allowing future VLM
    adapters and human-prior configuration to fill the assets later.
    """

    citation_files = _citation_files(citations)
    relation_items = list(relations or ())
    if not citation_files and not diagram and not relation_items:
        return []

    page_slug = page_id or "page"
    page_title = title or page_slug
    slots: list[WikiMediaSlot] = []

    if diagram:
        slots.append(
            WikiMediaSlot(
                id=f"{page_slug}-structure-diagram",
                kind="diagram",
                placement="lead",
                title=f"{page_title} structure",
                purpose=(
                    "Render the page's deterministic structure diagram as a "
                    "compact visual overview."
                ),
                source_citations=citation_files,
                prompt=(
                    "Create a compact architecture diagram grounded only in "
                    "the cited source files and the existing wiki graph."
                ),
            )
        )

    if citation_files:
        slots.append(
            WikiMediaSlot(
                id=f"{page_slug}-concept-illustration",
                kind="image",
                placement="section",
                title=f"{page_title} concept illustration",
                purpose=(
                    "Illustrate the main code concept using only the page's "
                    "source-grounded evidence."
                ),
                source_citations=citation_files,
                prompt=(
                    "Create a clean teaching-blog style illustration for this "
                    "wiki page. Use the cited files as the only technical "
                    "source of truth and avoid decorative or unrelated imagery."
                ),
            )
        )

    if relation_items:
        slots.append(
            WikiMediaSlot(
                id=f"{page_slug}-flow-storyboard",
                kind="storyboard",
                placement="appendix",
                title=f"{page_title} flow storyboard",
                purpose=(
                    "Plan a short visual sequence that explains how cited "
                    "components hand work to each other."
                ),
                source_citations=citation_files,
                prompt=(
                    "Create a three-to-five panel storyboard grounded in the "
                    "page citations and static relation evidence. Each panel "
                    "should name the component it explains."
                ),
            )
        )

    return [slot.to_dict() for slot in slots]


__all__ = ["WikiMediaSlot", "plan_media_slots"]
