# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Discovery manifest for repository-native multimodal artifacts."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlsplit

from ..repository_filters import walk_repository_files
from ..repository_source_selection import (
    DEFAULT_REPOSITORY_SOURCE_SELECTION,
    RepositorySourceSelection,
)

MEDIA_MANIFEST_SCHEMA = "codenib.media-manifest.v1"
MEDIA_MANIFEST_VERSION = 1
SUPPORTED_MEDIA_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".svg", ".webp"})

_MARKDOWN_EXTENSIONS = frozenset({".md", ".mdx"})
_MAX_MEDIA_ARTIFACTS = 4096
_MAX_MEDIA_BYTES = 32 * 1024 * 1024
_MAX_MARKDOWN_BYTES = 2 * 1024 * 1024
_MAX_TEXT_BYTES = 4096
_MARKDOWN_IMAGE_RE = re.compile(
    r"!\[(?P<alt>[^\]]{0,2048})\]\((?P<target>[^)\s]{1,4096})(?:\s+\"(?P<title>[^\"]{0,2048})\")?\)"
)
_HTML_IMAGE_RE = re.compile(
    r"<img\b[^>]*\bsrc=[\"'](?P<target>[^\"']{1,4096})[\"'][^>]*>",
    flags=re.IGNORECASE,
)
_HTML_ALT_RE = re.compile(
    r"\balt=[\"'](?P<alt>[^\"']{0,2048})[\"']",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class MediaReference:
    """One markdown/html reference that points to a repository media artifact."""

    markdown_path: str
    line: int
    alt_text: str = ""
    title: str = ""
    surrounding_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MediaArtifact:
    """A repository-native visual asset prepared for later VLM extraction."""

    path: str
    media_type: str
    mime_type: str
    sha256: str
    size_bytes: int
    role_hint: str = "repository_image"
    references: tuple[MediaReference, ...] = ()
    caption: str = ""
    surrounding_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["references"] = [reference.to_dict() for reference in self.references]
        return data


@dataclass(frozen=True)
class MediaManifest:
    """Stable manifest for media artifacts discovered inside one repository."""

    schema: str
    version: int
    commit: str
    source_selection_digest: str
    artifacts: tuple[MediaArtifact, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "commit": self.commit,
            "source_selection_digest": self.source_selection_digest,
            "artifact_count": len(self.artifacts),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metadata": dict(self.metadata),
            "manifest_sha256": self.manifest_sha256,
        }

    @property
    def manifest_sha256(self) -> str:
        payload = {
            "schema": self.schema,
            "version": self.version,
            "commit": self.commit,
            "source_selection_digest": self.source_selection_digest,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metadata": dict(self.metadata),
        }
        return _sha256_json(payload)


def discover_media_manifest(
    repo_path: str | Path,
    *,
    commit: str | None = None,
    exclude_roots: Iterable[str | Path] = (),
    selection: RepositorySourceSelection = DEFAULT_REPOSITORY_SOURCE_SELECTION,
    max_artifacts: int = _MAX_MEDIA_ARTIFACTS,
) -> dict[str, Any]:
    """Discover repository-native visual artifacts and return a stable manifest."""

    root = Path(repo_path).expanduser().resolve()
    selected = RepositorySourceSelection(selection.exclude_subtrees)
    references = _discover_markdown_references(
        root,
        exclude_roots=exclude_roots,
        selection=selected,
    )
    artifacts: list[MediaArtifact] = []
    for path in walk_repository_files(
        root,
        exclude_roots=exclude_roots,
        selection=selected,
    ):
        if len(artifacts) >= max(0, max_artifacts):
            break
        if path.is_symlink() or path.suffix.lower() not in SUPPORTED_MEDIA_EXTENSIONS:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size < 0 or size > _MAX_MEDIA_BYTES:
            continue
        relative = path.relative_to(root).as_posix()
        artifact_references = tuple(references.get(relative, ()))
        caption = _caption(artifact_references)
        artifacts.append(
            MediaArtifact(
                path=relative,
                media_type=_media_type(path),
                mime_type=_mime_type(path),
                sha256=_sha256_file(path),
                size_bytes=size,
                role_hint=_role_hint(relative, artifact_references),
                references=artifact_references,
                caption=caption,
                surrounding_text=_surrounding_text(artifact_references),
            )
        )

    manifest = MediaManifest(
        schema=MEDIA_MANIFEST_SCHEMA,
        version=MEDIA_MANIFEST_VERSION,
        commit=commit if commit is not None else _git_commit(root),
        source_selection_digest=selected.digest,
        artifacts=tuple(sorted(artifacts, key=lambda artifact: artifact.path)),
        metadata={
            "supported_extensions": sorted(SUPPORTED_MEDIA_EXTENSIONS),
            "max_media_bytes": _MAX_MEDIA_BYTES,
        },
    )
    return manifest.to_dict()


def _discover_markdown_references(
    root: Path,
    *,
    exclude_roots: Iterable[str | Path],
    selection: RepositorySourceSelection,
) -> dict[str, list[MediaReference]]:
    references: dict[str, list[MediaReference]] = {}
    for path in walk_repository_files(
        root,
        exclude_roots=exclude_roots,
        selection=selection,
    ):
        if path.is_symlink() or path.suffix.lower() not in _MARKDOWN_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > _MAX_MARKDOWN_BYTES:
                continue
            markdown = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        lines = markdown.splitlines()
        relative_markdown = path.relative_to(root).as_posix()
        for line_number, line in enumerate(lines, start=1):
            for target, alt, title in _line_image_links(line):
                target_path = _resolve_media_target(
                    target,
                    markdown_path=Path(relative_markdown),
                )
                if target_path is None:
                    continue
                references.setdefault(target_path, []).append(
                    MediaReference(
                        markdown_path=relative_markdown,
                        line=line_number,
                        alt_text=_safe_text(alt),
                        title=_safe_text(title),
                        surrounding_text=_safe_text(
                            _markdown_window(lines, line_number)
                        ),
                    )
                )
    return {
        key: sorted(value, key=lambda ref: (ref.markdown_path, ref.line))
        for key, value in references.items()
    }


def _line_image_links(line: str) -> list[tuple[str, str, str]]:
    links = [
        (
            match.group("target"),
            match.group("alt") or "",
            match.group("title") or "",
        )
        for match in _MARKDOWN_IMAGE_RE.finditer(line)
    ]
    for match in _HTML_IMAGE_RE.finditer(line):
        tag = match.group(0)
        alt_match = _HTML_ALT_RE.search(tag)
        links.append(
            (match.group("target"), alt_match.group("alt") if alt_match else "", "")
        )
    return links


def _resolve_media_target(target: str, *, markdown_path: Path) -> str | None:
    value = unquote(str(target or "").strip()).split("#", 1)[0].split("?", 1)[0]
    if not value:
        return None
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or value.startswith("//"):
        return None
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    resolved = (markdown_path.parent / candidate).as_posix()
    normalized = PurePosixPath(resolved).as_posix()
    if normalized == "." or normalized.startswith("../"):
        return None
    if PurePosixPath(normalized).suffix.lower() not in SUPPORTED_MEDIA_EXTENSIONS:
        return None
    return normalized


def _markdown_window(lines: list[str], line_number: int) -> str:
    start = max(0, line_number - 2)
    end = min(len(lines), line_number + 1)
    return "\n".join(line.strip() for line in lines[start:end] if line.strip())


def _caption(references: tuple[MediaReference, ...]) -> str:
    for reference in references:
        caption = reference.alt_text or reference.title
        if caption:
            return caption
    return ""


def _surrounding_text(references: tuple[MediaReference, ...]) -> str:
    for reference in references:
        if reference.surrounding_text:
            return reference.surrounding_text
    return ""


def _media_type(path: Path) -> str:
    return "svg" if path.suffix.lower() == ".svg" else "image"


def _mime_type(path: Path) -> str:
    if path.suffix.lower() == ".svg":
        return "image/svg+xml"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _role_hint(path: str, references: tuple[MediaReference, ...]) -> str:
    text = " ".join(
        [
            path,
            *(reference.alt_text for reference in references),
            *(reference.surrounding_text for reference in references),
        ]
    ).lower()
    if any(token in text for token in ("architecture", "diagram", "sequence", "flow")):
        return "architecture_diagram"
    if any(token in text for token in ("screenshot", "screen shot", "ui", "dashboard")):
        return "ui_screenshot"
    return "repository_image"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_commit(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(root), "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _safe_text(value: Any, *, max_bytes: int = _MAX_TEXT_BYTES) -> str:
    text = str(value or "").strip()
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    encoded = text.encode("utf-8")[: max(0, max_bytes - 1)]
    return encoded.decode("utf-8", errors="ignore").rstrip() + "…"


__all__ = [
    "MEDIA_MANIFEST_SCHEMA",
    "MEDIA_MANIFEST_VERSION",
    "SUPPORTED_MEDIA_EXTENSIONS",
    "MediaArtifact",
    "MediaManifest",
    "MediaReference",
    "discover_media_manifest",
]
