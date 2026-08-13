# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral materialization for planned wiki media slots."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

_GENERATABLE_IMAGE_KINDS = frozenset({"diagram", "image", "storyboard"})


@dataclass(frozen=True)
class WikiMediaAsset:
    """A generated media asset with source and model provenance."""

    slot_id: str
    kind: str
    uri: str
    mime_type: str
    model: str
    provider: str
    prompt: str
    source_citations: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_citations"] = list(self.source_citations)
        return data


class OpenAICompatibleImageGenerator:
    """Generate wiki media through an OpenAI-compatible images endpoint.

    The adapter targets ``POST /images/generations`` and accepts either
    ``b64_json`` or hosted ``url`` responses. It is intentionally isolated from
    page planning so CodeNib can keep deterministic slots even when no media
    provider is configured.
    """

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str | None = None,
        size: str = "1024x1024",
        timeout: float = 120.0,
        urlopen: Callable[..., Any] | None = None,
        provider: str = "openai-compatible",
    ) -> None:
        self.model = str(model or "").strip()
        self.api_base = str(api_base or "").strip()
        self.api_key = api_key
        self.size = size
        self.timeout = timeout
        self.provider = provider
        self._urlopen = urlopen or urllib.request.urlopen
        if not self.model:
            raise ValueError("wiki media model is required")
        if not self.api_base:
            raise ValueError("wiki media api_base is required")

    @property
    def endpoint(self) -> str:
        base = self.api_base.rstrip("/")
        if base.endswith("/images/generations"):
            return base
        return f"{base}/images/generations"

    def generate(
        self,
        slot: Mapping[str, Any],
        *,
        output_dir: str | Path | None = None,
        asset_base_path: str = "assets/wiki-media",
        reuse_existing: bool = True,
    ) -> dict[str, Any]:
        kind = str(slot.get("kind") or "").strip()
        if kind not in _GENERATABLE_IMAGE_KINDS:
            raise ValueError(f"unsupported image media slot kind: {kind!r}")
        slot_id = str(slot.get("id") or "").strip()
        if not slot_id:
            raise ValueError("media slot id is required")

        prompt = _generation_prompt(slot)
        filename = f"{_safe_filename(slot_id)}.png"
        cached = _read_cached_asset(
            output_dir,
            filename,
            expected={
                "slot_id": slot_id,
                "model": self.model,
                "provider": self.provider,
                "prompt_sha256": _sha256_text(prompt),
                "size": self.size,
            },
        )
        if cached is not None and reuse_existing:
            return cached

        payload = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "size": self.size,
            "response_format": "b64_json",
        }
        response = self._post_json(payload)
        item = _first_image_response(response)
        citations = tuple(str(value) for value in slot.get("source_citations") or ())

        if item.get("b64_json"):
            data = base64.b64decode(str(item["b64_json"]), validate=True)
            if output_dir is None:
                raise ValueError("output_dir is required for b64_json image responses")
            target_dir = Path(output_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / filename
            target.write_bytes(data)
            uri = f"{asset_base_path.rstrip('/')}/{filename}"
            mime_type = "image/png"
        elif item.get("url"):
            uri = str(item["url"])
            mime_type = "image/*"
        else:
            raise ValueError("image response must contain b64_json or url")

        asset = WikiMediaAsset(
            slot_id=slot_id,
            kind=kind,
            uri=uri,
            mime_type=mime_type,
            model=self.model,
            provider=self.provider,
            prompt=prompt,
            source_citations=citations,
            metadata={"size": self.size},
        ).to_dict()
        _write_asset_manifest(
            output_dir,
            filename,
            asset,
            prompt_sha256=_sha256_text(prompt),
        )
        return asset

    def _post_json(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with self._urlopen(request, timeout=self.timeout) as response:
            raw = response.read()
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("image response must be a JSON object")
        return data


class DeterministicSvgMediaGenerator:
    """Generate local, source-grounded SVG previews with the same asset contract.

    This gives CodeNib a zero-credential multimodal path for demos, tests, and
    offline exports. It is deliberately named as a local renderer rather than a
    VLM, so teams can distinguish deterministic scaffolding from provider
    output while keeping the page schema identical.
    """

    model = "local/svg"
    provider = "local"

    def generate(
        self,
        slot: Mapping[str, Any],
        *,
        output_dir: str | Path | None = None,
        asset_base_path: str = "assets/wiki-media",
        reuse_existing: bool = True,
    ) -> dict[str, Any]:
        kind = str(slot.get("kind") or "").strip()
        if kind not in _GENERATABLE_IMAGE_KINDS:
            raise ValueError(f"unsupported image media slot kind: {kind!r}")
        slot_id = str(slot.get("id") or "").strip()
        if not slot_id:
            raise ValueError("media slot id is required")
        if output_dir is None:
            raise ValueError("output_dir is required for local SVG generation")

        prompt = _generation_prompt(slot)
        filename = f"{_safe_filename(slot_id)}.svg"
        cached = _read_cached_asset(
            output_dir,
            filename,
            expected={
                "slot_id": slot_id,
                "model": self.model,
                "provider": self.provider,
                "prompt_sha256": _sha256_text(prompt),
            },
        )
        if cached is not None and reuse_existing:
            return cached

        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        target.write_text(_svg_for_slot(slot), encoding="utf-8")
        citations = tuple(str(value) for value in slot.get("source_citations") or ())
        asset = WikiMediaAsset(
            slot_id=slot_id,
            kind=kind,
            uri=f"{asset_base_path.rstrip('/')}/{filename}",
            mime_type="image/svg+xml",
            model=self.model,
            provider=self.provider,
            prompt=prompt,
            source_citations=citations,
            metadata={"deterministic": True},
        ).to_dict()
        _write_asset_manifest(
            output_dir,
            filename,
            asset,
            prompt_sha256=_sha256_text(prompt),
        )
        return asset


def materialize_media_slots(
    page: Mapping[str, Any],
    *,
    generator: Any,
    output_dir: str | Path,
    asset_base_path: str = "assets/wiki-media",
) -> dict[str, Any]:
    """Generate assets for supported slots and attach them to a page payload."""

    slots = []
    for slot in page.get("media_slots") or ():
        if not isinstance(slot, Mapping):
            continue
        updated = dict(slot)
        if "asset" not in updated:
            kind = str(updated.get("kind") or "")
            if kind in _GENERATABLE_IMAGE_KINDS:
                updated["asset"] = generator.generate(
                    updated,
                    output_dir=output_dir,
                    asset_base_path=asset_base_path,
                )
        slots.append(updated)
    return {**dict(page), "media_slots": slots}


def image_generator_from_config(
    config: Any,
) -> OpenAICompatibleImageGenerator | DeterministicSvgMediaGenerator | None:
    """Build a media generator from ``QAConfig``-shaped settings."""

    if not bool(getattr(config, "wiki_media_generation_enabled", False)):
        return None
    model = str(getattr(config, "wiki_media_model") or "").strip()
    options = dict(getattr(config, "wiki_media_options", {}) or {})
    provider = str(options.get("provider") or "").strip().lower()
    if model.lower() in {"local/svg", "local-svg"} or provider in {
        "local",
        "local-svg",
    }:
        return DeterministicSvgMediaGenerator()
    return OpenAICompatibleImageGenerator(
        model=model,
        api_base=str(getattr(config, "wiki_media_api_base") or ""),
        api_key=getattr(config, "wiki_media_api_key", None),
        size=str(options.get("size") or "1024x1024"),
        timeout=float(options.get("timeout") or 120.0),
        provider=str(options.get("provider") or "openai-compatible"),
    )


def _generation_prompt(slot: Mapping[str, Any]) -> str:
    parts = [str(slot.get("prompt") or "").strip()]
    purpose = str(slot.get("purpose") or "").strip()
    if purpose:
        parts.append(f"Purpose: {purpose}")
    citations = [str(value) for value in slot.get("source_citations") or ()]
    if citations:
        parts.append("Source citations: " + ", ".join(citations))
    return "\n\n".join(part for part in parts if part)


def _first_image_response(response: Mapping[str, Any]) -> Mapping[str, Any]:
    data = response.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], Mapping):
        raise ValueError("image response must contain data[0]")
    return data[0]


def _safe_filename(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return name or "wiki-media"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _manifest_path(output_dir: str | Path | None, filename: str) -> Path | None:
    if output_dir is None:
        return None
    return Path(output_dir) / f"{filename}.json"


def _read_cached_asset(
    output_dir: str | Path | None,
    filename: str,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any] | None:
    manifest_path = _manifest_path(output_dir, filename)
    if manifest_path is None or not manifest_path.is_file():
        return None
    asset_path = Path(output_dir) / filename
    if not asset_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    asset = payload.get("asset")
    return dict(asset) if isinstance(asset, Mapping) else None


def _write_asset_manifest(
    output_dir: str | Path | None,
    filename: str,
    asset: Mapping[str, Any],
    *,
    prompt_sha256: str,
) -> None:
    manifest_path = _manifest_path(output_dir, filename)
    if manifest_path is None:
        return
    payload = {
        "slot_id": asset.get("slot_id"),
        "model": asset.get("model"),
        "provider": asset.get("provider"),
        "prompt_sha256": prompt_sha256,
        "size": (asset.get("metadata") or {}).get("size"),
        "asset": dict(asset),
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _svg_for_slot(slot: Mapping[str, Any]) -> str:
    title = html.escape(str(slot.get("title") or "Wiki media"))
    kind = html.escape(str(slot.get("kind") or "media").title())
    purpose = html.escape(str(slot.get("purpose") or "Source-grounded visual"))
    citations = [str(value) for value in slot.get("source_citations") or ()][:4]
    files = citations or ["source evidence"]
    palette = {
        "diagram": ("#2563eb", "#eff6ff"),
        "image": ("#7c3aed", "#f5f3ff"),
        "storyboard": ("#0891b2", "#ecfeff"),
    }
    accent, wash = palette.get(str(slot.get("kind") or ""), ("#2563eb", "#eff6ff"))
    kind_value = str(slot.get("kind") or "")
    prompt_hash = _sha256_text(_generation_prompt(slot))[:12]
    if kind_value == "diagram":
        body = _svg_diagram_body(files, accent, wash)
    elif kind_value == "storyboard":
        body = _svg_storyboard_body(files, accent, wash)
    else:
        body = _svg_concept_body(files, accent, wash)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540" role="img" aria-label="{title}">
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="{wash}"/>
      <stop offset="1" stop-color="#ffffff"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="12" stdDeviation="18" flood-color="#0f172a" flood-opacity="0.14"/>
    </filter>
  </defs>
  <rect width="960" height="540" fill="url(#bg)"/>
  <circle cx="828" cy="84" r="94" fill="{accent}" opacity="0.11"/>
  <circle cx="112" cy="468" r="132" fill="{accent}" opacity="0.08"/>
  <g filter="url(#shadow)">
    <rect x="56" y="56" width="848" height="428" rx="28" fill="#ffffff" stroke="#dbe3ef"/>
  </g>
  <rect x="84" y="86" width="96" height="30" rx="15" fill="{wash}" stroke="{accent}"/>
  <text x="104" y="106" font-size="14" fill="{accent}" font-weight="750" font-family="ui-sans-serif, system-ui">{kind}</text>
  <text x="84" y="152" font-size="31" fill="#0f172a" font-weight="780" font-family="ui-sans-serif, system-ui">{title}</text>
  {_svg_multiline_text(purpose, x=84, y=180, width=760, max_lines=2, font_size=15, color="#475569")}
  {body}
  <rect x="84" y="448" width="792" height="22" rx="11" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="100" y="464" font-size="12" fill="#64748b" font-family="ui-sans-serif, system-ui">local/svg · source-grounded media slot · prompt sha256:{prompt_hash} · provider-neutral asset contract</text>
</svg>
"""


def _svg_diagram_body(files: list[str], accent: str, wash: str) -> str:
    cards = [
        ("Source", "cited files"),
        ("Facts", "graph/LSP"),
        ("Plan", "media slot"),
        ("Asset", "wiki media"),
    ]
    card_markup = []
    for index, (label, detail) in enumerate(cards):
        x = 92 + index * 202
        card_markup.append(
            f'<rect x="{x}" y="252" width="154" height="88" rx="20" fill="{wash}" '
            f'stroke="{accent}" stroke-width="1.5"/>'
            f'<circle cx="{x + 28}" cy="281" r="12" fill="{accent}" opacity="0.18"/>'
            f'<text x="{x + 48}" y="286" font-size="18" fill="#0f172a" '
            f'font-weight="760" font-family="ui-sans-serif, system-ui">{label}</text>'
            f'<text x="{x + 24}" y="318" font-size="13" fill="#475569" '
            f'font-family="ui-sans-serif, system-ui">{detail}</text>'
        )
        if index < len(cards) - 1:
            ax = x + 154
            card_markup.append(
                f'<path d="M{ax + 10} 296 L{ax + 44} 296" stroke="{accent}" '
                f'stroke-width="3.5" stroke-linecap="round"/>'
                f'<path d="M{ax + 38} 288 L{ax + 48} 296 L{ax + 38} 304" '
                f'stroke="{accent}" stroke-width="3.5" fill="none" '
                f'stroke-linecap="round" stroke-linejoin="round"/>'
            )
    citation_markup = _svg_citation_chips(files, x=92, y=370, width=760)
    return "\n  ".join(
        [
            '<text x="92" y="230" font-size="14" fill="#64748b" '
            'font-weight="700" font-family="ui-sans-serif, system-ui">'
            "Deterministic planning path</text>",
            *card_markup,
            citation_markup,
        ]
    )


def _svg_concept_body(files: list[str], accent: str, wash: str) -> str:
    return f"""
  <rect x="92" y="236" width="360" height="178" rx="22" fill="#0f172a"/>
  <rect x="116" y="264" width="184" height="12" rx="6" fill="#94a3b8"/>
  <rect x="116" y="294" width="284" height="10" rx="5" fill="#38bdf8" opacity="0.84"/>
  <rect x="116" y="320" width="224" height="10" rx="5" fill="#a78bfa" opacity="0.90"/>
  <rect x="116" y="346" width="260" height="10" rx="5" fill="#34d399" opacity="0.86"/>
  <rect x="116" y="372" width="198" height="10" rx="5" fill="#fbbf24" opacity="0.82"/>
  <text x="116" y="446" font-size="13" fill="#64748b" font-family="ui-sans-serif, system-ui">source code stays the technical source of truth</text>
  <path d="M474 324 C526 272 572 272 624 324" stroke="{accent}" stroke-width="4" fill="none" stroke-linecap="round"/>
  <path d="M612 314 L626 324 L612 334" stroke="{accent}" stroke-width="4" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  <rect x="642" y="236" width="210" height="178" rx="28" fill="{wash}" stroke="{accent}" stroke-width="1.5"/>
  <circle cx="708" cy="302" r="38" fill="{accent}" opacity="0.18"/>
  <path d="M684 304 L704 324 L738 282" stroke="{accent}" stroke-width="7" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  <text x="684" y="364" font-size="20" fill="#0f172a" font-weight="780" font-family="ui-sans-serif, system-ui">teaching visual</text>
  <text x="684" y="388" font-size="13" fill="#475569" font-family="ui-sans-serif, system-ui">bounded by citations</text>
  {_svg_citation_chips(files, x=92, y=204, width=760)}
"""


def _svg_storyboard_body(files: list[str], accent: str, wash: str) -> str:
    panels = [
        ("1", "Evidence", "collect cited files"),
        ("2", "Relations", "trace graph facts"),
        ("3", "Narrative", "shape teaching flow"),
        ("4", "Asset", "render local media"),
    ]
    markup = [
        '<text x="92" y="226" font-size="14" fill="#64748b" '
        'font-weight="700" font-family="ui-sans-serif, system-ui">'
        "Storyboard for future video generation</text>"
    ]
    for index, (num, title, detail) in enumerate(panels):
        x = 92 + index * 196
        markup.append(
            f'<rect x="{x}" y="250" width="166" height="136" rx="22" fill="{wash}" '
            f'stroke="{accent}" stroke-width="1.5"/>'
            f'<circle cx="{x + 34}" cy="286" r="17" fill="{accent}"/>'
            f'<text x="{x + 29}" y="292" font-size="17" fill="#ffffff" '
            f'font-weight="800" font-family="ui-sans-serif, system-ui">{num}</text>'
            f'<text x="{x + 24}" y="330" font-size="18" fill="#0f172a" '
            f'font-weight="760" font-family="ui-sans-serif, system-ui">{title}</text>'
            f'<text x="{x + 24}" y="356" font-size="13" fill="#475569" '
            f'font-family="ui-sans-serif, system-ui">{detail}</text>'
        )
        if index < len(panels) - 1:
            markup.append(
                f'<path d="M{x + 170} 318 L{x + 190} 318" stroke="{accent}" '
                f'stroke-width="3" stroke-linecap="round"/>'
            )
    markup.append(_svg_citation_chips(files, x=92, y=410, width=760))
    return "\n  ".join(markup)


def _svg_citation_chips(files: list[str], *, x: int, y: int, width: int) -> str:
    chips = [
        '<text x="{x}" y="{y}" font-size="13" fill="#64748b" font-weight="700" '
        'font-family="ui-sans-serif, system-ui">Source citations</text>'.format(
            x=x, y=y
        )
    ]
    cursor_x = x
    cursor_y = y + 18
    for file in files[:4]:
        label = _svg_shorten(file, 34)
        chip_width = min(width, max(132, len(label) * 8 + 30))
        if cursor_x + chip_width > x + width:
            cursor_x = x
            cursor_y += 32
        chips.append(
            f'<rect x="{cursor_x}" y="{cursor_y}" width="{chip_width}" height="24" '
            f'rx="12" fill="#f8fafc" stroke="#dbe3ef"/>'
            f'<text x="{cursor_x + 14}" y="{cursor_y + 17}" font-size="12" '
            f'fill="#334155" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">'
            f"{html.escape(label)}</text>"
        )
        cursor_x += chip_width + 10
    return "\n  ".join(chips)


def _svg_multiline_text(
    text: str,
    *,
    x: int,
    y: int,
    width: int,
    max_lines: int,
    font_size: int,
    color: str,
) -> str:
    max_chars = max(24, width // max(1, font_size // 2))
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and words:
        consumed = " ".join(lines)
        if len(consumed) < len(text):
            lines[-1] = _svg_shorten(lines[-1], max_chars - 1) + "…"
    return "\n  ".join(
        f'<text x="{x}" y="{y + index * (font_size + 7)}" font-size="{font_size}" '
        f'fill="{color}" font-family="ui-sans-serif, system-ui">'
        f"{html.escape(line)}</text>"
        for index, line in enumerate(lines)
    )


def _svg_shorten(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return "…" + value[-(limit - 1) :]


__all__ = [
    "DeterministicSvgMediaGenerator",
    "OpenAICompatibleImageGenerator",
    "WikiMediaAsset",
    "image_generator_from_config",
    "materialize_media_slots",
]
