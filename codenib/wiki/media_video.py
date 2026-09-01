# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Render validated visual storyboards as local, source-grounded MP4 assets."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .._bounded_json import validate_bounded_json_stream, validate_json_complexity
from ._safe_file_reads import read_regular_bytes
from .media_storyboard import (
    validate_visual_storyboard,
    validate_visual_storyboard_manifest,
)

VIDEO_RENDERER_ID = "local/ffmpeg-storyboard-v1"
_MAX_VIDEO_BYTES = 64 * 1024 * 1024
_MAX_FFMPEG_VERSION_BYTES = 512
_MAX_VIDEO_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_VIDEO_MANIFEST_NODES = 100_000
_MAX_VIDEO_MANIFEST_TOKENS = 200_000
_MAX_VIDEOS = 4096
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_VIDEO_FIELDS = frozenset(
    {
        "schema",
        "renderer",
        "storyboard_sha256",
        "artifact_path",
        "output_path",
        "mime_type",
        "content_sha256",
        "size_bytes",
        "width",
        "height",
        "fps",
        "frame_count",
        "duration_ms",
        "source_citations",
        "ffmpeg_version",
    }
)

# A compact built-in bitmap alphabet keeps the renderer dependency-free apart
# from ffmpeg itself. Unknown glyphs intentionally become a visible box.
_FONT = {
    " ": ("00000",) * 7,
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    "/": ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    ":": ("00000", "00110", "00110", "00000", "00110", "00110", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
}


def _add_glyphs() -> None:
    rows = {
        "A": "01110 10001 10001 11111 10001 10001 10001",
        "B": "11110 10001 10001 11110 10001 10001 11110",
        "C": "01111 10000 10000 10000 10000 10000 01111",
        "D": "11110 10001 10001 10001 10001 10001 11110",
        "E": "11111 10000 10000 11110 10000 10000 11111",
        "F": "11111 10000 10000 11110 10000 10000 10000",
        "G": "01111 10000 10000 10111 10001 10001 01111",
        "H": "10001 10001 10001 11111 10001 10001 10001",
        "I": "11111 00100 00100 00100 00100 00100 11111",
        "J": "00111 00010 00010 00010 10010 10010 01100",
        "K": "10001 10010 10100 11000 10100 10010 10001",
        "L": "10000 10000 10000 10000 10000 10000 11111",
        "M": "10001 11011 10101 10101 10001 10001 10001",
        "N": "10001 11001 10101 10011 10001 10001 10001",
        "O": "01110 10001 10001 10001 10001 10001 01110",
        "P": "11110 10001 10001 11110 10000 10000 10000",
        "Q": "01110 10001 10001 10001 10101 10010 01101",
        "R": "11110 10001 10001 11110 10100 10010 10001",
        "S": "01111 10000 10000 01110 00001 00001 11110",
        "T": "11111 00100 00100 00100 00100 00100 00100",
        "U": "10001 10001 10001 10001 10001 10001 01110",
        "V": "10001 10001 10001 10001 10001 01010 00100",
        "W": "10001 10001 10001 10101 10101 10101 01010",
        "X": "10001 10001 01010 00100 01010 10001 10001",
        "Y": "10001 10001 01010 00100 00100 00100 00100",
        "Z": "11111 00001 00010 00100 01000 10000 11111",
        "0": "01110 10001 10011 10101 11001 10001 01110",
        "1": "00100 01100 00100 00100 00100 00100 01110",
        "2": "01110 10001 00001 00010 00100 01000 11111",
        "3": "11110 00001 00001 01110 00001 00001 11110",
        "4": "00010 00110 01010 10010 11111 00010 00010",
        "5": "11111 10000 10000 11110 00001 00001 11110",
        "6": "01110 10000 10000 11110 10001 10001 01110",
        "7": "11111 00001 00010 00100 01000 01000 01000",
        "8": "01110 10001 10001 01110 10001 10001 01110",
        "9": "01110 10001 10001 01111 00001 00001 01110",
    }
    _FONT.update({key: tuple(value.split()) for key, value in rows.items()})


_add_glyphs()


def render_visual_storyboard_video(
    storyboard: Mapping[str, Any],
    output_path: str | Path,
    *,
    ffmpeg: str | None = None,
    width: int = 960,
    height: int = 540,
    fps: int = 24,
) -> dict[str, Any]:
    """Render one validated storyboard to an atomically published MP4."""

    normalized = validate_visual_storyboard(storyboard)
    executable = _ffmpeg_executable(ffmpeg)
    width = _bounded_integer(width, label="video width", minimum=320, maximum=1920)
    height = _bounded_integer(height, label="video height", minimum=180, maximum=1080)
    fps = _bounded_integer(fps, label="video fps", minimum=1, maximum=60)
    destination = Path(output_path).expanduser()
    if destination.suffix.lower() != ".mp4":
        raise ValueError("storyboard video output must use the .mp4 extension")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="codenib-storyboard-") as temporary_name:
        temporary = Path(temporary_name)
        concat_lines = []
        frame_paths = []
        for index, frame in enumerate(normalized["frames"], start=1):
            frame_path = temporary / f"frame-{index:03d}.ppm"
            _write_storyboard_frame(
                frame_path,
                frame,
                index=index,
                frame_count=len(normalized["frames"]),
                width=width,
                height=height,
            )
            frame_paths.append(frame_path)
            concat_lines.extend(
                [
                    f"file '{frame_path.name}'",
                    f"duration {frame['duration_ms'] / 1000:.3f}",
                ]
            )
        concat_lines.append(f"file '{frame_paths[-1].name}'")
        concat_path = temporary / "frames.txt"
        concat_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

        descriptor, unpublished_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.", suffix=".tmp.mp4", dir=destination.parent
        )
        os.close(descriptor)
        unpublished = Path(unpublished_name)
        try:
            command = [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "1",
                "-i",
                str(concat_path),
                "-vf",
                f"fps={fps},format=yuv420p",
                "-c:v",
                "libx264",
                "-movflags",
                "+faststart",
                str(unpublished),
            ]
            _run_ffmpeg(command, cwd=temporary)
            payload = read_regular_bytes(unpublished, max_bytes=_MAX_VIDEO_BYTES)
            if payload is None or len(payload) < 12 or b"ftyp" not in payload[:32]:
                raise ValueError("ffmpeg did not produce a bounded MP4 asset")
            os.replace(unpublished, destination)
        finally:
            try:
                unpublished.unlink()
            except FileNotFoundError:
                pass

    payload = read_regular_bytes(destination, max_bytes=_MAX_VIDEO_BYTES)
    if payload is None:
        raise ValueError("rendered storyboard video is not a stable regular file")
    citations = sorted(
        {
            citation["source_path"]
            for frame in normalized["frames"]
            for citation in frame["source_citations"]
        }
    )
    return {
        "schema": "codenib.storyboard-video-provenance.v1",
        "renderer": VIDEO_RENDERER_ID,
        "storyboard_sha256": normalized["storyboard_sha256"],
        "artifact_path": normalized["artifact_path"],
        "output_path": destination.name,
        "mime_type": "video/mp4",
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": len(normalized["frames"]),
        "duration_ms": normalized["total_duration_ms"],
        "source_citations": citations,
        "ffmpeg_version": _ffmpeg_version(executable),
    }


def render_visual_storyboard_manifest_videos(
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    *,
    ffmpeg: str | None = None,
    width: int = 960,
    height: int = 540,
    fps: int = 24,
) -> dict[str, Any]:
    """Render every storyboard and persist a canonical provenance manifest."""

    normalized = validate_visual_storyboard_manifest(manifest)
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    videos = []
    for index, storyboard in enumerate(normalized["storyboards"], start=1):
        stem = _safe_stem(storyboard["artifact_path"], index)
        videos.append(
            render_visual_storyboard_video(
                storyboard,
                destination / f"{stem}.mp4",
                ffmpeg=ffmpeg,
                width=width,
                height=height,
                fps=fps,
            )
        )
    output = {
        "schema": "codenib.storyboard-video-manifest.v1",
        "storyboard_manifest_sha256": normalized["manifest_sha256"],
        "video_count": len(videos),
        "videos": videos,
    }
    _atomic_json(destination / "manifest.json", output)
    return output


def load_storyboard_video_manifest(path: str | Path) -> dict[str, Any]:
    """Load and authenticate a generated video manifest and every MP4 asset."""

    source = Path(path).expanduser()
    raw = read_regular_bytes(source, max_bytes=_MAX_VIDEO_MANIFEST_BYTES)
    if raw is None:
        raise ValueError("storyboard video manifest must be a bounded regular file")
    validate_bounded_json_stream(
        io.BytesIO(raw),
        label="storyboard video manifest",
        max_bytes=_MAX_VIDEO_MANIFEST_BYTES,
        max_nodes=_MAX_VIDEO_MANIFEST_NODES,
        max_lexical_tokens=_MAX_VIDEO_MANIFEST_TOKENS,
    )
    try:
        decoded = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("storyboard video manifest contains invalid JSON") from exc
    normalized = validate_storyboard_video_manifest(decoded)
    for video in normalized["videos"]:
        payload = read_regular_bytes(
            source.parent / video["output_path"], max_bytes=_MAX_VIDEO_BYTES
        )
        if payload is None:
            raise ValueError("storyboard video asset is missing or unsafe")
        if len(payload) != video["size_bytes"]:
            raise ValueError("storyboard video asset size does not match")
        if hashlib.sha256(payload).hexdigest() != video["content_sha256"]:
            raise ValueError("storyboard video asset hash does not match")
    return normalized


def read_storyboard_video_asset(
    manifest_path: str | Path, filename: str
) -> bytes | None:
    """Read an authenticated MP4 that is declared by a validated manifest."""

    source = Path(manifest_path).expanduser()
    manifest = load_storyboard_video_manifest(source)
    video = next(
        (item for item in manifest["videos"] if item["output_path"] == filename),
        None,
    )
    if video is None:
        return None
    return read_regular_bytes(source.parent / filename, max_bytes=_MAX_VIDEO_BYTES)


def validate_storyboard_video_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the bounded public projection of rendered video provenance."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "storyboard_manifest_sha256",
        "video_count",
        "videos",
    }:
        raise ValueError("storyboard video manifest fields are invalid")
    if value["schema"] != "codenib.storyboard-video-manifest.v1":
        raise ValueError("storyboard video manifest schema is unsupported")
    videos_value = value["videos"]
    if not isinstance(videos_value, list) or len(videos_value) > _MAX_VIDEOS:
        raise ValueError("storyboard video manifest videos are invalid")
    videos = [_validated_video(video) for video in videos_value]
    if type(value["video_count"]) is not int or value["video_count"] != len(videos):
        raise ValueError("storyboard video manifest video_count is invalid")
    output_paths = [video["output_path"] for video in videos]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("storyboard video output paths must be unique")
    normalized = {
        "schema": "codenib.storyboard-video-manifest.v1",
        "storyboard_manifest_sha256": _digest(value["storyboard_manifest_sha256"]),
        "video_count": len(videos),
        "videos": videos,
    }
    validate_json_complexity(
        normalized,
        label="storyboard video manifest",
        max_nodes=_MAX_VIDEO_MANIFEST_NODES,
    )
    return normalized


def _validated_video(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _VIDEO_FIELDS:
        raise ValueError("storyboard video provenance fields are invalid")
    if value["schema"] != "codenib.storyboard-video-provenance.v1":
        raise ValueError("storyboard video provenance schema is unsupported")
    if value["renderer"] != VIDEO_RENDERER_ID or value["mime_type"] != "video/mp4":
        raise ValueError("storyboard video renderer or MIME type is unsupported")
    output_path = _single_line(
        value["output_path"], label="video output path", limit=128
    )
    if (
        PurePosixPath(output_path).name != output_path
        or Path(output_path).suffix.lower() != ".mp4"
    ):
        raise ValueError("storyboard video output path must be a flat MP4 filename")
    citations_value = value["source_citations"]
    if not isinstance(citations_value, list) or len(citations_value) > 192:
        raise ValueError("storyboard video source citations are invalid")
    citations = [_relative_path(item) for item in citations_value]
    if citations != sorted(set(citations)):
        raise ValueError("storyboard video source citations must be unique and sorted")
    return {
        "schema": "codenib.storyboard-video-provenance.v1",
        "renderer": VIDEO_RENDERER_ID,
        "storyboard_sha256": _digest(value["storyboard_sha256"]),
        "artifact_path": _relative_path(value["artifact_path"]),
        "output_path": output_path,
        "mime_type": "video/mp4",
        "content_sha256": _digest(value["content_sha256"]),
        "size_bytes": _integer(value["size_bytes"], 1, _MAX_VIDEO_BYTES),
        "width": _integer(value["width"], 320, 1920),
        "height": _integer(value["height"], 180, 1080),
        "fps": _integer(value["fps"], 1, 60),
        "frame_count": _integer(value["frame_count"], 1, 12),
        "duration_ms": _integer(value["duration_ms"], 1, 300_000),
        "source_citations": citations,
        "ffmpeg_version": _single_line(
            value["ffmpeg_version"], label="ffmpeg version", limit=512
        ),
    }


def _write_storyboard_frame(
    path: Path,
    frame: Mapping[str, Any],
    *,
    index: int,
    frame_count: int,
    width: int,
    height: int,
) -> None:
    pixels = bytearray((12, 19, 33)) * (width * height)
    _fill_rect(pixels, width, height, 0, 0, 18, height, (14, 165, 233))
    _fill_rect(pixels, width, height, 48, 46, width - 96, height - 92, (24, 35, 54))
    _draw_text(
        pixels,
        width,
        height,
        74,
        74,
        f"CODENIB VISUAL STORY {index}/{frame_count}",
        3,
        (56, 189, 248),
        52,
    )
    y = 132
    for line in _wrap_text(str(frame["title"]), 34)[:3]:
        _draw_text(pixels, width, height, 74, y, line, 5, (248, 250, 252), 34)
        y += 46
    y += 14
    for line in _wrap_text(str(frame["narration"]), 66)[:4]:
        _draw_text(pixels, width, height, 76, y, line, 2, (203, 213, 225), 66)
        y += 24
    citations = frame["source_citations"]
    if citations:
        y = height - 96
        source = citations[0]
        evidence = (
            f"SOURCE  {source['source_path']}:{source['line']}  {source['symbol']}"
        )
        _draw_text(pixels, width, height, 76, y, evidence, 2, (134, 239, 172), 76)
    else:
        _draw_text(
            pixels,
            width,
            height,
            76,
            height - 96,
            "SOURCE GROUNDING PENDING",
            2,
            (251, 191, 36),
            76,
        )
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + pixels)


def _draw_text(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    value: str,
    scale: int,
    color: tuple[int, int, int],
    limit: int,
) -> None:
    cursor = x
    for character in value.upper()[:limit]:
        glyph = _FONT.get(character, _FONT["?"])
        for row, bits in enumerate(glyph):
            for column, enabled in enumerate(bits):
                if enabled == "1":
                    _fill_rect(
                        pixels,
                        width,
                        height,
                        cursor + column * scale,
                        y + row * scale,
                        scale,
                        scale,
                        color,
                    )
        cursor += 6 * scale


def _fill_rect(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    rect_width: int,
    rect_height: int,
    color: tuple[int, int, int],
) -> None:
    left, top = max(0, x), max(0, y)
    right, bottom = min(width, x + rect_width), min(height, y + rect_height)
    row = bytes(color) * max(0, right - left)
    for current_y in range(top, bottom):
        start = (current_y * width + left) * 3
        pixels[start : start + len(row)] = row


def _wrap_text(value: str, width: int) -> list[str]:
    words = value.replace("\n", " ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > width:
            lines.append(current)
            current = word[:width]
        else:
            current = candidate[:width]
    if current:
        lines.append(current)
    return lines or [""]


def _safe_stem(path: str, index: int) -> str:
    stem = _SAFE_STEM_RE.sub("-", Path(path).stem).strip(".-") or "visual"
    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:10]
    return f"{index:03d}-{stem[:64]}-{digest}"


def _ffmpeg_executable(value: str | None) -> str:
    if value:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute() or not candidate.is_file():
            raise ValueError("ffmpeg must be an absolute regular executable path")
        executable = str(candidate)
    else:
        executable = shutil.which("ffmpeg") or ""
    if not executable or not os.access(executable, os.X_OK):
        raise ValueError("ffmpeg executable is required to render storyboard video")
    return executable


def _run_ffmpeg(command: list[str], *, cwd: Path) -> None:
    try:
        subprocess.run(
            command,
            cwd=cwd,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("ffmpeg storyboard render timed out") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()[:500]
        raise ValueError(f"ffmpeg storyboard render failed: {detail}") from exc


def _ffmpeg_version(executable: str) -> str:
    try:
        result = subprocess.run(
            [executable, "-version"],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    first_line = result.stdout.splitlines()[0] if result.stdout else b"unknown"
    return first_line[:_MAX_FFMPEG_VERSION_BYTES].decode("utf-8", errors="replace")


def _bounded_integer(value: Any, *, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return value


def _integer(value: Any, minimum: int, maximum: int) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise ValueError("storyboard video integer is out of bounds")
    return value


def _digest(value: Any) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError("storyboard video digest is invalid")
    return value


def _single_line(value: Any, *, label: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    text = value.strip()
    if (
        not text
        or len(text.encode("utf-8")) > limit
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in text)
    ):
        raise ValueError(f"{label} is invalid")
    return text


def _relative_path(value: Any) -> str:
    text = _single_line(value, label="storyboard video source path", limit=4096)
    if "\\" in text:
        raise ValueError("storyboard video source path must use POSIX separators")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("storyboard video source path is unsafe")
    return path.as_posix()


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"storyboard video manifest contains duplicate key: {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"storyboard video manifest contains non-finite number: {value}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


__all__ = [
    "VIDEO_RENDERER_ID",
    "load_storyboard_video_manifest",
    "read_storyboard_video_asset",
    "render_visual_storyboard_manifest_videos",
    "render_visual_storyboard_video",
    "validate_storyboard_video_manifest",
]
