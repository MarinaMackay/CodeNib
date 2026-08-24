# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Small descriptor-authenticated reads for repository discovery helpers."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def read_regular_bytes(path: Path, *, max_bytes: int) -> bytes | None:
    """Read one stable regular file without following a replaceable symlink."""

    try:
        before = path.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
        return None

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_size > max_bytes
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            return None

        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)

        finished = os.fstat(descriptor)
        if (
            finished.st_size,
            finished.st_mtime_ns,
            finished.st_ctime_ns,
        ) != (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns):
            return None
        payload = b"".join(chunks)
        return payload if len(payload) <= max_bytes else None
    finally:
        os.close(descriptor)


__all__ = ["read_regular_bytes"]
