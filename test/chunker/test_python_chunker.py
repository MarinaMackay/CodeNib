#!/usr/bin/env python3
"""
Regression tests for the Python code chunker using the httpie CLI repository.
"""

from pathlib import Path
from textwrap import dedent

from codeminer.code_chunker import CodeChunker, RepoChunkingConfig


def _collect_chunks(repo_root: Path, chunk_depth: int, l2_level_exclusive: bool = True):
    repo_config = RepoChunkingConfig(languages=["python"])
    chunker = CodeChunker(
        language="python",
        repo_config=repo_config,
        max_lines_per_chunk=None,
        chunk_depth=chunk_depth,
        l2_level_exclusive=l2_level_exclusive,
    )
    return chunker.chunk_repository(str(repo_root))


def test_python_chunker_level_zero(httpie_cli_repo):
    chunks = _collect_chunks(httpie_cli_repo, chunk_depth=0)
    assert chunks, "Python chunker returned no chunks at depth=0"
    assert all(chunk.chunk_type == "file" for chunk in chunks)


def test_python_chunker_top_level_only(httpie_cli_repo):
    chunks = _collect_chunks(httpie_cli_repo, chunk_depth=1)
    chunk_types = {chunk.chunk_type for chunk in chunks}
    assert "function" in chunk_types
    assert "method" not in chunk_types


def test_python_chunker_includes_methods(httpie_cli_repo):
    chunks = _collect_chunks(httpie_cli_repo, chunk_depth=2)
    chunk_types = {chunk.chunk_type for chunk in chunks}
    assert "method" in chunk_types


def test_python_chunker_includes_classes_when_enabled(httpie_cli_repo):
    chunks = _collect_chunks(httpie_cli_repo, chunk_depth=2, l2_level_exclusive=False)
    chunk_types = {chunk.chunk_type for chunk in chunks}
    assert "class" in chunk_types


def test_python_chunker_chunk_file(httpie_cli_repo):
    target_file = Path(httpie_cli_repo) / "httpie" / "core.py"
    assert target_file.exists(), f"Target Python file not found: {target_file}"

    chunker = CodeChunker(language="python", chunk_depth=2)
    chunks = chunker.chunk_file(str(target_file))

    assert chunks, "Chunker did not return any chunks for httpie/core.py"
    assert any(chunk.chunk_type == "function" for chunk in chunks)
    assert any(chunk.name == "raw_main" for chunk in chunks)


def test_python_chunker_skeleton_mode(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        dedent(
            """
            def top_level(a, b):
                return a + b

            class MyClass:
                def method_one(self, x: int) -> int:
                    return x
            """
        ).lstrip()
    )

    chunker = CodeChunker(
        language="python",
        chunk_depth=2,
        l2_level_exclusive=False,
    )
    chunks = chunker.chunk_file(str(sample), skeleton_mode=True)

    # Class skeleton should carry method signatures but no bodies.
    class_chunk = next(chunk for chunk in chunks if chunk.chunk_type == "class")
    assert "class MyClass" in class_chunk.content
    assert "def method_one" in class_chunk.content
    assert "return x" not in class_chunk.content

    method_chunk = next(chunk for chunk in chunks if chunk.chunk_type == "method")
    assert "def method_one" in method_chunk.content
    assert "return x" not in method_chunk.content
