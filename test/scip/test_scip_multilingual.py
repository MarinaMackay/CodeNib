#!/usr/bin/env python3
"""
Integration tests for multilingual SCIP indexing via unified run_pipeline().
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from codeminer.dataset.swebench_multilingual import SwebenchMultilingualDataset
from codeminer.scip_interface import SCIPIndexer


def _tools_ready(language: str) -> bool:
    if language == "cpp":
        return bool(shutil.which("scip-clang")) and bool(shutil.which("cmake"))
    if language == "rust":
        return bool(shutil.which("rust-analyzer"))
    if language == "ts":
        return bool(shutil.which("scip-typescript"))
    return False


def _ensure_cpp_compdb(repo: Path) -> None:
    compdb = repo / "build" / "compile_commands.json"
    if compdb.exists():
        return
    subprocess.run(
        [
            "cmake",
            "-S",
            str(repo),
            "-B",
            str(repo / "build"),
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _repo_keywords(language: str) -> list[str]:
    if language == "cpp":
        return [
            "fmtlib/",
            "nlohmann/",
            "jqlang/",
            "redis/",
            "valkey-io/",
            "micropython/",
        ]
    if language == "rust":
        return [
            "astral-sh/ruff",
            "pola-rs/polars",
            "tokio-rs/",
            "rust-lang/",
        ]
    if language == "ts":
        return [
            "vuejs/",
            "mui/",
            "darkreader/",
            "sveltejs/",
            "axios/",
            "expressjs/",
            "insomnia/",
            "dayjs/",
        ]
    return []


def _pick_swebench_multilingual_instance(language: str) -> dict:
    dataset_obj = SwebenchMultilingualDataset(split="test", filter_instance=".*")
    rows = dataset_obj.load()
    keywords = _repo_keywords(language)
    matches = [row for row in rows if any(k in row["repo"] for k in keywords)]
    if not matches:
        raise RuntimeError(f"No SWE-bench_Multilingual instances for {language}")
    return dict(matches[0])


@pytest.mark.parametrize("language", ["cpp", "rust", "ts"])
def test_run_pipeline_swebench_multilingual_instance(
    tmp_path: Path,
    language: str,
) -> None:
    if not _tools_ready(language):
        pytest.skip(f"Required tooling for {language} is not available in PATH")

    try:
        instance = _pick_swebench_multilingual_instance(language)
    except Exception as exc:
        pytest.skip(
            f"SWE-bench_Multilingual unavailable or no matching instance: {exc}"
        )

    dataset_obj = SwebenchMultilingualDataset(split="test", filter_instance=".*")
    dataset_obj.process_instance(instance)
    repo_path = Path(dataset_obj.get_repo_path(instance))

    if language == "cpp":
        _ensure_cpp_compdb(repo_path)

    kwargs = {"infer_tsconfig": True} if language == "ts" else {}
    indexer = SCIPIndexer(
        project_root=repo_path,
        output_dir=tmp_path / f"scip_multi_{language}",
        language=language,
    )
    graph = indexer.run_pipeline(skip_level="graph", report_profile=False, **kwargs)

    assert graph is not None, f"run_pipeline returned None for {language} (dataset)"
    assert len(graph.graph.vs) > 0, f"no graph nodes for {language} (dataset)"
