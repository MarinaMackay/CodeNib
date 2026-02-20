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


def _has_node(ig, name_sub, node_type=None):
    """Check if any node name or display_name contains *name_sub* (optionally filtered by type)."""
    for v in ig.vs:
        matched = name_sub in v["name"]
        if not matched and "display_name" in ig.vs.attributes():
            dn = v["display_name"] or ""
            matched = name_sub in dn
        if matched:
            if node_type is None or v["type"] == node_type:
                return True
    return False


def _has_edge(ig, src_sub, tgt_sub, edge_type=None):
    """Check if an edge exists where source/target names or display_names contain the substrings."""
    has_dn = "display_name" in ig.vs.attributes()
    for e in ig.es:
        sv = ig.vs[e.source]
        tv = ig.vs[e.target]
        s_match = src_sub in sv["name"] or (has_dn and src_sub in (sv["display_name"] or ""))
        t_match = tgt_sub in tv["name"] or (has_dn and tgt_sub in (tv["display_name"] or ""))
        if s_match and t_match:
            if edge_type is None or e["type"] == edge_type:
                return True
    return False


# Expected nodes / edges for the first SWE-bench instance of each language.
# cpp: matched by display_name (clangd uses hash-based node names)
# rust/ts: matched by node name (SCIP symbol format)
_EXPECTED = {
    # fmtlib/fmt (fmtlib__fmt-1683) — clangd indexer
    "cpp": {
        "nodes": [
            ("include/fmt/core.h", "file"),
            ("fmt::print", "function"),
            ("fmt::to_string", "function"),
            ("fmt::basic_string_view", "class"),
            ("fmt::buffered_file", "class"),
            ("fmt::file", "class"),
            ("fmt::buffered_file::get", "method"),
            ("fmt::file::write", "method"),
        ],
        "edges": [
            ("include/fmt/core.h", "fmt::to_string_view", "contain"),
            ("fmt::detail::make_arg", "fmt::basic_format_arg", "reference"),
        ],
    },
    # astral-sh/ruff (astral-sh__ruff-15309)
    "rust": {
        "nodes": [
            ("crates/ruff_linter/src/linter.rs", "file"),
            ("linter/check_path", "function"),
            ("directives/extract_directives", "function"),
            ("resolver/resolve_import", "function"),
            ("settings/types/PythonVersion", "class"),
            ("line_width/IndentWidth", "class"),
            ("source_kind/SourceKind", "class"),
            ("source_kind/impl#[SourceKind]source_code", "method"),
            ("locator/impl#[`Locator", "method"),
        ],
        "edges": [
            ("crates/ruff_linter/src/linter.rs", "linter/check_path", "contain"),
            ("linter/check_path", "settings/LinterSettings", "reference"),
            ("linter/check_path", "source_kind/SourceKind", "reference"),
        ],
    },
    # axios/axios (axios__axios-4731)
    "ts": {
        "nodes": [
            ("index.d.ts", "file"),
            ("Axios.request()", "method"),
            ("Axios.get()", "method"),
            ("Axios.post()", "method"),
            ("AxiosError", "class"),
            ("AxiosHeaders", "class"),
            ("AxiosRequestConfig", "class"),
            ("AxiosResponse", "class"),
            ("CancelToken", "class"),
        ],
        "edges": [
            ("index.d.ts", "Axios.request()", "contain"),
            ("test/typescript/axios.ts", "Axios.get()", "reference"),
        ],
    },
}


def _tools_ready(language: str) -> bool:
    if language == "cpp":
        return bool(shutil.which("clangd")) and bool(shutil.which("cmake"))
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
    ig = graph.graph
    assert len(ig.vs) > 0, f"no graph nodes for {language} (dataset)"

    # Validate expected nodes and edges
    expected = _EXPECTED.get(language)
    if expected:
        for name_sub, node_type in expected["nodes"]:
            assert _has_node(ig, name_sub, node_type), (
                f"[{language}] missing node: '{name_sub}' (type={node_type})"
            )
        for src_sub, tgt_sub, edge_type in expected["edges"]:
            assert _has_edge(ig, src_sub, tgt_sub, edge_type), (
                f"[{language}] missing edge: '{src_sub}' -> '{tgt_sub}' ({edge_type})"
            )
