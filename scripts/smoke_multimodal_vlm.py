#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Run a minimal multimodal knowledge smoke test on a synthetic repository."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from codenib.wiki import (
    OpenAICompatibleVisualFactExtractor,
    build_multimodal_repository_knowledge,
    save_multimodal_knowledge_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to write the smoke-test multimodal knowledge bundle JSON",
    )
    parser.add_argument(
        "--keep-repo",
        default=None,
        help="Optional path where the synthetic repository should be kept",
    )
    parser.add_argument(
        "--visual-facts-model",
        default=None,
        help="Optional OpenAI-compatible VLM model for extracting visual facts",
    )
    parser.add_argument(
        "--visual-facts-api-base",
        default=None,
        help="OpenAI-compatible API base URL for --visual-facts-model",
    )
    parser.add_argument(
        "--visual-facts-api-key-env",
        default="CODENIB_WIKI_VISUAL_FACTS_API_KEY",
        help="Environment variable that contains the visual-facts API key",
    )
    parser.add_argument(
        "--visual-facts-provider",
        default="openai-compatible",
        help="Provider label recorded in extracted visual facts",
    )
    parser.add_argument(
        "--visual-facts-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for each visual-fact VLM request",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.keep_repo:
        repo = Path(args.keep_repo).expanduser().resolve()
        repo.mkdir(parents=True, exist_ok=True)
        _write_synthetic_repo(repo)
        return _run_smoke(args, repo)

    with tempfile.TemporaryDirectory(prefix="codenib-mmwiki-smoke-") as temp_dir:
        repo = Path(temp_dir) / "repo"
        repo.mkdir()
        _write_synthetic_repo(repo)
        return _run_smoke(args, repo)


def _run_smoke(args: argparse.Namespace, repo: Path) -> int:
    extractor = _build_visual_fact_extractor(args, repo)
    bundle = build_multimodal_repository_knowledge(
        repo,
        commit="smoke-test",
        extractor=extractor,
    )
    save_multimodal_knowledge_bundle(bundle, args.output)
    facts = bundle["visual_facts_manifest"]["facts"]
    counts = {
        "bundle": str(Path(args.output).expanduser().resolve()),
        "repo": str(repo),
        "extractor": facts[0]["extractor"] if facts else "",
        "media_artifacts": bundle["media_manifest"]["artifact_count"],
        "visual_fact_packs": bundle["visual_facts_manifest"]["fact_count"],
        "visual_code_bindings": bundle["grounding_manifest"]["binding_count"],
        "knowledge_entries": bundle["knowledge_view"]["entry_count"],
    }
    print(json.dumps(counts, sort_keys=True))
    return 0


def _build_visual_fact_extractor(args: argparse.Namespace, repo: Path):
    model = str(args.visual_facts_model or "").strip()
    api_base = str(args.visual_facts_api_base or "").strip()
    if not model and not api_base:
        return None
    extractor = OpenAICompatibleVisualFactExtractor(
        model=model,
        api_base=api_base,
        api_key=os.environ.get(str(args.visual_facts_api_key_env or "")),
        timeout=args.visual_facts_timeout,
        provider=args.visual_facts_provider,
    )
    return lambda artifact: extractor.extract(artifact, repo_path=repo)


def _write_synthetic_repo(repo: Path) -> None:
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "src").mkdir(exist_ok=True)
    (repo / "README.md").write_text(
        "\n".join(
            [
                "# CodeNib multimodal smoke repo",
                "",
                "The architecture diagram shows how WikiRenderer calls "
                "IndexCompiler before writing to VectorStore.",
                "",
                "![WikiRenderer architecture](docs/architecture.svg)",
            ]
        ),
        encoding="utf-8",
    )
    (repo / "docs" / "architecture.svg").write_text(
        "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="220">',
                "<title>WikiRenderer architecture</title>",
                "<desc>WikiRenderer calls IndexCompiler and VectorStore</desc>",
                '<text x="40" y="70">WikiRenderer</text>',
                '<text x="260" y="70">IndexCompiler</text>',
                '<text x="470" y="70">VectorStore</text>',
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )
    (repo / "src" / "compiler.py").write_text(
        "\n".join(
            [
                "class WikiRenderer:",
                "    def render(self, compiler, store):",
                "        return compiler.compile(store)",
                "",
                "class IndexCompiler:",
                "    def compile(self, store):",
                "        return store.write()",
                "",
                "class VectorStore:",
                "    def write(self):",
                "        return True",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
