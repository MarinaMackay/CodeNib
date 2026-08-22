#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Run a minimal multimodal knowledge smoke test on a synthetic repository."""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import tempfile
from pathlib import Path

from codenib.wiki import (
    OpenAICompatibleVisualFactExtractor,
    OpenAICompatibleVisualGraphPlanExtractor,
    build_multimodal_repository_knowledge,
    compile_visual_graph_plan_to_mermaid,
    compile_visual_storyboard_to_markdown,
    save_multimodal_knowledge_bundle,
)
from codenib.wiki.media_tools import MultimodalKnowledgeToolRouter


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
        "--preview-html",
        default=None,
        help="Optional path to write a self-contained HTML preview",
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
    parser.add_argument(
        "--visual-plan-model",
        default=None,
        help="Optional OpenAI-compatible VLM model for planning visual graphs",
    )
    parser.add_argument(
        "--visual-plan-api-base",
        default=None,
        help="OpenAI-compatible API base URL for --visual-plan-model",
    )
    parser.add_argument(
        "--visual-plan-api-key-env",
        default="CODENIB_WIKI_VISUAL_GRAPH_API_KEY",
        help="Environment variable that contains the visual-plan API key",
    )
    parser.add_argument(
        "--visual-plan-provider",
        default="openai-compatible",
        help="Provider label recorded in VLM-planned graph metadata",
    )
    parser.add_argument(
        "--visual-plan-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for each visual graph planning request",
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
    graph_planner = _build_visual_graph_planner(args, repo)
    bundle = build_multimodal_repository_knowledge(
        repo,
        commit="smoke-test",
        extractor=extractor,
        graph_planner=graph_planner,
    )
    save_multimodal_knowledge_bundle(bundle, args.output)
    if args.preview_html:
        _write_preview_html(bundle, repo, Path(args.preview_html))
    facts = bundle["visual_facts_manifest"]["facts"]
    counts = {
        "bundle": str(Path(args.output).expanduser().resolve()),
        "repo": str(repo),
        "extractor": facts[0]["extractor"] if facts else "",
        "media_artifacts": bundle["media_manifest"]["artifact_count"],
        "visual_fact_packs": bundle["visual_facts_manifest"]["fact_count"],
        "visual_code_bindings": bundle["grounding_manifest"]["binding_count"],
        "visual_graph_plans": bundle["visual_graph_manifest"]["plan_count"],
        "visual_storyboards": bundle["visual_storyboard_manifest"]["storyboard_count"],
        "knowledge_entries": bundle["knowledge_view"]["entry_count"],
    }
    print(json.dumps(counts, sort_keys=True))
    return 0


def _write_preview_html(bundle: dict, repo: Path, output: Path) -> None:
    view = bundle["knowledge_view"]
    router = MultimodalKnowledgeToolRouter(view)
    entry = view["entries"][0] if view.get("entries") else {}
    artifact = entry.get("artifact") or {}
    fact = entry.get("facts") or {}
    bindings = entry.get("bindings") or []
    graph_plans = (bundle.get("visual_graph_manifest") or {}).get("plans") or []
    graph_plan = graph_plans[0] if graph_plans else {}
    mermaid = compile_visual_graph_plan_to_mermaid(graph_plan) if graph_plan else ""
    storyboards = (bundle.get("visual_storyboard_manifest") or {}).get(
        "storyboards"
    ) or []
    storyboard = storyboards[0] if storyboards else {}
    storyboard_markdown = (
        compile_visual_storyboard_to_markdown(storyboard) if storyboard else ""
    )
    image_uri = _image_data_uri(repo / artifact.get("path", ""))
    explore = router.call_tool(
        "explore_visual_context",
        {
            "query": "WikiRenderer architecture VectorStore",
            "artifact_path": artifact.get("path", ""),
            "source_path": "src/compiler.py",
            "symbol": "IndexCompiler",
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _preview_html(
            image_uri=image_uri,
            artifact=artifact,
            fact=fact,
            bindings=bindings,
            mermaid=mermaid,
            storyboard_markdown=storyboard_markdown,
            explore=explore,
        ),
        encoding="utf-8",
    )


def _preview_html(
    *,
    image_uri: str,
    artifact: dict,
    fact: dict,
    bindings: list,
    mermaid: str,
    storyboard_markdown: str,
    explore: dict,
) -> str:
    entities = fact.get("entities") or []
    claims = fact.get("claims") or []
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CodeNib multimodal smoke preview</title>
  <style>
    body {{ margin: 0; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0f172a; color: #e5e7eb; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    h2 {{ margin: 0 0 12px; font-size: 17px; color: #bfdbfe; }}
    .sub {{ color: #94a3b8; margin-bottom: 24px; }}
    .grid {{ display: grid; grid-template-columns: 1.1fr 1fr; gap: 16px; }}
    .card {{ background: #111827; border: 1px solid #334155; border-radius: 14px; padding: 16px; box-shadow: 0 20px 50px rgba(0,0,0,.25); }}
    img {{ max-width: 100%; background: white; border-radius: 10px; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    pre {{ overflow: auto; background: #020617; border-radius: 10px; padding: 12px; color: #d1fae5; }}
    .pill {{ display: inline-block; margin: 0 8px 8px 0; padding: 6px 10px; border-radius: 999px; background: #1e3a8a; color: #dbeafe; }}
    .binding {{ padding: 8px 0; border-bottom: 1px solid #1f2937; }}
    .muted {{ color: #94a3b8; }}
  </style>
</head>
<body>
<main>
  <h1>CodeNib Multimodal CodeWiki Smoke Preview</h1>
  <div class="sub">Real local chain: SVG artifact → visual facts → source grounding → multimodal tool response.</div>
  <section class="grid">
    <div class="card">
      <h2>Repository visual artifact</h2>
      <img alt="{_h(artifact.get("caption", "architecture"))}" src="{image_uri}">
      <p><code>{_h(artifact.get("path", ""))}</code></p>
      <p class="muted">{_h(artifact.get("embedded_text", ""))}</p>
    </div>
    <div class="card">
      <h2>Extracted visual entities</h2>
      {''.join(f'<span class="pill">{_h(entity.get("name", ""))}</span>' for entity in entities)}
      <h2>Source-grounded bindings</h2>
      {''.join(_binding_html(binding) for binding in bindings)}
    </div>
    <div class="card">
      <h2>Validated graph plan as Mermaid</h2>
      <pre>{_h(mermaid)}</pre>
    </div>
    <div class="card">
      <h2>Video-ready storyboard</h2>
      <pre>{_h(storyboard_markdown)}</pre>
    </div>
    <div class="card">
      <h2>Claims and explore_visual_context output</h2>
      <ul>{''.join(f'<li>{_h(claim.get("text", ""))}</li>' for claim in claims)}</ul>
      <pre>{_h(json.dumps(explore, ensure_ascii=False, indent=2))}</pre>
    </div>
  </section>
</main>
</body>
</html>
"""


def _binding_html(binding: dict) -> str:
    return (
        '<div class="binding">'
        f'<strong>{_h(binding.get("entity_name", ""))}</strong>'
        f' → <code>{_h(binding.get("source_path", ""))}</code>'
        f' :: <code>{_h(binding.get("symbol", ""))}</code>'
        f' <span class="muted">line {binding.get("line", 0)}</span>'
        "</div>"
    )


def _image_data_uri(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(data).decode("ascii")


def _h(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


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


def _build_visual_graph_planner(args: argparse.Namespace, repo: Path):
    model = str(args.visual_plan_model or "").strip()
    api_base = str(args.visual_plan_api_base or "").strip()
    if not model and not api_base:
        return None
    planner = OpenAICompatibleVisualGraphPlanExtractor(
        model=model,
        api_base=api_base,
        api_key=os.environ.get(str(args.visual_plan_api_key_env or "")),
        timeout=args.visual_plan_timeout,
        provider=args.visual_plan_provider,
    )
    return lambda entry: planner.plan(entry, repo_path=repo)


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
