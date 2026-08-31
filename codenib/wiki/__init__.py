# SPDX-FileCopyrightText: 2025-2026 CodeNib Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Index-derived wiki generation for the DeepWiki-style demo.

Produces a per-repo page tree and source-grounded page content from the
already-loaded BM25 / vector indexes — no LLM required. When an LLM is
configured it can later refine the prose, but every code anchor here resolves
to a real symbol span pulled from the indexes (no fabricated lines).
"""

from .builder import WikiBuilder
from .media_archify import (
    ARCHIFY_DIAGRAM_TYPE,
    ARCHIFY_SCHEMA_VERSION,
    compile_visual_graph_plan_to_archify,
    save_archify_architecture,
)
from .media_artifacts import discover_media_manifest
from .media_eval import (
    evaluate_mmwiki_predictions,
    evaluate_visual_code_grounding,
    evaluate_visual_fact_extraction,
)
from .media_evidence import build_media_evidence_pack
from .media_facts import build_visual_facts_manifest, deterministic_visual_facts
from .media_graph_plan import (
    VISUAL_GRAPH_MANIFEST_SCHEMA,
    VISUAL_GRAPH_MANIFEST_VERSION,
    VISUAL_GRAPH_PLAN_SCHEMA,
    VISUAL_GRAPH_PLAN_VERSION,
    build_visual_graph_manifest,
    build_visual_graph_plan,
    compile_visual_graph_plan_to_mermaid,
    load_visual_graph_manifest,
    save_visual_graph_manifest,
    validate_visual_graph_manifest,
    validate_visual_graph_plan,
)
from .media_grounding import (
    VisualGroundingScorer,
    discover_source_symbol_candidates,
    ground_visual_facts_to_sources,
)
from .media_incremental import (
    diff_media_manifests,
    merge_incremental_visual_facts,
    plan_incremental_visual_fact_update,
)
from .media_knowledge import (
    build_multimodal_knowledge_view,
    find_visual_code_links,
    get_visual_evidence,
    search_visual_context,
)
from .media_pipeline import build_multimodal_repository_knowledge
from .media_storage import (
    MULTIMODAL_KNOWLEDGE_BUNDLE_SCHEMA,
    MULTIMODAL_KNOWLEDGE_BUNDLE_VERSION,
    build_multimodal_knowledge_bundle,
    load_multimodal_knowledge_bundle,
    save_multimodal_knowledge_bundle,
    validate_multimodal_knowledge_bundle,
)
from .media_tools import MultimodalKnowledgeToolRouter, multimodal_tool_schemas
from .media_vlm import (
    OpenAICompatibleVisualFactExtractor,
    visual_fact_extractor_from_config,
)

__all__ = [
    "ARCHIFY_DIAGRAM_TYPE",
    "ARCHIFY_SCHEMA_VERSION",
    "MULTIMODAL_KNOWLEDGE_BUNDLE_SCHEMA",
    "MULTIMODAL_KNOWLEDGE_BUNDLE_VERSION",
    "MultimodalKnowledgeToolRouter",
    "VISUAL_GRAPH_MANIFEST_SCHEMA",
    "VISUAL_GRAPH_MANIFEST_VERSION",
    "VISUAL_GRAPH_PLAN_SCHEMA",
    "VISUAL_GRAPH_PLAN_VERSION",
    "WikiBuilder",
    "OpenAICompatibleVisualFactExtractor",
    "VisualGroundingScorer",
    "build_media_evidence_pack",
    "build_multimodal_knowledge_view",
    "build_multimodal_knowledge_bundle",
    "build_multimodal_repository_knowledge",
    "build_visual_facts_manifest",
    "build_visual_graph_manifest",
    "build_visual_graph_plan",
    "compile_visual_graph_plan_to_mermaid",
    "compile_visual_graph_plan_to_archify",
    "deterministic_visual_facts",
    "diff_media_manifests",
    "discover_media_manifest",
    "discover_source_symbol_candidates",
    "evaluate_mmwiki_predictions",
    "evaluate_visual_code_grounding",
    "evaluate_visual_fact_extraction",
    "find_visual_code_links",
    "get_visual_evidence",
    "ground_visual_facts_to_sources",
    "merge_incremental_visual_facts",
    "multimodal_tool_schemas",
    "plan_incremental_visual_fact_update",
    "load_multimodal_knowledge_bundle",
    "load_visual_graph_manifest",
    "save_multimodal_knowledge_bundle",
    "save_archify_architecture",
    "save_visual_graph_manifest",
    "search_visual_context",
    "validate_multimodal_knowledge_bundle",
    "validate_visual_graph_manifest",
    "validate_visual_graph_plan",
    "visual_fact_extractor_from_config",
]
