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
from .media_index_grounding import (
    IndexBackedVisualGroundingScorer,
    build_index_backed_visual_grounding_scorer,
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
from .media_storyboard import (
    VISUAL_STORYBOARD_MANIFEST_SCHEMA,
    VISUAL_STORYBOARD_MANIFEST_VERSION,
    VISUAL_STORYBOARD_SCHEMA,
    VISUAL_STORYBOARD_VERSION,
    build_visual_storyboard,
    build_visual_storyboard_manifest,
    compile_visual_storyboard_to_markdown,
    load_visual_storyboard_manifest,
    save_visual_storyboard_manifest,
    validate_visual_storyboard,
    validate_visual_storyboard_manifest,
)
from .media_tools import MultimodalKnowledgeToolRouter, multimodal_tool_schemas
from .media_vector import (
    VisualDocumentEmbedder,
    VisualEmbeddingDocument,
    VisualTextEmbedder,
    build_visual_vector_index,
    deterministic_visual_text_embeddings,
    load_visual_vector_index,
    save_visual_vector_index,
    search_visual_vector_index,
    validate_visual_vector_index,
)
from .media_vlm import (
    OpenAICompatibleVisualFactExtractor,
    visual_fact_extractor_from_config,
)

__all__ = [
    "MULTIMODAL_KNOWLEDGE_BUNDLE_SCHEMA",
    "MULTIMODAL_KNOWLEDGE_BUNDLE_VERSION",
    "MultimodalKnowledgeToolRouter",
    "IndexBackedVisualGroundingScorer",
    "VISUAL_GRAPH_MANIFEST_SCHEMA",
    "VISUAL_GRAPH_MANIFEST_VERSION",
    "VISUAL_GRAPH_PLAN_SCHEMA",
    "VISUAL_GRAPH_PLAN_VERSION",
    "VISUAL_STORYBOARD_MANIFEST_SCHEMA",
    "VISUAL_STORYBOARD_MANIFEST_VERSION",
    "VISUAL_STORYBOARD_SCHEMA",
    "VISUAL_STORYBOARD_VERSION",
    "WikiBuilder",
    "OpenAICompatibleVisualFactExtractor",
    "VisualDocumentEmbedder",
    "VisualEmbeddingDocument",
    "VisualGroundingScorer",
    "VisualTextEmbedder",
    "build_media_evidence_pack",
    "build_index_backed_visual_grounding_scorer",
    "build_multimodal_knowledge_view",
    "build_multimodal_knowledge_bundle",
    "build_multimodal_repository_knowledge",
    "build_visual_vector_index",
    "build_visual_facts_manifest",
    "build_visual_graph_manifest",
    "build_visual_graph_plan",
    "build_visual_storyboard",
    "build_visual_storyboard_manifest",
    "compile_visual_graph_plan_to_mermaid",
    "compile_visual_storyboard_to_markdown",
    "deterministic_visual_text_embeddings",
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
    "load_visual_storyboard_manifest",
    "save_visual_graph_manifest",
    "save_visual_storyboard_manifest",
    "load_visual_vector_index",
    "save_multimodal_knowledge_bundle",
    "save_visual_vector_index",
    "search_visual_context",
    "search_visual_vector_index",
    "validate_multimodal_knowledge_bundle",
    "validate_visual_graph_manifest",
    "validate_visual_graph_plan",
    "validate_visual_storyboard",
    "validate_visual_storyboard_manifest",
    "validate_visual_vector_index",
    "visual_fact_extractor_from_config",
]
