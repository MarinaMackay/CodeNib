#!/usr/bin/env python3
"""
End-to-end example of the skill agent pipeline.

Demonstrates **three execution paths**:

1. **Standalone (baseline)** — the agent calls a skill executor directly.
   No compiler, no DAG, no scheduling.

2. **Compiled** — multiple skill invocations are compiled into a DAG,
   scheduled for parallel execution, and run through the kernel system.

3. **Two-phase** — the recommended architecture:
   - Phase 1 (index compilation): ``IndexCompiler`` builds all indexes
     and writes ``repo_manifest.json``.
   - Phase 2 (query compilation): ``SkillCompiler`` reads the manifest
     to resolve index freshness, compiles the skill graph, and executes.

Usage:
    # Standalone BM25 search:
    python examples/skill_agent.py --repo . \
        --query "how does the skill registry work" --path standalone

    # Compiler-optimized sparse search:
    python examples/skill_agent.py --repo . \
        --query "how does the skill registry work" --path compiled --mode sparse

    # Two-phase (index compile → query compile):
    python examples/skill_agent.py --repo . \
        --query "how does the skill registry work" --path two-phase --mode sparse
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when running as a script
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from codeminer.agent.skills.loader import SkillLoader
from codeminer.agent.skills.registry import SkillRegistry
from codeminer.code_chunker import CodeChunker, RepoChunkingConfig
from codeminer.compiler.execution import ExecutionEngine
from codeminer.compiler.params import SessionContext
from codeminer.compiler.resources import (
    IndexState,
    IndexStatus,
    InMemoryIndexStateStore,
    ResourceResolver,
)
from codeminer.compiler.skills.compiler import SkillCompiler
from codeminer.compiler.skills.scheduler import SkillScheduler
from codeminer.index.sparse_idx.bm25_index import BM25CodeIndexer
from codeminer.ops.rerank import RerankContext, register_rerank_ops
from codeminer.ops.retrieve import RetrieveContext, register_retrieve_ops
from codeminer.types import QueriedNode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Skill agent demo (standalone + compiled + two-phase)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo", type=str, default=".")
    parser.add_argument(
        "--query",
        type=str,
        default="how does the skill registry work",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="standalone",
        choices=["standalone", "compiled", "two-phase"],
        help="Execution path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sparse",
        choices=["sparse", "dense", "hybrid"],
        help="Retrieval mode (used with --path compiled / two-phase)",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--languages", type=str, nargs="+", default=["python"])
    parser.add_argument("--index-path", type=str, default=None)
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nomic-ai/CodeRankEmbed",
    )
    parser.add_argument("--embedding-dimension", type=int, default=768)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Index builders (used by standalone and compiled paths)
# ---------------------------------------------------------------------------


def build_bm25_index(
    repo_path: str, languages: List[str], max_k: int = 128
) -> BM25CodeIndexer:
    """Chunk the repo and build a BM25 index."""
    primary = languages[0] if languages else "python"
    chunker = CodeChunker(
        language=primary,
        repo_config=RepoChunkingConfig(languages=languages),
        max_lines_per_chunk=300,
    )
    chunks = chunker.chunk_repository(repo_path=repo_path)
    if not chunks:
        raise RuntimeError(f"No code chunks generated from {repo_path}")
    print(f"  BM25: indexed {len(chunks)} chunks")
    return BM25CodeIndexer(chunks=chunks, max_k=max_k)


def build_vector_store(
    repo_path: str,
    index_path: str,
    languages: List[str],
    embedding_model: str,
    embedding_dimension: int,
):
    """Build (or load) hierarchical embedding index."""
    from codeminer.index.embedding import (
        CodeVectorStore,
        build_hierarchical_vector_store,
    )

    store_path = Path(index_path)
    store_path.mkdir(parents=True, exist_ok=True)

    l0 = store_path / "l0"
    l2 = store_path / "l2"
    if l0.exists() and l2.exists():
        print("  Embedding: loading cached index")
        vs = CodeVectorStore(
            embedding_model=embedding_model,
            embedding_provider="huggingface",
            dimension=embedding_dimension,
            store_path=str(store_path),
        )
        vs.load(str(store_path))
        if vs.l0_documents and vs.l2_documents:
            return vs
        print("  Embedding: cache incomplete, rebuilding")

    print("  Embedding: building hierarchical index (may take a minute)...")
    vs = build_hierarchical_vector_store(
        repo_path=repo_path,
        index_path=str(store_path),
        plan_name=None,
        languages=languages,
        max_lines_per_chunk=300,
        build_levels=["l0", "l2"],
        embedding_model=embedding_model,
        embedding_provider="huggingface",
        embedding_dimension=embedding_dimension,
        embedding_kwargs={"model_kwargs": {"trust_remote_code": True}},
        index_metric="ip",
    )
    return vs


def count_files(repo_path: str) -> int:
    """Count source files for SessionContext.repo_size."""
    total = 0
    for _, _, files in os.walk(repo_path):
        total += len(files)
    return total


# ---------------------------------------------------------------------------
# Skill invocation builders (compiler path)
# ---------------------------------------------------------------------------


def build_invocations(mode: str, query: str, top_k: int) -> List[Dict[str, Any]]:
    """Build the list of skill invocations for the compiler."""
    if mode == "sparse":
        return [
            {
                "skill_id": "bm25_search",
                "inputs": {
                    "query": query,
                    "top_k": top_k,
                    "return_content": True,
                },
            },
        ]

    if mode == "dense":
        return [
            {
                "skill_id": "embedding_search",
                "inputs": {
                    "query": query,
                    "top_k": top_k,
                    "level": "l2",
                    "return_content": True,
                },
            },
        ]

    # hybrid: bm25 + embedding -> fusion -> rerank
    return [
        {
            "skill_id": "bm25_search",
            "inputs": {
                "query": query,
                "top_k": top_k * 5,
                "return_content": True,
            },
        },
        {
            "skill_id": "embedding_search",
            "inputs": {
                "query": query,
                "top_k": top_k * 5,
                "level": "l2",
                "return_content": True,
            },
        },
        {
            "skill_id": "hybrid_search",
            "inputs": {
                "query": query,
                "top_k": top_k * 3,
                "weights": [0.4, 0.6],
            },
            "depends_on": ["bm25_search", "embedding_search"],
        },
        {
            "skill_id": "embedding_rerank",
            "inputs": {"query": query, "top_k": top_k},
            "depends_on": ["hybrid_search"],
        },
    ]


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def print_compilation(result) -> None:
    """Print the compilation plan."""
    plan = result.execution_plan
    dag = result.skill_dag

    print("\n=== Compilation Result ===")
    if result.warnings:
        print("Warnings:")
        for w in result.warnings:
            print(f"  - {w}")

    if result.resource_plan:
        rp = result.resource_plan
        print(f"\nResource Plan (can_execute={rp.can_execute}):")
        for d in rp.decisions:
            tag = f"[{d.state.value}]"
            print(f"  {d.index_type}: {tag} action={d.action}")
            if d.warning:
                print(f"    -> {d.warning}")

    print(f"\nSkill DAG ({len(dag.nodes)} nodes):")
    for node in dag.iter_topo():
        deps = f" <- [{', '.join(node.dependencies)}]" if node.dependencies else ""
        print(f"  {node.skill_id}{deps}")

    print(f"\nExecution Plan ({len(plan.stages)} stages):")
    for stage in plan.stages:
        par = f"(parallelism={stage.parallelism})" if stage.parallelism > 1 else ""
        print(f"  Stage {stage.stage_id}: {stage.skill_ids} {par}")

    if plan.policy_metadata:
        print(f"  Policy: {plan.policy_metadata.get('policy', '?')}")


def print_manifest(manifest) -> None:
    """Print the repo manifest summary."""
    print(f"\n=== Repo Manifest ===")
    print(f"  repo     : {manifest.repo_path}")
    print(f"  commit   : {manifest.commit[:12] if manifest.commit else '(none)'}...")
    print(f"  languages: {manifest.languages}")
    print(f"  files    : {manifest.file_count}")
    print(f"\nIndexes:")
    for idx_type, entry in manifest.indexes.items():
        status_tag = f"[{entry.status}]"
        print(f"  {idx_type}: {status_tag} path={entry.path}")
        if entry.metadata.get("build_duration_seconds"):
            print(f"    built in {entry.metadata['build_duration_seconds']:.1f}s")
        if entry.metadata.get("file_count"):
            print(f"    {entry.metadata['file_count']} chunks")
        if entry.metadata.get("document_count"):
            print(f"    documents: {entry.metadata['document_count']}")
    print(f"\nCapabilities: {manifest.capabilities}")


def print_results(results: List[Any], top_k: int) -> None:
    """Print search results."""
    nodes = [n for n in results if isinstance(n, QueriedNode)][:top_k]

    print(f"\n=== Search Results ({len(nodes)} hits) ===")
    for i, node in enumerate(nodes, 1):
        loc = node.file or "?"
        if node.start_line is not None:
            loc += f":{node.start_line}"
            if node.end_line is not None and node.end_line != node.start_line:
                loc += f"-{node.end_line}"
        score = f"{node.score:.4f}" if node.score is not None else "n/a"
        print(f"\n  [{i}] {node.node_name or '(unnamed)'}  (score={score})")
        print(f"      {loc}  type={node.type or '?'}")
        if node.content:
            snippet = node.content[:200]
            if len(node.content) > 200:
                snippet += "..."
            for line in snippet.splitlines()[:5]:
                print(f"      | {line}")
            if len(node.content.splitlines()) > 5:
                print(f"      | ... ({len(node.content.splitlines())} lines total)")


# ---------------------------------------------------------------------------
# Path 1: Standalone execution (baseline)
# ---------------------------------------------------------------------------


async def run_standalone(args: argparse.Namespace) -> None:
    """
    The agent calls a skill executor directly — no compiler, no DAG.

    This is the baseline path: the agent picks the right skill by reading
    its skill.md, resolves parameters from config defaults + query inputs,
    and invokes the executor function.
    """
    repo_path = os.path.abspath(args.repo)

    print(f"Repository : {repo_path}")
    print(f"Query      : {args.query}")
    print(f"Path       : standalone (agent calls executor directly)")
    print(f"Top-k      : {args.top_k}")

    # Step 1: Build BM25 index
    print("\n--- Building index ---")
    bm25_index = build_bm25_index(repo_path, args.languages)

    # Step 2: Create context and load skills
    print("\n--- Loading skills ---")
    retrieve_ctx = RetrieveContext(
        bm25=bm25_index,
        default_top_k=args.top_k,
        default_level="l2",
    )
    skills_dir = os.path.join(_PROJECT_ROOT, "codeminer", "agent", "skills")
    loader = SkillLoader()
    loaded = loader.load_all(skills_dir, contexts={"retrieve": retrieve_ctx})
    print(f"  Loaded {len(loaded)} skills: {[s.skill_id for s in loaded]}")

    # Step 3: Agent reads skill.md to decide which skill to use
    registry = SkillRegistry()
    bm25_meta = registry.get("bm25_search")
    if bm25_meta is None:
        print("Error: bm25_search skill not loaded", file=sys.stderr)
        sys.exit(1)

    print("\n--- Agent reads skill.md ---")
    if bm25_meta.skill_doc:
        for line in bm25_meta.skill_doc.strip().splitlines()[:8]:
            print(f"  {line}")

    # Step 4: Agent calls executor directly with query-time params
    print("\n--- Executing bm25_search (standalone) ---")
    executor = bm25_meta.executor_fn
    if executor is None:
        print("Error: no executor loaded for bm25_search", file=sys.stderr)
        sys.exit(1)

    results = executor(query=args.query, top_k=args.top_k)
    print(f"  Got {len(results)} results")

    print_results(results, args.top_k)


# ---------------------------------------------------------------------------
# Path 2: Compiler-optimized execution
# ---------------------------------------------------------------------------


async def run_compiled(args: argparse.Namespace) -> None:
    """
    Multiple skills compiled into a DAG, scheduled, and executed via kernels.

    Uses layered parameter resolution (config defaults -> session -> query)
    and resource resolution (index freshness checks).
    """
    repo_path = os.path.abspath(args.repo)
    index_path = args.index_path or os.path.join(repo_path, ".codeminer_cache")

    print(f"Repository : {repo_path}")
    print(f"Query      : {args.query}")
    print(f"Path       : compiled (compiler-optimized DAG)")
    print(f"Mode       : {args.mode}")
    print(f"Top-k      : {args.top_k}")

    # ------------------------------------------------------------------
    # Step 1: Build indexes
    # ------------------------------------------------------------------
    print("\n--- Building indexes ---")
    needs_sparse = args.mode in ("sparse", "hybrid")
    needs_dense = args.mode in ("dense", "hybrid")

    bm25_index = None
    vector_store = None

    if needs_sparse:
        bm25_index = build_bm25_index(repo_path, args.languages)

    if needs_dense:
        vector_store = build_vector_store(
            repo_path,
            index_path,
            args.languages,
            args.embedding_model,
            args.embedding_dimension,
        )

    # ------------------------------------------------------------------
    # Step 2: Wire execution engine with kernels
    # ------------------------------------------------------------------
    print("\n--- Registering kernels ---")
    engine = ExecutionEngine()

    retrieve_ctx = RetrieveContext(
        bm25=bm25_index,
        vector_store=vector_store,
        default_top_k=args.top_k,
        default_level="l2",
    )
    register_retrieve_ops(engine, retrieve_ctx)
    print(f"  Registered retrieve ops: {list(engine._kernels.keys())}")

    rerank_ctx = None
    if needs_dense:
        rerank_ctx = RerankContext(embedding_store=vector_store)
        register_rerank_ops(engine, rerank_ctx)
        print(f"  Registered rerank ops (total): {list(engine._kernels.keys())}")

    # ------------------------------------------------------------------
    # Step 3: Load skill packages
    # ------------------------------------------------------------------
    print("\n--- Loading skills ---")
    skills_dir = os.path.join(_PROJECT_ROOT, "codeminer", "agent", "skills")
    contexts: Dict[str, Any] = {"retrieve": retrieve_ctx}
    if rerank_ctx:
        contexts["rerank"] = rerank_ctx

    loader = SkillLoader()
    loaded = loader.load_all(skills_dir, contexts=contexts)
    print(f"  Loaded {len(loaded)} skills: {[s.skill_id for s in loaded]}")

    # ------------------------------------------------------------------
    # Step 4: Set up session context (Layer 3 params)
    # ------------------------------------------------------------------
    session_ctx = SessionContext(
        repo_path=repo_path,
        repo_size=count_files(repo_path),
        primary_language=args.languages[0] if args.languages else "python",
    )
    print("\n--- Session context ---")
    print(f"  repo_size={session_ctx.repo_size} files")
    print(f"  primary_language={session_ctx.primary_language}")

    # ------------------------------------------------------------------
    # Step 5: Set up resource resolver (index freshness)
    # ------------------------------------------------------------------
    state_store = InMemoryIndexStateStore()
    now = time.time()
    if bm25_index:
        state_store.set_status(
            IndexStatus(
                index_type="bm25",
                state=IndexState.FRESH,
                last_built=now,
                age_seconds=0,
                scope="current_repo",
            )
        )
    if vector_store:
        state_store.set_status(
            IndexStatus(
                index_type="vector",
                state=IndexState.FRESH,
                last_built=now,
                age_seconds=0,
                scope="current_repo",
            )
        )
    resource_resolver = ResourceResolver(state_store)

    # ------------------------------------------------------------------
    # Step 6: Compile skill invocations
    # ------------------------------------------------------------------
    print("\n--- Compiling skill graph ---")
    invocations = build_invocations(args.mode, args.query, args.top_k)

    registry = SkillRegistry()
    compiler = SkillCompiler(registry=registry, resource_resolver=resource_resolver)
    compilation = compiler.compile(invocations)
    print_compilation(compilation)

    # ------------------------------------------------------------------
    # Step 7: Execute via SkillScheduler (with session context)
    # ------------------------------------------------------------------
    print("\n--- Executing skills ---")
    scheduler = SkillScheduler(
        registry=registry,
        engine=engine,
        session_ctx=session_ctx,
    )
    state = await scheduler.execute(
        compilation.execution_plan,
        resource_plan=compilation.resource_plan,
    )

    # Report per-skill status
    for skill_id in compilation.skill_dag.nodes:
        output = state.get(skill_id)
        if isinstance(output, list):
            print(f"  {skill_id}: {len(output)} results")
        elif output is not None:
            print(f"  {skill_id}: completed")
        else:
            print(f"  {skill_id}: no output (check errors)")

    # ------------------------------------------------------------------
    # Step 8: Print results from the final skill
    # ------------------------------------------------------------------
    topo = compilation.skill_dag.iter_topo()
    final_skill = topo[-1].skill_id if topo else invocations[-1]["skill_id"]
    output = state.get(final_skill)
    if output is None:
        print(f"\nNo output from {final_skill!r} -- check errors above.")
    else:
        nodes = output if isinstance(output, list) else [output]
        print_results(nodes, args.top_k)


# ---------------------------------------------------------------------------
# Path 3: Two-phase compilation (recommended)
# ---------------------------------------------------------------------------


def run_index_compilation(args: argparse.Namespace) -> str:
    """
    Phase 1: Index Compilation (static, repo-time).

    Uses ``IndexCompiler`` to build all requested indexes and write a
    ``repo_manifest.json``. This replaces manual ``build_bm25_index()``
    and ``build_vector_store()`` calls.
    """
    from codeminer.compiler.index_builders import (
        BM25IndexBuilder,
        IndexBuilderRegistry,
        VectorIndexBuilder,
    )
    from codeminer.compiler.index_compiler import IndexCompiler, IndexCompilerConfig

    repo_path = os.path.abspath(args.repo)
    cache_dir = args.index_path or os.path.join(repo_path, ".codeminer_cache")

    # Determine which indexes to build based on mode
    index_types = []
    if args.mode in ("sparse", "hybrid"):
        index_types.append("bm25")
    if args.mode in ("dense", "hybrid"):
        index_types.append("vector")

    # Register concrete builders
    registry = IndexBuilderRegistry()
    registry.register(
        "bm25",
        BM25IndexBuilder(languages=args.languages),
    )
    registry.register(
        "vector",
        VectorIndexBuilder(
            languages=args.languages,
            embedding_model=args.embedding_model,
            embedding_dimension=args.embedding_dimension,
        ),
    )

    # Compile the repository
    config = IndexCompilerConfig(
        index_types=index_types,
        languages=args.languages,
    )
    compiler = IndexCompiler(registry, config)

    print(f"  Building indexes: {index_types}")
    manifest = compiler.compile_repo(repo_path, cache_dir=cache_dir)
    print_manifest(manifest)

    manifest_path = os.path.join(cache_dir, "repo_manifest.json")
    return manifest_path


async def run_query_compilation(args: argparse.Namespace, manifest_path: str) -> None:
    """
    Phase 2: Query Compilation (dynamic, query-time).

    Loads the manifest, creates a ``ManifestIndexStateStore`` for resource
    resolution, loads the actual index objects from the manifest paths,
    wires up the engine, compiles the skill graph, and executes.
    """
    from codeminer.compiler.manifest import ManifestIndexStateStore, RepoManifest

    repo_path = os.path.abspath(args.repo)

    # Step 1: Load the manifest (output of Phase 1)
    manifest = RepoManifest.load(manifest_path)
    print(f"  Loaded manifest: {len(manifest.indexes)} indexes")
    print(f"  Capabilities: {manifest.capabilities}")

    # Step 2: Create resource resolver from manifest
    state_store = ManifestIndexStateStore(manifest)
    resource_resolver = ResourceResolver(state_store)

    # Step 3: Load actual index objects from manifest paths
    bm25_index = None
    vector_store = None

    if "bm25" in manifest.indexes and manifest.indexes["bm25"].status == "fresh":
        bm25_index = BM25CodeIndexer()
        bm25_index.load_index(manifest.indexes["bm25"].path)
        print(f"  Loaded BM25 from {manifest.indexes['bm25'].path}")

    if "vector" in manifest.indexes and manifest.indexes["vector"].status == "fresh":
        from codeminer.index.embedding import CodeVectorStore

        emb_model = manifest.indexes["vector"].config.get(
            "embedding_model",
            args.embedding_model,
        )
        emb_dim = manifest.indexes["vector"].config.get(
            "embedding_dimension",
            args.embedding_dimension,
        )
        vector_store = CodeVectorStore(
            embedding_model=emb_model,
            embedding_provider="huggingface",
            dimension=emb_dim,
            store_path=manifest.indexes["vector"].path,
        )
        vector_store.load(manifest.indexes["vector"].path)
        print(f"  Loaded vector store from {manifest.indexes['vector'].path}")

    # Step 4: Wire execution engine
    engine = ExecutionEngine()
    retrieve_ctx = RetrieveContext(
        bm25=bm25_index,
        vector_store=vector_store,
        default_top_k=args.top_k,
        default_level="l2",
    )
    register_retrieve_ops(engine, retrieve_ctx)

    rerank_ctx = None
    if vector_store:
        rerank_ctx = RerankContext(embedding_store=vector_store)
        register_rerank_ops(engine, rerank_ctx)

    # Step 5: Load skill packages
    skills_dir = os.path.join(_PROJECT_ROOT, "codeminer", "agent", "skills")
    contexts: Dict[str, Any] = {"retrieve": retrieve_ctx}
    if rerank_ctx:
        contexts["rerank"] = rerank_ctx

    loader = SkillLoader()
    loaded = loader.load_all(skills_dir, contexts=contexts)
    print(f"  Loaded {len(loaded)} skills: {[s.skill_id for s in loaded]}")

    # Step 6: Set up session context
    session_ctx = SessionContext(
        repo_path=repo_path,
        repo_size=manifest.file_count,
        primary_language=manifest.languages[0] if manifest.languages else "python",
    )

    # Step 7: Compile skill invocations (with manifest-backed resource resolution)
    print("\n--- Compiling skill graph ---")
    invocations = build_invocations(args.mode, args.query, args.top_k)

    registry = SkillRegistry()
    compiler = SkillCompiler(registry=registry, resource_resolver=resource_resolver)
    compilation = compiler.compile(invocations)
    print_compilation(compilation)

    if compilation.resource_plan and not compilation.resource_plan.can_execute:
        print(
            f"\nCannot execute: missing indexes {compilation.resource_plan.blocking_builds}"
        )
        print("Re-run Phase 1 with the needed index types.")
        return

    # Step 8: Execute
    print("\n--- Executing skills ---")
    scheduler = SkillScheduler(
        registry=registry,
        engine=engine,
        session_ctx=session_ctx,
    )
    state = await scheduler.execute(
        compilation.execution_plan,
        resource_plan=compilation.resource_plan,
    )

    for skill_id in compilation.skill_dag.nodes:
        output = state.get(skill_id)
        if isinstance(output, list):
            print(f"  {skill_id}: {len(output)} results")
        elif output is not None:
            print(f"  {skill_id}: completed")
        else:
            print(f"  {skill_id}: no output (check errors)")

    topo = compilation.skill_dag.iter_topo()
    final_skill = topo[-1].skill_id if topo else invocations[-1]["skill_id"]
    output = state.get(final_skill)
    if output is None:
        print(f"\nNo output from {final_skill!r} -- check errors above.")
    else:
        nodes = output if isinstance(output, list) else [output]
        print_results(nodes, args.top_k)


async def run_two_phase(args: argparse.Namespace) -> None:
    """Demonstrate the full two-phase compilation architecture."""
    repo_path = os.path.abspath(args.repo)

    print(f"Repository : {repo_path}")
    print(f"Query      : {args.query}")
    print(f"Path       : two-phase (index compile -> query compile)")
    print(f"Mode       : {args.mode}")
    print(f"Top-k      : {args.top_k}")

    # ============================
    # Phase 1: Index Compilation
    # ============================
    print("\n" + "=" * 50)
    print("  PHASE 1: Index Compilation (repo-time)")
    print("=" * 50)
    manifest_path = run_index_compilation(args)

    # ============================
    # Phase 2: Query Compilation
    # ============================
    print("\n" + "=" * 50)
    print("  PHASE 2: Query Compilation (query-time)")
    print("=" * 50)
    await run_query_compilation(args, manifest_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.path == "standalone":
        asyncio.run(run_standalone(args))
    elif args.path == "compiled":
        asyncio.run(run_compiled(args))
    else:
        asyncio.run(run_two_phase(args))


if __name__ == "__main__":
    main()
