# Source-grounded multimodal repository knowledge

This experiment extends the wiki media slot work into a first-class multimodal
repository knowledge pipeline.

The goal is not to make wiki pages more decorative. The goal is to make
repository-native images, diagrams, and screenshots reusable as source-grounded
context for wiki pages and coding agents.

## Pipeline

```text
repository images / svg / screenshots
  -> MediaManifest
  -> VisualFactPack
  -> VisualGroundingManifest
  -> MultimodalKnowledgeView
  -> Wiki / future MCP query APIs
```

## Components

### MediaManifest

`codenib.wiki.media_artifacts.discover_media_manifest()` scans a repository for
supported visual artifacts:

- `.png`
- `.jpg`
- `.jpeg`
- `.svg`
- `.webp`

It respects the shared repository traversal and source-selection policy, skips
symlinks, reads bounded stable regular files, hashes media content, and records
bounded markdown references, alt text, captions, and surrounding documentation
context. Repository-contained parent references such as
`../assets/architecture.svg` are normalized while paths that escape the
repository are ignored.

### VisualFactPack

`codenib.wiki.media_facts` defines the structured output expected from a VLM:

- visual entities
- visual relations
- source-grounded claims
- grounding candidates

The local `deterministic_visual_facts()` fallback extracts conservative facts
from artifact metadata only. Future VLM backends can replace that extractor
while keeping the same schema.

`OpenAICompatibleVisualFactExtractor` provides the first provider-neutral VLM
adapter. It targets an OpenAI-compatible `/chat/completions` endpoint, sends a
bounded local artifact as a data URL, asks for JSON-only structured visual
facts, and normalizes the response into the same `VisualFactPack` schema. This
keeps the multimodal knowledge pipeline independent of a specific model family.

The extractor is disabled by default. It can be configured through `QAConfig`
or environment variables:

```yaml
wiki_visual_facts_enabled: true
wiki_visual_facts_model: qwen-vl
wiki_visual_facts_api_base: http://localhost:8000/v1
wiki_visual_facts_options:
  provider: qwen
  timeout: 120
```

Equivalent environment variables:

```text
CODENIB_WIKI_VISUAL_FACTS_ENABLED=true
CODENIB_WIKI_VISUAL_FACTS_MODEL=qwen-vl
CODENIB_WIKI_VISUAL_FACTS_API_BASE=http://localhost:8000/v1
CODENIB_WIKI_VISUAL_FACTS_API_KEY=...
CODENIB_WIKI_VISUAL_FACTS_OPTIONS='{"provider":"qwen","timeout":120}'
```

Offline and CI runs keep using deterministic local extraction unless the VLM is
explicitly enabled and both model and endpoint are provided.

Callers can bind the validated configuration to the repository root and pass
the resulting extractor directly into the one-step pipeline:

```python
from codenib.web.config import load_config
from codenib.wiki import (
    build_multimodal_repository_knowledge,
    visual_fact_extractor_from_config,
)

repo = "/path/to/repository"
config = load_config()
extractor = visual_fact_extractor_from_config(config, repo_path=repo)
bundle = build_multimodal_repository_knowledge(repo, extractor=extractor)
```

### VisualGroundingManifest

`codenib.wiki.media_grounding` grounds extracted visual entities to repository
files and symbols. The first implementation uses deterministic lexical scoring
against a bounded source-symbol inventory derived from the shared language
registry. Later versions can replace the scorer with BM25, embeddings,
CodeGraph, LSP facts, or `FactQueryIndex` /
`FactBatch`. The `ground_visual_facts_to_sources(..., scorer=...)` hook already
accepts a custom scorer, so graph/fact-backed ranking can be added without
changing the binding manifest schema. Custom scorers return a positive, finite
relevance score; values are not clipped, so backend ranking order is preserved.

### MultimodalKnowledgeView

`codenib.wiki.media_knowledge` joins artifacts, facts, and source bindings into
a queryable view. It exposes three functions that future MCP tools can wrap:

- `search_visual_context`
- `get_visual_evidence`
- `find_visual_code_links`

`codenib.wiki.media_tools.MultimodalKnowledgeToolRouter` exposes the same
surface as an MCP-compatible tool router with stable tool schemas and bounded
input validation. This keeps the query surface testable before wiring it into a
server-specific MCP registration path.

### Visual semantic vector sidecar

`codenib.wiki.media_vector` maps the existing source-grounded knowledge entries
into a separate semantic vector view. The image/VLM extraction path and the
embedding path remain independent: a VLM produces structured visual facts,
while an embedding backend can map the artifact itself and text queries into a
shared space. Its bounded document contract includes the repository-relative
artifact path, verified content hash, MIME type, and the source-grounded text
assembled from captions, claims, entities, and bindings. Text-only backends can
use the assembled text; multimodal backends can resolve and verify the artifact
before encoding it. This separation lets either model change without changing
the multimodal knowledge contract.

The sidecar records the embedding provider, model, optional immutable model
revision, dimensions, input contract, document/query modalities, per-entry
input digest, normalized vector digest, and outer index digest.
Unchanged entries can reuse their previous vectors only when both the input
digest and complete embedding policy match. A model or dimension change forces
re-embedding. The default deterministic feature-hash embedder keeps local and
CI workflows functional without a model server; callers can supply a real
embedding function through the same contract.

When a vector index is attached to
`MultimodalKnowledgeToolRouter`, the router additionally exposes
`search_visual_semantic_context`. Routers without an index do not advertise the
tool, so consumers never receive an unavailable capability.

### Multimodal knowledge bundle

`codenib.wiki.media_storage` wraps the pipeline output as a versioned bundle:

```text
schema: codenib.multimodal-knowledge-bundle.v1
schema_version: 1
media_manifest
visual_facts_manifest
grounding_manifest
knowledge_view
component_sha256
bundle_sha256
```

The storage helper writes bundle JSON atomically and validates loaded bundles.
Validation recomputes every component digest, checks the cross-component digest
bindings and outer bundle hash, bounds JSON bytes and structure, and rejects
duplicate keys, non-finite numbers, and unstable or symlinked input files. This
gives downstream consumers a stable artifact boundary instead of an ad hoc
script JSON dump.

### Incremental updates

`codenib.wiki.media_incremental` provides deterministic update planning for
multimodal views. It compares two media manifests by path and by a stable
fingerprint of every extraction input: media bytes, MIME/role metadata,
captions, surrounding Markdown, and references. It marks artifacts as added,
removed, changed, or unchanged and identifies which visual fact packs can be
reused without another VLM call. Reused packs are re-normalized against the
current trusted artifact record. Missing, stale, invalid, or digest-mismatched
packs are scheduled for extraction instead.

Reuse is opt-in through `expected_extractor`. Omitting it safely schedules all
current artifacts for extraction. Callers should use a distinct extractor
identifier whenever the model or extraction policy changes so an upgrade
cannot silently retain facts produced by an older policy.

This is the first step toward incremental multimodal maintenance:

```text
media unchanged -> reuse existing VisualFactPack
media changed   -> rerun VLM/extractor for that artifact
media removed   -> drop stale visual facts and bindings
```

### MMWiki-style evaluation

`codenib.wiki.media_eval` defines a small evaluation protocol for the first
benchmark seed. It does not try to replace SWE-bench Multimodal or MM-IssueLoc.
Instead, it measures whether repository visuals can be compiled into persistent
wiki knowledge:

- visual entity extraction precision / recall / F1;
- visual-code grounding path hit@k;
- visual-code grounding symbol hit@k.

Inputs and report payloads are bounded and normalized. Grounding `k` is limited
to 1-20, non-finite ranking scores cannot destabilize ordering, and reports
emit only canonical binding fields rather than arbitrary prediction metadata.

Gold instances use this shape:

```json
{
  "instances": [
    {
      "artifact_path": "docs/architecture.svg",
      "gold_entities": [
        {"name": "IndexCompiler", "type": "component"}
      ],
      "gold_bindings": [
        {
          "entity_name": "IndexCompiler",
          "source_path": "codenib/compiler/index_compiler.py",
          "symbol": "IndexCompiler"
        }
      ]
    }
  ]
}
```

## Why evidence stays server-side

Media generation may use bounded source snippets inside provider prompts. Those
prompts should not be returned to the browser. Public asset payloads expose
safe provenance such as source citations and evidence-pack hashes, while the
full evidence pack remains a transient server-side input.

## Minimal local example

```python
from codenib.wiki.media_artifacts import discover_media_manifest
from codenib.wiki.media_facts import build_visual_facts_manifest
from codenib.wiki.media_grounding import (
    discover_source_symbol_candidates,
    ground_visual_facts_to_sources,
)
from codenib.wiki.media_knowledge import build_multimodal_knowledge_view

repo = "/path/to/repository"

media = discover_media_manifest(repo)
facts = build_visual_facts_manifest(media)
sources = discover_source_symbol_candidates(repo)
grounding = ground_visual_facts_to_sources(facts, sources)
view = build_multimodal_knowledge_view(media, facts, grounding)

print(view["entry_count"])
```

This creates a deterministic local view. A VLM extractor can be added later by
passing a custom extractor into `build_visual_facts_manifest()`.

For callers that want the full deterministic pipeline in one step:

```python
from codenib.wiki import build_multimodal_repository_knowledge

bundle = build_multimodal_repository_knowledge(repo)
view = bundle["knowledge_view"]
```

The same deterministic bundle can be written from the command line:

```text
python scripts/build_multimodal_knowledge.py /path/to/repository \
  --output /tmp/multimodal-knowledge.json \
  --exclude-root /path/to/repository/generated
```

The default grounding remains dependency-free lexical matching. To require
evidence returned by CodeNib's persisted BM25 index instead, use:

```text
python scripts/build_multimodal_knowledge.py /path/to/repository \
  --output /tmp/multimodal-knowledge.json \
  --grounding-indexes bm25 \
  --grounding-cache-dir /path/to/index-cache \
  --grounding-language python
```

`bm25` mode accepts only exact returned definitions and exact identifier
occurrences; it does not treat an arbitrary top-k result as evidence. Use
`--grounding-indexes bm25+lsp` to also build or load CodeNib's symbol graph and
query its LSP-shaped definition/reference provider. That provider selects the
available CodeGraph, SCIP FactQueryIndex, or native clangd query backend, so
the wiki layer does not introduce another graph implementation.

Index-backed scores remain bounded by the visual entity's extraction
confidence. An exact source hit therefore cannot turn a weak metadata-derived
entity into a high-confidence claim. Symbol candidates require symbol-specific
evidence; path-only index evidence can bind only to a path-level source
candidate. Index-returned targets are inserted before the bounded lexical
candidate inventory, so a definition later in a large repository cannot be
lost merely because the lexical scan reached its candidate limit first. The
total candidate count still respects `--max-source-candidates`. Backend errors
fail the build rather than silently reverting to lexical matches; an ordinary
missing or ambiguous LSP symbol is skipped as a non-match.

To derive validated diagram plans at the same time, request the graph sidecar
and optional Mermaid sources:

```text
python scripts/build_multimodal_knowledge.py /path/to/repository \
  --output /tmp/multimodal-knowledge.json \
  --visual-graph-output /tmp/visual-graphs.json \
  --visual-graph-mermaid-dir /tmp/visual-graphs
```

Each plan is bound to the knowledge-view hash and one repository artifact. Its
nodes retain source paths, symbols, lines, grounding scores, and grounding
evidence when a binding exists;
its edges come only from explicit visual fact relations. CodeNib does not infer
call edges from captions. Node ids, edge endpoints, repository-relative paths,
counts, fields, and content hashes are validated before a plan can compile to
Mermaid. The generated `.mmd` files are an inspectable renderer input; the JSON
manifest remains the provider-neutral contract for later Wiki UI or storyboard
renderers.

The local metadata extractor may produce node-only plans because it does not
invent visual relations. A configured VLM can populate `facts.relations`; only
those validated relations become graph edges.

To produce a video-ready but provider-neutral storyboard alongside the graph
plans:

```text
python scripts/build_multimodal_knowledge.py /path/to/repository \
  --output /tmp/multimodal-knowledge.json \
  --visual-storyboard-output /tmp/visual-storyboards.json \
  --visual-storyboard-markdown-dir /tmp/visual-storyboards
```

Each frame records a controlled visual prompt, narration, duration, graph-node
focus, and source citations down to path, symbol, and line. Relation frames are
created only for validated graph edges. Node-only plans instead produce entity
frames that explicitly avoid inferring unseen relationships. Weak lexical
bindings remain visible in graph metadata but are not promoted to storyboard
citations; exact or higher-confidence custom grounding is required. The resulting
manifest is a shot-list contract for a future image/video backend; it does not
claim that video has already been rendered. The Markdown files let reviewers
inspect the same production plan without a model key.

To expose a validated bundle in the local Wiki overview, write it to the
repository-local discovery path and then serve the Wiki normally:

```text
python scripts/build_multimodal_knowledge.py /path/to/repository \
  --output /path/to/repository/.codenib/multimodal-knowledge.json
codenib wiki /path/to/repository
```

The overview requests a bounded projection from
`/api/repos/{repo_id}/wiki-multimodal`. It displays only persisted visual facts
and source-binding candidates from a bundle that passes the normal storage
validation. Repositories without a bundle show the normal Wiki with no empty
placeholder panel. Matches below `0.8` are labelled as candidates; stronger
matches retain a high-confidence label and their underlying evidence instead
of being presented as independently verified citations.

The incremental updater can materialize a visual vector sidecar from the same
newly validated knowledge view:

```text
python scripts/update_multimodal_knowledge.py /path/to/repository \
  --bundle-output /tmp/multimodal-knowledge.json \
  --visual-vector-output /tmp/visual-vector-index.json \
  --visual-vector-store-output /tmp/visual-vector-faiss
```

The JSON sidecar is the portable provenance contract. The optional FAISS
directory is a real CodeNib `CodeVectorStore` materialization for low-latency
semantic search. Its schema-8 row mapping is stored as inert canonical JSON;
search hits are joined back to the validated sidecar to restore source paths,
symbols, captions, MIME types, and entry hashes.

On the next run, unchanged visual entries can reuse their vectors:

```text
python scripts/update_multimodal_knowledge.py /path/to/repository \
  --previous /tmp/multimodal-knowledge.json \
  --output /tmp/multimodal-knowledge-next.json \
  --visual-vector-output /tmp/visual-vector-index-next.json \
  --previous-visual-vector-index /tmp/visual-vector-index.json \
  --visual-vector-store-output /tmp/visual-vector-faiss-next \
  --previous-visual-vector-store /tmp/visual-vector-faiss
```

For a flat FAISS store, a small change set uses CodeNib's native
`CodeVectorStore.delta_update()` path. Larger updates, policy changes, IVF
stores, and first-time builds use a full rebuild. The output directory must be
empty unless it is the same directory as the explicitly supplied previous
store; this prevents unrelated or differently configured native artifacts from
being overwritten.

The default embedding remains the deterministic local text fallback. A
non-local multimodal embedding policy must supply both a document embedder when
building the sidecar and a query embedder when creating the FAISS store. This
keeps image/document generation and embedding as separate, auditable stages.

To use an OpenAI-compatible VLM for visual fact extraction:

```text
export CODENIB_WIKI_VISUAL_FACTS_API_KEY=...
python scripts/build_multimodal_knowledge.py /path/to/repository \
  --output /tmp/multimodal-knowledge.json \
  --visual-facts-model qwen-vl \
  --visual-facts-api-base http://localhost:8000/v1 \
  --visual-facts-provider qwen
```

After the initial bundle exists, update it without paying to re-extract
unchanged media:

```text
python scripts/update_multimodal_knowledge.py /path/to/repository \
  --previous /tmp/multimodal-knowledge.json \
  --grounding-indexes bm25+lsp \
  --grounding-cache-dir /path/to/index-cache \
  --grounding-language python
```

The command validates the previous bundle, rediscovers the current repository
media, and reuses a visual fact pack only when both its extraction inputs and
extractor identity still match. Added or changed media is extracted, removed
media is dropped, and source candidates plus visual-code grounding are always
rebuilt. Rebuilding grounding means a code-only change can improve or remove a
binding without another VLM request. The completed bundle is validated and
written atomically; an extraction or validation failure leaves an in-place
bundle untouched.

Use `--dry-run` to inspect the deterministic update plan without extraction,
grounding, or filesystem writes:

```text
python scripts/update_multimodal_knowledge.py /path/to/repository \
  --previous /tmp/multimodal-knowledge.json \
  --dry-run
```

`--output` retains the previous bundle and writes the result elsewhere, while
`--force-reextract` explicitly refreshes all current artifacts. For a VLM
update, pass the same model, API base and provider options accepted by the
initial build command. Incremental reuse is model-specific: the persisted
extractor identity includes both provider and model (or a bounded digest for a
long identity), so switching models cannot silently retain facts from the old
model.

```python
from codenib.wiki import OpenAICompatibleVisualFactExtractor

extractor = OpenAICompatibleVisualFactExtractor(
    model="qwen-vl",
    api_base="http://localhost:8000/v1",
    api_key=None,
    repo_path=repo,
)

facts = build_visual_facts_manifest(
    media,
    extractor=extractor,
)
```
