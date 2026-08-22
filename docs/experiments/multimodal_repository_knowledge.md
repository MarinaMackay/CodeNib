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
  -> VisualGraphManifest
  -> Wiki / MCP query APIs / local preview
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
symlinks, hashes media content, and records markdown references, alt text,
captions, and surrounding documentation context.

For SVG artifacts, it also extracts bounded embedded text from `<title>`,
`<desc>`, `<text>`, and `<tspan>` elements. This makes the deterministic local
fallback useful for real repository diagrams without requiring a VLM key.

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

### VisualGroundingManifest

`codenib.wiki.media_grounding` grounds extracted visual entities to repository
files and symbols. The first implementation uses deterministic lexical scoring
against a bounded source-symbol inventory. Later versions can replace the
scorer with BM25, embeddings, CodeGraph, LSP facts, or `FactQueryIndex` /
`FactBatch`. The `ground_visual_facts_to_sources(..., scorer=...)` hook already
accepts a custom scorer, so graph/fact-backed ranking can be added without
changing the binding manifest schema.

### MultimodalKnowledgeView

`codenib.wiki.media_knowledge` joins artifacts, facts, and source bindings into
a queryable view. It exposes three functions that future MCP tools can wrap:

- `search_visual_context`
- `get_visual_evidence`
- `find_visual_code_links`

`codenib.wiki.media_tools.MultimodalKnowledgeToolRouter` exposes the same
surface as an MCP-compatible tool router with stable tool schemas and bounded
input validation. The production MCP server uses this router when it is started
with a persisted multimodal bundle.

The production MCP server can load a persisted bundle and expose these tools
directly:

```text
codenib mcp /path/to/repository \
  --multimodal-bundle /tmp/multimodal-knowledge.json
```

The bundle-backed MCP tools are:

- `explore_visual_context`
- `search_visual_context`
- `get_visual_evidence`
- `find_visual_code_links`

If the MCP server starts without `--multimodal-bundle`, these tools return a
clear runtime error rather than silently fabricating visual context.

`explore_visual_context` is the agent-friendly entry point. It composes visual
search, optional artifact evidence, and optional visual-code links in one
bounded response so agents do not need to decide which lower-level visual tool
to call first.

### VisualGraphManifest

`codenib.wiki.media_graph_plan` converts source-grounded visual knowledge into
a validated renderable graph plan. The plan follows the same separation used by
diagram-first systems: first produce a structured graph, validate it, then
compile it to a rendering format such as Mermaid.

Each `VisualGraphPlan` validates:

- unique node ids;
- bounded node and edge counts;
- repository-relative source paths;
- edges whose endpoints reference existing nodes;
- stable plan hashes.

The deterministic compiler can render a validated plan as Mermaid:

```text
flowchart LR
  WikiRenderer["WikiRenderer"]
  IndexCompiler["IndexCompiler"]
  WikiRenderer -->|calls| IndexCompiler
```

### Multimodal knowledge bundle

`codenib.wiki.media_storage` wraps the pipeline output as a versioned bundle:

```text
schema: codenib.multimodal-knowledge-bundle.v1
schema_version: 1
media_manifest
visual_facts_manifest
grounding_manifest
knowledge_view
visual_graph_manifest
component_sha256
bundle_sha256
```

The storage helper writes bundle JSON atomically and validates loaded bundles,
including schema version, required object fields, byte limits, and bundle hash.
This gives downstream consumers a stable artifact boundary instead of an ad hoc
script JSON dump.

### Incremental updates

`codenib.wiki.media_incremental` provides deterministic update planning for
multimodal views. It compares two media manifests by path and content hash,
marks artifacts as added, removed, changed, or unchanged, and identifies which
visual fact packs can be reused without another VLM call.

This is the first step toward incremental multimodal maintenance:

```text
media unchanged -> reuse existing VisualFactPack
media changed   -> rerun VLM/extractor for that artifact
media removed   -> drop stale visual facts and bindings
```

The update path is exposed as a CLI:

```text
python scripts/update_multimodal_knowledge.py /path/to/repository \
  --previous /tmp/old-multimodal-knowledge.json \
  --output /tmp/new-multimodal-knowledge.json
```

It prints a compact summary with added/removed/changed/unchanged media counts,
reused visual fact packs, regenerated fact packs, bindings, and graph plans.

### MMWiki-style evaluation

`codenib.wiki.media_eval` defines a small evaluation protocol for the first
benchmark seed. It does not try to replace SWE-bench Multimodal or MM-IssueLoc.
Instead, it measures whether repository visuals can be compiled into persistent
wiki knowledge:

- visual entity extraction precision / recall / F1;
- visual-code grounding path hit@k;
- visual-code grounding symbol hit@k.

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
  --output /tmp/multimodal-knowledge.json
```

To use an OpenAI-compatible VLM for visual fact extraction:

```text
export CODENIB_WIKI_VISUAL_FACTS_API_KEY=...
python scripts/build_multimodal_knowledge.py /path/to/repository \
  --output /tmp/multimodal-knowledge.json \
  --visual-facts-model qwen-vl \
  --visual-facts-api-base http://localhost:8000/v1 \
  --visual-facts-provider qwen
```

A minimal smoke test can create a synthetic repository with a real SVG diagram
and generate a bundle in one command:

```text
python scripts/smoke_multimodal_vlm.py \
  --output /tmp/mmwiki-smoke-bundle.json \
  --keep-repo /tmp/mmwiki-smoke-repo \
  --preview-html /tmp/mmwiki-smoke-preview.html
```

The same smoke test can call an OpenAI-compatible VLM endpoint:

```text
export CODENIB_WIKI_VISUAL_FACTS_API_KEY=...
python scripts/smoke_multimodal_vlm.py \
  --output /tmp/mmwiki-smoke-bundle.json \
  --visual-facts-model qwen-vl \
  --visual-facts-api-base http://localhost:8000/v1 \
  --visual-facts-provider qwen
```

```python
from codenib.wiki import OpenAICompatibleVisualFactExtractor

extractor = OpenAICompatibleVisualFactExtractor(
    model="qwen-vl",
    api_base="http://localhost:8000/v1",
    api_key=None,
)

facts = build_visual_facts_manifest(
    media,
    extractor=lambda artifact: extractor.extract(artifact, repo_path=repo),
)
```
