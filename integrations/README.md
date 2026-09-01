# Agent harness integrations

CodeNib's core remains harness-neutral. Thin packages in this directory expose
the same local runtime contracts to individual coding agents without copying
retrieval or multimodal logic into each harness.

- [`pi-codenib`](./pi-codenib/) registers read-only Pi tools for visual
  semantic search, persisted visual evidence, Archify IR, and authenticated
  storyboard videos.

Adapters must preserve cancellation, response bounds, source provenance, and
the local-first network policy of the CodeNib runtime.
