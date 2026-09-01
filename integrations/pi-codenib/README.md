# CodeNib for Pi

This Pi package adds four read-only tools backed by a locally served CodeNib
repository:

- `codenib_visual_search` searches repository visuals by natural language;
- `codenib_visual_evidence` returns persisted visual facts and source bindings;
- `codenib_architecture` returns validated Archify IR;
- `codenib_storyboard_videos` lists authenticated local MP4 walkthroughs.

## Requirements

- Node.js 22.19 or newer, matching current Pi;
- a running CodeNib local Wiki;
- the repository id shown by `GET /api/repos`.

## Run

```text
export CODENIB_API_BASE=http://127.0.0.1:8765
export CODENIB_REPO_ID=codenib-local
codenib wiki /path/to/repository
pi -e ./integrations/pi-codenib
```

Use `/codenib-status` inside Pi to confirm the connection. To install the
package for the current project after reviewing its source:

```text
pi install -l ./integrations/pi-codenib
```

The adapter accepts loopback HTTP by default. A non-loopback runtime must use
HTTPS and requires `CODENIB_ALLOW_REMOTE=1`. Responses, queries, timeouts, and
result counts are bounded; redirects are rejected. The plugin does not mutate
the repository or CodeNib indexes.
