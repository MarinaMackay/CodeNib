#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/swebench_sampling}"
LANGS=("Python" "Rust" "JavaScript/TypeScript" "C++" "C")

python scripts/collect_and_synthesize_swebench.py \
  --output-dir "${OUTPUT_DIR}" \
  --languages "${LANGS[@]}" \
  --stage collect \
  --print-sample 5
