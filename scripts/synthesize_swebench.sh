#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CACHE_DIR="${CACHE_DIR:-$HOME/.codeminer}"
OUTPUT_DIR="${OUTPUT_DIR:-$CACHE_DIR/swebench_sampling}"
SELECTED_INSTANCES="${SELECTED_INSTANCES:-$OUTPUT_DIR/selected_instances.json}"
SYNTHESIS_LIMIT="${SYNTHESIS_LIMIT:-}"
REPEAT_PER_INSTANCE="${REPEAT_PER_INSTANCE:-1}"
INSTANCE_ID="${INSTANCE_ID:-}"
QUERY_TYPES="${QUERY_TYPES:-behavioral,module_hint,file_hint,symbol_hint,reasoning}"

cd "${ROOT_DIR}"

EXTRA_ARGS=()
if [[ -n "${SYNTHESIS_LIMIT}" ]]; then
  EXTRA_ARGS+=(--synthesis-limit "${SYNTHESIS_LIMIT}")
fi
if [[ -n "${INSTANCE_ID}" ]]; then
  EXTRA_ARGS+=(--instance-id "${INSTANCE_ID}")
fi
PYTHONPATH=. python scripts/synthesize_swebench.py \
  --cache-dir "${CACHE_DIR}" \
  --repo-cache-dir "${CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --selected-instances "${SELECTED_INSTANCES}" \
  --query-types "${QUERY_TYPES}" \
  --repeat-per-instance "${REPEAT_PER_INSTANCE}" \
  "${EXTRA_ARGS[@]}"
