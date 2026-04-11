#!/usr/bin/env bash
set -euo pipefail
#
# Build embeddings for the CodeMiner-base dataset with three models:
#   1. Salesforce/SweRankEmbed-Small   (768d)
#   2. fishmingyu/SweRankEmbed-Large   (3584d)
#   3. jinaai/jina-code-embeddings-1.5b (1536d)
#
# Usage:
#   # Full run (all instances, all models)
#   bash scripts/embeddings/build_codeminer_base_embeddings.sh
#
#   # Single instance smoke test
#   FILTER="^(astropy__astropy-12907)$" bash scripts/embeddings/build_codeminer_base_embeddings.sh
#
#   # Override storage / dataset
#   STORAGE_DIR=/tmp/emb DATASET=fishmingyu/codeminer-base-dataset \
#     bash scripts/embeddings/build_codeminer_base_embeddings.sh

STORAGE_DIR="${STORAGE_DIR:-/mnt/data/codeminer}"
DATASET="${DATASET:-fishmingyu/codeminer-base-dataset}"
SPLIT="${SPLIT:-test}"
FILTER="${FILTER:-.*}"
PROFILE_TAG="${PROFILE_TAG:-codeminer_base_${SPLIT}}"

# model:dimension pairs
MODELS=(
  "Salesforce/SweRankEmbed-Small:768"
  "fishmingyu/SweRankEmbed-Large:3584"
  "jinaai/jina-code-embeddings-1.5b:1536"
)

mkdir -p "${STORAGE_DIR}"

for entry in "${MODELS[@]}"; do
  MODEL="${entry%%:*}"
  DIM="${entry##*:}"
  echo ""
  echo "================================================================"
  echo "Building embeddings: ${MODEL} (dim=${DIM})"
  echo "  dataset=${DATASET}  split=${SPLIT}  filter=${FILTER}"
  echo "================================================================"
  python scripts/embeddings/build_embeddings.py \
    --dataset-class codeminer_base \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --filter-instance "${FILTER}" \
    --storage-dir "${STORAGE_DIR}" \
    --enable-profiler \
    --profile-tag "${PROFILE_TAG}" \
    --embedding-model "${MODEL}" \
    --embedding-provider huggingface \
    --embedding-dimension "${DIM}" \
    --trust-remote-code \
    "$@"
done

echo ""
echo "Done. Embeddings stored under ${STORAGE_DIR}"
echo "Profile logs stored under ${STORAGE_DIR}/profile_log"
