python examples/retrieve_rerank.py     \
    --dataset swebench_lite     \
    --top-k 100     \
    --metrics-k 5     \
    --eval-instances $EVAL_INSTANCES     \
    --embedding-model nomic-ai/CodeRankEmbed     \
    --embedding-provider huggingface     \
    --retrieval-mode dense \
    --result-path $RESULT_PATH \
    --retrieval-only
