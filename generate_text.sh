#!/bin/bash
# Start the LLM server first (./serve_llm.sh); set LLM_BASE_URL if it runs on another node.
# The run is resumable: re-running skips prompts that are already generated.

output_dir="all/0_sentence_collection"
num_per_k=20000  # 10 values of k x 20000 = 200K prompts
concurrency=${CONCURRENCY:-256}  # roughly max-num-seqs x number of server replicas

python ./dynamix/batch_sentence_generation.py \
    --k_min 1 --k_max 10 \
    --num_per_k $num_per_k \
    --output_dir $output_dir \
    --concurrency $concurrency

echo "Sentence generation completed."
