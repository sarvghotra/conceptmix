#!/bin/bash
# Serve the local LLM used by dynamix/{batch,main}_sentence_generation.py (OpenAI-compatible API).
#   eval "$(mamba shell hook --shell bash)"; mamba activate vllm; module load cuda/12.6.0
#
# The 27B model (~54GB bf16) fits on one 80GB GPU, so by default each GPU runs its own
# replica (data parallel), which gives more throughput than tensor parallel.
# On GPUs with <80GB memory use TP=2 (or more); replicas = num_gpus / TP.
# Prefix caching is off: prompts (~600 tokens) are shorter than one KV block of this hybrid
# model (784 tokens), so it never hits and only costs memory. max-num-seqs is sized so the
# KV cache holds all running requests without preemption.

# keep compile caches on node-local disk: $HOME hits its quota (~/.cache/vllm grows to >10GB),
# and on shared scratch the replicas race on Triton cache files
cache_dir=${CACHE_DIR:-${SLURM_TMPDIR:-/tmp/$USER}/vllm_cache}
export VLLM_CACHE_ROOT=$cache_dir/vllm
export TRITON_CACHE_DIR=$cache_dir/triton
export TORCHINDUCTOR_CACHE_DIR=$cache_dir/torchinductor

# Triton compiles its kernel launchers with gcc against the Python headers; read from scratch
# (Lustre) by many concurrent compiles, headers randomly come back as missing
# ("fatal error: pyhash.h: No such file or directory") and kill a replica at startup.
# So copy the headers to node-local disk and have Triton's compiler ($CC) use the copy.
py_include=$(python -c "import sysconfig; print(sysconfig.get_paths()['include'])")
mkdir -p "$cache_dir/bin"
rm -rf "$cache_dir/py_include" && cp -r "$py_include" "$cache_dir/py_include"
cat > "$cache_dir/bin/cc" <<EOF
#!/bin/bash
args=()
for a in "\$@"; do
    [ "\$a" = "-I$py_include" ] && a="-I$cache_dir/py_include"
    args+=("\$a")
done
exec gcc "\${args[@]}"
EOF
chmod +x "$cache_dir/bin/cc"
export CC=$cache_dir/bin/cc

model_path=${MODEL_PATH:-/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/Qwen3.6-27B}
num_gpus=${NUM_GPUS:-$(python -c "import torch; print(torch.cuda.device_count())")}
tp=${TP:-1}
dp=$((num_gpus / tp))

exec vllm serve $model_path \
    --served-model-name Qwen3.6-27B \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --tensor-parallel-size $tp \
    --data-parallel-size $dp \
    --max-model-len 40960 \
    --max-num-seqs ${MAX_NUM_SEQS:-192} \
    --gpu-memory-utilization 0.95 \
    --no-enable-prefix-caching \
    --language-model-only \
    --reasoning-parser qwen3
