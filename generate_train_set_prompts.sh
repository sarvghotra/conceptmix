#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:4
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=conceptmix
#SBATCH --partition short-unkillable
#SBATCH -o /home/mila/s/sarvjeet-singh.ghotra/scratch/git/conceptmix/slurm_logs/%x_%j.out
#SBATCH -e /home/mila/s/sarvjeet-singh.ghotra/scratch/git/conceptmix/slurm_logs/%x_%j.out

# Generate the training prompt set (k = 1..10 skills, NUM_PER_K prompts each) with a local
# Qwen3.6-27B server on this node.
#
# Each prompt comes with GenEval2-style VQA questions built from its config
# (utils.geneval2_vqa); {NUM_PER_K}_geneval2_k={k}.jsonl holds them in the GenEval2
# benchmark format (prompt, atom_count, vqa_list, skills) for GenEval2's evaluation.py.
#
#   sbatch generate_train_set_prompts.sh
#
# A full run (~200K prompts) takes ~35h on 4x A100, longer than the 3h limit of
# short-unkillable. Each job starts the server, generates until shortly before its time
# limit, writes the .json files, and resubmits itself until every prompt exists.
# Generation is resumable, so no work is lost between jobs.
#
# Overridable with environment variables (sbatch passes them through), e.g.
#   NUM_PER_K=100 MAX_JOBS=1 sbatch generate_train_set_prompts.sh
#
# VQA check: an extra LLM call per prompt answers its VQA questions from the prompt alone,
# to catch prompts whose questions would be judged wrong (missing or misattached attributes).
# By default it runs once all prompts are generated, and failing prompts are removed.
#
# Failed prompts are not regenerated: an index for which every attempt fails
# ({NUM_PER_K}_failed_k={k}.jsonl) or whose prompt fails the VQA check
# ({NUM_PER_K}_vqa_rejected_k={k}.jsonl) is left out, so the final set can have fewer
# than NUM_PER_K prompts per k. Only prompts lost to server errors are generated again.
# The chain works out its next step from the output files:
#   generate -> VQA check (POST_VQA_CHECK=1) -> finished
#   POST_VQA_CHECK=0 sbatch generate_train_set_prompts.sh  # no check
#   VQA_CHECK=1 sbatch generate_train_set_prompts.sh       # check while generating instead of after
#   CHECK_ONLY=1 sbatch generate_train_set_prompts.sh      # (re)check generated prompts, remove failing ones, stop
#
# Thinking: every LLM call (config refinement, prompt writing, verification, VQA check) runs
# in Qwen3.6's thinking mode. It makes each prompt several times more expensive (see
# serve_llm.sh); THINKING=0 turns it off.

set -uo pipefail

# repo = the script's own directory; under sbatch the script runs from a spool copy,
# so fall back to the submission directory
repo_dir=$(cd "$(dirname "$0")" && pwd)
if [ ! -f "$repo_dir/serve_llm.sh" ]; then
    repo_dir=${SLURM_SUBMIT_DIR:-}
fi
if [ ! -f "$repo_dir/serve_llm.sh" ]; then
    echo "Cannot find the conceptmix repo (serve_llm.sh); run from the repo or sbatch from there." >&2
    exit 1
fi
output_dir=${OUTPUT_DIR:-$repo_dir/all/train_set_prompts}
k_min=${K_MIN:-1}
k_max=${K_MAX:-10}
num_per_k=${NUM_PER_K:-20000}            # 10 values of k x 20000 = 200K prompts
concurrency=${CONCURRENCY:-768}          # max-num-seqs (192) x 4 server replicas
max_jobs=${MAX_JOBS:-30}                 # safety cap on the resubmission chain
job_index=${JOB_INDEX:-1}                # position of this job in the chain
margin_s=${MARGIN_S:-600}                # stop generating this long before the time limit
object_prob=${OBJECT_PROB:-0.35}         # chance, per skill, of adding another object (original: 0.25)
category_weights=${CATEGORY_WEIGHTS:-number=6,style=0.2,spatial=3,texture=2,color=2,shape=0.2,verb=0.5,text=0.5}  # relative sampling weights of attribute categories (original: all 1)
vqa_check=${VQA_CHECK:-0}                # 1: check each prompt against its VQA questions while generating
post_vqa_check=${POST_VQA_CHECK:-$([ "$vqa_check" = 1 ] && echo 0 || echo 1)}  # 1: check all prompts once generated
check_only=${CHECK_ONLY:-0}              # 1: only check already generated prompts, removing those that fail
thinking_flag=$([ "${THINKING:-0}" = 1 ] && echo --thinking)  # Qwen3.6 thinking mode for every LLM call

seed=$(( ${SEED:-19} + ${SLURM_JOB_ID:-0} ))
echo "SEED: ${seed}"


cd "$repo_dir"
mkdir -p "$output_dir" "$repo_dir/slurm_logs"

# environment (non-interactive shells may not define `module`)
type module >/dev/null 2>&1 || source /etc/profile
eval "$(/network/scratch/s/sarvjeet-singh.ghotra/installs/miniforge3/bin/mamba shell hook --shell bash)"
mamba activate vllm
module load cuda/12.6.0

# count settled prompts: unique indices that were generated, or failed and are not retried
# (in the per-k .jsonl, failed_ and vqa_rejected_ files)
count_done() {
    python - "$output_dir" "$num_per_k" "$k_min" "$k_max" <<'EOF'
import json, os, sys
out, n, kmin, kmax = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
total = 0
for k in range(kmin, kmax + 1):
    idx = set()
    for name in ['', 'failed_', 'vqa_rejected_']:
        path = os.path.join(out, f'{n}_{name}k={k}.jsonl')
        if os.path.exists(path):
            for line in open(path):
                try:
                    idx.add(json.loads(line)['index'])
                except (json.JSONDecodeError, KeyError):
                    pass
    total += len(idx)
print(total)
EOF
}

# count VQA-checked prompts (lines in the per-k check files)
count_checked() {
    cat "$output_dir"/${num_per_k}_vqa_check_k=*.jsonl 2>/dev/null | wc -l
}

target=$(( (k_max - k_min + 1) * num_per_k ))
# written when the VQA check of all prompts has finished (and failing prompts were removed);
# prompts generated after it (e.g. after CHECK_ONLY=1 on a partial set) are checked as they are written
check_marker="$output_dir/${num_per_k}_vqa_check.done"
# a requested check always runs; it only rechecks prompts it has not seen
[ "$check_only" = 1 ] && [ "${JOB_INDEX:-1}" = 1 ] && rm -f "$check_marker"

# what is left to do: generate, check or finished
next_step() {
    if [ "$check_only" = 1 ]; then
        [ -f "$check_marker" ] && echo finished || echo check
    elif [ "$(count_done)" -lt "$target" ]; then
        echo generate
    elif [ "$post_vqa_check" = 1 ] && [ ! -f "$check_marker" ]; then
        echo check
    else
        echo finished
    fi
}
# changes whenever a job gets anything done
state() {
    echo "$(count_done) $(count_checked) $([ -f "$check_marker" ] && echo checked)"
}
finalize() {
    # rebuild the per-k .json files that main_image_generation.py reads
    python ./dynamix/batch_sentence_generation.py --k_min $k_min --k_max $k_max --num_per_k $num_per_k --output_dir "$output_dir" --finalize_only
}

echo "[$(date)] job ${SLURM_JOB_ID:-local} (#$job_index in chain) on $(hostname): $(count_done)/$target prompts generated or failed, $(count_checked) VQA-checked, next step: $(next_step)"
if [ "$(next_step)" = finished ]; then
    finalize
    echo "Nothing left to do."
    exit 0
fi

# when this job ends (from squeue's time left), so work can stop cleanly before it
time_left_s=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null | python -c "
import sys
s = sys.stdin.read().strip()
try:
    d, _, hms = s.rpartition('-')
    parts = [int(x) for x in hms.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    print((int(d) if d else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2])
except ValueError:
    print(3 * 3600)
")
job_end=$(( $(date +%s) + time_left_s ))


# start the server on a per-job port, so it cannot clash with other servers on the node
port=$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 ))
export LLM_BASE_URL="http://127.0.0.1:$port/v1"
server_log="$repo_dir/slurm_logs/vllm_${SLURM_JOB_ID:-local}.log"
PORT=$port ./serve_llm.sh > "$server_log" 2>&1 &
server_pid=$!

stop_server() {
    kill $server_pid 2>/dev/null  # serve_llm.sh execs vllm, so this is the server itself
    wait $server_pid 2>/dev/null
}
trap stop_server EXIT

echo "[$(date)] waiting for the server (log: $server_log)"
until curl -sf -m 5 "$LLM_BASE_URL/models" >/dev/null; do
    if ! kill -0 $server_pid 2>/dev/null; then
        echo "Server exited during startup; last lines of $server_log:"
        tail -20 "$server_log"
        exit 1
    fi
    sleep 10
done
echo "[$(date)] server ready"

# run the steps until done or shortly before the time limit; both steps flush their output
# per prompt, so work in flight at the cutoff is simply redone by the next job
state_before=$(state)
while step=$(next_step); [ "$step" != finished ]; do
    run_s=$(( job_end - $(date +%s) - margin_s ))
    [ $run_s -le 60 ] && break
    if [ "$step" = check ]; then
        # failing prompts are only removed once every prompt has been checked
        echo "[$(date)] VQA check of the generated prompts"
        timeout --signal=TERM $run_s python ./dynamix/batch_sentence_generation.py \
            --k_min $k_min --k_max $k_max \
            --num_per_k $num_per_k \
            --output_dir "$output_dir" \
            --concurrency $concurrency \
            $thinking_flag \
            --check_only
        rc=$?
        [ $rc -eq 124 ] && reason="time limit reached" || reason="exit code $rc"
        echo "[$(date)] VQA check stopped ($reason)"
        [ $rc -eq 0 ] || break
        touch "$check_marker"
    else
        # prompts generated after the VQA check are checked as they are written
        check_flag=""
        if [ "$vqa_check" = 1 ] || [ -f "$check_marker" ]; then
            check_flag="--vqa_check"
        fi
        echo "[$(date)] generating prompts $check_flag"
        timeout --signal=TERM $run_s python ./dynamix/batch_sentence_generation.py \
            --k_min $k_min --k_max $k_max \
            --num_per_k $num_per_k \
            --output_dir "$output_dir" \
            --concurrency $concurrency \
            --object_prob $object_prob \
            --category_weights "$category_weights" \
            $check_flag \
            $thinking_flag \
            --seed $seed
        rc=$?
        [ $rc -eq 124 ] && reason="time limit reached" || reason="exit code $rc"
        echo "[$(date)] generation stopped ($reason)"
        [ $rc -eq 0 ] || break
        # prompts lost to server errors are generated by the next job
        [ "$(next_step)" = generate ] && break
    fi
done

stop_server
trap - EXIT
finalize

step=$(next_step)
echo "[$(date)] $(count_done)/$target prompts generated or failed, $(count_checked) VQA-checked, next step: $step"

if [ "$step" = finished ]; then
    echo "All done in $output_dir"
elif [ "$(state)" = "$state_before" ]; then
    echo "No progress in this job; not resubmitting. Check $server_log"
    exit 1
elif [ "$job_index" -ge "$max_jobs" ]; then
    echo "Reached MAX_JOBS=$max_jobs; not resubmitting. Run sbatch again to continue."
else
    JOB_INDEX=$((job_index + 1)) sbatch --export=ALL "$repo_dir/generate_train_set_prompts.sh"
fi
