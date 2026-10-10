"""High-throughput version of main_sentence_generation.py.

Runs many prompt generations concurrently in one process against a local
OpenAI-compatible server (see serve_llm.sh), so the server can batch requests.

Results are appended as JSON lines (one per prompt, holding both the sentence
data and the refined config) to {output_dir}/{num_per_k}_k={k}.jsonl.
The run is resumable: indices already present in that file are skipped.
Prompts that fail are not regenerated: an index for which every attempt fails is
recorded in {num_per_k}_failed_k={k}.jsonl, and an index whose prompt fails the VQA
check in {num_per_k}_vqa_rejected_k={k}.jsonl; later runs skip both.
At the end, each k is also written in the original JSON-list format
({num_per_k}_sentences_k={k}.json / {num_per_k}_refined_config_k={k}.json)
that main_image_generation.py reads, plus {num_per_k}_geneval2_k={k}.jsonl in the
GenEval2 benchmark format (prompt, atom_count, vqa_list, skills) for its evaluation.py.

The VQA questions are GenEval2-style. Most are built from the refined config (see
utils.geneval2_vqa); an extra LLM call then adds questions for important content of the
written prompt that the config does not cover, e.g. an object the writer added (see
utils.prompt_vqa). With --vqa_check, each prompt is also checked against its questions
by an extra LLM call; --check_only runs that check later on prompts generated without it.
"""
import os
import sys
import json
import time
import random
import argparse
import threading
import contextlib
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from utils import generate_sentence, load_configs_new, load_local_chat_api, geneval2_vqa, prompt_vqa, config_has_skills, vqa_mismatches, LLM_USAGE


# why attempts were discarded, for diagnosing low yield
STATS = collections.Counter()
STATS_LOCK = threading.Lock()


def count(key, n=1):
    with STATS_LOCK:
        STATS[key] += n


def server_unreachable(e):
    """Whether e (or an error it was raised from) means the LLM server could not be reached."""
    while e is not None:
        if isinstance(e, requests.ConnectionError):
            return True
        e = e.__cause__
    return False


def generate_one(chat_api, k, index, args):
    """Same logic as main_sentence_generation.main, minus the file writing."""
    for attempt in range(args.max_resamples):
        count(f'attempts_k={k}')
        try:
            skills, descriptions, categories, refined_config = load_configs_new(chat_api, k, config_path=args.config_path, specified_skill=args.specified_skill, object_prob=args.object_prob, category_weights=args.category_weights)
            # VQA first: configs it cannot ask about are resampled before spending LLM calls on a sentence
            if not config_has_skills(refined_config, skills, categories):
                count('config_dropped_skill')
                continue
            config_vqa, config_vqa_skills, _ = geneval2_vqa(refined_config)
            # with --vqa_check, a prompt whose VQA answers do not follow from it is rewritten for the same config
            for _ in range(args.sentence_retries if args.vqa_check else 1):
                sentence, question = generate_sentence(chat_api, skills, descriptions, json_config=refined_config, generate_questions=False)
                if sentence.startswith("WRONG") or question is None:
                    break
                # questions for what the written prompt adds to the config
                vqa_list, vqa_skills, atom_count = prompt_vqa(chat_api, sentence, config_vqa, config_vqa_skills)
                count('prompt_vqa_added', len(vqa_list) - len(config_vqa))
                if not args.vqa_check:
                    break
                mismatches = vqa_mismatches(chat_api, sentence, vqa_list)
                if not mismatches:
                    break
                count('vqa_mismatch')
                if args.verbose:
                    print(f"[k={k} idx={index}] VQA mismatches {mismatches}: {sentence}", file=sys.stderr, flush=True)
            else:
                continue
        except Exception as e:
            # a server outage is not the prompt's fault: give up on the index without
            # recording it as failed, so a later run generates it
            if server_unreachable(e):
                raise
            # e.g. the model returned an invalid refined config; resample
            count(f'error:{type(e).__name__}')
            if args.verbose:
                print(f"[k={k} idx={index}] attempt {attempt+1} failed: {e}", file=sys.stderr, flush=True)
            continue
        if sentence.startswith("WRONG") or question is None:
            count('rejected_by_llm')
            if args.verbose:
                print(f"[k={k} idx={index}] rejected: {sentence[:300]}", file=sys.stderr, flush=True)
            continue
        intermediate_data = {
            'index': index,
            'sentence': sentence,
            'categories': categories,
            'skills': skills,
            'question': [q for q, _ in vqa_list],
            'vqa_list': vqa_list,
            'vqa_skills': vqa_skills,
            'atom_count': atom_count,
            'num_config_questions': len(config_vqa),  # the rest of vqa_list comes from prompt_vqa
        }
        refined_config['index'] = index
        return intermediate_data, refined_config
    return None


def usage_summary(n_done):
    """LLM calls and tokens so far, also per generated prompt (counting discarded attempts)."""
    usage = dict(LLM_USAGE)
    per_prompt = {k: round(v / max(n_done, 1)) for k, v in usage.items()}
    return f"LLM usage {usage}, per prompt {per_prompt}"


def read_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # partially written last line from an interrupted run
    return rows


def skipped_indices(output_dir, num_per_k, k):
    """Indices not to generate again: every attempt failed, or the prompt failed the VQA check."""
    return {r['index'] for name in ['failed_', 'vqa_rejected_']
            for r in read_jsonl(os.path.join(output_dir, f'{num_per_k}_{name}k={k}.jsonl'))}


def finalize(output_dir, num_per_k, k):
    rows = sorted(read_jsonl(os.path.join(output_dir, f'{num_per_k}_k={k}.jsonl')), key=lambda r: r['index'])
    for name in ['sentences', 'refined_config']:
        with open(os.path.join(output_dir, f'{num_per_k}_{name}_k={k}.json'), 'w') as f:
            json.dump([r[name] for r in rows], f, indent=4)
    with open(os.path.join(output_dir, f'{num_per_k}_geneval2_k={k}.jsonl'), 'w') as f:
        for r in rows:
            s = r['sentences']
            f.write(json.dumps({'prompt': s['sentence'], 'atom_count': s['atom_count'], 'vqa_list': s['vqa_list'], 'skills': s['vqa_skills']}) + '\n')


def check_existing(args):
    """VQA check (see utils.vqa_mismatches) of prompts that were generated without it.

    Results are appended to {num_per_k}_vqa_check_k={k}.jsonl, so an interrupted check
    resumes where it stopped. Once every prompt is checked, failing rows are moved from
    {num_per_k}_k={k}.jsonl to {num_per_k}_vqa_rejected_k={k}.jsonl; generation runs
    do not regenerate those indices. The .json files are rebuilt without them.
    """
    chat_api = load_local_chat_api(base_url=args.llm_base_url, model=args.llm_model, thinking=args.thinking)
    ks = list(range(args.k_min, args.k_max + 1))
    path = lambda name, k: os.path.join(args.output_dir, f'{args.num_per_k}_{name}k={k}.jsonl')
    # a check applies to one sentence of an index, so regenerated prompts are checked again
    key = lambda index, sentence: f'{index}\t{sentence}'

    checked, tasks, files = {}, [], {}
    for k in ks:
        checked[k] = {key(r['index'], r['sentence']): r['mismatches'] for r in read_jsonl(path('vqa_check_', k))}
        tasks += [(k, r) for r in read_jsonl(path('', k)) if key(r['index'], r['sentences']['sentence']) not in checked[k]]
        files[k] = open(path('vqa_check_', k), 'a')
    total = len(tasks)
    print(f"{total} prompts to check ({sum(map(len, checked.values()))} already checked)", file=sys.stderr, flush=True)

    def check_one(row):
        s = row['sentences']
        vqa_list = s.get('vqa_list')
        if vqa_list is None:  # rows from before the VQA questions were stored
            try:
                vqa_list = geneval2_vqa(row['refined_config'])[0]
            except ValueError as e:
                return [['(config)', 'askable', str(e)]]
        return vqa_mismatches(chat_api, s['sentence'], vqa_list)

    lock = threading.Lock()
    n_done = n_failed = n_mismatch = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(check_one, r): (k, r) for k, r in tasks}
        for fut in as_completed(futures):
            k, r = futures[fut]
            with lock:
                try:
                    mismatches = fut.result()
                except Exception as e:
                    print(f"[k={k} idx={r['index']}] check failed: {e}", file=sys.stderr, flush=True)
                    n_failed += 1
                    continue
                files[k].write(json.dumps({'index': r['index'], 'sentence': r['sentences']['sentence'], 'mismatches': mismatches}) + '\n')
                files[k].flush()
                checked[k][key(r['index'], r['sentences']['sentence'])] = mismatches
                n_done += 1
                n_mismatch += bool(mismatches)
                if args.verbose and mismatches:
                    print(f"[k={k} idx={r['index']}] VQA mismatches {mismatches}: {r['sentences']['sentence']}", file=sys.stderr, flush=True)
                if (n_done + n_failed) % args.log_every == 0:
                    rate = n_done / (time.time() - start)
                    eta = (total - n_done - n_failed) / max(rate, 1e-9) / 3600
                    print(f"{n_done}/{total} checked ({n_mismatch} with mismatches), {n_failed} failed, {rate:.2f} prompts/s, ETA {eta:.1f}h", file=sys.stderr, flush=True)
    for f in files.values():
        f.close()
    if n_failed:
        print(f"{n_failed} checks failed; re-run to retry them before rejected prompts are removed.", file=sys.stderr)
        sys.exit(1)

    for k in ks:
        rows = read_jsonl(path('', k))
        keep = [r for r in rows if not checked[k].get(key(r['index'], r['sentences']['sentence']))]
        rejected = [r for r in rows if checked[k].get(key(r['index'], r['sentences']['sentence']))]
        if rejected:
            with open(path('vqa_rejected_', k), 'a') as f:
                for r in rejected:
                    f.write(json.dumps({**r, 'mismatches': checked[k][key(r['index'], r['sentences']['sentence'])]}) + '\n')
            tmp = path('', k) + '.tmp'
            with open(tmp, 'w') as f:
                for r in keep:
                    f.write(json.dumps(r) + '\n')
            os.replace(tmp, path('', k))
        finalize(args.output_dir, args.num_per_k, k)
        print(f"k={k}: {len(rejected)}/{len(rows)} prompts rejected", file=sys.stderr)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.check_only:
        check_existing(args)
        return
    if args.finalize_only:
        for k in range(args.k_min, args.k_max + 1):
            finalize(args.output_dir, args.num_per_k, k)
        return
    random.seed(args.seed)
    chat_api = load_local_chat_api(base_url=args.llm_base_url, model=args.llm_model, thinking=args.thinking)

    ks = list(range(args.k_min, args.k_max + 1))
    files, failed_files, tasks = {}, {}, []
    n_skipped = 0
    for k in ks:
        path = os.path.join(args.output_dir, f'{args.num_per_k}_k={k}.jsonl')
        done = {r['index'] for r in read_jsonl(path)}
        skipped = skipped_indices(args.output_dir, args.num_per_k, k) - done
        n_skipped += len(skipped)
        tasks += [(k, i) for i in range(1, args.num_per_k + 1) if i not in done and i not in skipped]
        files[k] = open(path, 'a')
        failed_files[k] = open(os.path.join(args.output_dir, f'{args.num_per_k}_failed_k={k}.jsonl'), 'a')
    # interleave k values so the server sees a mix of short and long requests
    random.shuffle(tasks)
    total = len(tasks)
    print(f"{total} prompts to generate ({len(ks) * args.num_per_k - total - n_skipped} already done, {n_skipped} failed earlier and skipped)", file=sys.stderr, flush=True)

    lock = threading.Lock()
    n_done = n_failed = n_lost = 0
    start = time.time()

    # utils prints a lot per prompt; silence stdout unless asked
    quiet = open(os.devnull, 'w') if not args.verbose else None
    with contextlib.redirect_stdout(quiet) if quiet else contextlib.nullcontext():
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(generate_one, chat_api, k, i, args): (k, i) for k, i in tasks}
            for fut in as_completed(futures):
                k, i = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    # e.g. the server went down: the index is generated by a later run
                    print(f"[k={k} idx={i}] lost: {e}", file=sys.stderr, flush=True)
                    with lock:
                        n_lost += 1
                    continue
                with lock:
                    if result is None:
                        # every attempt failed; recorded so later runs skip the index
                        failed_files[k].write(json.dumps({'index': i}) + '\n')
                        failed_files[k].flush()
                        n_failed += 1
                    else:
                        f = files[k]
                        f.write(json.dumps({'index': i, 'sentences': result[0], 'refined_config': result[1]}) + '\n')
                        f.flush()
                        n_done += 1
                    if (n_done + n_failed) % args.log_every == 0:
                        rate = n_done / (time.time() - start)
                        eta = (total - n_done - n_failed) / max(rate, 1e-9) / 3600
                        print(f"{n_done}/{total} done, {n_failed} failed, {rate:.2f} prompts/s, ETA {eta:.1f}h | {dict(STATS)} | {usage_summary(n_done)}", file=sys.stderr, flush=True)

    for f in [*files.values(), *failed_files.values()]:
        f.close()
    for k in ks:
        finalize(args.output_dir, args.num_per_k, k)
    print(f"Discarded attempts: {dict(STATS)}", file=sys.stderr)
    print(f"LLM usage: {usage_summary(n_done)}", file=sys.stderr)
    print(f"Finished: {n_done} generated, {n_failed} failed (not retried), {n_lost} lost to server errors (re-run to generate them) in {(time.time() - start) / 3600:.2f}h.", file=sys.stderr)


def parse_weights(text):
    """'number=3,color=2' -> {'number': 3.0, 'color': 2.0}"""
    weights = {}
    for item in filter(None, text.split(',')):
        name, _, value = item.partition('=')
        weights[name.strip()] = float(value)
    return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompts concurrently with a local LLM server.")
    parser.add_argument("--k_min", type=int, default=1, help="Smallest number of skills")
    parser.add_argument("--k_max", type=int, default=10, help="Largest number of skills")
    parser.add_argument("--num_per_k", type=int, default=20000, help="Number of prompts per k")
    parser.add_argument("--specified_skill", type=str, default=None, help="Specify a skill to generate prompts for")
    parser.add_argument("--output_dir", type=str, default="all/0_sentence_collection", help="Output directory")
    parser.add_argument("--config_path", type=str, default="./config", help="Path to the config files")
    parser.add_argument("--concurrency", type=int, default=256, help="Number of prompts generated in parallel")
    parser.add_argument("--object_prob", type=float, default=0.35, help="Chance, per skill, of adding another object (original ConceptMix: 0.25)")
    parser.add_argument("--category_weights", type=parse_weights, default="number=5,style=0.2,spatial=3,texture=2,color=2", help="Relative sampling weights of attribute categories, e.g. 'number=3,color=2' (others 1; original ConceptMix: all 1)")
    parser.add_argument("--max_resamples", type=int, default=10, help="Resample attempts when the LLM rejects a combination")
    parser.add_argument("--vqa_check", action="store_true", help="Check each prompt with an extra LLM call that answers its VQA questions from the prompt alone, and rewrite prompts that fail")
    parser.add_argument("--check_only", action="store_true", help="Only run the VQA check on already generated prompts, and remove the ones that fail (they are not regenerated)")
    parser.add_argument("--sentence_retries", type=int, default=3, help="With --vqa_check, rewrites of a prompt that fails the check before resampling the config")
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen3.6 thinking mode for every LLM call (more accurate, several times slower)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for shuffling task order")
    parser.add_argument("--log_every", type=int, default=100, help="Print progress every N prompts")
    parser.add_argument("--finalize_only", action="store_true", help="Only rebuild the .json files from the .jsonl files (e.g. after a job hit its time limit)")
    parser.add_argument("--verbose", action="store_true", help="Show per-prompt LLM outputs")
    parser.add_argument("--llm_base_url", type=str, default=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"), help="Base URL of the local server")
    parser.add_argument("--llm_model", type=str, default=os.environ.get("LLM_MODEL", "Qwen3.6-27B"), help="Model name as served by the local server")
    args = parser.parse_args()
    main(args)
