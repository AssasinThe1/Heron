#!/usr/bin/env python3
"""vLLM inference over a prompts JSON. COLAB. Never touches a database.

Two modes:

  greedy (default)  temperature 0, n=1 -> results_{tag}.json  [{id, sql, finish_reason}]
  sampling (--n K)  temperature ~0.8, n=K -> candidates_{tag}.json
                    [{id, candidates: [{sql, finish_reason}, ...]}]

The sampling mode is both the pass@k measurement and the rejection-sampling
generation run. Measure pass@k on a train-fit SAMPLE first (--limit 500): if the
teacher's pass@8 is only a few points above its greedy 63%, the survivor set
will be thin on exactly the hard questions you need, and you should know that
before spending hours generating the full set.

Install note: Colab's nvidia-smi reports the DRIVER CUDA version (13.0) while
nvcc reports the TOOLKIT (12.8). vLLM auto-detects the driver, pulls a cu130
wheel, and dies on libcudart.so.13. Install with:
    pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
or pin an explicit +cu128 wheel URL to bypass detection entirely.

Examples
--------
  # greedy eval
  python run_inference.py --prompts prompts_train-val-1k_fk+samples.json

  # pass@8 probe on a subset
  python run_inference.py --prompts prompts_train-fit_fk+samples.json \
      --n 8 --temperature 0.8 --limit 500 --out candidates_probe.json

  # the tuned student (merged weights)
  python run_inference.py --prompts prompts_dev_fk+samples.json \
      --model ./qwen7b-t2s-merged --quantization ''
"""

import argparse
import json
import os
import time


def load_done(out_path, resume):
    if not (resume and os.path.exists(out_path)):
        return [], set()
    existing = json.load(open(out_path))
    return existing, {r["id"] for r in existing}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default=None)

    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ")
    ap.add_argument("--quantization", default="awq",
                    help="'awq' for the AWQ build; '' for unquantized/merged")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--enable-lora", action="store_true",
                    help="serve a LoRA adapter without merging (slower; prefer merged)")
    ap.add_argument("--lora-path", default=None)

    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=1,
                    help=">1 switches to candidate-sampling mode")

    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    tag = os.path.basename(args.prompts).replace("prompts_", "").replace(".json", "")
    sampling_mode = args.n > 1
    default_out = f"{'candidates' if sampling_mode else 'results'}_{tag}.json"
    out_path = args.out or default_out

    if sampling_mode and args.temperature == 0.0:
        raise SystemExit("--n > 1 with temperature 0 produces n identical "
                         "completions. Set --temperature 0.8.")

    prompts = json.load(open(args.prompts))
    if args.limit:
        prompts = prompts[:args.limit]

    out, done = load_done(out_path, args.resume)
    todo = [p for p in prompts if p["id"] not in done]
    print(f"{len(prompts)} prompts | {len(done)} done | {len(todo)} to run")
    print(f"mode={'sampling n=' + str(args.n) if sampling_mode else 'greedy'} "
          f"-> {out_path}")
    if not todo:
        return

    from vllm import LLM, SamplingParams
    lora_req = None
    llm = LLM(
        model=args.model,
        quantization=args.quantization or None,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_lora=args.enable_lora,
    )
    if args.enable_lora and args.lora_path:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("t2s", 1, args.lora_path)

    sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                              max_tokens=args.max_tokens, n=args.n)

    t0 = time.time()
    step = args.checkpoint_every
    for start in range(0, len(todo), step):
        batch = todo[start:start + step]
        kw = {"lora_request": lora_req} if lora_req else {}
        outs = llm.chat([p["messages"] for p in batch], sampling, **kw)
        for p, o in zip(batch, outs):
            if sampling_mode:
                # finish_reason travels with every candidate: at T=0.8 you will
                # get occasional rambling completions that hit max_tokens, and a
                # truncated survivor must never enter the training set.
                out.append({"id": p["id"], "candidates": [
                    {"sql": c.text.strip(), "finish_reason": c.finish_reason}
                    for c in o.outputs]})
            else:
                c = o.outputs[0]
                out.append({"id": p["id"], "sql": c.text.strip(),
                            "finish_reason": c.finish_reason,
                            "n_prompt_tokens": len(o.prompt_token_ids),
                            "n_completion_tokens": len(c.token_ids)})
        json.dump(out, open(out_path, "w"))
        print(f"  checkpoint {len(out)}/{len(prompts)}  "
              f"({time.time() - t0:.0f}s elapsed)")

    out.sort(key=lambda r: r["id"])
    json.dump(out, open(out_path, "w"))

    if sampling_mode:
        n_trunc = sum(1 for r in out for c in r["candidates"]
                      if c["finish_reason"] == "length")
        print(f"done: {len(out)} questions x {args.n} candidates -> {out_path}")
        print(f"truncated candidates: {n_trunc} (these will be dropped by "
              "build_sft_data.py)")
    else:
        n_trunc = sum(1 for r in out if r["finish_reason"] == "length")
        print(f"done: {len(out)} results -> {out_path}")
        print(f"truncated: {n_trunc}")
    print(f"wall clock: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
