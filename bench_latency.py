#!/usr/bin/env python3
"""Measure latency and derive cost. COLAB (needs the GPU).

Reports TWO regimes, explicitly labelled, because conflating them is how these
comparisons get attacked:

  single-stream   batch size 1, one query at a time. This is what one user
                  experiences. Report p50/p95.
  saturated       all queries submitted at once, vLLM batches freely. This is
                  what throughput and therefore cost derive from.

Also records calls-per-question (1 for this system, by design) and token counts.
Lead the pitch with calls and tokens, not dollars: call counts are hardware- and
price-independent and they do not rot. Dollar figures are an illustration with a
cited price, not the headline.

Multi-call baselines: do not estimate their call counts. Take them from the
papers (DIN-SQL, MAC-SQL, CHESS all describe their call structure) or reproduce
one pipeline yourself if you want it bulletproof.

Example
-------
  python bench_latency.py --prompts prompts_dev_fk+samples.json \
      --model ./qwen7b-t2s-merged --quantization '' \
      --n-single 100 --gpu-hour-price 1.10 --label "7B tuned (merged fp16)"
"""

import argparse
import json
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--label", default="system-under-test")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--quantization", default="")
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-tokens", type=int, default=1024)

    ap.add_argument("--n-single", type=int, default=100,
                    help="queries to run one-at-a-time for p50/p95")
    ap.add_argument("--n-saturated", type=int, default=500,
                    help="queries to submit at once for throughput")
    ap.add_argument("--gpu-hour-price", type=float, default=None,
                    help="cited on-demand price for the GPU, USD/hour")
    ap.add_argument("--price-source", default="",
                    help="where the price came from, and the date")
    ap.add_argument("--out", default="bench.json")
    args = ap.parse_args()

    prompts = json.load(open(args.prompts))
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, quantization=args.quantization or None,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util)
    sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    # ---- warmup: first call includes graph capture / allocator warmup, and
    # including it would inflate p50 on small samples.
    llm.chat([prompts[0]["messages"]], sampling)

    # ---- single-stream --------------------------------------------------
    lat, ptoks, ctoks = [], [], []
    for p in prompts[:args.n_single]:
        t0 = time.perf_counter()
        out = llm.chat([p["messages"]], sampling)[0]
        lat.append((time.perf_counter() - t0) * 1000)
        ptoks.append(len(out.prompt_token_ids))
        ctoks.append(len(out.outputs[0].token_ids))
    lat = np.array(lat)

    # ---- saturated ------------------------------------------------------
    batch = prompts[:args.n_saturated]
    t0 = time.perf_counter()
    outs = llm.chat([p["messages"] for p in batch], sampling)
    wall = time.perf_counter() - t0
    qps = len(batch) / wall
    sat_ptoks = sum(len(o.prompt_token_ids) for o in outs)
    sat_ctoks = sum(len(o.outputs[0].token_ids) for o in outs)

    rec = {
        "label": args.label, "model": args.model,
        "quantization": args.quantization or "none",
        "calls_per_question": 1,
        "single_stream": {
            "n": int(len(lat)),
            "p50_ms": round(float(np.percentile(lat, 50)), 1),
            "p95_ms": round(float(np.percentile(lat, 95)), 1),
            "mean_ms": round(float(lat.mean()), 1),
            "mean_prompt_tokens": round(float(np.mean(ptoks)), 1),
            "mean_completion_tokens": round(float(np.mean(ctoks)), 1),
        },
        "saturated": {
            "n": len(batch), "wall_s": round(wall, 2),
            "queries_per_sec": round(qps, 2),
            "queries_per_gpu_hour": round(qps * 3600, 0),
            "total_prompt_tokens": sat_ptoks,
            "total_completion_tokens": sat_ctoks,
        },
    }

    if args.gpu_hour_price:
        per_1k = args.gpu_hour_price / (qps * 3600) * 1000
        rec["cost"] = {
            "gpu_hour_price_usd": args.gpu_hour_price,
            "price_source": args.price_source,
            "usd_per_1k_queries": round(per_1k, 4),
            "formula": "gpu_hour_price / queries_per_gpu_hour * 1000",
        }

    json.dump(rec, open(args.out, "w"), indent=2)

    s, t = rec["single_stream"], rec["saturated"]
    print(f"\n{args.label}")
    print(f"  calls per question:      {rec['calls_per_question']}")
    print(f"  single-stream p50:       {s['p50_ms']:,} ms")
    print(f"  single-stream p95:       {s['p95_ms']:,} ms")
    print(f"  mean prompt tokens:      {s['mean_prompt_tokens']:,}")
    print(f"  mean completion tokens:  {s['mean_completion_tokens']:,}")
    print(f"  saturated throughput:    {t['queries_per_sec']} q/s "
          f"({t['queries_per_gpu_hour']:,.0f} q/GPU-hour)")
    if "cost" in rec:
        print(f"  cost:                    ${rec['cost']['usd_per_1k_queries']}"
              f" / 1k queries  (at ${args.gpu_hour_price}/GPU-hr)")
        if not args.price_source:
            print("  WARNING: --price-source empty. Cite the price and date or "
                  "the cost claim is not defensible.")
    print(f"\n-> {args.out}")
    print("\nFor the writeup: compare calls-per-question and tokens-per-question "
          "against published multi-call pipelines, and present dollars as a "
          "cited illustration rather than the headline claim.")


if __name__ == "__main__":
    main()
