#!/usr/bin/env python3
"""Measure latency and derive cost. COLAB (needs the GPU).

Reports THREE regimes, explicitly labelled, because conflating them is how these
comparisons get attacked:

  single-stream   batch size 1, one query at a time. This is what one user
                  experiences. Report p50/p95.
  saturated       all queries submitted at once, vLLM batches freely. This is
                  what throughput and therefore cost derive from.
  concurrency     a sweep C = 1,2,4,... The load pool is fed in chunks of C so
                  vLLM sees exactly C in flight; per-request end-to-end and
                  time-to-first-token latency come from RequestOutput.metrics.
                  This is the curve between the two above: where throughput
                  saturates and where p99/TTFT knee upward as the KV cache fills.

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
    ap.add_argument("--concurrency", type=int, nargs="*",
                    default=[1, 2, 4, 8, 16, 32, 64],
                    help="in-flight request counts to sweep")
    ap.add_argument("--n-load", type=int, default=256,
                    help="queries processed per concurrency level")
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

    # ---- concurrency sweep ----------------------------------------------
    def pct(a, q):
        return round(float(np.percentile(a, q)), 1) if a else None

    # Fixed-size load pool per level so throughput is a steady-state measure,
    # not a single-batch fluke. Feeding it in chunks of C pins in-flight count.
    pool = (prompts * (args.n_load // max(1, len(prompts)) + 1))[:args.n_load]
    sweep = []
    for c in args.concurrency:
        e2e, ttft, ctok = [], [], 0
        ct0 = time.perf_counter()
        for i in range(0, len(pool), c):
            chunk = pool[i:i + c]
            k0 = time.perf_counter()
            outs = llm.chat([p["messages"] for p in chunk], sampling)
            k_wall = (time.perf_counter() - k0) * 1000
            for o in outs:
                ctok += len(o.outputs[0].token_ids)
                m = getattr(o, "metrics", None)
                if m and m.arrival_time and m.finished_time:
                    e2e.append((m.finished_time - m.arrival_time) * 1000)
                    if m.first_token_time:
                        ttft.append((m.first_token_time - m.arrival_time) * 1000)
                else:
                    e2e.append(k_wall)   # metrics absent: coarse chunk-wall proxy
        c_wall = time.perf_counter() - ct0
        sweep.append({
            "concurrency": c, "n": len(pool), "wall_s": round(c_wall, 2),
            "queries_per_sec": round(len(pool) / c_wall, 2),
            "tokens_per_sec": round(ctok / c_wall, 1),
            "p50_ms": pct(e2e, 50), "p95_ms": pct(e2e, 95),
            "p99_ms": pct(e2e, 99), "ttft_p95_ms": pct(ttft, 95),
        })
    # Saturation = first level whose throughput gain over the previous is <10%.
    sat_c = sweep[-1]["concurrency"]
    for a, b in zip(sweep, sweep[1:]):
        if b["queries_per_sec"] < a["queries_per_sec"] * 1.10:
            sat_c = a["concurrency"]
            break
    peak_qps = max(r["queries_per_sec"] for r in sweep)

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
        "concurrency_sweep": sweep,
        "saturation_concurrency": sat_c,
        "peak_queries_per_sec": round(peak_qps, 2),
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

    print("\n  concurrency sweep (end-to-end ms under load):")
    print("     C      q/s    tok/s     p50     p95     p99   TTFT-p95")
    for r in rec["concurrency_sweep"]:
        g = lambda k: f"{r[k]:.0f}" if r[k] is not None else "n/a"
        print(f"   {r['concurrency']:>3}  {r['queries_per_sec']:>7.1f}  "
              f"{r['tokens_per_sec']:>7.0f}  {g('p50_ms'):>6}  {g('p95_ms'):>6}  "
              f"{g('p99_ms'):>6}  {g('ttft_p95_ms'):>8}")
    print(f"  saturation ~ C={rec['saturation_concurrency']}  "
          f"(peak {rec['peak_queries_per_sec']:.1f} q/s)")
    if all(r["ttft_p95_ms"] is None for r in rec["concurrency_sweep"]):
        print("  WARNING: per-request metrics did not populate (TTFT n/a). The "
              "under-load p50/p95/p99 above are chunk-batch times, not per-request\n"
              "  latency -- the tail is unreliable (one long generation taints its "
              "whole chunk). For a defensible p99-under-load, benchmark a live\n"
              "  `vllm serve` endpoint with vLLM's benchmark_serving.py "
              "(open-loop --request-rate). Single-stream and throughput above stand.")

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
