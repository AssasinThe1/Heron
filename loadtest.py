#!/usr/bin/env python3
"""Open-loop load test against a deployed OpenAI-compatible endpoint. LOCAL.

Fires requests at a fixed ARRIVAL RATE (Poisson gaps) regardless of whether prior
ones have returned -- the open-loop discipline that reveals real tail latency as
queues build and the KV cache fills. This is the p99-under-load the offline chunk
sweep in bench_latency.py could not give (that one was closed-loop + chunk-batch
artifacts).

Sweeps request rates and reports achieved throughput + p50/p95/p99 per rate. Run
it from your laptop against the public Modal URL -- the numbers then include real
network RTT, which is honest (it's what a user experiences), just don't compare
them to on-box vLLM numbers.

Install:  pip install aiohttp numpy
Example
-------
  python loadtest.py --url https://xxx-serve.modal.run \
      --model AssassinThe1/qwen7b-t2s-ddl-merged \
      --prompts prompts_dev_ddl.json --rates 2 5 10 20 40 --n 200 --max-tokens 256 \
      --out load_dev.json
"""

import argparse
import asyncio
import json
import random
import time

import numpy as np


async def _one(session, url, model, messages, max_tokens, out):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0}
    t0 = time.perf_counter()
    ok = False
    try:
        async with session.post(f"{url}/v1/chat/completions", json=body) as r:
            await r.read()
            ok = r.status == 200
    except Exception:
        ok = False
    out.append(((time.perf_counter() - t0) * 1000.0, ok))


async def _run_rate(url, model, prompts, rate, n, max_tokens):
    import aiohttp
    out, tasks = [], []
    conn = aiohttp.TCPConnector(limit=0)          # no client-side cap
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s:
        # warm the endpoint so a cold start doesn't poison the first rate's tail
        await _one(s, url, model, prompts[0]["messages"], max_tokens, [])
        t_start = time.perf_counter()
        for i in range(n):
            msgs = prompts[i % len(prompts)]["messages"]
            tasks.append(asyncio.create_task(
                _one(s, url, model, msgs, max_tokens, out)))
            await asyncio.sleep(random.expovariate(rate))   # Poisson arrivals
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - t_start
    lat = np.array([x[0] for x in out])
    errs = sum(1 for x in out if not x[1])
    return {"target_rate": rate, "n": n, "wall_s": round(wall, 2),
            "achieved_qps": round(n / wall, 2),
            "p50_ms": round(float(np.percentile(lat, 50)), 1),
            "p95_ms": round(float(np.percentile(lat, 95)), 1),
            "p99_ms": round(float(np.percentile(lat, 99)), 1),
            "errors": errs}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="base URL, no trailing /v1")
    ap.add_argument("--model", required=True, help="model id the endpoint serves")
    ap.add_argument("--prompts", required=True, help="prompts_*.json (id+messages)")
    ap.add_argument("--rates", type=float, nargs="+", default=[2, 5, 10, 20, 40],
                    help="target arrival rates (req/s) to sweep")
    ap.add_argument("--n", type=int, default=200, help="requests per rate")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prompts = json.load(open(args.prompts))
    url = args.url.rstrip("/")

    print(f"\nopen-loop load test -> {url}  ({args.n} req/rate)")
    print(" target  achieved     p50     p95     p99   errors")
    sweep = []
    for rate in args.rates:
        r = asyncio.run(_run_rate(url, args.model, prompts, rate, args.n,
                                  args.max_tokens))
        sweep.append(r)
        print(f" {r['target_rate']:>6}  {r['achieved_qps']:>8.1f}  "
              f"{r['p50_ms']:>6.0f}  {r['p95_ms']:>6.0f}  {r['p99_ms']:>6.0f}  "
              f"{r['errors']:>6}")

    # saturation = last rate where achieved ~ target (queue not yet exploding)
    sat = None
    for r in sweep:
        if r["achieved_qps"] >= 0.9 * r["target_rate"] and r["errors"] == 0:
            sat = r["target_rate"]
    print(f"\nsustains up to ~{sat} req/s before achieved throughput or errors "
          "diverge from target")

    if args.out:
        json.dump({"url": url, "model": args.model, "sweep": sweep,
                   "sustained_rate": sat}, open(args.out, "w"), indent=2)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
