"""Serverless deployment of the tuned 7B on Modal — the serving-economics lab.

Serves an OpenAI-compatible vLLM endpoint that SCALES TO ZERO, with the fp16
merged model cached on a persistent Volume so cold starts skip the 15GB Hugging
Face download (the slow, variable part). Idle cost ~$0; you pay GPU-seconds only
while a request is being served.

The economics levers, all here and commented:
  * VOLUME cache      -> removes the 15GB download from cold start
  * SCALEDOWN_WINDOW  -> scale-to-zero: pay nothing idle, pay a cold start on wake
  * GPU choice        -> L4 (24GB) fits the 15GB fp16 model + KV cache, cheap

--------------------------------------------------------------------------------
NOTE: Modal's SDK renames things between versions. If `modal deploy` errors on a
name, check current docs -- common renames: `scaledown_window` was
`container_idle_timeout`; `gpu="L4"` vs `gpu=modal.gpu.L4()`; `@modal.web_server`
vs `@modal.asgi_app`. The shape below is right; the spelling may need a nudge.
--------------------------------------------------------------------------------

Prereqs
  pip install modal
  modal token new                        # free account, one-time
Pre-cache the weights into the Volume (run ONCE; CPU-only, no GPU):
  modal run deploy_modal.py::download_model
Deploy the endpoint (prints a public URL):
  modal deploy deploy_modal.py

Measure COLD START (client-side, the honest number):
  # wait > SCALEDOWN_SECONDS so it scales to zero, then time the first hit:
  time curl -s $URL/v1/chat/completions -H 'content-type: application/json' \
       -d '{"model":"'$MODEL'","max_tokens":256,"messages":[...]}'
  # immediately repeat for the WARM number; the difference is your cold-start tax.

Load test (open-loop, the REAL p99 the offline sweep couldn't give): point an
open-loop client (or vLLM's bench serve) at $URL at rising --request-rate.
"""

import modal

MODEL = "AssassinThe1/qwen7b-t2s-ddl-merged"   # fp16 merged model
GPU = "L4"                    # 24GB: fits the 15GB fp16 model + KV cache, cheap
SCALEDOWN_SECONDS = 300       # scale to zero after 5 min idle

app = modal.App("qwen7b-t2s")

# Persistent weight cache. First run writes 15GB here; every later cold start
# reads it locally instead of pulling from HF. THIS makes cold start cheap.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    # The exact stack that served on Colab: vllm 0.9.2 + a pre-`aimv2` transformers.
    # aimv2 landed in transformers 4.54 and vllm 0.9.2 double-registers it -> crash,
    # so pin transformers explicitly below 4.54 (a bare install pulls 4.54+).
    .pip_install("vllm==0.9.2", "transformers==4.53.3", "huggingface_hub")
    .env({"HF_HOME": "/cache"})
)


@app.function(image=image, volumes={"/cache": hf_cache}, timeout=1800)
def download_model():
    """Run once: pull the model into the Volume and commit it.

    No `gpu=` argument -> Modal runs this on a cheap CPU box. Downloading weights
    is pure network/disk; a GPU would sit idle and cost money."""
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL)
    hf_cache.commit()
    print(f"cached {MODEL} into the Volume")


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache": hf_cache},
    scaledown_window=SCALEDOWN_SECONDS,   # <- scale-to-zero window (see NOTE)
    max_containers=1,                     # ONE GPU. Load then hits vLLM's own
                                          # batching (what we want to measure),
                                          # not Modal fan-out to N GPUs / the quota.
                                          # (older Modal: concurrency_limit=1)
    timeout=600,
)
@modal.concurrent(max_inputs=32)          # let the one container take 32 concurrent
                                          # requests so vLLM BATCHES them. Without
                                          # this Modal serializes 1-at-a-time -> the
                                          # queue + 180s timeouts you just saw.
                                          # (older Modal: allow_concurrent_inputs=32
                                          #  as an @app.function arg instead)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    """Launch vLLM's OpenAI-compatible server; Modal proxies port 8000 to a URL.

    --max-model-len 4096 keeps the KV cache light (dev prompts are ~1.1k tokens).
    Generation length is capped per-request by the client's `max_tokens` (use 256
    -- SQL is ~60 tokens, and it bounds the tail we saw in the offline load test)."""
    import subprocess
    # --enforce-eager skips torch.compile (~80-100s) + CUDA-graph capture (~40s),
    # which dominate cold start. For a scale-to-zero endpoint that's the right
    # trade: ~150s init -> ~15-30s, at a modest steady-state decode cost. (Drop it
    # for an always-on / high-throughput deployment, where the compile pays off.)
    subprocess.Popen(
        f"vllm serve {MODEL} --host 0.0.0.0 --port 8000 --enforce-eager "
        f"--max-model-len 4096 --max-num-seqs 32",
        shell=True,
    )
