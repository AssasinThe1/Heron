"""Heron on Modal, serving the LoRA adapter on the base model (no merged weights).

Why this exists: the uploaded merged model (AssassinThe1/qwen7b-t2s-ddl-merged) does not
reproduce Heron's offline outputs. On 300 BIRD dev prompts, greedy, vLLM 0.9.2:

    uploaded merged model (A100 or L4, fp16)   identical 94/300   unbalanced brackets ~98/300
    base + LoRA adapter (L4, fp16)             identical 297/300  unbalanced brackets 1/300

This app is a separate "branch" of deploy_modal.py: its own app name and URL, the same
hf-cache Volume. The original `qwen7b-t2s` endpoint is untouched.

  modal run deploy_modal_lora.py::download_model     # once: base + adapter into the Volume
  modal deploy deploy_modal_lora.py                  # -> https://<user>--qwen7b-t2s-lora-serve.modal.run

Clients must request model "heron" (the LoRA name). In PayInsight:
  HERON_URL=https://<user>--qwen7b-t2s-lora-serve.modal.run  HERON_MODEL=heron
"""

import modal

BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER = "AssassinThe1/qwen7b-t2s-ddl"
LORA_NAME = "heron"
GPU = "L4"
SCALEDOWN_SECONDS = 300

app = modal.App("qwen7b-t2s-lora")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    # same stack as deploy_modal.py (vllm 0.9.2 needs transformers < 4.54)
    .pip_install("vllm==0.9.2", "transformers==4.53.3", "huggingface_hub")
    .env({"HF_HOME": "/cache"})
)


@app.function(image=image, volumes={"/cache": hf_cache}, timeout=1800)
def download_model():
    """CPU-only: cache the base model and the adapter in the Volume."""
    from huggingface_hub import snapshot_download
    snapshot_download(BASE)
    snapshot_download(ADAPTER)
    hf_cache.commit()
    print(f"cached {BASE} and {ADAPTER}")


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache": hf_cache},
    scaledown_window=SCALEDOWN_SECONDS,
    max_containers=1,
    timeout=600,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    """vLLM OpenAI server with the adapter registered as model "heron".

    --dtype float16 matches the offline evaluation explicitly. --max-lora-rank 32 is the
    adapter's r. --enforce-eager keeps cold starts short, as in deploy_modal.py."""
    import subprocess
    from huggingface_hub import snapshot_download
    adapter_dir = snapshot_download(ADAPTER)          # local path from the Volume cache
    subprocess.Popen(
        f"vllm serve {BASE} --host 0.0.0.0 --port 8000 --enforce-eager "
        f"--max-model-len 4096 --max-num-seqs 32 --dtype float16 "
        f"--enable-lora --max-lora-rank 32 --max-loras 1 "
        f"--lora-modules {LORA_NAME}={adapter_dir}",
        shell=True,
    )
