#!/usr/bin/env python3
"""Merge a LoRA adapter into its base and optionally push the merged fp16 model
to the Hugging Face Hub. COLAB (needs enough RAM or a GPU).

Why this is separate from train_qlora.py --merge: the adapter now lives on the
Hub, not local disk, so this recreates the *servable* full model from the
published adapter without retraining. vLLM / serverless pull the merged model at
container start; a LoRA adapter alone is not servable by vLLM directly.

GOTCHA that bit us once: an old `torchao` (0.10) makes peft's LoRA dispatcher
raise `ImportError` on adapter load. Uninstall it first:  pip uninstall -y torchao

Install:  pip install -U transformers peft accelerate huggingface_hub
Login:    huggingface-cli login        (needs a WRITE token)

Memory: fp16 7B needs ~15GB. --device cpu wants ~15GB system RAM (High-RAM
Colab); --device cuda wants a ~24GB+ GPU (L4/A100). Use whichever you have.

Example
-------
  python merge_and_push.py \
      --adapter AssassinThe1/qwen7b-t2s-ddl \
      --push-to AssassinThe1/qwen7b-t2s-ddl-merged
"""

import argparse


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--adapter", required=True,
                    help="local path OR Hugging Face repo id of the LoRA adapter")
    ap.add_argument("--out", default="./qwen7b-t2s-merged",
                    help="where to save the merged fp16 model locally")
    ap.add_argument("--push-to", default=None,
                    help="HF repo id to push the merged model to (optional)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cpu (~15GB RAM) or cuda (~24GB VRAM)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"loading base {args.base} in fp16 on {args.device} ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map=args.device)

    print(f"applying + merging adapter {args.adapter} ...")
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()

    merged.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"merged fp16 -> {args.out}")

    if args.push_to:
        print(f"pushing to the Hub as {args.push_to} (~15GB upload) ...")
        merged.push_to_hub(args.push_to)
        AutoTokenizer.from_pretrained(args.out).push_to_hub(args.push_to)
        print(f"done -> https://huggingface.co/{args.push_to}")
    else:
        print("(no --push-to; merged saved locally only)")


if __name__ == "__main__":
    main()
