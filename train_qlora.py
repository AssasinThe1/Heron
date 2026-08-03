#!/usr/bin/env python3
"""QLoRA fine-tune Qwen2.5-Coder-7B on execution-verified SQL. COLAB.

Full fine-tuning a 7B will not fit in 40GB (weights + Adam states + activations
runs well past 100GB), so this is 4-bit NF4 base + LoRA adapters. The output
artifact is a ~150MB adapter rather than a 15GB checkpoint, which is also much
easier to move and to serve.

Three things here matter more than any hyperparameter:

1. CHAT TEMPLATE PARITY. Training uses tokenizer.apply_chat_template, the exact
   same path vLLM's llm.chat() takes at serving time. A template mismatch between
   training and serving is the classic silent way to lose several points and
   never find out why.

2. COMPLETION-ONLY LOSS. Loss is masked to the SQL tokens. Without masking, most
   of the gradient signal goes into reproducing schema text, which the model has
   no reason to learn and which drowns out the actual task.

3. DROP, NEVER TRUNCATE. Rows exceeding --max-seq-len are dropped and logged.
   Truncating would cut the tail off a schema block and teach column
   hallucination -- the exact failure being fixed.

Select on train-val execution accuracy, not eval loss. Loss and EX correlate
only loosely here: a model can get better at reproducing the teacher's token
distribution while getting worse at producing queries that execute correctly.

Install (Colab):
  pip install -q "torch" --extra-index-url https://download.pytorch.org/whl/cu128
  pip install -q transformers peft trl bitsandbytes accelerate datasets

Example
-------
  python train_qlora.py --data sft_train-fit_fk+samples.jsonl \
      --max-seq-len 12288 --out-dir ./qwen7b-t2s --epochs 2
"""

import argparse
import json
import os


def build_dataset(path, tok, max_seq_len, drop_log):
    """Render each row to the serving-time chat format, mask to completion."""
    from datasets import Dataset

    rows = [json.loads(l) for l in open(path) if l.strip()]
    kept, dropped = [], []
    for r in rows:
        prompt = tok.apply_chat_template(r["messages"], tokenize=False,
                                         add_generation_prompt=True)
        completion = r["completion"].strip() + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(completion, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(c_ids) > max_seq_len:
            dropped.append({"id": r.get("id"), "db_id": r.get("db_id"),
                            "n_tokens": len(p_ids) + len(c_ids)})
            continue
        input_ids = p_ids + c_ids
        # -100 masks a position out of the loss. Everything before the
        # completion is context, not a prediction target.
        labels = [-100] * len(p_ids) + c_ids
        kept.append({"input_ids": input_ids, "labels": labels,
                     "attention_mask": [1] * len(input_ids)})

    if dropped:
        json.dump(dropped, open(drop_log, "w"), indent=2)
        from collections import Counter
        by_db = Counter(d["db_id"] for d in dropped)
        print(f"DROPPED {len(dropped)} rows over {max_seq_len} tokens "
              f"-> {drop_log}")
        for db, n in by_db.most_common(5):
            print(f"    {db}: {n}")
        print("    If drops concentrate in one db, fix the schema variant's "
              "degradation rule rather than accepting the loss.")
    print(f"training rows: {len(kept)}")
    return Dataset.from_list(kept)


def collate(features, pad_id):
    """Right-pad to the longest sequence in the batch; pad labels with -100."""
    import torch
    n = max(len(f["input_ids"]) for f in features)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for f in features:
        d = n - len(f["input_ids"])
        out["input_ids"].append(f["input_ids"] + [pad_id] * d)
        out["labels"].append(f["labels"] + [-100] * d)
        out["attention_mask"].append(f["attention_mask"] + [0] * d)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="sft_*.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--out-dir", default="./qwen7b-t2s")
    ap.add_argument("--drive-dir", default=None,
                    help="e.g. /content/drive/MyDrive/t2s_ckpt -- checkpoint here, "
                         "not to Colab local disk; losing a training run costs hours")

    ap.add_argument("--max-seq-len", type=int, required=True,
                    help="from measure_lengths.py. Do not guess.")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=32,
                    help="effective batch = batch_size * grad_accum")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--exclude-gold", action="store_true",
                    help="ablation: drop source=='gold' fallback rows")
    ap.add_argument("--merge", action="store_true",
                    help="after training, merge adapter into base and save fp16")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    data_path = args.data
    if args.exclude_gold:
        data_path = args.data.replace(".jsonl", ".teacher_only.jsonl")
        with open(data_path, "w") as f:
            for line in open(args.data):
                if json.loads(line).get("source") == "teacher":
                    f.write(line)
        print(f"ablation: gold rows excluded -> {data_path}")

    os.makedirs(args.out_dir, exist_ok=True)
    ds = build_dataset(data_path, tok, args.max_seq_len,
                       os.path.join(args.out_dir, "dropped_rows.json"))

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,      # ~0.4 bits/param saved, free
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0},
        attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model,
                                            use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        bias="none", task_type="CAUSAL_LM",
        # ALL linear projections, not attention-only. Attention-only LoRA
        # measurably underperforms when the task requires shifting behaviour
        # (which columns to use, which joins to take) rather than style.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.drive_dir or args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=lambda f: collate(f, tok.pad_token_id),
    )
    trainer.train()

    adapter = os.path.join(args.out_dir, "adapter")
    model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    print(f"adapter -> {adapter}")

    if args.merge:
        # Merged fp16 weights are faster to serve than runtime LoRA and are the
        # configuration you would actually deploy, so latency numbers should be
        # measured on the merged model.
        del model, trainer
        torch.cuda.empty_cache()
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="cpu",
            trust_remote_code=True)
        merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
        mpath = args.out_dir + "-merged"
        merged.save_pretrained(mpath, safe_serialization=True)
        tok.save_pretrained(mpath)
        print(f"merged fp16 -> {mpath}")
        print("Serve this with run_inference.py --model <path> --quantization ''")


if __name__ == "__main__":
    main()
