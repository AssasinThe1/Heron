# Runbook

`[L]` = local (has the databases) · `[C]` = Colab (has the GPU)
The two exchange only flat JSON aligned by `id`. Databases never leave local.

Install locally: `pip install pandas sqlglot tqdm`
Install on Colab: see the header of `run_inference.py` (the cu128 wheel note)
and `train_qlora.py`.


---

## Stage 0 — splits `[L]`

```bash
python make_splits.py --n-val-dbs 10 --subsample 1000 --seed 13
```

Writes `splits/{train_fit,train_val,train_val_1k,manifest}.json`. Asserts no
database appears in both halves. **Do not re-run with a different seed after
you have generated SFT data** — the fit/val boundary would move and the
validation number becomes meaningless.

---

## Stage 1 — failure taxonomy on the current baseline `[L]` + `[C]`

Build this before the schema arms. You are about to optimize against 562
failures; you need to know how many are actually the model's fault.

```bash
# [L]
python build_prompts.py --split train-val-1k --variant ddl
# [C]
python run_inference.py --prompts prompts_train-val-1k_ddl.json --resume
# [L]
python score.py --payload payload_train-val-1k_ddl.json \
                --results results_train-val-1k_ddl.json \
                --ast --guard --out-csv errors_ddl.csv
```

Read in this order:

| Look at | Because |
|---|---|
| `missing_column` split | `wrong_table` → arm (a) is the fix; `bad_name` → the guard is |
| `shape` = `pred_empty` | wrong literal format; direct before/after measure for arm (b) |
| `shape` = `ncol_mismatch` | SELECT-clause disagreement; a chunk are strict-metric artifacts |
| `shape × ast` crosstab | cell counts tell you which arm to build first |
| `EX with repair guard` | free points; report separately, never folded in |

Then eyeball the categories:

```bash
python sample_errors.py --errors errors_ddl.csv --shape pred_empty -n 8
python sample_errors.py --errors errors_ddl.csv --subkind bad_name -n 8
python sample_errors.py --errors errors_ddl.csv --ast join_path -n 8
```

**Gate:** if `ncol_mismatch` + arguable-gold rows are a large share, say so in the
writeup and report a noise-adjusted number alongside strict EX. That finding is
the most differentiated thing in this project — the published audits found BIRD's
EX agrees with human judgment only ~62% of the time.

---

## Stage 2 — schema arms `[L]` + `[C]`

Four arms. `fk+samples` is included because (a) and (b) fix nearly disjoint
failure populations, so testing them only in isolation would leave one class
permanently unaddressed once the format freezes.

```bash
# [L]
for v in ddl fk samples fk+samples; do
  python build_prompts.py --split train-val-1k --variant "$v"
done
# [C]  (one per arm)
python run_inference.py --prompts prompts_train-val-1k_fk+samples.json --resume
# [L]
python score.py --payload payload_train-val-1k_fk+samples.json \
                --results results_train-val-1k_fk+samples.json \
                --ast --guard --out-csv errors_fk+samples.csv

# churn, not just aggregate
python sample_errors.py --errors errors_fk+samples.csv \
       --compare errors_ddl.csv --only regressions
```

Select on **accuracy per prompt token**, not raw accuracy — prompt tokens are
latency and cost, which is what the project sells. `build_prompts.py` prints mean
prompt size per arm; `bench_latency.py` gives exact token counts.

**Gate before freezing:** re-run the top two arms on the 7B base. Format effects
do not always transfer down; smaller models are sometimes hurt by longer, denser
context. This is one extra Colab session protecting the single most irreversible
decision in the project.

```bash
python run_inference.py --prompts prompts_train-val-1k_fk+samples.json \
    --model Qwen/Qwen2.5-Coder-7B-Instruct --quantization '' \
    --max-model-len 12288 --out results_7Bbase_fk+samples.json
```

**Then freeze the variant.** Everything downstream assumes it.

---

## Stage 3 — teacher pass@k probe `[C]` + `[L]`

Cheap measurement that de-risks the most expensive commitment.

```bash
# [L]
python build_prompts.py --split train-fit --variant fk+samples
# [C]  probe on 500 questions only
python run_inference.py --prompts prompts_train-fit_fk+samples.json \
    --n 8 --temperature 0.8 --limit 500 --out candidates_probe.json
# [L]
python build_sft_data.py --candidates candidates_probe.json \
    --payload payload_train-fit_fk+samples.json --cap 3
```

Read `pass@k` and `coverage` from the printout.

**Gate:** if pass@8 is only a few points over greedy, survivors will be thin on
exactly the hard questions the student needs. Enable `--gold-fallback` (rows get
tagged `source:"gold"` so `--exclude-gold` ablates them later) or reconsider SFT
in favour of the repair pass and schema work.

---

## Stage 4 — full generation and SFT `[C]` + `[L]`

```bash
# [C]  the big run; checkpoints every 200
python run_inference.py --prompts prompts_train-fit_fk+samples.json \
    --n 8 --temperature 0.8 --resume
# [L]
python build_sft_data.py --candidates candidates_train-fit_fk+samples.json \
    --payload payload_train-fit_fk+samples.json \
    --cap 3 --gold-fallback --majority-filter
# [C]  measure before you train — do not guess max_seq_len
python measure_lengths.py --input sft_train-fit_fk+samples.jsonl \
    --model Qwen/Qwen2.5-Coder-7B-Instruct --budget 12288
python train_qlora.py --data sft_train-fit_fk+samples.jsonl \
    --max-seq-len 12288 --epochs 2 --merge \
    --drive-dir /content/drive/MyDrive/t2s_ckpt
```

If `measure_lengths.py` reports drops concentrated in one database, fix the
variant's degradation rule and rebuild — do **not** accept silent truncation, and
do not raise the ceiling past what fits.

Select checkpoints on **train-val EX**, not eval loss. Loss and execution
accuracy correlate only loosely: a model can get better at reproducing the
teacher's token distribution while getting worse at producing queries that run.

---

## Stage 5 — final dev run `[C]` + `[L]`

**Second and last look at dev.** Three configurations, one pass each.

```bash
# [L]
python build_prompts.py --split dev --variant fk+samples
# [C]  32B teacher / 7B base / 7B tuned
python run_inference.py --prompts prompts_dev_fk+samples.json --out results_dev_32b.json
python run_inference.py --prompts prompts_dev_fk+samples.json \
    --model Qwen/Qwen2.5-Coder-7B-Instruct --quantization '' --out results_dev_7b_base.json
python run_inference.py --prompts prompts_dev_fk+samples.json \
    --model ./qwen7b-t2s-merged --quantization '' --out results_dev_7b_tuned.json
# [L]
for r in 32b 7b_base 7b_tuned; do
  python score.py --payload payload_dev_fk+samples.json \
      --results results_dev_$r.json --expect-variant fk+samples \
      --ast --guard --out-csv errors_dev_$r.csv
done
python sample_errors.py --errors errors_dev_7b_tuned.csv \
    --compare errors_dev_7b_base.csv --only regressions
```

Latency, on the merged model:

```bash
# [C]
python bench_latency.py --prompts prompts_dev_fk+samples.json \
    --model ./qwen7b-t2s-merged --quantization '' \
    --gpu-hour-price 1.10 --price-source "provider X on-demand A100 40GB, 2026-07-30" \
    --label "7B tuned (merged fp16)" --out bench_7b_tuned.json
```

---

## Reporting checklist

- Pin the **dev split version** (BIRD has re-cleaned dev more than once and
  numbers moved by ~3 points; a cleaner `bird-sql-dev-1106` split exists).
- Report **strict EX** as the headline, plus EX-on-winnable and the
  `ncol_mismatch` count as a measured noise floor.
- Report the repair guard **as its own row**, guarded and unguarded, so nobody
  can claim it is doing the work.
- Report **McNemar** for tuned-vs-base and tuned-vs-teacher.
- Report **calls per question** and **tokens per question** first; dollars second,
  with a cited price and date.
- State the pretraining-contamination limitation explicitly, and that the
  disjoint-database design is the mitigation: what is measured is reading an
  unseen schema cold.

## Expectation setting

The tuned 7B will likely land a few points *below* the 32B teacher, not at it.
After SFT on teacher survivors the student inherits the teacher's interpretation
conventions, so the ceiling is the teacher's accuracy on the student-reachable
subset. Recovering ~7 of the 10pp gap (0.53 → ~0.60) is a success. Note also
that a published 7B (Arctic-Text2SQL-R1-7B, 68.47% dev) already exceeds your
32B baseline — the differentiated contribution here is the harness, the
validation discipline, and the error taxonomy, not the headline number.
