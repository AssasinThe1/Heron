# Cost-Efficient Text-to-SQL via Teacher–Student Distillation

A single-call 7B text-to-SQL model, distilled from a 32B teacher on the
[BIRD](https://bird-bench.github.io/) benchmark. The goal is **accuracy per token
and per dollar at one model call**, not leaderboard rank — a model you could
actually put in front of a warehouse.

The differentiated contribution is the **methodology and the findings**: a
disjoint-database validation discipline, a shipped-data-corruption discovery, a
mechanical failure taxonomy, and an honestly-reported distillation result with its
cost tradeoff measured — not the headline accuracy.

## Results — BIRD dev (1,534 questions, execution accuracy)

| metric | 32B teacher | **7B tuned** | 7B advantage |
|---|---|---|---|
| execution accuracy | 0.627 | **0.564** | −6.3 pp |
| calls / question | 1 | **1** | — |
| p50 latency | 13,730 ms | **830 ms** | **16.5× faster** |
| cost / 1k queries | $0.0369 | **$0.0053** | **7.0× cheaper** |

*(A100 40GB @ $1.10/GPU-hr, 2026-08-03. 7B base, untuned, scores 0.537.)*

**The tuned 7B trades 6.3 pp of accuracy for ~16× lower interactive latency and
~7× lower cost — at a single call each.** Tuning is statistically significant
(+41 questions over base, McNemar χ²=8.8), driven by the student distilling the
teacher's *schema-faithfulness*: hard errors more than halved (121 → 58) as it
learned to stop hallucinating columns and join paths.

Full numbers and provenance: [stats.md](stats.md).

## Key findings

- **A packaging bug the benchmark ships with.** One BIRD database (`retail_world`)
  is Microsoft Northwind gold queries run against a *reduced, renamed* `.sqlite` —
  three tables missing, columns renamed. **88% of its questions are unanswerable
  by construction.** This is a *packaging* error (the DB file doesn't match its
  own golds), distinct from the annotation-error class in the literature, and it's
  mechanically detectable — see `audit_gold.py`.
- **Schema-format richness doesn't help a strong model.** Four prompt formats
  (bare DDL, +foreign keys, +sample values, both) come out **statistically tied**
  on the 32B; `ddl` wins on accuracy-per-token because BIRD's DDL already embeds
  foreign keys inline, so extra context is redundant cost.
- **The single-call constraint is the whole cost story.** No agentic loop, no
  multi-call repair — which is what keeps latency and cost flat and comparable.

## Method

- **Disjoint-database splits.** Train is split into `train-fit` / `train-val` *by
  database*, so validation measures reading an *unseen* schema cold — the
  capability that transfers to dev — rather than memorized schema facts.
- **Execution-verified rejection sampling.** The 32B teacher generates 4 candidates
  per question (temp 0.8); only completions whose *result set* matches gold survive
  (12,769 verified rows). Training on verified survivors avoids BIRD's noisy gold.
- **Cardinality-gated value profiler.** Sample values are listed only for
  low-cardinality (enum) columns, under a per-prompt budget. The *rules* are frozen
  on train; the *values* are computed per-database — the leakage boundary is a
  module boundary (`derive_profile_rules.py` → `utils.py`).
- **QLoRA.** 4-bit NF4 base + LoRA adapters on Qwen2.5-Coder-7B, completion-only
  loss, chat-template parity with serving.

## Repository

**Pipeline (local — has the databases):**

| file | role |
|---|---|
| `utils.py` | splits, schema rendering, query execution — shared core |
| `make_splits.py` | disjoint-database train-fit / train-val splits |
| `audit_gold.py` | gold-integrity audit (found `retail_world`) |
| `derive_profile_rules.py` | train-only value-profiler rule derivation |
| `build_prompts.py` | prompt + payload builder (frozen schema variant) |
| `build_sft_data.py` | rejection-sampling → execution-verified SFT dataset |
| `score.py` | execution accuracy + full failure taxonomy |
| `analysis.py` | failure taxonomy: result-shape + AST diff |
| `sample_errors.py` | error sampling and multi-arm comparison |
| `repair.py` | mechanical identifier-repair guard |

**Compute (Colab — has the GPU):**

| file | role |
|---|---|
| `run_inference.py` | vLLM inference (greedy eval or candidate sampling) |
| `train_qlora.py` | QLoRA fine-tune + merge |
| `measure_lengths.py` | tokenizer-exact sequence lengths before training |
| `bench_latency.py` | latency + cost benchmark |

**Docs:** [RUNBOOK.md](RUNBOOK.md) — step-by-step operational guide ·
[stats.md](stats.md) — all findings and numbers · [Notes.md](Notes.md) — journal.

*(`Evaluator.py` is a legacy Day-1 Gemini-API evaluator, superseded by the vLLM
pipeline above; not part of the current flow.)*

## Reproduce

The BIRD data lives outside the repo at `../Data/` (`train/` and `dev/`), from the
[BIRD download](https://bird-bench.github.io/). Then follow [RUNBOOK.md](RUNBOOK.md)
end to end — splits → taxonomy → schema arms → teacher probe → generation + SFT →
QLoRA → dev run. Local needs `pip install pandas sqlglot tqdm`; Colab install notes
are in the script headers.

## Data & artifacts

Generated artifacts are **not committed** (see `.gitignore`) — they total ~360MB.
What's what:

- **Regenerate for free:** `prompts_*` / `payload_*` are deterministic outputs of
  `build_prompts.py` (CPU). `results_*` / `errors_*` / `bench_*` re-run from
  `run_inference.py` / `score.py` / `bench_latency.py` (needs a GPU).
- **Committed on purpose:** `splits/*.json` and `profile_rules.json` — small, and
  they pin the exact train/val boundary and frozen profiler rules that everything
  downstream depends on. Reproducing the split requires the seed **and** the
  exclusion lists recorded in `splits/manifest.json`.
- **Not reproducible → on Drive:** `candidates_train-fit_ddl.json` (teacher
  generations, temperature 0.8) and `sft_train-fit_ddl.jsonl` (the derived
  training set) cannot be recreated exactly, along with the trained LoRA adapter.

**Exact artifacts** — the non-reproducible teacher generations
(`candidates_train-fit_ddl.json`), the SFT training set
(`sft_train-fit_ddl.jsonl`), and the run outputs (`results_*`, `errors_*`,
`bench_*`): [Google Drive ↗](https://drive.google.com/drive/folders/1IxN3reqrVu649hPq1QUdHyXpULmu6btd?usp=sharing)

## Limitations

- **The headline accuracy is modest** and well below SOTA 7B (Arctic-Text2SQL-R1-7B,
  68.5% dev). This project optimizes cost-at-fixed-simplicity and methodology, not
  peak accuracy.
- **Pretraining contamination** can't be excluded for a public benchmark; the
  disjoint-database design is the mitigation, not a guarantee.
- **The student's ceiling is the teacher.** ~30% of the base→teacher gap was
  recovered; the rest is bounded by the teacher's accuracy on the reachable subset.
- The largest remaining failure class is **schema linking** (`table_set`), not
  prompt format — the natural next direction.
