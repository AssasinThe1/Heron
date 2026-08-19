# Stats & Findings

Consolidated record of results and decisions. Numbers are from the artifacts in
this folder (the `errors_*.csv`, `splits/manifest.json`, `profile_rules.json`,
and the audit). This is the project's reference sheet — every number with its source.

Last updated: 2026-08-02.

---

## Models

| Role | Model | Notes |
|---|---|---|
| **Teacher** | `Qwen2.5-Coder-32B-Instruct-AWQ` | default in `run_inference.py`. Every number below without an explicit "7B" is this model. |
| Student (base) | `Qwen2.5-Coder-7B-Instruct` | not yet run for the arm gate |
| Student (tuned) | `./qwen7b-t2s-merged` | not yet built (Stage 4) |

All eval is **greedy** (temp 0, n=1). Data generation will be n=8, temp 0.8.

---

## Headline numbers

| Set | Questions | EX | Notes |
|---|---|---|---|
| train raw (pre-clean), 32B | 1000 | **0.4980** | dragged down by `retail_world` |
| train, winnable only, 32B | 799 | 0.6233 | excludes broken gold |
| **train, excl. `retail_world`, 32B** | 774 | **0.6111** | the real baseline |
| train-val-1k, clean, 32B (ddl) | 887 | **0.6674** | teacher working baseline |
| **dev — 32B teacher** (ddl) | 1534 | **0.6271** | exact `ddl` greedy under the frozen prompt builder |
| **dev — 7B base** | 1534 | 0.5372 | untuned |
| **dev — 7B tuned (QLoRA, ddl)** | 1534 | **0.5639** | +41 q vs base, McNemar χ²=8.8 **SIG** |

train-val (0.667) and dev-32B (0.627) land ~4pp apart across disjoint database
sets → the split is a sound proxy and the harness is correct end-to-end. **Carry
0.56–0.63 forward for the 7B story, not the 32B's 0.667.**

---

## Broken data

### `retail_world` — corrupted database (EXCLUDED)

The single biggest data problem. It is Microsoft **Northwind**, but the shipped
`.sqlite` is a reduced, renamed variant while the gold queries were written
against canonical Northwind.

- **199 of 201 broken golds in the whole benchmark are in this one db** (4.3×
  enrichment over chance: 197 co-occurrences vs 46 expected).
- **199 of 226 questions (88%) are unanswerable by construction.**
- Gold expects `Order Details`, `EmployeeTerritories`, `Territories`, `Region`,
  `CompanyName`, `Salary`, `Title`, `Discontinued`, `UnitsInStock`,
  `TitleOfCourtesy`; the file has `orderdetails` (renamed) and is **missing three
  whole tables** (`Territories`, `EmployeeTerritories`, `Region`).
- This is a **packaging error** (db file doesn't match its own golds), distinct
  from the annotation-error class in the literature (wrong gold for the right
  db). That distinction is a differentiated finding for the writeup.
- The model scored **25/27 (93%) on the winnable `retail_world` questions** — it
  handles the schema fine. The database was the problem, not the model.
- Explains the repair guard's +0 here: the model invents `Territories` /
  `CompanyName` because the *question* asks for them; they aren't typos of
  anything in the file, so the guard correctly declines. Guard works; data was
  broken.

### Scattered breakage

- **31 broken question ids** across otherwise-sound databases (ordinary
  annotation noise) — excluded at the question level.
- `shooting`: 2 broken golds, unrelated to `retail_world` (gold references
  `incidents`, which exists) — column-level noise.

### Audit tool

`audit_gold.py` runs every gold once, two tiers:
- `corrupted_dbs` (fail_rate ≥ 0.5 → drop whole db): **`["retail_world"]`**
- `broken_ids` (individual failures outside corrupted dbs): **31**
- `referenced_but_absent_tables` per db (the diagnostic that made `retail_world`
  obvious at a glance).

**OPEN:** dev has **not** been audited yet (run was interrupted). A packaging
casualty in dev would be more significant than the train one, since dev is what
the leaderboard reports. `python audit_gold.py --split dev --out audit_dev.json`.

---

## Splits (current)

Regenerated with `retail_world` and the 31 ids excluded:

```
train-fit:  7189 questions over 58 dbs
train-val:  1835 questions over 10 dbs
train-val-1k: 887 questions (fixed subsample)
```

### Held-out validation databases (10)

| db | tables | questions |
|---|---|---|
| airline | 3 | 92 |
| craftbeer | 2 | 6 |
| language_corpus | 7 | 136 |
| mondial_geo | 34 | 293 |
| movie_platform | 5 | 167 |
| shooting | 3 | 24 |
| simpson_episodes | 7 | 210 |
| soccer_2016 | 21 | 258 |
| synthea | 11 | 185 |
| works_cycles | 66 | 464 |

### Why train-val-1k is 887, not 1000

The 15%-per-db cap (`int(1000*0.15)=150`) clips the two widest val dbs:
`works_cycles` 253→150 (−103) and `mondial_geo` 160→150 (−10) = **−113**. The
subsampler trims when over target but doesn't redistribute freed budget, so it
lands at 887. Accepted as-is (still fixed, stratified, reproducible). A
redistribute-to-1000 option exists in the discussion but was not applied.

`make_splits.py` now supports `--exclude-dbs` and `--exclude-ids`; both recorded
in `splits/manifest.json` for reproducibility.

---

## Schema-format arms (32B, train-val-1k, 887 q)

| arm | EX | correct | hard_err | truncated | timeouts |
|---|---|---|---|---|---|
| **ddl** | **0.6674** | 592/887 | 30 | 0 | 4 |
| samples | 0.6629 | 588/887 | 33 | 0 | 6 |
| fk | 0.6584 | 584/887 | 30 | 0 | 4 |
| fk+samples | 0.6539 | 580/887 | 40 | 6 | 6 |

### The ranking is within noise — McNemar vs ddl

| arm | +gained | −lost | net | χ² (1 dof) | verdict |
|---|---|---|---|---|---|
| fk | 6 | 14 | −8 | 2.45 | not significant |
| samples | 20 | 24 | −4 | 0.20 | not significant |
| fk+samples | 19 | 31 | −12 | 2.42 | not significant |

**All four formats are statistically tied.** "ddl highest" is not a real
ordering — the 1.35pp spread is inside the error bars.

### Consensus over 887

- all arms correct: **556 (62.7%)**
- all arms wrong: **268 (30.2%)** ← unreachable by any format
- contested: **63 (7.1%)** ← the only format-sensitive questions
- best-per-question oracle ceiling: **0.6979** (+3.0pp over ddl, not achievable
  without an oracle)
- unique wins (only this arm right): ddl 7, samples 5, fk+samples 5, fk 2

### Per-db winners (where formats disagree)

`shooting`→fk+samples, `works_cycles`→ddl, `movie_platform`→samples,
`language_corpus`→ddl, `airline`→fk, `simpson_episodes`→ddl, `soccer_2016`→samples,
`synthea`→ddl, `mondial_geo`→samples.

### Why context doesn't help

- **fk is redundant** — BIRD's DDL already embeds `references` inline, so the
  explicit FK block re-states known info. Pure token cost.
- Only 7.1% of questions are format-sensitive at all.
- Added context mildly *distracts* a model this strong; samples' enum values are
  the kind of signal a weaker 7B is more likely to need.

### Decision

- **ddl wins on accuracy-per-token** (best nominal EX *and* cheapest prompt).
- **Drop `fk` and `fk+samples`** (dominated; fk+samples owns the only truncations).
- **Gate `ddl` vs `samples` on the 7B base before freezing** — samples is the
  closest tie (χ²=0.20) with a mechanism to help down-scale. If it doesn't beat
  ddl on the 7B, freeze ddl.
- The real headroom is the **30.2% all-wrong bucket**, not format — that's
  SFT/repair territory.

Compare arms anytime: `python sample_errors.py --arms errors_ddl.csv errors_samples.csv --baseline ddl`

---

## Value profiler (the `samples` variant)

Cardinality-gated, replaces the naive "3 values per column". Rules frozen on
**train only** in `profile_rules.json`; values computed per-db at build time
(incl. dev) — the rules/values split is the leakage boundary.

**Frozen rules:** `K=10` (enum cutoff), `value_budget=60` (per-prompt),
`scan_cap=20000`, `str_trunc=20`. Derived on train-fit (58 dbs, 2029 listable
columns).

### Cardinality histogram (train-fit, non-id non-date cols)

- **32.5% of columns are 200+ distinct** → skipped (the pure token waste).
- ~38% sit at ≤20 distinct (the enum band).
- by kind: text 1131, numeric 686, date 212. id cols skipped: 534.

### Payoff on `works_cycles` (66 tables)

| variant | chars | ~tokens |
|---|---|---|
| naive fk+samples | 59,001 | 17,415 (measured) |
| **profiled** (K=10/60) | 42,207 | ~9,500 |
| ddl floor | 38,174 | ~8,000 |

The value bloat is gone; the ~8k-token 66-table DDL floor is structural.
**Consequence: run the arms at `--max-model-len 12288`** (fits everything;
default 8192 rejected the naive works_cycles prompt outright).

Build cost is negligible: 58 dbs profiled in ~5s at `scan_cap=20000`.

---

## Teacher rejection-sampling probe (Stage 3)

500-question probe on **train-fit**, ddl, n=8, temp 0.8. (First aborted attempt
ran on a stale train-val-1k prompts file — the probe must run on train-fit, which
is the data the student is allowed to learn.)

### pass@k curve

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| pass@k | 0.750 | 0.776 | 0.784 | **0.800** | 0.816 | 0.824 | 0.824 | 0.832 |

Most of the lift lands by **k=4** (0.800); k=5→8 buys 1.6pt total, k=6→7 nothing.
**Decision: k=4 for the full run** — ~60% of the available lift at half the
compute, and less label noise.

### Survivor funnel (k=8, cap=3)

- candidates seen: 4000 → execution-correct: 2991 → after dedup: **1124** (62% of
  survivors canonically identical) → after cap=3: **942** (~2.3 rows/covered q).
- covered by teacher: **416/500 (83.2%)**; uncovered: 84.

### CAVEATS — do not extrapolate the 83.2% naively

- **The probe subset is NOT stratified.** `--limit 500` took the first 500 by
  orig_id → only **4 of 58 train-fit dbs**: `retail_complains` 168, `movies_4`
  158, `codebase_comments` 101, `book_publishing_company` 73. Coverage is over 4
  (easy) dbs → an **upper bound**. Real full-run coverage will be lower.
- This also explains **k=1 = 0.750 vs eval 0.667**: train-fit is a different,
  easier split than train-val-1k, and these 4 dbs are ones the teacher handles
  well. Sampling at 0.8 is not beating greedy on like-for-like.
- Extrapolation base is **7,189** (train-fit), not 9,024. At a realistic ~75%
  coverage × ~2.3 rows ≈ **~12–13k teacher rows** before gold fallback — ample
  for QLoRA on a 7B.

### Full run settings

`--n 4 --temperature 0.8`, then `build_sft_data.py --cap 3 --majority-filter
--gold-fallback`. `--majority-filter` is on because pure set-equality lets some
survivors match by coincidence; preferring the modal canonical form is a cheap
correctness proxy. The **84 uncovered** overlap heavily with the 30.2%
universal-fail population — gold fallback (tagged `source:"gold"`) covers them for
training, but they're what the student will never learn well from this teacher.

---

## Full generation run (Stage 4) — the SFT dataset

Full rejection sampling over train-fit, ddl, **n=4**, temp 0.8. 2h51m of query
execution. Output: `sft_train-fit_ddl.jsonl` (+ `.stats.json`).

| stage | count | note |
|---|---|---|
| questions | 7189 | train-fit |
| broken gold (skipped) | 17 | see below — likely timeout-flaky, not the excluded 31 |
| evaluable | 7172 | ×4 = 28,688 candidates (exact) |
| execution-correct | 19,771 | |
| after dedup | 11,578 | **−41% — dedup does the real work** |
| after cap=3 | 11,066 | cap removed only 512; nearly non-binding |
| covered by teacher | **5,469 (76.2%)** | |
| uncovered → gold fallback | 1,703 | tagged `source:"gold"` |
| **training rows** | **12,769** | 11,066 teacher + 1,703 gold |

pass@k (full): k1 **0.686**, k2 0.732, k3 0.752, k4 **0.762**.

### Probe over-promised (as predicted by the 4-db concentration)

| | probe (4 easy dbs) | full (58 dbs) |
|---|---|---|
| k=1 | 0.750 | 0.686 |
| k=4 | 0.800 | 0.762 |

Full k=1 0.686 ≈ eval 0.667 — the reassuring cross-check. **Lesson: probe on a
stratified subset next time, not the first-N by orig_id.**

### Findings

- **Dedup >> cap at k=4.** 41% of execution-correct survivors are canonically
  identical; the teacher's correct answers are mostly the *same* query, not
  diverse phrasings. Multiple samples buy coverage, not much phrasing diversity.
- **Hard-tail under-weighting.** Covered ≈2.0 rows/q, uncovered (gold) =1 row/q,
  so uncovered = 23.7% of questions but only 13.3% of rows. NOT fixed by
  upweighting yet: gold is the noisy signal, so 2× gold doubles noise on the
  hardest questions. Plan: train as-is → ablate gold-fallback (teacher-only) →
  only upweight if gold helps.
- **17 broken golds** in train-fit — **confirmed timeout-flaky**, not genuine
  breakage. A standalone `audit_gold --split train-fit` found **0 broken golds**
  (all 7,189 execute cleanly); the 17 only failed under the load of the 2h51m run
  (concurrent candidate execution pushed heavy golds past `--timeout 30`). A
  different set from the excluded 31 (all 31 correctly removed, 0 residual).
  Broken gold is skipped entirely (no teacher rows, no fallback), so these are 17
  winnable questions dropped — negligible (17/7,189); recoverable with
  `--timeout 60` if ever regenerating. Not worth redoing the run.

---

## Stage 5 — final dev run (the headline result)

1534 dev questions, ddl, greedy. Three configs on the identical prompts.

| config | dev EX | correct |
|---|---|---|
| 32B teacher | 0.6271 | 962/1534 |
| 7B base (untuned) | 0.5372 | 824/1534 |
| **7B tuned (QLoRA)** | **0.5639** | **865/1534** |

### Significance (McNemar, 1 dof)

| pair | +gain | −lose | net | χ² | verdict |
|---|---|---|---|---|---|
| tuned vs base | 111 | 70 | **+41** | 8.8 | **SIGNIFICANT** |
| tuned vs teacher | 80 | 177 | −97 | 35.9 | SIG (student below teacher) |
| teacher vs base | 205 | 67 | +138 | 69.0 | SIG (the ceiling) |

- **Tuning is a real, significant gain**: recovered **30% of the 9pp base→teacher
  gap** (0.537 → 0.564). Below the runbook's ~0.60 aspiration, and **not
  monotonic — 70 base-correct questions broke** while 111 were fixed.
- **Cleanest effect — schema-faithfulness distilled.** Hard errors **121 → 58**
  (teacher 46); hallucinated joins (`wrong_table`) **110 → 49** (teacher 39). The
  student learned to stop inventing identifiers. Of 63 fewer hard errors, ~41
  became correct, ~22 converted to silent wrong-rows.

### Remaining 6.3pp gap to the teacher (tuned taxonomy)

- `table_set` **241** (largest) — wrong tables; schema-linking-hard, the dominant
  unfixed class.
- `same_n_diff_vals` **309** (shape) — right rows, wrong projection/values.
- `pred_empty` **128** — wrong literal format. **This is the `samples`-arm target,
  and we trained on `ddl`** (deferred the 7B gate). Regenerating SFT on `samples`
  and retraining is the **highest-value next experiment** to push past 0.564 —
  full generation+train cycle, so only if the number matters over the writeup.

### Per-db (tuned vs base)

Gains broad and small — financial +5.7, superhero +5.4, formula_1 +4.6,
debit_card +4.7, thrombosis +4.3. Regressions mild and localized —
california_schools −2.2, european_football_2 −1.6. No catastrophic breakage.

Weakest tuned dbs: california_schools 0.337, financial 0.406, toxicology 0.407,
formula_1 0.500 — the hard multi-table schemas.

### Framing for the writeup

The headline number is modest and well below SOTA 7B (Arctic-Text2SQL-R1-7B
68.5%). Per the project's own thesis, the contribution is the **harness, the
disjoint-split discipline, the data-corruption finding, and the error taxonomy** —
not the EX. The tuning result is a clean, significant, honestly-reported
distillation gain with its churn and its ceiling stated.

---

## Latency & cost (the actual thesis)

7B tuned (merged fp16), benched on dev ddl prompts. Price basis: **A100 40GB @
$1.10/GPU-hr, 2026-08-03**. Source: `bench_7b_tuned.json`.

| metric | value |
|---|---|
| **calls per question** | **1** (single call — no agentic loop, no multi-call repair) |
| single-stream p50 / p95 | **830ms / 1,218ms** (mean 835ms) |
| tokens per question | **~1,121 prompt + 60 completion** (~1,181 total) |
| saturated throughput | **57.3 q/s = 206,213 queries/GPU-hour** |
| **cost** | **$0.0053 / 1,000 queries** (~$5.30 / million) |

**The deployable pitch: EX 0.564 at one model call, sub-second p50, half a cent
per 1,000 queries.** The single-call design is what keeps latency and cost flat.

### Direct tradeoff — 32B teacher vs 7B tuned (same harness, dev ddl)

| metric | 32B teacher | 7B tuned | 7B advantage |
|---|---|---|---|
| dev EX | 0.6271 | 0.5639 | **−6.3pp** (−10% rel.) |
| calls / question | 1 | 1 | — |
| p50 latency | 13,730ms | 830ms | **16.5×** |
| p95 latency | 22,391ms | 1,218ms | 18.4× |
| throughput | 29,782 q/GPU-hr | 206,213 q/GPU-hr | 6.9× |
| cost / 1k queries | $0.0369 | $0.0053 | **7.0×** |

**Headline: the tuned 7B trades 6.3pp EX for ~16× lower interactive latency and
~7× lower cost, single call each.** The latency gap (16.5×) exceeds the cost gap
(7×) because single-stream decode is where the 32B is slowest (~4 tok/s vs ~72),
while batching amortizes its throughput — use 16× for the interactive story, 7×
for the $/query-at-scale story; they measure different things, don't average them.
Sources: `bench_7b_tuned.json`, `bench_32b.json`.

### Load / concurrency (closed-loop, offline) — `bench_7b_load.json`

7B tuned served from `AssassinThe1/qwen7b-t2s-ddl-merged`.

- **Trustworthy:** single-stream p50/p95 = 845/1362 ms; throughput scales 1.1 → 12.1
  q/s over C = 1..64; **saturated (all-at-once) = 23.85 q/s — the real max.**
- **NOT trustworthy — the under-load p50/p95/p99 curve.** vLLM's offline `LLM` did
  not populate per-request metrics (TTFT n/a everywhere), so the sweep fell back to
  chunk-batch time, which tags every request in a chunk with its slowest member.
  One `max_tokens=1024` runaway generation (~14 s at ~70 tok/s) pins p99 at ~14 s
  from C=2 on. Measurement artifact **and** a real config smell.
- **Actions:** (1) cap `--max-tokens 256` — SQL completions average ~60 tokens, so
  this bounds the tail from ~14 s to ~3.5 s with no accuracy cost; (2) real
  p99-under-load needs a live `vllm serve` endpoint benchmarked open-loop
  (vLLM `benchmark_serving.py --request-rate`) — folded into the deploy step.

---

## Deployment (Modal, serverless) — the live serving numbers

fp16 merged model served on **Modal**, one **L4** GPU, OpenAI-compatible vLLM
endpoint, **scale-to-zero**, weights cached on a persistent Volume. Config in
`deploy_modal.py`; open-loop client `loadtest.py`. Measured client-side from a
laptop, so latencies include real network RTT (honest, not on-box).

### Config that actually works (after real dependency pain)

- python 3.11, `vllm==0.9.2`, `transformers==4.53.3` — newer transformers ships a
  native `aimv2` config that vllm 0.9.2 double-registers → startup crash; latest
  vLLM instead pulls a flashinfer needing py3.12. Pin the pre-`aimv2` stack.
- `--enforce-eager` (skip torch.compile + CUDA-graph capture).
- `max_containers=1` (measure vLLM batching on one GPU, not Modal fan-out / the
  10-GPU quota) + `@modal.concurrent(max_inputs=32)` (let the one container batch;
  without it Modal serializes 1-at-a-time → a queue + 180s timeouts).

### Cold start

`--enforce-eager` cut cold start from ~150 s to **77 s** — torch.compile (~80–100 s)
+ CUDA-graph capture (~40 s) were ~90% of it; what's left is weights→VRAM (~13 s) +
container boot. Trade: faster cold start, modestly slower steady-state decode.

### Warm load curve (open-loop, dev prompts, max_tokens=256)

| target req/s | achieved | p50 | p95 | p99 | errors |
|---|---|---|---|---|---|
| 2 | 1.9 | 4.5 s | 7.2 s | 8.6 s | 0 |
| 5 | **4.3** | 5.1 s | 7.8 s | 10 s | 0 |
| 10 | 1.8 ↓ | 11.1 s | 15.8 s | 20 s | 0 |
| 20 | 1.1 ↓ | 15.0 s | 22.7 s | 27.8 s | 2 |

**One L4 sustains ~5 req/s** (achieved ≈ target, 0 errors); beyond that it saturates
and throughput *collapses* (1.8, 1.1) rather than plateauing.

### Why it saturates exactly there — the KV ceiling

- 14.25 GiB fp16 weights + 5.21 GiB KV headroom → **97,520 KV tokens → ~24 concurrent
  @ 4096**.
- Little's Law: at rate 5, 4.3 q/s × ~5 s ≈ **~21 in flight ≈ the 24 ceiling** — it
  saturates precisely when the KV cache fills.
- Past ~5 req/s, `max_inputs=32` **over-admits** beyond the 24 the cache holds → vLLM
  preempts + recomputes → throughput collapses. `max_inputs≈20` would give a graceful
  plateau instead.

### Economics + tradeoffs (both measured, not hand-waved)

- **Idle cost $0** (scale-to-zero). Per session ≈ (77 s cold + warm seconds) × the L4
  per-second rate — cents for personal use; contrast always-on L4 ~$300+/mo.
- **eager ↔ compile:** 77 s cold + ~4.5 s warm  vs  ~150 s cold + ~2.5 s warm. Scale-
  to-zero favors eager; always-on favors compile.
- **fp16 ↔ quant:** 14.25 GiB weights leave only 5.21 GiB KV (~24 concurrent). 4-bit
  would free ~9 GiB → far more KV → higher concurrency at ~1 pp accuracy. Quant was
  cut (AutoAWQ/llm-compressor both broke on Colab's torch/torchvision), so this cost
  is stated, not paid.

**Takeaway:** a working scale-to-zero 7B SQL endpoint on one cheap GPU — ~5 req/s at
p95 ~8 s, $0 idle — where every limit (cold start, throughput ceiling) is explained
by a measured cause, not a guess.

---

## Phase 2B preconditions — router / cascade (32B ⇄ 7B)

Computed **before** writing any routing code, from per-query dev results
(`errors_dev_32b.csv`, `errors_dev_7b_tuned.csv`) and train-val 7B
(`errors_train_val_ddl_7b.csv`).

### Gate 1 — cross-tab: is there accuracy headroom?

|  | 32B right | 32B wrong |
|---|---|---|
| **7B right** | A = 785 | **B = 80** |
| **7B wrong** | C = 177 | D = 492 |

- 7B = A+B = 865 (0.564); 32B = A+C = 962 (0.627).
- **B = 80** (7B right, 32B wrong) — above the 60 threshold. The student's correct
  set is **not** nested inside the teacher's.
- **Oracle best-of-both ceiling = 1042/1534 = 0.679 → +5.2pp / +80 questions over
  always-32B (0.627).** The router is a cost play with a real accuracy ceiling
  *above the teacher* stacked on top.
- **D = 492 (32.1%)** both-wrong floor — unreachable by any router.

### Gate 1b — are the 80 B-wins genuine? (the proof)

The 32B's failure mode on the 80 questions the 7B wins:
- **68 / 80 (85%) are 32B *silent* failures** — clean-executing queries returning
  wrong rows (same_n_diff_vals 36, pred_empty 15, ncol_mismatch 7, …). Genuine
  interpretation wins, not mechanical teacher failures.
- **12 / 80 (15%) are 32B hard errors** (10 missing_column, 1 syntax, 1 truncation).
- So "the 7B knows things the 32B doesn't" is defensible: on 68 dev questions the
  distilled student reads the question correctly where the greedy teacher is
  confidently wrong — plausibly because execution-verified SFT selected for
  gold-aligned interpretations. Wins concentrate in hard multi-table schemas
  (formula_1 15, thrombosis_prediction 14, codebase_community 9).

### Gate 2 — is difficulty predictable from cheap features?

Logistic regression predicting "7B gets it wrong" from table count, total columns,
question length, aggregation/superlative word count. **Leave-one-database-out** CV
(generalization to unseen schemas, matching the dev routing setting).

- **AUC = 0.605** (fold range 0.50–0.75), base 7B-wrong rate 0.388.
- Dominant feature: **table count (+0.83 standardized)** — confirms the schema-size
  prior; join-heavy schemas are where the 7B fails.
- **Verdict: weak but real, below the 0.75 "build a classifier" bar → a static
  feature pre-router is NOT viable.** Points to the **cascade** (run the 7B,
  escalate on the model's *own* confidence). Whether sequence logprob separates the
  89.8%-silent failures is untested here and is the real 2B open question.

### Implication for 2B

Build the **cascade**, not the feature pre-router. Economics already favored it
(~9% more cost for a better signal); Gate 2 adds the reason (static features can't
route). Gate 1 says the payoff is cost **and** up to +5.2pp accuracy if routing is
good. **Next experiment:** run the 7B on dev emitting sequence logprobs and measure
right/wrong separation (AUC) — that decides whether the cascade realizes the oracle
headroom.

---

## Known weak databases (genuine model failure, golds execute)

- **`professional_basketball` — EX 0.326 over 95 questions** (from the full
  1000-q pass; it's a train-fit db, *not* in the 887 val set). All golds execute
  → 64 real misses in one schema. Largest clean failure pocket; first to sample
  once you drill into model errors.
- `mondial_geo` — 0.59 over the val set, largest absolute contributor there.

---

## Tooling changes made this stream

- `analysis.py` — `ast_kind` now captures `order_limit` (ORDER BY / LIMIT),
  `join_types`, `having`, `distinct`. `order_limit` sits after `grouping` in the
  precedence tuple. Targets the superlative-direction failures in the
  `equivalent`/`same_n_diff_vals` buckets.
- `audit_gold.py` — new; two-tier corruption audit (above).
- `make_splits.py` — `--exclude-dbs`, `--exclude-ids`, both in the manifest;
  empty-db prune after exclusions.
- `utils.py` — cardinality-gated profiler + `_load_profile_rules`,
  `_bounded_distinct` (bounded + deterministic), `_col_kind`.
- `derive_profile_rules.py` — new; train-only rule derivation + histogram.
- `sample_errors.py` — `--arms` multi-arm comparison mode.

---

## Open items / next steps

**Done:** data clean-up, splits, arms (ddl chosen), profiler, Stage 4 generation
(12,769 SFT rows), QLoRA train (loss 0.076), Stage 5 dev run (tuned 0.564, SIG),
latency/cost bench, **Phase 2A deployment** (Modal serverless, live serving numbers
in Deployment), **Phase 2B router preconditions** (B=80, cascade chosen).

**Dev is clean** — the 32B dev run reported only 2 broken golds (normal noise), no
`retail_world`-style packaging casualty, so the dev EX stands unqualified. (A
formal `audit_gold --split dev` is still worth one run for the writeup, but the
risk is effectively cleared.)

**The core pipeline + deployment are complete** and writeup-ready: trained model,
accuracy-vs-cost table, and a live scale-to-zero endpoint with measured serving
economics.

Remaining threads (none blocking a writeup):
1. **Phase 2B — the cascade experiment:** run the 7B on dev emitting sequence
   logprobs, measure right/wrong separation (AUC). Decides whether the cascade
   realizes the **0.679 oracle** (Gate 1) or falls back to the pure-cost floor. The
   one genuinely open empirical question left.
2. **Deployment polish (optional):** `max_inputs≈20` for a graceful throughput
   plateau instead of the overload collapse; `--max-tokens 128` to ease KV pressure.
3. **Push past 0.564:** `samples` retrain (targets the 128 `pred_empty` misses), or
   attack `table_set` (schema linking, the largest failure class).
4. Formal `audit_gold --split dev` for the writeup (risk already cleared).

Reporting checklist status: McNemar ✓, tradeoff table ✓, guard-as-its-own-row ✓,
calls/tokens per question ✓. Still add strict-EX headline + ncol_mismatch noise
floor when writing up.
