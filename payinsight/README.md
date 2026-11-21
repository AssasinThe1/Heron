# PayInsight — an AI business analyst for payments

Ask *"why did our success rate drop last week?"* and get, in under a minute, an answer that
keeps three things apart: **what the data shows** (facts, each tagged to the query behind it),
**what might explain it** (hypotheses with a visible confidence rating), and **what to do next**
(recommendations). Every number comes from a checked query you can open and copy.

> **AI decides what to compute; ordinary, testable code checks every query, runs it and
> computes every number.** SQL is written by [Heron](../README.md), our fine-tuned
> Qwen2.5-Coder-7B. Answers are filled into templates from computed results, and a final check
> rejects any answer that quotes a number no query produced.

## Quick start

```bash
cd payinsight
pip install -r requirements.txt
python data/generate.py          # ~10 s: 607k payment attempts, 6 tables, 5 planted problems
streamlit run app.py             # the app
python -m pytest -q              # 80 tests, no network needed
python -m eval.run --heron off   # 30 known-answer questions, offline (~5 s)
python -m eval.run               # same, with Heron live on Modal (~3 min + cold start)
```

Nothing needs a key. Without one, the rules planner handles understanding, and if Heron is
unreachable the semantic-layer compiler writes the SQL. Optional settings are in `.env.example`
(Groq/Gemini keys for the hosted planner, `HERON_MODE=off` for fully offline).

**Before a demo:** press *Wake Heron* in the sidebar. A cold start of the LoRA endpoint took **128 s** when
measured on 21 Nov 2026 (the old merged endpoint took 163 s; Heron's README says ~77 s). The endpoint scales to zero after 5 idle minutes.

## Results (real scores, including failures)

| Group | Heron live | Offline (compiler only) | Target |
|---|---|---|---|
| Simple metrics | 10/10 | 10/10 | ≥ 85% |
| Segment and trend | 8/8 | 8/8 | ≥ 85% |
| Root cause (names P1 mix shift + P2 Coral Bank) | 4/4 | 4/4 | 4/4 |
| Ambiguous (asks, or states the assumption) | 3/3 | 3/3 | — |
| Must refuse (a miss is a blocker) | 5/5 | 5/5 | 5/5 |
| **Heron's own SQL** matched the independent check | **14/18** | — | — |

Full per-question tables: [eval/results_heron_live.md](eval/results_heron_live.md),
[eval/results_heron_off.md](eval/results_heron_off.md).

**Heron is scored as its own row.** The system's number can come from the semantic-layer
cross-check (below), so a system pass never stands in for "Heron was right". All 4 misses are the
same wrong-but-safe error: asked for the best or worst *success rate* by bank, method, category or
device, Heron ranks by a *count* of payments, even after the checks send that reason back to it.
In each case the cross-check caught the mismatch, and the answer used the definition-built query
and said so in the caveats. For "which X" questions Heron answers BIRD-style, with only the
winning label; that counts as a match when the label is the definition-built query's top segment,
and the numbers are then cited from the definition-built query.

Latency: root-cause answers take under 1 s (27 template queries on DuckDB). Simple and segment
questions with Heron live had a **median of 7.1 s**. Most of that is Heron itself; a query that fails
the checks goes back to Heron with the reason, up to twice, before the definition-built query is
used. That retry is a deliberate safety trade-off. Offline answers take under 0.2 s.

## What the demo shows

| Brief requirement | Where | What happens |
|---|---|---|
| Question answered end to end | "Why did our success rate drop last week?" | Asks which week and which denominator, and offers **12–18 Oct as the largest weekly drop** (found by a query). Then: headline, mix-vs-rate chart, ranked segments, drill-down, evidence panel |
| Decline explained by mix shift | same answer | 62% of the −2.53 pp is mix: PlayArena (onboarded 1 Oct) went from 4.6% to 17.7% of attempts, 90% card at 64%. Card's own rate stayed flat (64.9% → 65.1%, within noise) |
| Driver finding and drill-down | same, then "Break Coral Bank UPI down by hour" | Coral Bank → UPI → 14–15 Oct, 18:00–22:00 → ISSUER_DOWN, matched to the outage log. **Against:** 20% of the extra failures fall outside the logged hours |
| Duplicated / missing records | caveats | "2,400 duplicate attempt records were removed (2,400 on 9 Oct)"; blank `bank_id` on 20–22 Sep; the 6-hour gap on 17 Aug when the window covers it |
| Sources disagree | "What was October's successful payment value?" | ₹38.98 crore (transactions) vs ₹38.36 crore (settlements before fees), −1.59%. Bridge: UTC cut-off −₹2.97 lakh, next-day payout +₹11.99 lakh, refunds −₹66.31 lakh, 12 missing payout rows −₹4.71 lakh, **unexplained ₹0** |
| Request outside access | Regional analyst (South) asks about North, or for phone numbers | Refused before any query runs, offers "Show the South region instead", logged |
| Expensive / unsafe query | "Show me every transaction", "Delete the test merchants", "last 3 years" | Cost check refuses (~605k rows) and offers a daily summary or a 1,000-row sample; write refused; 180-day limit |

## How it works

```
question ─▶ 1 understand ─▶ 2 clarify? ─▶ 3 policy pre-check ─▶ 4 write SQL ─▶ 5 check + run ─▶ 6 analyse ─▶ 7 explain ─▶ 8 verify + audit
            rules / Groq     code decides   write·PII·region·range  Heron (AI)    guardrails·cache   plain Python  templates    number check
```

AI is used in steps 1, 4 and (optionally) 7. Steps 3, 5, 6 and 8 are deterministic code.

* **Heron, in its frozen training format.** `engine/heron_bridge.py` imports Heron's own `SYSTEM`
  prompt, `clean_sql` and `repair_identifiers` from the parent repo. Heron sees SQLite-style DDL
  of *only the role's tables*, BIRD-style "External knowledge" generated from the semantic layer,
  and the question. Its SQLite output is transpiled to DuckDB with sqlglot.
* **Independent cross-check.** Every Heron query is compared with the same question compiled
  straight from `semantic_layer.yaml` (`engine/compiler.py`). If they match, Heron's SQL is the
  cited evidence. If not, the answer uses the definition and the caveats say so.
* **Two layers of protection.** (1) Each role gets its own in-memory DuckDB holding filtered copies
  of its allowed tables, with PII dropped or masked. The source file is then detached and external
  access disabled, so even SQL that slipped past the checks has nothing else to read (tested).
  (2) `engine/guardrails.py` parses every query, including the engine's own templates, and rejects
  it unless: exactly one read-only statement; no file/network/settings functions; only allowed
  tables; no PII; no out-of-scope region; transactions carry a date filter of ≤ 180 days; joins
  have conditions; and the EXPLAIN estimate is ≤ 5M rows scanned and ≤ 10k rows returned. A LIMIT
  is added if missing.
* **Analysis engine** (`engine/analysis.py`). It compares two periods, splits the change *exactly*
  into mix and rate, sweeps six dimensions, and drills down (≤ 3 levels, ≥ 500 attempts). It
  checks the outage log and merchant start dates, and computes Wilson 95% intervals.
  **One design point matters:** rate drivers are searched *after setting aside the mix driver*.
  Without that, "West region" and "android" look like performance drops, because PlayArena sits
  inside them: the same mix shift one level down. This is the trap the product exists to catch.
* **Answers** (`engine/answer.py`). Every number is rendered through `Fmt`, which records it, and
  `number_check` rejects text containing any other number. Hosted-model polishing (`POLISH=on`)
  must pass the same check, or the template wording is kept.
* **Run-time controls.** 10 s timeout (timer thread → `interrupt()`, as in Heron's `run_query`); a
  cache keyed on normalised SQL + role + data version; and an audit log of every question, role,
  SQL, check result, run time and answer in a separate DuckDB file.

## Files

```
data/generate.py            seeded generator (all knobs in data/generator_config.yaml)
config/semantic_layer.yaml  business definitions, synonyms, analysis settings
config/policy.yaml          roles, row filters, PII handling, limits, blocked functions
engine/                     planner · sqlgen · guardrails · compiler · analysis · dq · reconcile · answer · pipeline
app.py                      Streamlit: Ask · Data & rules · Audit log · Evaluation
eval/questions.yaml         30 questions with hand-written gold SQL; eval/run.py scores them
tests/                      guardrails, maths, planner, planted data, end-to-end, role isolation
```

## Heron serving: what was wrong and how it was fixed

The original endpoint (`deploy_modal.py`, merged model `AssassinThe1/qwen7b-t2s-ddl-merged`) did
not reproduce Heron's offline outputs: about a third of its queries had a stray `))`. Offline vLLM
runs on 300 BIRD dev prompts, greedy (21 Nov 2026), located the cause:

| Setup (vLLM 0.9.2, offline `LLM.chat`) | Identical to offline results | Unbalanced brackets |
|---|---|---|
| A100, uploaded merged model, fp16 | 94/300 | 97/300 |
| L4, uploaded merged model, fp16 | 94/300 | 98/300 |
| L4, uploaded merged model, dtype `auto` (resolved to fp16) | 94/300 | 98/300 |
| L4, base Qwen2.5-Coder-7B-Instruct + LoRA adapter `AssassinThe1/qwen7b-t2s-ddl`, fp16 | **297/300** | **1/300** |
| (offline `results_dev_7b_tuned.json`) | — | 1/300 |

GPU type and dtype make no difference; the adapter is fine; **the uploaded merged weights are not
the evaluated model.** `merge_and_push.py` looks correct, but the upload was saved by
transformers 5.15, not the 4.x stack the offline merge used. Also ruled out: the chat template
(identical to the base model's). A smaller difference was real: `vllm serve` applied the merged
repo's `generation_config.json` (`repetition_penalty 1.1`), which the offline run never used, so
PayInsight now pins plain greedy decoding (`engine/llm.py`).

**Fix in use:** `deploy_modal_lora.py` (in the Heron root) serves the base model with the adapter
as model `heron`, as a separate Modal app (`qwen7b-t2s-lora`); the original endpoint is untouched.
On 100 dev prompts it matched the offline outputs 99/100 with 0 unbalanced brackets, and Heron's
score in PayInsight went from 11/18 to 14/18 (the label-only match above accounts for 3 of those).
The merged upload should still be re-made in a transformers 4.53 environment and re-checked
before anyone else serves it.

The earlier workarounds stay as safety nets: `balance_parens` in `engine/sqlgen.py` removes surplus
`)` only at `))` sites and keeps the result only if it parses (not triggered in the latest run);
resolved dates are stated explicitly in Heron's question; and semantic lint rewrites
`BETWEEN 'day' AND 'day'` on timestamps and sends missing-cancel-exclusion and rank-by-count
errors back to Heron as retry feedback.

## Assumptions, mocks and limitations

* Success rate = successful ÷ all attempts, excluding customer-cancelled attempts; unique-order
  rate is offered. Weeks run Monday–Sunday IST. "Today" is 1 Nov 2026, with data complete through
  31 Oct.
* Mocked: login (a role dropdown), the warehouse (synthetic DuckDB), the outage feed (a 15-row
  table), monitoring (the audit log and evaluation page).
* Causes are hypotheses: patterns that line up in time, not proof. Every answer says so.
* Synthetic data is cleaner than reality; the planted problems are known in advance.
* The finance role sees hourly totals only, which can't split the hour containing the 05:30 IST
  cut-off. Its reconciliation shows a small "unexplained" remainder (₹27.5k) instead of ₹0, and
  the caveats say why.
* Text-to-SQL can still be wrong-but-safe. The checks stop unsafe queries; the cross-check stops
  wrong numbers only where a semantic-layer definition exists. The root-cause and reconciliation
  paths use fixed templates for that reason.
* Free tiers cap usage (Groq ~1,000 requests/day/model). Modal's credit covers a demo plus a few
  evaluation runs.
