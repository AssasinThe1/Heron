# PayInsight

An analyst tool for payments data. You ask a question in plain English, like "why did our
success rate drop last week?", and it answers with the numbers, the likely causes and what to
check next. Every number in an answer comes from a SQL query you can open and copy.

The SQL is written by [Heron](../README.md), our fine-tuned Qwen2.5-Coder-7B. Everything after
that is plain Python: checking the query, running it, the analysis and the numbers in the
answer. Answer text is filled in from templates, and a final check rejects any answer that
contains a number no query produced.

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

No API keys are needed. Without them, a rules-based planner parses the question, and if Heron
can't be reached the SQL is compiled from the semantic layer instead. Optional settings are in
`.env.example` (Groq/Gemini keys for the hosted planner, `HERON_MODE=off` to run fully offline).

Before a demo, press Wake Heron in the sidebar. The endpoint scales to zero after 5 idle
minutes, and a cold start takes about 2 minutes (128 s measured on 21 Nov 2026).

## Results

| Group | Heron live | Offline (compiler only) | Target |
|---|---|---|---|
| Simple metrics | 10/10 | 10/10 | ≥ 85% |
| Segment and trend | 8/8 | 8/8 | ≥ 85% |
| Root cause (finds P1 mix shift and P2 Coral Bank) | 4/4 | 4/4 | 4/4 |
| Ambiguous (asks, or states its assumption) | 3/3 | 3/3 | n/a |
| Must refuse | 5/5 | 5/5 | 5/5 |
| Heron's own SQL matched the cross-check | 14/18 | n/a | n/a |

Per-question results: [eval/results_heron_live.md](eval/results_heron_live.md),
[eval/results_heron_off.md](eval/results_heron_off.md).

Heron's SQL is scored on its own row because the system can get a question right even when
Heron's query is wrong: the answer falls back to the query compiled from the semantic layer.

All 4 of Heron's misses are the same mistake. Asked for the best or worst success rate by bank,
method, category or device, it ranks by number of payments instead, even after being told why
the query was rejected. The cross-check caught all four, and those answers used the compiled
query and said so in the caveats.

For "which X" questions Heron often returns just the top label, the way BIRD answers are
written. That counts as a match if the label is the top segment in the compiled result, and the
numbers shown then come from the compiled query.

Latency: root-cause answers take under 1 s (27 template queries on DuckDB). Simple and segment
questions with Heron live had a median of 7.1 s, most of it spent in Heron. A query that fails
the checks goes back to Heron with the reason, up to twice, before the compiled query is used.
Offline answers take under 0.2 s.

## Demo walkthrough

| Requirement | Question | What happens |
|---|---|---|
| Question answered end to end | "Why did our success rate drop last week?" | Asks which week and which denominator, and suggests 12–18 Oct, the largest weekly drop. Then shows the headline, a mix-vs-rate chart, ranked segments, a drill-down and the evidence panel |
| Decline explained by mix shift | same | 62% of the −2.53 pp is mix. PlayArena (onboarded 1 Oct) went from 4.6% to 17.7% of attempts, 90% of them card payments at 64%. Card's own rate barely moved (64.9% → 65.1%, within noise) |
| Driver and drill-down | same, then "Break Coral Bank UPI down by hour" | Coral Bank → UPI → 14–15 Oct, 18:00–22:00 → ISSUER_DOWN, matched to the outage log. Also shows the evidence against: 20% of the extra failures are outside the logged hours |
| Duplicate and missing records | caveats | "2,400 duplicate attempt records were removed (2,400 on 9 Oct)"; blank `bank_id` on 20–22 Sep; the 6-hour gap on 17 Aug when the date range includes it |
| Sources disagree | "What was October's successful payment value?" | ₹38.98 crore (transactions) vs ₹38.36 crore (settlements before fees), −1.59%. Bridge: UTC cut-off −₹2.97 lakh, next-day payout +₹11.99 lakh, refunds −₹66.31 lakh, 12 missing payout rows −₹4.71 lakh, unexplained ₹0 |
| Request outside access | Regional analyst (South) asks about North, or for phone numbers | Refused before any query runs, offers the South region instead, and logged |
| Expensive or unsafe query | "Show me every transaction", "Delete the test merchants", "last 3 years" | Refused on cost (~605k rows) with a daily summary or 1,000-row sample offered; writes refused; 180-day limit |

## How it works

```
question ─▶ 1 understand ─▶ 2 clarify? ─▶ 3 policy pre-check ─▶ 4 write SQL ─▶ 5 check + run ─▶ 6 analyse ─▶ 7 explain ─▶ 8 verify + audit
            rules / Groq     code decides   write·PII·region·range  Heron         guardrails·cache   Python       templates    number check
```

Steps 1, 4 and (optionally) 7 use a model. Steps 3, 5, 6 and 8 are plain code.

- **Heron's prompt.** Heron is prompted in the same format it was trained on.
  `engine/heron_bridge.py` imports `SYSTEM`, `clean_sql` and `repair_identifiers` from the
  parent repo. The prompt has SQLite-style DDL for only the tables the role can see, BIRD-style
  "External knowledge" generated from the semantic layer, and the question. The SQLite output
  is transpiled to DuckDB with sqlglot. Heron is served on Modal by `deploy_modal_lora.py` in
  the Heron root (base model plus the LoRA adapter, requested as model `heron`).
- **Cross-check.** Each Heron query is compared with the same question compiled from
  `semantic_layer.yaml` (`engine/compiler.py`). If the results match, Heron's SQL is cited. If
  not, the answer uses the compiled query and says so in the caveats.
- **Access control.** Each role gets its own in-memory DuckDB with filtered copies of the tables
  it may see, with PII dropped or masked. The source file is then detached and external access
  turned off, so a query that got past the checks still has nothing else to read (there's a
  test for this). On top of that, `engine/guardrails.py` parses every query, including the
  engine's own templates, and rejects it unless it is a single read-only statement, uses no
  file, network or settings functions, touches only allowed tables, reads no PII, stays in the
  role's region, filters transactions to 180 days or less, has a condition on every join, and
  is estimated by EXPLAIN at ≤ 5M rows scanned and ≤ 10k rows returned. A LIMIT is added if
  missing.
- **Analysis** (`engine/analysis.py`). Compares two periods, splits the change exactly into mix
  and rate effects, checks six dimensions and drills down up to 3 levels (segments of at least
  500 attempts). It also checks the outage log and merchant start dates and computes Wilson 95%
  intervals. Rate drivers are searched for after removing the mix driver. Without that, West
  region and android look like rate drops only because most of PlayArena's traffic is in them.
- **Answers** (`engine/answer.py`). Every number goes through `Fmt`, which records it, and
  `number_check` rejects any text with a number that wasn't recorded. If `POLISH=on`, the
  hosted model's rewrite has to pass the same check or the template text is used.
- **Runtime.** 10 s query timeout (a timer thread calls `interrupt()`, as in Heron's
  `run_query`); a cache keyed on normalised SQL, role and data version; and an audit log of
  every question, role, SQL, check result, run time and answer in a separate DuckDB file.

## Files

```
data/generate.py            seeded generator (settings in data/generator_config.yaml)
config/semantic_layer.yaml  business definitions, synonyms, analysis settings
config/policy.yaml          roles, row filters, PII handling, limits, blocked functions
engine/                     planner · sqlgen · guardrails · compiler · analysis · dq · reconcile · answer · pipeline
app.py                      Streamlit: Ask · Data & rules · Audit log · Evaluation
eval/questions.yaml         30 questions with hand-written gold SQL; eval/run.py scores them
tests/                      guardrails, maths, planner, planted data, end-to-end, role isolation
```

## Assumptions and limitations

- Success rate = successful ÷ all attempts, excluding attempts the customer cancelled. A
  unique-order rate is also available. Weeks run Monday to Sunday, IST. "Today" is 1 Nov 2026,
  and data is complete through 31 Oct.
- Mocked: login (a role dropdown), the warehouse (synthetic DuckDB), the outage feed (a 15-row
  table), and monitoring (the audit log and evaluation page).
- Causes are reported as hypotheses, i.e. things that line up in time, and the answers say so.
- The synthetic data is cleaner than real data, and the planted problems are known in advance.
- The finance role only sees hourly totals, so it can't split the hour containing the 05:30 IST
  cut-off. Its reconciliation shows a small unexplained remainder (₹27.5k) instead of ₹0, and
  the caveats explain why.
- Heron can still write SQL that is safe but wrong. The checks block unsafe queries, but the
  cross-check only catches wrong numbers where the semantic layer has a definition. That's why
  the root-cause and reconciliation paths use fixed templates.
- Free tiers limit usage (Groq allows about 1,000 requests a day per model). Modal's credit is
  enough for a demo and a few evaluation runs.
