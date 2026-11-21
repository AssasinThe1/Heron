"""The "why did this metric change" engine. Plain Python; no AI step.

Five steps, the same for every question:
  1. compare two periods (default: the named week vs the 4 weeks before it)
  2. split the change exactly into mix and rate:
        dSR = sum_i (w1_i - w0_i) * r0_i   +   sum_i w1_i * (r1_i - r0_i)
              ----------- mix -----------      ---------- rate ----------
     Per segment we report mix_i = (w1_i - w0_i)(r0_i - R0), which sums to the
     same mix total (because sum_i (w1_i - w0_i) = 0) but reads correctly: a
     growing segment only drags the total if it is below the baseline average.
  3. rank segments by contribution across every swept dimension
  4. drill into the top rate contributor (bank -> method -> hour & failure code)
     and the top mix contributor (method -> merchant), stopping at 3 levels or
     500 attempts
  5. look for corroborating events: logged bank outages, merchant start dates

All SQL is fixed templates run through the checked Runner (see evidence.py).
"""

import math
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from .compiler import DIM_EXPR, FILTER_EXPR
from .settings import semantic

Z = 1.959964


def wilson(k, n, z=Z):
    """95% Wilson interval for k successes out of n, on the 0-1 scale."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def overlaps(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def decompose(df):
    """df: period ('baseline'|'current'), segment, attempts, successes.

    -> (segments DataFrame, totals dict). Exact: mix + rate == R1 - R0.
    """
    p = df.pivot_table(index="segment", columns="period", values=["attempts", "successes"],
                       aggfunc="sum", fill_value=0)
    seg = pd.DataFrame(index=p.index)
    for col, short in (("attempts", "n"), ("successes", "k")):
        for per, s in (("baseline", "0"), ("current", "1")):
            seg[f"{short}{s}"] = p[col][per] if per in p[col] else 0
    N0, N1 = seg.n0.sum(), seg.n1.sum()
    K0, K1 = seg.k0.sum(), seg.k1.sum()
    R0 = K0 / N0 if N0 else float("nan")
    R1 = K1 / N1 if N1 else float("nan")
    seg["w0"] = seg.n0 / N0 if N0 else 0.0
    seg["w1"] = seg.n1 / N1 if N1 else 0.0
    r0 = np.where(seg.n0 > 0, seg.k0 / seg.n0.where(seg.n0 > 0, 1), np.nan)
    r1 = np.where(seg.n1 > 0, seg.k1 / seg.n1.where(seg.n1 > 0, 1), np.nan)
    # a segment absent in one period: its rate there is taken as the other period's
    seg["r0"] = np.where(np.isnan(r0), r1, r0)
    seg["r1"] = np.where(np.isnan(r1), r0, r1)
    seg["mix"] = (seg.w1 - seg.w0) * (seg.r0 - R0)
    seg["rate"] = seg.w1 * (seg.r1 - seg.r0)
    seg["contrib"] = seg.mix + seg.rate
    ci0 = [wilson(k, n) for k, n in zip(seg.k0, seg.n0)]
    ci1 = [wilson(k, n) for k, n in zip(seg.k1, seg.n1)]
    seg["ci0_lo"], seg["ci0_hi"] = [c[0] for c in ci0], [c[1] for c in ci0]
    seg["ci1_lo"], seg["ci1_hi"] = [c[0] for c in ci1], [c[1] for c in ci1]
    seg["noise"] = [overlaps(a, b) if seg.n0.iloc[i] and seg.n1.iloc[i] else True
                    for i, (a, b) in enumerate(zip(ci0, ci1))]
    seg = seg.sort_values("contrib")
    totals = {"N0": int(N0), "N1": int(N1), "K0": int(K0), "K1": int(K1), "R0": R0, "R1": R1,
              "delta": R1 - R0, "mix": float(seg.mix.sum()), "rate": float(seg.rate.sum()),
              "ci0": wilson(K0, N0), "ci1": wilson(K1, N1)}
    return seg, totals


@dataclass
class WhyResult:
    current: tuple
    baseline: tuple
    primary_dim: str
    totals: dict
    tables: dict                       # dim -> decomposed DataFrame
    ev: dict                           # dim -> evidence id
    ranked: pd.DataFrame
    mix_path: dict = None
    rate_path: dict = None
    hypotheses: list = field(default_factory=list)
    extra_ev: dict = field(default_factory=dict)


def _q(v):
    return "'" + str(v).replace("'", "''") + "'"


def _filters_sql(filters, exclude=None):
    out, joins = [], set()
    for neg, spec in ((False, filters), (True, exclude)):
        for d, vals in (spec or {}).items():
            if not vals:
                continue
            col = FILTER_EXPR.get(d, f"t.{d}")
            if d in FILTER_EXPR:
                joins.add("bank" if d.startswith("bank") else "merchant")
            op = "NOT IN" if neg else "IN"
            out.append(f"{col} {op} ({', '.join(_q(v) for v in vals)})")
    return out, joins


def _from(joins):
    f = "transactions t"
    if "bank" in joins:
        f += " LEFT JOIN banks b ON t.bank_id = b.bank_id"
    if "merchant" in joins:
        f += " LEFT JOIN merchants m ON t.merchant_id = m.merchant_id"
    return f


def window_sql(*periods):
    parts = [f"(t.created_at >= '{a}' AND t.created_at < '{b + timedelta(days=1)}')"
             for a, b in periods]
    return "(" + " OR ".join(parts) + ")"


def segment_sql(dim, current, baseline, filters, metric="success_rate", exclude=None):
    expr, j = DIM_EXPR[dim]
    fs, joins = _filters_sql(filters, exclude)
    if j:
        joins.add(j)
    unit = "t.order_id" if metric == "order_success_rate" else "t.txn_id"
    c0, c1 = current[0], current[1] + timedelta(days=1)
    where = ["t.status <> 'user_cancelled'", window_sql(baseline, current)] + fs
    return (f"SELECT CASE WHEN t.created_at >= '{c0}' AND t.created_at < '{c1}' THEN 'current' "
            f"ELSE 'baseline' END AS period, {expr} AS segment, "
            f"COUNT(DISTINCT {unit}) AS attempts, "
            f"COUNT(DISTINCT {unit}) FILTER (WHERE t.status = 'success') AS successes "
            f"FROM {_from(joins)} WHERE {' AND '.join(where)} GROUP BY 1, 2")


def default_baseline(current, weeks=None):
    weeks = weeks or semantic()["analysis"]["baseline_weeks"]
    a = current[0] - timedelta(days=7 * weeks)
    return (a, current[0] - timedelta(days=1))


def _flt_label(filters):
    if not filters:
        return "all payments"
    return "; ".join(f"{d} = {', '.join(map(str, v))}" for d, v in filters.items())


def why(it, runner, fmt_window, dq_warned=()):
    """Run steps 1-5. `fmt_window(a, b)` renders a period label for evidence."""
    sem = semantic()["analysis"]
    current = it.period
    baseline = it.compare or default_baseline(current)
    filters = it.filters or {}
    sweep = [d for d in sem["sweep"] if not (d in filters and len(filters[d]) == 1)]
    primary = sem["primary_mix_dimension"]
    if primary not in sweep:
        primary = sweep[0]
    window = f"{fmt_window(*current)} vs {fmt_window(*baseline)}"
    flabel = _flt_label(filters)

    tables, ev = {}, {}
    for dim in sweep:
        df, e = runner.run(segment_sql(dim, current, baseline, filters, it.metric),
                           title=f"Success rate by {dim}, current vs baseline",
                           dataset="transactions" + (" + banks" if dim.startswith("bank") else "")
                           + (" + merchants" if dim in ("merchant", "category", "tier") else ""),
                           filters=f"{flabel}; excludes user_cancelled", window=window,
                           source="template")
        seg, tot = decompose(df)
        tables[dim], ev[dim] = seg, e.id
    totals = _totals_from(tables[primary])

    rows = []
    for dim, seg in tables.items():
        for name, r in seg.iterrows():
            rows.append({"dim": dim, "segment": name, **r.to_dict()})
    ranked = pd.DataFrame(rows)
    ranked["abs"] = ranked.contrib.abs()
    ranked = ranked.sort_values("abs", ascending=False).reset_index(drop=True)

    res = WhyResult(current=current, baseline=baseline, primary_dim=primary, totals=totals,
                    tables=tables, ev=ev, ranked=ranked)
    sign = np.sign(totals["delta"]) or -1.0
    min_n = sem["min_segment_attempts"]

    # ---------------- mix path ------------------------------------------------
    pt = tables[primary]
    mix_seg = pt[(np.sign(pt.mix) == sign)].sort_values("mix", key=lambda s: -s.abs() * 1)
    if len(mix_seg) and abs(totals["mix"]) > 0.15 * abs(totals["delta"]) \
            and np.sign(totals["mix"]) == sign:
        top = mix_seg.index[0]
        r = pt.loc[top]
        mp = {"segment": top, "dim": primary, "row": r, "explains": totals["mix"] / totals["delta"]}
        driver = None
        for dim in ("merchant", "category"):
            if dim in tables:
                t = tables[dim]
                cand = t[(np.sign(t.mix) == sign) & (t.n1 >= min_n)]
                if len(cand):
                    c = cand.reindex(cand.mix.abs().sort_values(ascending=False).index).iloc[0]
                    driver = driver or {"dim": dim, "segment": c.name, "row": c}
                    mp[f"top_{dim}"] = {"segment": c.name, "row": c}
        mp["driver"] = driver
        if driver and driver["dim"] == "merchant":
            name = driver["segment"]
            info, e1 = runner.run(
                f"SELECT name, category, tier, region, onboarded_at FROM merchants "
                f"WHERE name = {_q(name)}", title=f"Merchant record: {name}", dataset="merchants",
                filters=f"name = {name}", window="all time")
            mp["merchant_info"] = info.iloc[0].to_dict() if len(info) else None
            mp["ev_merchant"] = e1.id
            fs, joins = _filters_sql({**filters, "merchant": [name]})
            joins.add("merchant")
            c0, c1 = current[0], current[1] + timedelta(days=1)
            mix_df, e2 = runner.run(
                f"SELECT t.method AS method, COUNT(DISTINCT t.txn_id) AS attempts, "
                f"COUNT(DISTINCT t.txn_id) FILTER (WHERE t.status = 'success') AS successes "
                f"FROM {_from(joins)} WHERE t.status <> 'user_cancelled' AND "
                f"t.created_at >= '{c0}' AND t.created_at < '{c1}' AND {' AND '.join(fs)} "
                f"GROUP BY 1 ORDER BY 2 DESC",
                title=f"{name}: attempts by payment method (current period)",
                dataset="transactions + merchants", filters=f"merchant = {name}",
                window=fmt_window(*current))
            mix_df["share"] = mix_df.attempts / mix_df.attempts.sum()
            mix_df["rate"] = mix_df.successes / mix_df.attempts
            mp["merchant_methods"] = mix_df
            mp["ev_merchant_methods"] = e2.id
            # how much of the primary segment's growth is this merchant?
            if primary == "method":
                fs2, j2 = _filters_sql({**filters, "method": [top]})
                j2.add("merchant")
                share_df, e3 = runner.run(
                    f"SELECT CASE WHEN t.created_at >= '{c0}' THEN 'current' ELSE 'baseline' END AS period, "
                    f"COUNT(DISTINCT t.txn_id) AS attempts, "
                    f"COUNT(DISTINCT t.txn_id) FILTER (WHERE m.name = {_q(name)}) AS from_merchant, "
                    f"COUNT(DISTINCT t.txn_id) FILTER (WHERE m.name <> {_q(name)} AND t.status = 'success') AS ok_excl, "
                    f"COUNT(DISTINCT t.txn_id) FILTER (WHERE m.name <> {_q(name)}) AS n_excl "
                    f"FROM {_from(j2)} WHERE t.status <> 'user_cancelled' AND {window_sql(baseline, current)} "
                    f"AND {' AND '.join(fs2)} GROUP BY 1",
                    title=f"{top} attempts from {name}, current vs baseline",
                    dataset="transactions + merchants", filters=f"method = {top}", window=window)
                mp["primary_from_driver"] = share_df.set_index("period")
                mp["ev_primary_from_driver"] = e3.id
            onb = mp["merchant_info"]["onboarded_at"] if mp.get("merchant_info") else None
            if onb is not None and baseline[0] - timedelta(days=14) <= onb.date() <= current[1]:
                mp["event"] = {"type": "onboarded", "date": onb.date()}
        res.mix_path = mp

    # ---------------- rate path -----------------------------------------------
    # Set the mix driver aside first. Otherwise a coarse segment that merely
    # CONTAINS the new low-success merchant (its region, its category, the
    # device most of its users have) shows a "rate drop" that is really the
    # same mix shift one level down -- the trap this engine exists to avoid.
    exclude = None
    rate_tables, scale = tables, 1.0
    if res.mix_path and res.mix_path.get("driver"):
        drv = res.mix_path["driver"]
        exclude = {drv["dim"]: [drv["segment"]]}
        rate_tables = {}
        for dim in [d for d in ("bank", "merchant", "category", "region", "device", primary)
                    if d in tables and d != drv["dim"]]:
            df, e = runner.run(segment_sql(dim, current, baseline, filters, it.metric, exclude),
                               title=f"Success rate by {dim}, excluding {drv['segment']}",
                               dataset="transactions", filters=f"{flabel}; {drv['dim']} <> {drv['segment']}",
                               window=window)
            rate_tables[dim], _ = decompose(df)
            res.extra_ev[f"excl_{dim}"] = e.id
        any_t = next(iter(rate_tables.values()))
        scale = any_t.n1.sum() / totals["N1"]
        res.extra_ev["excluded"] = exclude

    best = None
    for dim in [d for d in ("bank", "merchant", "category", "region", "device", primary)
                if d in rate_tables]:
        t = rate_tables[dim].copy()
        t["rate_overall"] = t.rate * scale
        cand = t[(np.sign(t.rate) == sign) & (t.n1 >= min_n) & (~t.noise)]
        if len(cand):
            c = cand.reindex(cand.rate.abs().sort_values(ascending=False).index).iloc[0]
            # prefer a specific dimension over the primary one when they tie
            better = best is None or abs(c.rate_overall) > abs(best["row"].rate_overall) * (
                1.6 if dim == primary else 1.0)
            if better:
                best = {"dim": dim, "segment": c.name, "row": c, "ev": res.extra_ev.get(
                    f"excl_{dim}", ev.get(dim))}
    if best is not None and abs(best["row"].rate_overall) > 0.1 * abs(totals["delta"]):
        res.rate_path = _drill(best, it, runner, current, baseline, filters, fmt_window,
                               totals, min_n, sem["max_drill_levels"], exclude)
    return res


def _totals_from(seg):
    N0, N1, K0, K1 = seg.n0.sum(), seg.n1.sum(), seg.k0.sum(), seg.k1.sum()
    R0, R1 = K0 / N0, K1 / N1
    return {"N0": int(N0), "N1": int(N1), "K0": int(K0), "K1": int(K1), "R0": R0, "R1": R1,
            "delta": R1 - R0, "mix": float(seg.mix.sum()), "rate": float(seg.rate.sum()),
            "ci0": wilson(K0, N0), "ci1": wilson(K1, N1)}


def _drill(best, it, runner, current, baseline, filters, fmt_window, totals, min_n, max_levels,
           exclude=None):
    """Level 1: the segment. Level 2: its most concentrated child. Level 3: hours & codes."""
    path = {"levels": [best], "explains": best["row"].rate_overall / totals["delta"],
            "exclude": exclude}
    flt = {**filters, best["dim"]: [best["segment"]]}
    window = f"{fmt_window(*current)} vs {fmt_window(*baseline)}"
    parent_rate = best["row"].rate_overall

    # level 2: which child dimension concentrates the drop?
    if max_levels >= 2 and best["row"].n1 >= min_n:
        options = [d for d in ("method", "bank", "device", "category", "region")
                   if d != best["dim"] and not (d in filters and len(filters[d]) == 1)]
        chosen = None
        for dim in options:
            df, e = runner.run(segment_sql(dim, current, baseline, flt, it.metric, exclude),
                               title=f"{best['segment']}: success rate by {dim}",
                               dataset="transactions", filters=_flt_label(flt), window=window)
            seg, _ = decompose(df)
            if len(seg) < 2:                 # one value only (e.g. a single-region role): no split
                continue
            # child rate contribution expressed on the overall total
            scale = seg.n1.sum() / totals["N1"]
            seg["rate_overall"] = seg.rate * scale
            cand = seg[(seg.n1 >= min_n) & (~seg.noise)]
            if not len(cand):
                continue
            top = cand.reindex(cand.rate_overall.abs().sort_values(ascending=False).index).iloc[0]
            conc = top.rate_overall / parent_rate if parent_rate else 0
            if chosen is None or conc > chosen["concentration"]:
                chosen = {"dim": dim, "segment": top.name, "row": top, "concentration": conc,
                          "ev": e.id, "table": seg}
        if chosen and chosen["concentration"] >= 0.5:
            path["levels"].append(chosen)
            flt = {**flt, chosen["dim"]: [chosen["segment"]]}

    # level 3: time and failure codes inside the drilled segment
    if max_levels >= 3:
        fs, joins = _filters_sql(flt, exclude)
        c0, c1 = current[0], current[1] + timedelta(days=1)
        hourly, e_h = runner.run(
            f"SELECT date_trunc('hour', t.created_at) AS hour, COUNT(DISTINCT t.txn_id) AS attempts, "
            f"COUNT(DISTINCT t.txn_id) FILTER (WHERE t.status = 'success') AS successes, "
            f"COUNT(DISTINCT t.txn_id) FILTER (WHERE t.status = 'failed') AS failures "
            f"FROM {_from(joins)} WHERE t.status <> 'user_cancelled' AND t.created_at >= '{c0}' "
            f"AND t.created_at < '{c1}' AND {' AND '.join(fs)} GROUP BY 1 ORDER BY 1",
            title=f"{_seg_name(path)}: hourly success rate", dataset="transactions",
            filters=_flt_label(flt), window=fmt_window(*current))
        leaf = path["levels"][-1]["row"]
        r0 = leaf.r0
        hourly["rate"] = hourly.successes / hourly.attempts
        hourly["expected_fail"] = hourly.attempts * (1 - r0)
        hourly["excess"] = hourly.failures - hourly.expected_fail
        hourly["day"] = hourly.hour.dt.date
        daily = hourly.groupby("day").agg(attempts=("attempts", "sum"), successes=("successes", "sum"),
                                          excess=("excess", "sum"))
        daily["rate"] = daily.successes / daily.attempts
        lo0 = wilson(leaf.k0, leaf.n0)[0]
        bad_days = [d for d, r in daily.iterrows()
                    if wilson(r.successes, r.attempts)[1] < lo0]
        bad_hours = hourly[(hourly.day.isin(bad_days)) & (hourly.attempts >= 10)
                           & (hourly.rate < r0 - 0.15)]
        path.update({"hourly": hourly, "daily": daily, "bad_days": bad_days,
                     "bad_hours": bad_hours, "ev_hourly": e_h.id, "r0": r0})
        if bad_days:
            day_sql = " OR ".join(f"CAST(t.created_at AS DATE) = '{d}'" for d in bad_days)
            day_df, e_d = runner.run(
                f"SELECT CAST(t.created_at AS DATE) AS day, COUNT(DISTINCT t.txn_id) AS attempts, "
                f"COUNT(DISTINCT t.txn_id) FILTER (WHERE t.status = 'success') AS successes "
                f"FROM {_from(joins)} WHERE t.status <> 'user_cancelled' AND t.created_at >= '{c0}' "
                f"AND t.created_at < '{c1}' AND {' AND '.join(fs)} AND ({day_sql}) GROUP BY 1 ORDER BY 1",
                title=f"{_seg_name(path)}: success rate on the affected days", dataset="transactions",
                filters=_flt_label(flt), window=fmt_window(*current))
            path["ev_days"] = e_d.id
            path["bad_days_rate"] = day_df.successes.sum() / day_df.attempts.sum()

        codes, e_c = runner.run(
            f"SELECT CASE WHEN t.created_at >= '{c0}' THEN 'current' ELSE 'baseline' END AS period, "
            f"t.failure_code AS code, COUNT(DISTINCT t.txn_id) AS failures "
            f"FROM {_from(joins)} WHERE t.status = 'failed' AND {window_sql(baseline, current)} "
            f"AND {' AND '.join(fs)} GROUP BY 1, 2",
            title=f"{_seg_name(path)}: failure codes, current vs baseline", dataset="transactions",
            filters=_flt_label(flt) + "; status = failed", window=window)
        pv = codes.pivot_table(index="code", columns="period", values="failures", fill_value=0)
        for c in ("baseline", "current"):
            if c not in pv:
                pv[c] = 0
        pv["share0"] = pv.baseline / max(pv.baseline.sum(), 1)
        pv["share1"] = pv.current / max(pv.current.sum(), 1)
        pv["lift"] = pv.share1 - pv.share0
        pv = pv.sort_values("lift", ascending=False)
        path.update({"codes": pv, "ev_codes": e_c.id})

        # corroboration: logged outages for this bank (and method)
        bank = next((lv["segment"] for lv in path["levels"] if lv["dim"] == "bank"), None)
        method = next((lv["segment"] for lv in path["levels"] if lv["dim"] == "method"), None)
        if bank:
            cond = [f"b.name = {_q(bank)}", f"e.started_at < '{c1}'", f"e.ended_at > '{c0}'"]
            if method:
                cond.append(f"e.method = {_q(method)}")
            evs, e_o = runner.run(
                f"SELECT b.name AS bank, e.method, e.started_at, e.ended_at, e.reason "
                f"FROM bank_downtime_events e JOIN banks b ON e.bank_id = b.bank_id "
                f"WHERE {' AND '.join(cond)} ORDER BY e.started_at",
                title=f"Logged outages: {bank}" + (f" {method}" if method else ""),
                dataset="bank_downtime_events + banks", filters=f"bank = {bank}",
                window=fmt_window(*current))
            path["events"] = evs
            path["ev_events"] = e_o.id
            if len(evs):
                inside = np.zeros(len(hourly), dtype=bool)
                for ev in evs.itertuples():
                    inside |= ((hourly.hour >= ev.started_at.floor("h"))
                               & (hourly.hour < ev.ended_at))
                on_bad = hourly.day.isin(bad_days) if bad_days else np.ones(len(hourly), bool)
                ex = hourly.excess.clip(lower=0)
                tot = ex[on_bad].sum()
                ins = ex[on_bad & inside].sum()
                path["excess_total"] = float(tot)
                path["excess_inside"] = float(ins)
                path["outside_share"] = float(1 - ins / tot) if tot > 0 else 0.0
    return path


def _seg_name(path):
    return " ".join(str(lv["segment"]) for lv in path["levels"])


def confidence(explains, n, event, dq_warn, noise):
    """Visible rules -> (rating, list of reasons)."""
    reasons = [f"explains {explains:.0%} of the change"]
    low = []
    if explains < 0.30:
        low.append("explains under 30%")
    if n < 500:
        low.append("small sample")
    if noise:
        low.append("confidence ranges overlap")
    if dq_warn:
        low.append("data-quality warning on the key column")
    if low:
        return "low", reasons + low
    missing = []
    if not event:
        missing.append("no matching logged event")
    if n <= 5000:
        missing.append("sample of 5,000 attempts or fewer")
    if explains > 0.70 and not missing:
        return "high", reasons + ["sample above 5,000", "matching logged event",
                                  "no data-quality warning"]
    if explains > 0.70:
        return "medium", reasons + missing
    return "medium", reasons + (missing or ["sample above 5,000", "matching logged event"])
