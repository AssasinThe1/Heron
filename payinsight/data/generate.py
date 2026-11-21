#!/usr/bin/env python3
"""Synthetic payments data with five planted scenarios (P1-P5).

Deterministic: a fixed seed in generator_config.yaml fully determines every row,
so the known answers in eval/ stay valid until somebody changes a knob.

    python data/generate.py              # -> data/*.parquet + data/payinsight.duckdb

Everything is fictional: bank names, merchant names, phone numbers, UPI IDs.
Times are naive timestamps in IST (Asia/Kolkata); amounts are INR.

The planted scenarios are TARGETS. tests/test_planted.py checks with
hand-written SQL that each one actually landed in the data.
"""

import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "generator_config.yaml")
DB_PATH = os.path.join(HERE, "payinsight.duckdb")

# Hourly share of daily volume (IST). Peaks at lunch and evening.
HOUR_PROFILE = np.array([
    0.8, 0.5, 0.35, 0.3, 0.3, 0.45, 0.9, 1.6, 2.6, 3.6, 4.6, 5.2,
    6.0, 6.1, 5.5, 5.0, 4.9, 5.3, 6.2, 7.0, 7.4, 6.9, 5.2, 2.9,
])
HOUR_PROFILE = HOUR_PROFILE / HOUR_PROFILE.sum()

FAILURE_CODES = {
    "card": (["INSUFFICIENT_FUNDS", "DO_NOT_HONOR", "3DS_AUTH_FAILED", "RISK_DECLINED", "TIMEOUT"],
             [0.30, 0.25, 0.25, 0.10, 0.10]),
    "upi": (["INSUFFICIENT_FUNDS", "INVALID_PIN", "BANK_DECLINED", "TIMEOUT", "RISK_DECLINED"],
            [0.30, 0.25, 0.20, 0.15, 0.10]),
    "netbanking": (["TIMEOUT", "BANK_DECLINED", "INVALID_CREDENTIALS"], [0.35, 0.35, 0.30]),
    "wallet": (["INSUFFICIENT_FUNDS", "TIMEOUT"], [0.60, 0.40]),
}

NAME_A = ["Blue", "Silver", "Urban", "Prime", "Bright", "Green", "Swift", "Royal",
          "Little", "Golden", "Nova", "Quick", "True", "Happy", "Metro", "Coastal",
          "Daily", "Smart", "Fresh", "Grand"]
NAME_B = ["Basket", "Cart", "Kitchen", "Trails", "Bazaar", "Wheels", "Threads",
          "Gadgets", "Learn", "Pixel", "Bites", "Nest", "Power", "Spark", "Leaf",
          "Market", "Tickets", "Studio", "Hub", "Works"]


def load_cfg(path=CFG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def _ts(s):
    return np.datetime64(datetime.fromisoformat(s))


def make_merchants(cfg, rng):
    n = cfg["n_merchants"] - 1                      # the last slot is PlayArena
    cats = list(cfg["categories"])
    cat_p = np.array([cfg["categories"][c]["share"] for c in cats])
    regions = list(cfg["regions"])
    reg_p = np.array(list(cfg["regions"].values()))

    category = rng.choice(cats, size=n, p=cat_p / cat_p.sum())
    region = rng.choice(regions, size=n, p=reg_p / reg_p.sum())
    weight = (np.arange(1, n + 1) ** -0.9)
    rng.shuffle(weight)
    rank = np.argsort(np.argsort(-weight))
    tier = np.where(rank < n * 0.10, "enterprise", np.where(rank < n * 0.40, "mid", "small"))

    # 92% onboarded before the window; the rest join during it (small ones).
    start = datetime.fromisoformat(cfg["start"])
    old = rng.random(n) < 0.92
    days_before = rng.integers(30, 5 * 365, size=n)
    days_into = rng.integers(0, 85, size=n)
    onboarded = [start - timedelta(days=int(b)) if o else start + timedelta(days=int(d))
                 for o, b, d in zip(old, days_before, days_into)]
    weight = np.where(old, weight, weight * 0.3)

    names, used = [], set()
    for i in range(n):
        while True:
            nm = f"{NAME_A[rng.integers(len(NAME_A))]}{NAME_B[rng.integers(len(NAME_B))]}"
            if nm not in used:
                break
            nm = f"{nm} {rng.integers(2, 99)}"
            if nm not in used:
                break
        used.add(nm)
        names.append(nm)

    m = pd.DataFrame({
        "merchant_id": [f"M{i + 1:04d}" for i in range(n)],
        "name": names,
        "category": category,
        "tier": tier,
        "region": region,
        "onboarded_at": pd.to_datetime(onboarded).normalize(),
        "_weight": weight,
    })
    pa = cfg["p1_playarena"]
    m.loc[len(m)] = {
        "merchant_id": pa["merchant_id"], "name": pa["name"], "category": pa["category"],
        "tier": "enterprise", "region": pa["region"],
        "onboarded_at": pd.Timestamp(pa["onboarded_at"]), "_weight": 0.0,
    }
    return m


def windows_mask(ts, windows):
    mask = np.zeros(len(ts), dtype=bool)
    for a, b in windows:
        mask |= (ts >= _ts(a)) & (ts < _ts(b))
    return mask


def generate(cfg):
    rng = np.random.default_rng(cfg["seed"])
    banks = pd.DataFrame(cfg["banks"]).rename(columns={"id": "bank_id", "type": "bank_type"})
    methods = list(cfg["methods"])
    merchants = make_merchants(cfg, rng)
    days = pd.date_range(cfg["start"], cfg["end"], freq="D")
    cats = cfg["categories"]
    pa = cfg["p1_playarena"]

    # ---------------- orders -------------------------------------------------
    order_parts = []
    reg_idx = merchants.index[merchants.merchant_id != pa["merchant_id"]]
    ramp_start = pd.Timestamp(pa["ramp_start"])
    for di, day in enumerate(days):
        growth = (1 + cfg["growth_per_month"]) ** (di / 30.0)
        wk = cfg["weekend_factor"] if day.dayofweek >= 5 else 1.0
        n = int(cfg["base_orders_per_day"] * growth * wk * rng.normal(1, cfg["daily_noise_sd"]))
        active = merchants.loc[reg_idx]
        active = active[active.onboarded_at <= day]
        w = active._weight.to_numpy()
        midx = rng.choice(active.index.to_numpy(), size=n, p=w / w.sum())
        mth = rng.choice(methods, size=n, p=[cfg["methods"][k]["share"] for k in methods])
        part = pd.DataFrame({"day": day, "m_idx": midx, "method": mth})

        # P1: PlayArena ramps from 1 Oct to its plateau share of attempts.
        if day >= ramp_start:
            frac = min(1.0, ((day - ramp_start).days + 1) / pa["ramp_days"])
            s = pa["plateau_share"] * frac
            n_pa = int(n * s / (1 - s) * 0.93)       # PA retries more; this keeps share on target
            pm = list(pa["method_mix"])
            pa_part = pd.DataFrame({
                "day": day,
                "m_idx": merchants.index[merchants.merchant_id == pa["merchant_id"]][0],
                "method": rng.choice(pm, size=n_pa, p=list(pa["method_mix"].values())),
            })
            part = pd.concat([part, pa_part], ignore_index=True)
        order_parts.append(part)
    orders = pd.concat(order_parts, ignore_index=True)
    N = len(orders)

    hours = rng.choice(24, size=N, p=HOUR_PROFILE)
    secs = rng.integers(0, 3600, size=N)
    orders["t0"] = (orders.day.to_numpy() + hours.astype("timedelta64[h]")
                    + secs.astype("timedelta64[s]"))
    m = merchants.loc[orders.m_idx]
    orders["merchant_id"] = m.merchant_id.to_numpy()
    orders["region"] = m.region.to_numpy()
    orders["category"] = m.category.to_numpy()
    b_p = banks.share.to_numpy()
    orders["bank_idx"] = rng.choice(len(banks), size=N, p=b_p / b_p.sum())
    dev = list(cfg["devices"])
    orders["device"] = rng.choice(dev, size=N, p=list(cfg["devices"].values()))
    ticket = np.array([cats[c]["ticket"] for c in orders.category])
    is_pa = orders.merchant_id.to_numpy() == pa["merchant_id"]
    ticket = np.where(is_pa, pa["ticket"], ticket)
    orders["amount_inr"] = np.round(ticket * rng.lognormal(0, 0.75, size=N), 2).clip(10, 250000)
    orders["order_id"] = np.arange(N)

    # base success probability per order (time-independent part)
    base = np.array([cfg["methods"][k]["rate"] for k in orders.method])
    pa_rate = np.where(orders.method == "card", pa["card_rate"], pa["upi_rate"])
    base = np.where(is_pa, pa_rate, base)
    tweak = np.array([cats[c]["tweak"] for c in orders.category])
    tweak = np.where(is_pa, 1.0, tweak)
    factor = banks.factor.to_numpy()[orders.bank_idx]
    day_noise = rng.normal(1, 0.006, size=len(days))
    dnoise = day_noise[(orders.day - days[0]).dt.days.to_numpy()]
    orders["p_base"] = np.clip(base * tweak * factor * dnoise, 0.05, 0.985)
    cancel = np.array([cfg["methods"][k]["cancel"] for k in orders.method])
    orders["p_cancel"] = cancel

    # background outages: logged, small, on other banks
    bg = []
    other_banks = banks.bank_id[banks.bank_id != cfg["p2_coral_outage"]["bank_id"]].to_numpy()
    reasons = ["Scheduled maintenance", "Switch failure", "Network degradation",
               "Core banking upgrade", "Third-party processor issue"]
    for i in range(cfg["background_outages"]):
        d = days[rng.integers(0, len(days))]
        start = d + pd.Timedelta(hours=int(rng.integers(1, 22)), minutes=int(rng.choice([0, 15, 30, 45])))
        dur = int(rng.choice([20, 30, 45, 60, 90]))
        bg.append({"bank_id": other_banks[rng.integers(len(other_banks))],
                   "method": rng.choice(methods),
                   "started_at": start, "ended_at": start + pd.Timedelta(minutes=dur),
                   "reason": rng.choice(reasons)})
    co = cfg["p2_coral_outage"]
    events = pd.DataFrame(
        [{"bank_id": co["bank_id"], "method": co["method"], "started_at": pd.Timestamp(a),
          "ended_at": pd.Timestamp(b), "reason": "UPI switch outage"} for a, b in co["windows"]]
        + bg).sort_values("started_at").reset_index(drop=True)

    # ---------------- attempts (with retries) --------------------------------
    bank_ids = banks.bank_id.to_numpy()
    o_bank = bank_ids[orders.bank_idx.to_numpy()]
    o_method = orders.method.to_numpy()
    attempts = []
    pending = np.arange(N)
    t = orders.t0.to_numpy().copy()
    for att in range(1, cfg["max_attempts"] + 1):
        if len(pending) == 0:
            break
        ts = t[pending]
        p = orders.p_base.to_numpy()[pending].copy()
        code_forced = np.zeros(len(pending), dtype=bool)

        coral = (o_bank[pending] == co["bank_id"]) & (o_method[pending] == co["method"])
        out = coral & windows_mask(ts, co["windows"])
        deg = coral & windows_mask(ts, co["degraded"])
        p = np.where(out, co["outage_rate"], p)
        p = np.where(deg, co["degraded_rate"], p)
        code_forced |= out | deg
        for e in bg:
            hit = ((o_bank[pending] == e["bank_id"]) & (o_method[pending] == e["method"])
                   & (ts >= np.datetime64(e["started_at"])) & (ts < np.datetime64(e["ended_at"])))
            p = np.where(hit, p * 0.35, p)
            code_forced |= hit

        u = rng.random(len(pending))
        cancelled = u < orders.p_cancel.to_numpy()[pending]
        success = (~cancelled) & (rng.random(len(pending)) < p)
        status = np.where(cancelled, "user_cancelled", np.where(success, "success", "failed"))
        attempts.append(pd.DataFrame({
            "order_idx": pending, "attempt_no": att, "created_at": ts, "status": status,
            "forced": code_forced & (status == "failed"),
        }))
        retry = (status == "failed") & (rng.random(len(pending)) < cfg["retry_prob"])
        pending = pending[retry]
        t[pending] = t[pending] + rng.integers(20, 300, size=len(pending)).astype("timedelta64[s]")

    a = pd.concat(attempts, ignore_index=True)
    o = orders.loc[a.order_idx]
    tx = pd.DataFrame({
        "order_id": o.order_id.to_numpy(),
        "created_at": a.created_at.to_numpy(),
        "merchant_id": o.merchant_id.to_numpy(),
        "bank_id": o_bank[a.order_idx.to_numpy()],
        "method": o.method.to_numpy(),
        "amount_inr": o.amount_inr.to_numpy(),
        "status": a.status.to_numpy(),
        "attempt_no": a.attempt_no.to_numpy(),
        "region": o.region.to_numpy(),
        "device": o.device.to_numpy(),
        "_forced": a.forced.to_numpy(),
    })
    # failure codes
    fc = np.full(len(tx), None, dtype=object)
    for mth, (codes, probs) in FAILURE_CODES.items():
        sel = (tx.status.to_numpy() == "failed") & (tx.method.to_numpy() == mth)
        fc[sel] = rng.choice(codes, size=sel.sum(), p=probs)
    fc[tx._forced.to_numpy()] = "ISSUER_DOWN"
    fc[tx.status.to_numpy() == "user_cancelled"] = "USER_ABORTED"
    tx["failure_code"] = fc
    tx = tx.drop(columns="_forced")
    tx = tx[tx.created_at <= np.datetime64(f"{cfg['end']}T23:59:59")]
    tx = tx.sort_values(["created_at", "order_id", "attempt_no"]).reset_index(drop=True)

    # ids after sorting, so they read chronologically
    order_map = {k: i for i, k in enumerate(pd.unique(tx.order_id))}
    tx["order_id"] = [f"O{order_map[k] + 1:07d}" for k in tx.order_id]
    tx.insert(0, "txn_id", [f"T{i + 1:08d}" for i in range(len(tx))])

    # PII (fictional): every attempt has a phone; UPI attempts have a VPA
    phone = rng.integers(10**8, 10**9, size=len(tx))
    tx["customer_phone"] = [f"+91-9{p:09d}" for p in phone]
    handle = {r.bank_id: "ok" + r.name.split()[0].lower() for r in banks.itertuples()}
    vpa_num = rng.integers(1000, 99999, size=len(tx))
    tx["customer_vpa"] = [f"user{n}@{handle[b]}" if mth == "upi" else None
                          for n, b, mth in zip(vpa_num, tx.bank_id, tx.method)]

    # ---------------- settlements (P5) — from the TRUE payments ---------------
    st_cfg = cfg["p5_settlements"]
    ok = tx[tx.status == "success"]
    utc_day = (ok.created_at - pd.Timedelta(hours=5, minutes=30)).dt.normalize()
    g = (ok.assign(utc_day=utc_day).groupby(["merchant_id", "utc_day"], as_index=False)
           .amount_inr.sum().rename(columns={"amount_inr": "gross_inr"}))
    g["settlement_date"] = g.utc_day + pd.Timedelta(days=st_cfg["payout_lag_days"])
    lo, hi = st_cfg["refund_rate"]
    g["refunds_inr"] = np.round(g.gross_inr * rng.uniform(lo, hi, size=len(g)), 2)
    tier = merchants.set_index("merchant_id").tier
    fee = np.array([st_cfg["fee_rate"][tier[mid]] for mid in g.merchant_id])
    g["fees_inr"] = np.round(g.gross_inr * fee, 2)
    g["gross_inr"] = np.round(g.gross_inr, 2)
    g["net_inr"] = np.round(g.gross_inr - g.refunds_inr - g.fees_inr, 2)
    oct_rows = g.index[(g.settlement_date.dt.month == 10) & (g.gross_inr > 20000)]
    drop = rng.choice(oct_rows, size=st_cfg["dropped_october_payouts"], replace=False)
    dropped = g.loc[drop, ["merchant_id", "settlement_date", "gross_inr"]].copy()
    g = g.drop(index=drop).sort_values(["settlement_date", "merchant_id"]).reset_index(drop=True)
    g.insert(0, "settlement_id", [f"S{i + 1:06d}" for i in range(len(g))])
    settlements = g[["settlement_id", "merchant_id", "settlement_date", "gross_inr",
                     "refunds_inr", "fees_inr", "net_inr"]].copy()
    settlements["settlement_date"] = settlements.settlement_date.dt.date

    # ---------------- P4: missing records ------------------------------------
    p4 = cfg["p4_missing"]
    gap = (tx.created_at >= pd.Timestamp(p4["gap_start"])) & (tx.created_at < pd.Timestamp(p4["gap_end"]))
    n_gap = int(gap.sum())
    tx = tx[~gap].reset_index(drop=True)
    d0, d1 = p4["null_bank_dates"]
    in_rng = (tx.created_at >= pd.Timestamp(d0)) & (tx.created_at < pd.Timestamp(d1) + pd.Timedelta(days=1))
    null_sel = in_rng & (rng.random(len(tx)) < p4["null_bank_share"])
    tx.loc[null_sel, "bank_id"] = None

    # ---------------- P3: duplicates -----------------------------------------
    p3 = cfg["p3_duplicates"]
    day9 = tx.index[tx.created_at.dt.strftime("%Y-%m-%d") == p3["date"]]
    w = np.where(tx.loc[day9, "status"] == "failed", p3["failed_weight"], 1.0)
    dup_idx = rng.choice(day9, size=p3["n_rows"], replace=False, p=w / w.sum())
    raw = pd.concat([tx, tx.loc[dup_idx]], ignore_index=True)
    raw = raw.sort_values(["created_at", "txn_id"]).reset_index(drop=True)

    pm = pd.DataFrame([{"method": k, "display_name": v["display"],
                        "typical_success_rate": v["rate"]} for k, v in cfg["methods"].items()])
    merchants_out = merchants.drop(columns="_weight")
    banks_out = banks[["bank_id", "name", "bank_type"]]

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": cfg["seed"], "start": cfg["start"], "end": cfg["end"], "as_of": cfg["as_of"],
        "rows": {"transactions": len(raw), "transactions_unique": len(tx),
                 "merchants": len(merchants_out), "banks": len(banks_out),
                 "payment_methods": len(pm), "settlements": len(settlements),
                 "bank_downtime_events": len(events)},
        "planted": {
            "P3_duplicate_rows": int(p3["n_rows"]),
            "P4_null_bank_rows": int(null_sel.sum()),
            "P4_gap_rows_removed": n_gap,
            "P5_dropped_payouts": [
                {"merchant_id": r.merchant_id, "settlement_date": str(r.settlement_date.date()),
                 "gross_inr": round(float(r.gross_inr), 2)} for r in dropped.itertuples()],
        },
    }
    return {
        "transactions": raw[["txn_id", "order_id", "created_at", "merchant_id", "bank_id", "method",
                             "amount_inr", "status", "failure_code", "attempt_no", "region",
                             "device", "customer_phone", "customer_vpa"]],
        "merchants": merchants_out, "banks": banks_out, "payment_methods": pm,
        "settlements": settlements, "bank_downtime_events": events,
    }, manifest


def write(tables, manifest, db_path=DB_PATH, out_dir=HERE):
    import duckdb
    import hashlib
    for name, df in tables.items():
        df.to_parquet(os.path.join(out_dir, f"{name}.parquet"), index=False)
    if os.path.exists(db_path):
        os.remove(db_path)
    con = duckdb.connect(db_path)
    for name, df in tables.items():
        con.register("_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _df")
        con.unregister("_df")
    # Deduplicated copy every role view reads from; the raw table keeps the
    # duplicates so the data-quality checks can count them.
    con.execute("""CREATE TABLE transactions_clean AS
                   SELECT * FROM transactions
                   QUALIFY row_number() OVER (PARTITION BY txn_id ORDER BY created_at) = 1""")
    h = hashlib.sha256(json.dumps(manifest["rows"], sort_keys=True).encode()
                       + str(manifest["seed"]).encode()).hexdigest()[:12]
    manifest["data_version"] = h
    con.execute("CREATE TABLE _meta (key VARCHAR, value VARCHAR)")
    con.executemany("INSERT INTO _meta VALUES (?, ?)",
                    [("data_version", h), ("as_of", manifest["as_of"]),
                     ("start", manifest["start"]), ("end", manifest["end"])])
    con.close()
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    cfg = load_cfg(sys.argv[1] if len(sys.argv) > 1 else CFG_PATH)
    tables, manifest = generate(cfg)
    write(tables, manifest)
    for k, v in manifest["rows"].items():
        print(f"  {k:22s} {v:>9,}")
    print(f"  data_version {manifest['data_version']}  ->  {DB_PATH}")


if __name__ == "__main__":
    main()
