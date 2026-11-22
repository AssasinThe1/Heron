"""Understand the question: metric, filters, period, grouping, denominator.

Two interchangeable planners produce the same `Intent`:
  * rules   deterministic regex + the semantic layer's synonyms (always available)
  * llm     a free hosted model (Groq, Gemini backup) asked for JSON, then
            validated field-by-field against the known metrics/dimensions/values

Safety flags (write / PII / dump requests, out-of-scope regions) ALWAYS come from
the rules planner and are OR-ed in, so a model can never talk its way past them.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

from . import llm
from .settings import PLANNER_MODE, data_meta, semantic

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
          "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}
MONTH_RE = (r"(january|february|march|april|may|june|july|august|september|october|"
            r"november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)")
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "twelve": 12}

DIM_WORDS = [
    ("failure_code", r"failure codes?|failure reasons?|error codes?|decline codes?|decline reasons?|reasons?"),
    ("bank_type", r"bank types?"),
    ("method", r"payment methods?|methods?|payment modes?|modes?|instruments?"),
    ("bank", r"issuer banks?|issuers?|banks?"),
    ("category", r"merchant categor(?:y|ies)|categor(?:y|ies)|verticals?|segments? of merchants"),
    ("tier", r"merchant tiers?|tiers?"),
    ("merchant", r"merchants?"),
    ("region", r"regions?|zones?"),
    ("device", r"devices?|platforms?"),
]
GRAIN_WORDS = [("hour", r"hours?|hourly"), ("day", r"days?|daily|date"),
               ("week", r"weeks?|weekly"), ("month", r"months?|monthly")]
METHOD_VALUES = [("netbanking", r"net ?banking|netbank|internet banking"),
                 ("upi", r"\bupi\b"), ("card", r"\b(?:credit |debit )?cards?\b"),
                 ("wallet", r"\bwallets?\b")]
DEVICE_VALUES = [("mobile_web", r"mobile web|mweb"), ("android", r"\bandroid\b"),
                 ("ios", r"\bios\b|\biphones?\b"), ("web", r"\b(?:desktop|web)\b")]
REGION_VALUES = ["North", "South", "East", "West"]

WRITE_RE = re.compile(
    r"\b(delete|truncate|purge|wipe|overwrite|grant|revoke)\b"
    r"|\bdrop (?:the )?(?:table|database|column|schema|merchants?|rows?|records?)\b"
    r"|\binsert (?:into|rows?|records?)\b|\bupdate (?:the )?\w+ (?:set|to)\b|\balter table\b"
    r"|\bremove (?:the |all |these |those )?(?:test |duplicate |old )?(?:merchants?|rows?|records?|transactions?|data|entries)\b"
    r"|\bcreate (?:a )?table\b|\bmodify (?:the )?(?:data|records?|rows?|table)\b", re.I)
PII_RE = re.compile(r"\b(phone|phones|mobile numbers?|contact numbers?|e-?mails?|vpas?|"
                    r"upi ids?|upi handles?|customer names?|addresses|pan numbers?|aadhaar|"
                    r"card numbers?|personal (?:data|details|information)|pii)\b", re.I)
DUMP_RE = re.compile(r"\b(every (?:single )?(?:transaction|payment|attempt|row|record)s?|"
                     r"all (?:the )?(?:raw )?(?:transactions|payments|attempts|rows|records)|"
                     r"raw (?:data|rows|table)|dump|export (?:all|everything|the table)|"
                     r"entire (?:table|dataset)|full (?:table|dataset|list of transactions))\b", re.I)
WHY_RE = re.compile(r"\b(why|what (?:caused|drove|explains|is behind|happened)|root cause|"
                    r"reason for|reasons for|explain (?:the|why)|what's behind|driv(?:er|ers) of)\b", re.I)
RECON_RE = re.compile(r"\b(reconcil\w*|settle\w*|payouts?|disagree\w*|mismatch\w*|"
                      r"differ\w*|gap between|tie out|match (?:the )?settlements?)\b", re.I)
VAGUE_RE = re.compile(r"\b(lately|recently|these days|of late|nowadays|right now|currently|"
                      r"how are we doing|how's it going|how is it going)\b", re.I)


@dataclass
class Intent:
    kind: str = "metric"           # metric | why | reconcile | dump | write | pii | unknown
    metric: str = "success_rate"
    filters: dict = field(default_factory=dict)       # dimension -> [values]
    period: tuple = None           # (date, date) inclusive
    period_label: str = ""
    period_source: str = ""        # explicit | clarified | default | inherited
    vague_period: bool = False
    period_candidates: list = field(default_factory=list)
    compare: tuple = None
    group_by: list = field(default_factory=list)
    grain: str = None              # hour | day | week | month
    order: str = None              # asc | desc
    limit: int = None
    denominator_explicit: bool = False
    direction: str = None          # drop | rise
    question: str = ""
    planner: str = "rules"
    notes: list = field(default_factory=list)

    def to_json(self):
        d = asdict(self)
        for k in ("period", "compare"):
            if d[k]:
                d[k] = [str(x) for x in d[k]]
        return d


# ----------------------------------------------------------------------------- dates
def today():
    """The day after the data ends: data is complete through 'yesterday'."""
    return data_meta()["as_of_date"] + timedelta(days=1)


def month_range(y, m):
    first = date(y, m, 1)
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    return first, nxt - timedelta(days=1)


def last_full_week(ref=None):
    """Last complete Monday-Sunday week strictly before `ref` (default today)."""
    ref = ref or today()
    this_monday = ref - timedelta(days=ref.weekday())
    start = this_monday - timedelta(days=7)
    end = start + timedelta(days=6)
    if end >= ref:                                   # ref is mid-week -> go back one more
        start -= timedelta(days=7)
        end -= timedelta(days=7)
    return start, end


def week_of(d):
    s = d - timedelta(days=d.weekday())
    return s, s + timedelta(days=6)


def fmt_range(a, b):
    if a == b:
        return f"{a.day} {a:%b} {a.year}"
    if a.month == b.month and a.year == b.year:
        if a.day == 1 and b == month_range(a.year, a.month)[1]:
            return f"{a:%B %Y}"
        return f"{a.day}–{b.day} {a:%b} {a.year}"
    return f"{a.day} {a:%b} – {b.day} {b:%b} {b.year}"


def _year_for(month, text):
    m = re.search(r"\b(20\d\d)\b", text)
    return int(m.group(1)) if m else data_meta()["end_date"].year


def _mnum(s):
    return MONTHS[s[:4].lower()] if s[:4].lower() in MONTHS else MONTHS[s[:3].lower()]


def parse_period(text):
    """-> (period | None, label, vague: bool, candidates: list).

    Explicit forms win over relative ones. "last week" and vague words return
    vague=True with candidate periods for the clarifying question.
    """
    t = text.lower()
    tdy = today()
    data_end = data_meta()["end_date"]

    # ISO dates
    iso = [date.fromisoformat(x) for x in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", t)]
    if len(iso) >= 2:
        a, b = min(iso), max(iso)
        return (a, b), fmt_range(a, b), False, []
    if len(iso) == 1:
        return (iso[0], iso[0]), fmt_range(iso[0], iso[0]), False, []

    # "12-18 Oct", "12 to 18 October", "14 October", "14th oct"
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:(?:-|–|to|and|until|till)\s*(\d{1,2})(?:st|nd|rd|th)?\s*)?"
                  + MONTH_RE + r"\b", t)
    m2 = re.search(MONTH_RE + r"\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*(?:-|–|to|and)\s*(\d{1,2})(?:st|nd|rd|th)?)?\b", t)
    week_of_hit = re.search(r"\bweek (?:of|starting|beginning|commencing)\b", t)
    if m:
        d1, d2, mon = int(m.group(1)), m.group(2), m.group(3)
        y = _year_for(mon, t)
        try:
            a = date(y, _mnum(mon), d1)
            b = date(y, _mnum(mon), int(d2)) if d2 else a
        except ValueError:
            a = b = None
        if a:
            # "14 Oct to 20 Oct" / "between 1 Sep and 15 Oct"
            rest = t[m.end():]
            m3 = re.search(r"(?:-|–|to|and|until|till)\s*(\d{1,2})(?:st|nd|rd|th)?\s*" + MONTH_RE, rest)
            if m3 and not d2:
                try:
                    b = date(y, _mnum(m3.group(2)), int(m3.group(1)))
                except ValueError:
                    pass
            if week_of_hit and a == b:
                a, b = week_of(a)
            return (min(a, b), max(a, b)), fmt_range(min(a, b), max(a, b)), False, []
    if m2 and not re.match(r"\s*20\d\d", t[m2.start(2):]):
        mon, d1, d2 = m2.group(1), int(m2.group(2)), m2.group(3)
        if d1 <= 31:
            y = _year_for(mon, t)
            try:
                a = date(y, _mnum(mon), d1)
                b = date(y, _mnum(mon), int(d2)) if d2 else a
                if week_of_hit and a == b:
                    a, b = week_of(a)
                return (a, b), fmt_range(a, b), False, []
            except ValueError:
                pass

    # mid / early / late month
    m = re.search(r"\b(mid|middle of|early|start of|beginning of|late|end of)[\s-]*" + MONTH_RE, t)
    if m:
        y = _year_for(m.group(2), t)
        mo = _mnum(m.group(2))
        a, b = month_range(y, mo)
        w = m.group(1)
        if w in ("mid", "middle of"):
            a, b = week_of(date(y, mo, 15))
        elif w in ("early", "start of", "beginning of"):
            b = date(y, mo, 10)
        else:
            a = date(y, mo, 21)
        return (a, b), fmt_range(a, b), False, []

    # multi-year / long relative ranges (these exist to be refused by the range limit)
    m = re.search(r"\b(?:last|past|previous)\s+(\d+|" + "|".join(NUM_WORDS) + r")\s+(years?|months?|weeks?|days?)\b", t)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else NUM_WORDS[m.group(1)]
        unit = m.group(2).rstrip("s")
        days = {"year": 365, "month": 30, "week": 7, "day": 1}[unit] * n
        b = data_end
        a = b - timedelta(days=days - 1)
        return (a, b), f"last {n} {unit}{'s' if n > 1 else ''} ({fmt_range(a, b)})", False, []
    m = re.search(r"\bsince (20\d\d)\b", t)
    if m:
        a = date(int(m.group(1)), 1, 1)
        return (a, data_end), fmt_range(a, data_end), False, []
    m = re.search(r"\b(?:from |between )?(20\d\d)\s*(?:-|–|to|and|through)\s*(20\d\d)\b", t)
    if m:
        a, b = date(int(m.group(1)), 1, 1), date(int(m.group(2)), 12, 31)
        return (a, b), f"{m.group(1)}–{m.group(2)}", False, []

    # quarters
    m = re.search(r"\bq([1-4])\b", t)
    if m:
        q = int(m.group(1))
        y = _year_for("", t)
        a = date(y, 3 * q - 2, 1)
        b = month_range(y, 3 * q)[1]
        return (a, b), f"Q{q} {y}", False, []

    # two named months: "September vs October", "from August to October"
    months = re.findall(r"\b" + MONTH_RE + r"\b", t)
    months = [x for x in months if x != "may" or re.search(r"\bmay\s+20\d\d", t)]
    if months:
        y = _year_for(months[0], t)
        if len(months) >= 2 and re.search(r"\b(from|between|to|through|until)\b", t) \
                and not re.search(r"\b(vs|versus|compared|against)\b", t):
            a = month_range(y, _mnum(months[0]))[0]
            b = month_range(y, _mnum(months[-1]))[1]
            return (a, b), fmt_range(a, b), False, []
        a, b = month_range(y, _mnum(months[-1] if re.search(r"\b(vs|versus|compared|against)\b", t)
                                    else months[0]))
        return (a, b), fmt_range(a, b), False, []

    # relative
    if re.search(r"\byesterday\b", t):
        d = tdy - timedelta(days=1)
        return (d, d), f"yesterday ({fmt_range(d, d)})", False, []
    m = re.search(r"\b(?:last|past) (\d+|" + "|".join(NUM_WORDS) + r") days\b", t)
    if m or re.search(r"\b(?:last|past) week of data\b", t):
        n = 7 if not m else (int(m.group(1)) if m.group(1).isdigit() else NUM_WORDS[m.group(1)])
        a, b = tdy - timedelta(days=n), tdy - timedelta(days=1)
        return (a, b), f"last {n} days ({fmt_range(a, b)})", False, []
    if re.search(r"\b(?:last|past|previous) (?:full |complete |calendar )week\b", t):
        a, b = last_full_week()
        return (a, b), f"last full week ({fmt_range(a, b)})", False, []
    if re.search(r"\bthis week\b|\bweek to date\b|\bwtd\b", t):
        a = tdy - timedelta(days=tdy.weekday())
        b = tdy - timedelta(days=1)
        if b < a:
            a, b = last_full_week()
        return (a, b), f"this week ({fmt_range(a, b)})", False, []
    if re.search(r"\b(?:last|previous|past) month\b", t):
        last = tdy.replace(day=1) - timedelta(days=1)
        a, b = month_range(last.year, last.month)
        return (a, b), fmt_range(a, b), False, []
    if re.search(r"\bthis month\b|\bmonth to date\b|\bmtd\b", t):
        a, b = month_range(data_end.year, data_end.month)
        return (a, b), f"month to date ({fmt_range(a, b)})", False, []
    if re.search(r"\b(?:last|past|previous) week\b", t):
        return None, "last week", True, period_candidates()
    if VAGUE_RE.search(t):
        return None, "unspecified", True, period_candidates()
    return None, "", False, []


def period_candidates():
    """Choices offered when the period is unclear."""
    tdy = today()
    wk = last_full_week()
    d7 = (tdy - timedelta(days=7), tdy - timedelta(days=1))
    mo = month_range(data_meta()["end_date"].year, data_meta()["end_date"].month)
    return [
        {"key": "last_full_week", "label": f"Last full week ({fmt_range(*wk)})", "period": wk},
        {"key": "last_7_days", "label": f"Last 7 days ({fmt_range(*d7)})", "period": d7},
        {"key": "month_to_date", "label": f"Month to date ({fmt_range(*mo)})", "period": mo},
    ]


# ----------------------------------------------------------------------------- rules
def _filters(text, lookups):
    t = text.lower()
    f = {}
    for val, pat in METHOD_VALUES:
        if re.search(pat, t):
            f.setdefault("method", []).append(val)
    for val, pat in DEVICE_VALUES:
        if re.search(pat, t):
            f.setdefault("device", []).append(val)
    for r in REGION_VALUES:
        if re.search(rf"\b{r.lower()}(?:ern)?\b(?! ?-? ?(?:facing|bound))", t):
            f.setdefault("region", []).append(r)
    for b in lookups["banks"]:
        short = b.lower().replace(" bank", "").replace(" cooperative", "")
        if b.lower() in t or re.search(rf"\b{re.escape(short)}\b", t):
            f.setdefault("bank", []).append(b)
    for c in lookups["categories"]:
        pat = c.replace("_", "[ _-]?")
        if re.search(rf"\b{pat}\b", t):
            f.setdefault("category", []).append(c)
    for name in lookups["merchants"].name:
        if re.search(rf"\b{re.escape(name.lower())}\b", t):
            f.setdefault("merchant", []).append(name)
    m = re.search(r"\b(ISSUER_DOWN|INSUFFICIENT_FUNDS|DO_NOT_HONOR|3DS_AUTH_FAILED|RISK_DECLINED|"
                  r"TIMEOUT|INVALID_PIN|BANK_DECLINED|INVALID_CREDENTIALS)\b", text, re.I)
    if m:
        f["failure_code"] = [m.group(1).upper()]
    # a merchant name like "PlayArena" also matches its category word? keep both.
    return f


def _metric(text):
    t = text.lower()
    sem = semantic()["metrics"]
    if re.search(r"\b(unique orders?|per order|order success|order-level|orders with)\b", t):
        return "order_success_rate", True
    best, best_len = None, 0
    for name, spec in sem.items():
        for syn in spec.get("synonyms", []):
            if re.search(rf"\b{re.escape(syn)}\b", t) and len(syn) > best_len:
                best, best_len = name, len(syn)
    if best:
        return best, best in ("order_success_rate",) or bool(re.search(r"\ball attempts\b", t))
    if re.search(r"\bfail", t):
        return "failures", False
    return "success_rate", bool(re.search(r"\ball attempts\b", t))


def _grouping(text):
    t = text.lower()
    orig = t
    groups, grain, order, limit = [], None, None, None
    lead = r"(?:by|per|each|every|across|which|what|top \d+|bottom \d+|worst|best|split by|break(?:ing)? (?:it )?down by|breakdown by|for each|wise)"
    for g, pat in GRAIN_WORDS:
        if re.search(rf"\b(?:by|per|each|every|over|which|what|across|break(?:ing)? (?:it )?down by|split by|worst|best)\s+(?:the\s+)?(?:{pat})\b|\b(?:hourly|daily|weekly|monthly)\b", t):
            if re.search(rf"\b(?:{pat})\b", t) or re.search(rf"\b{g}ly\b", t):
                grain = g
                break
    if not grain and re.search(r"\b(trend|over time|day by day|each day)\b", t):
        grain = "day"
    for dim, pat in DIM_WORDS:
        m = (re.search(rf"\b{lead}\s+(?:the\s+|our\s+)?(?:\w+\s+)?(?:{pat})\b", t)
             or re.search(rf"\b(?:{pat})[- ]wise\b", t))
        if m:
            if dim == "failure_code" and not re.search(r"\b(failure|error|decline)", t) \
                    and not re.search(r"\bby reason", t):
                continue
            if dim not in groups:
                groups.append(dim)
            # consume the phrase so "merchant category" can't also match "merchant"
            t = t[:m.start()] + " " * (m.end() - m.start()) + t[m.end():]
    t = orig
    if re.search(r"\b(worst|lowest|poorest|weakest|least|bottom|biggest drop|fell the most)\b", t):
        order = "asc"
    if re.search(r"\b(best|highest|top|most|largest|biggest|strongest)\b", t) and order is None:
        order = "desc"
    m = re.search(r"\b(?:top|bottom|worst|best)\s+(\d+|" + "|".join(NUM_WORDS) + r")\b", t)
    if m:
        limit = int(m.group(1)) if m.group(1).isdigit() else NUM_WORDS[m.group(1)]
    elif re.search(r"\bwhich\b|\bwhat (?:bank|merchant|method|region|device|category|day|week|hour)\b", t) \
            and order:
        limit = 1
    return groups, grain, order, limit


def rules_plan(question, lookups):
    t = question.lower()
    it = Intent(question=question)
    it.metric, it.denominator_explicit = _metric(question)
    it.filters = _filters(question, lookups)
    it.group_by, it.grain, it.order, it.limit = _grouping(question)
    # a dimension that is both filtered to one value and grouped is a filter
    for g in list(it.group_by):
        if g in it.filters and len(it.filters[g]) == 1 and not re.search(rf"\bby {g}", t):
            it.group_by.remove(g)
    period, label, vague, cands = parse_period(question)
    it.period, it.period_label, it.vague_period = period, label, vague
    it.period_candidates = cands
    it.period_source = "explicit" if period else ""

    if re.search(r"\b(drop|dropped|fell|fall|decline|declined|down|worse|dip|dipped|lower)\b", t):
        it.direction = "drop"
    elif re.search(r"\b(rise|rose|increase|increased|up|improve|improved|better|higher)\b", t):
        it.direction = "rise"

    if WRITE_RE.search(t):
        it.kind = "write"
    elif PII_RE.search(t):
        it.kind = "pii"
    elif DUMP_RE.search(t) and not it.group_by and not re.search(r"\b(count|how many|number of|rate)\b", t):
        it.kind = "dump"
    elif WHY_RE.search(t):
        it.kind = "why"
    elif RECON_RE.search(t) and re.search(r"\bsettle|payout|reconcil|disagree|mismatch|differ|gap", t):
        it.kind = "reconcile"
        it.metric = "successful_value"
    if it.kind in ("metric", "why") and not it.period and not it.vague_period:
        it.vague_period = True
        it.period_label = "unspecified"
        it.period_candidates = period_candidates()
    return it


# ----------------------------------------------------------------------------- llm
LLM_SYSTEM = """You turn a payments analytics question into JSON. Today is {today}; data covers {start} to {end} (IST).
Return ONLY a JSON object with these keys:
  kind: one of metric | why | reconcile | dump | write | pii
  metric: one of {metrics}
  filters: object mapping a dimension to a list of values. Dimensions: method {methods}, bank {banks}, region {regions}, device {devices}, category {categories}, merchant (merchant name), failure_code
  period_start, period_end: 'YYYY-MM-DD' (inclusive) or null if the question gives no time period. If the period is vague ("lately", "recently", "last week" without saying full week or last 7 days) return null and set vague_period true.
  vague_period: boolean
  group_by: list of dimensions to split by (may be empty)
  grain: one of hour | day | week | month | null (time series granularity)
  order: asc | desc | null  (asc for worst/lowest, desc for best/highest/most)
  limit: integer or null (1 for "which X")
  denominator_explicit: true only if the question says "all attempts" or "unique orders"
kind rules: why = asks for causes of a change; reconcile = compares transactions with settlements; dump = asks for every raw row; write = asks to change/delete data; pii = asks for customer personal data."""


def llm_plan(question, lookups, context=None):
    m = data_meta()
    sem = semantic()
    sys_prompt = LLM_SYSTEM.format(
        today=today(), start=m["start"], end=m["end"], metrics=list(sem["metrics"]),
        methods=["card", "upi", "netbanking", "wallet"], banks=lookups["banks"],
        regions=REGION_VALUES, devices=["android", "ios", "web", "mobile_web"],
        categories=lookups["categories"])
    user = question if not context else f"Previous question: {context}\nNew question: {question}"
    data, provider = llm.chat_json(sys_prompt, user)
    if not data:
        return None
    it = Intent(question=question, planner=provider)
    if data.get("kind") in ("metric", "why", "reconcile", "dump", "write", "pii"):
        it.kind = data["kind"]
    if data.get("metric") in sem["metrics"]:
        it.metric = data["metric"]
    valid = {"method": {"card", "upi", "netbanking", "wallet"}, "region": set(REGION_VALUES),
             "device": {"android", "ios", "web", "mobile_web"}, "bank": set(lookups["banks"]),
             "category": set(lookups["categories"]),
             "merchant": set(lookups["merchants"].name), "failure_code": None}
    for dim, vals in (data.get("filters") or {}).items():
        if dim not in valid or not vals:
            continue
        vals = vals if isinstance(vals, list) else [vals]
        if valid[dim] is None:
            it.filters[dim] = [str(v).upper() for v in vals]
            continue
        canon = {v.lower(): v for v in valid[dim]}
        keep = [canon[str(v).lower()] for v in vals if str(v).lower() in canon]
        if keep:
            it.filters[dim] = keep
    try:
        if data.get("period_start") and data.get("period_end"):
            a = date.fromisoformat(data["period_start"])
            b = date.fromisoformat(data["period_end"])
            it.period = (min(a, b), max(a, b))
            it.period_label = fmt_range(*it.period)
            it.period_source = "explicit"
    except (TypeError, ValueError):
        pass
    it.vague_period = bool(data.get("vague_period")) and not it.period
    it.group_by = [g for g in (data.get("group_by") or []) if g in semantic()["dimensions"]]
    it.grain = data.get("grain") if data.get("grain") in ("hour", "day", "week", "month") else None
    it.order = data.get("order") if data.get("order") in ("asc", "desc") else None
    it.limit = data.get("limit") if isinstance(data.get("limit"), int) and data["limit"] > 0 else None
    it.denominator_explicit = bool(data.get("denominator_explicit"))
    return it


def plan(question, lookups, context=None, mode=None):
    """Plan with the configured planner; rules supply safety flags regardless."""
    mode = (mode or PLANNER_MODE).lower()
    rules = rules_plan(question, lookups)
    if mode == "rules" or (mode == "auto" and not llm.hosted_available()):
        return rules
    it = llm_plan(question, lookups, context)
    if it is None:
        rules.notes.append("Hosted planner unavailable; used the rules planner.")
        return rules
    # Safety: code decides these, never the model.
    if rules.kind in ("write", "pii", "dump"):
        it.kind = rules.kind
    for dim in ("region",):
        if rules.filters.get(dim):
            it.filters[dim] = sorted(set(it.filters.get(dim, [])) | set(rules.filters[dim]))
    # Deterministic period parsing beats a model's date arithmetic when both exist.
    if rules.period:
        it.period, it.period_label = rules.period, rules.period_label
        it.vague_period = False
    elif rules.vague_period and it.kind in ("metric", "why"):
        it.period, it.vague_period = None, True
        it.period_label = rules.period_label
    it.period_candidates = rules.period_candidates or period_candidates()
    if not it.period and not it.vague_period and it.kind in ("metric", "why"):
        it.vague_period = True
    it.direction = rules.direction
    return it


def apply_patch(it, patch):
    """Apply a clarification choice: {'period': (a, b), 'metric': ...}."""
    if "period" in patch and patch["period"]:
        a, b = patch["period"]
        a = a if isinstance(a, date) else date.fromisoformat(a)
        b = b if isinstance(b, date) else date.fromisoformat(b)
        it.period = (a, b)
        it.period_label = patch.get("period_label") or fmt_range(a, b)
        it.vague_period = False
        it.period_source = patch.get("source", "clarified")
    if patch.get("metric"):
        it.metric = patch["metric"]
        it.denominator_explicit = True
    return it


def intent_from_json(d):
    d = dict(d)
    for k in ("period", "compare"):
        if d.get(k):
            d[k] = tuple(date.fromisoformat(x) for x in d[k])
    return Intent(**d)


def dumps(it):
    return json.dumps(it.to_json(), default=str)
