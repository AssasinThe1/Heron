"""The six-part answer, its number formatter, and the number-matching check.

Text is written by filling templates from computed results. Every number goes
through `Fmt`, which records the exact string it rendered. `number_check`
then extracts every number in the final text and rejects the answer if one was
not produced by Fmt from a query result. A hosted model may polish wording;
its output must pass the same check or the template text is kept.
"""

import json
import re
from dataclasses import dataclass, field

from . import llm
from .settings import POLISH

NUM_RE = re.compile(r"(?<![A-Za-z_])\d[\d,]*(?:\.\d+)?")
TAG_RE = re.compile(r"\[E\d+(?:,\s*E\d+)*\]")


class Fmt:
    """Formats numbers and remembers every number it produced."""

    def __init__(self):
        self.allowed = set()
        self.labels = set()

    def _reg(self, s):
        for tok in NUM_RE.findall(s):
            self.allowed.add(tok.replace(",", ""))
        return s

    def pct(self, x, d=1):            # 0-100 scale in, "80.1%" out
        return self._reg(f"{x:.{d}f}%".replace("-", "−"))

    def share(self, x, d=0):          # 0-1 scale in
        return self._reg(f"{100 * x:.{d}f}%")

    def pp(self, x, d=1, sign=True):
        s = f"{x:+.{d}f}" if sign else f"{abs(x):.{d}f}"
        s = s.replace("-", "−")
        return self._reg(f"{s} pp")

    def n(self, x):
        return self._reg(f"{int(round(x)):,}")

    def num(self, x, d=1):
        return self._reg(f"{x:,.{d}f}")

    def inr(self, x):
        """Indian convention: crore (1e7) and lakh (1e5)."""
        a = abs(x)
        sign = "−" if x < 0 and a >= 0.5 else ""
        if a >= 1e7:
            s = f"{sign}₹{a / 1e7:,.2f} crore"
        elif a >= 1e5:
            s = f"{sign}₹{a / 1e5:,.2f} lakh"
        else:
            s = f"{sign}₹{a:,.0f}"
        return self._reg(s)

    def date(self, d):
        return self._reg(f"{d.day} {d:%b}")

    def drange(self, a, b):
        from .planner import fmt_range
        return self._reg(fmt_range(a, b))

    def hour(self, h):
        return self._reg(f"{int(h):02d}:00")

    def label(self, s):
        """Names that contain digits (e.g. 3DS_AUTH_FAILED, 'Week 42')."""
        s = str(s)
        self.labels.add(s)
        return s

    def ts(self, t):
        return self._reg(f"{t:%d %b %H:%M}")


def number_check(text, fmt):
    """-> (ok, unmatched_numbers)."""
    t = TAG_RE.sub("", text)
    for lab in sorted(fmt.labels, key=len, reverse=True):
        t = t.replace(lab, "")
    bad = [tok for tok in NUM_RE.findall(t) if tok.replace(",", "") not in fmt.allowed]
    return (not bad), bad


@dataclass
class Answer:
    kind: str = "answer"            # answer | clarify | refusal | error | no_data
    headline: str = ""
    facts: list = field(default_factory=list)
    hypotheses: list = field(default_factory=list)   # dicts: text, confidence, why, for, against
    actions: list = field(default_factory=list)
    caveats: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    charts: list = field(default_factory=list)       # (title, figure, evidence_id)
    tables: list = field(default_factory=list)       # (title, DataFrame)
    clarify: dict = None
    refusal: dict = None
    number_check: dict = None
    value: object = None            # the primary number (for evaluation)
    top_segment: object = None      # the primary segment label (for evaluation)
    causes: list = field(default_factory=list)       # machine-readable cause tags (eval)
    meta: dict = field(default_factory=dict)
    fmt: object = None

    def text(self):
        parts = [self.headline] + self.facts
        for h in self.hypotheses:
            parts += [h["text"]] + h.get("for", []) + h.get("against", [])
        parts += self.actions + self.caveats
        return "\n".join(p for p in parts if p)

    def markdown(self):
        out = [f"**{self.headline}**", ""]
        if self.facts:
            out += ["**Observed facts**"] + [f"- {f}" for f in self.facts] + [""]
        if self.hypotheses:
            out.append("**Hypotheses**")
            for h in self.hypotheses:
                out.append(f"- *Hypothesis ({h['confidence']} confidence):* {h['text']}")
                out += [f"  - For: {x}" for x in h.get("for", [])]
                out += [f"  - Against: {x}" for x in h.get("against", [])]
            out.append("")
        if self.actions:
            out += ["**Recommended actions**"] + [f"- *Recommendation:* {a}" for a in self.actions] + [""]
        out += ["**Caveats**"] + ([f"- {c}" for c in self.caveats] or ["- None found."])
        return "\n".join(out)


def finalize(ans):
    """Run the number check; optionally polish wording (and re-check)."""
    fmt = ans.fmt or Fmt()
    ans.facts = [_ucfirst(f) for f in ans.facts]
    for h in ans.hypotheses:
        h["text"] = _ucfirst(h["text"])
        h["for"] = [_ucfirst(x) for x in h.get("for", [])]
        h["against"] = [_ucfirst(x) for x in h.get("against", [])]
    if POLISH and llm.hosted_available() and ans.kind == "answer":
        _polish(ans, fmt)
    ok, bad = number_check(ans.text(), fmt)
    ans.number_check = {"ok": ok, "unmatched": bad}
    if not ok:
        ans.caveats.append("Number check flagged values not traced to a query: "
                           + ", ".join(bad) + ". Treat them with care.")
    return ans


def _ucfirst(s):
    return s[:1].upper() + s[1:] if s and not s.startswith("iOS") else s


def _polish(ans, fmt):
    items = {"headline": ans.headline, "facts": ans.facts}
    sys = ("Rewrite each string for a busy payments executive: shorter, plainer. Keep every number, "
           "unit and [E#] tag exactly as written. Do not add facts or numbers. Return JSON with the "
           "same keys and list lengths.")
    data, _ = llm.chat_json(sys, json.dumps(items, ensure_ascii=False))
    if not data:
        return
    new_head = data.get("headline")
    new_facts = data.get("facts")
    if isinstance(new_head, str) and number_check(new_head, fmt)[0] \
            and set(TAG_RE.findall(new_head)) == set(TAG_RE.findall(ans.headline)):
        ans.headline = new_head
    if isinstance(new_facts, list) and len(new_facts) == len(ans.facts):
        for i, (old, new) in enumerate(zip(ans.facts, new_facts)):
            if isinstance(new, str) and number_check(new, fmt)[0] \
                    and set(TAG_RE.findall(new)) == set(TAG_RE.findall(old)):
                ans.facts[i] = new
    ans.meta["polished"] = True
