"""Paths, environment and the two YAML rule files. One place to look."""

import os
from datetime import date
from functools import lru_cache

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERON_ROOT = os.path.dirname(ROOT)                    # the Heron repo this lives in
DATA_DB = os.path.join(ROOT, "data", "payinsight.duckdb")
AUDIT_DB = os.path.join(ROOT, "data", "audit.duckdb")
SEMANTIC_PATH = os.path.join(ROOT, "config", "semantic_layer.yaml")
POLICY_PATH = os.path.join(ROOT, "config", "policy.yaml")
EVAL_RESULTS = os.path.join(ROOT, "eval", "results.json")


def _load_dotenv():
    """Minimal .env reader (KEY=VALUE lines); real env vars win."""
    for path in (os.path.join(ROOT, ".env"), os.path.join(HERON_ROOT, ".env")):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

HERON_URL = os.environ.get("HERON_URL", "https://assasinthe1--qwen7b-t2s-serve.modal.run").rstrip("/")
HERON_MODEL = os.environ.get("HERON_MODEL", "AssassinThe1/qwen7b-t2s-ddl-merged")
HERON_TIMEOUT = float(os.environ.get("HERON_TIMEOUT", "150"))     # cold start is ~77 s
# auto: try Heron, fall back to the semantic-layer compiler if it is unreachable.
# off:  never call Heron (fully offline; every query comes from the compiler).
HERON_MODE = os.environ.get("HERON_MODE", "auto").lower()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
# auto: use a hosted model for understanding if a key exists, else rules.
PLANNER_MODE = os.environ.get("PLANNER_MODE", "auto").lower()
POLISH = os.environ.get("POLISH", "off").lower() == "on"


@lru_cache(maxsize=1)
def semantic():
    with open(SEMANTIC_PATH) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def policy():
    with open(POLICY_PATH) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def data_meta():
    import duckdb
    if not os.path.exists(DATA_DB):
        raise FileNotFoundError(f"{DATA_DB} missing. Run: python data/generate.py")
    con = duckdb.connect(DATA_DB, read_only=True)
    try:
        meta = dict(con.execute("SELECT key, value FROM _meta").fetchall())
    finally:
        con.close()
    meta["as_of_date"] = date.fromisoformat(meta["as_of"])
    meta["start_date"] = date.fromisoformat(meta["start"])
    meta["end_date"] = date.fromisoformat(meta["end"])
    return meta
