"""Reuse Heron's own code rather than re-implementing it.

  * utils.SYSTEM     the exact system prompt Heron was fine-tuned with
  * utils.clean_sql  strip fences / trailing statements from a completion
  * repair.repair_identifiers  the scoped, deterministic identifier guard

Heron was trained on SQLite, so it writes SQLite. We keep its frozen prompt
format (system rules + schema DDL + external knowledge + question) and transpile
the output to DuckDB with sqlglot, instead of asking it for a dialect it never
saw in training.
"""

import sys

from .settings import HERON_ROOT

if HERON_ROOT not in sys.path:
    sys.path.insert(0, HERON_ROOT)

from repair import repair_identifiers  # noqa: E402
from utils import SYSTEM as HERON_SYSTEM  # noqa: E402
from utils import clean_sql  # noqa: E402


def heron_user_block(schema_ddl, knowledge, question):
    """Byte-for-byte the layout of utils.user_block() with variant='ddl'.

    utils.user_block needs a BIRD db_map, so the format is reproduced here:
        "Schema:\\n{ddl}\\n" + "\\n" + "External knowledge: ..." + "\\n" + "Question: ..."
    Feeding Heron a different layout silently costs accuracy (see Heron README).
    """
    parts = [f"Schema:\n{schema_ddl}\n", f"External knowledge: {knowledge}",
             f"Question: {question}"]
    return "\n".join(parts)


__all__ = ["HERON_SYSTEM", "clean_sql", "repair_identifiers", "heron_user_block"]
