import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# Tests never call external services: Heron and hosted models are off.
os.environ["HERON_MODE"] = "off"
os.environ["PLANNER_MODE"] = "rules"
os.environ["POLISH"] = "off"

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def leader():
    from engine.db import role_db
    return role_db("payments_leader")


@pytest.fixture(scope="session")
def south():
    from engine.db import role_db
    return role_db("regional_analyst_south")


@pytest.fixture
def audit_path(tmp_path):
    return str(tmp_path / "audit.duckdb")
