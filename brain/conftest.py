"""
conftest.py — pytest fixtures for brain/ test suite.

WS-33-01: Brain Directory Test Infrastructure.

Provides:
  - mock_db_conn: a fully mocked psycopg2 connection (no real DB needed)
  - mock_llm: a mocked ChatOpenAI instance (no real API calls)
  - mock_requests: patches the requests module in supervisor (no real HTTP)

Module stubs:
  langchain_openai, langgraph, and related packages are heavy Docker-only
  dependencies. We stub them in sys.modules so supervisor.py can be imported
  in the local test environment without the full LangChain stack.
"""

import sys
import types
from unittest.mock import MagicMock, patch
import pytest

# ── Stub Docker-only dependencies ────────────────────────────────────────────

def _make_stub(name):
    mod = types.ModuleType(name)
    mod.__path__ = []
    return mod

# langchain_openai
_lco = _make_stub("langchain_openai")
_lco.ChatOpenAI = MagicMock
sys.modules.setdefault("langchain_openai", _lco)

# langgraph hierarchy
for _pkg in ["langgraph", "langgraph.graph", "langgraph.graph.message"]:
    sys.modules.setdefault(_pkg, _make_stub(_pkg))

_lg_graph = sys.modules["langgraph.graph"]
# Use a lambda factory so each StateGraph(schema) call returns a fresh MagicMock
# with unrestricted attribute access (add_node, add_edge, compile, etc.)
_lg_graph.StateGraph = lambda *a, **kw: MagicMock()
_lg_graph.END = "__end__"

_lg_msg = sys.modules["langgraph.graph.message"]
_lg_msg.add_messages = lambda x, y: x  # identity reducer stub

# langchain_core (pulled in transitively)
for _pkg in ["langchain_core", "langchain_core.messages"]:
    sys.modules.setdefault(_pkg, _make_stub(_pkg))


@pytest.fixture
def mock_cursor():
    """A mock psycopg2 cursor with configurable fetchone result."""
    cursor = MagicMock()
    cursor.fetchone.return_value = None  # default: no rows
    return cursor


@pytest.fixture
def mock_db_conn(mock_cursor):
    """A mock psycopg2 connection that returns mock_cursor on cursor()."""
    conn = MagicMock()
    conn.cursor.return_value = mock_cursor
    return conn


@pytest.fixture
def patch_db_conn(mock_db_conn):
    """Patches supervisor.get_db_conn to return mock_db_conn."""
    with patch("supervisor.get_db_conn", return_value=mock_db_conn) as patched:
        yield patched


@pytest.fixture
def mock_llm():
    """A mock LLM that returns a canned AI message."""
    llm = MagicMock()
    response = MagicMock()
    response.content = "mocked llm response"
    llm.invoke.return_value = response
    return llm


@pytest.fixture
def patch_requests_post():
    """Patches requests.post in supervisor to avoid real HTTP calls."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True}
    with patch("supervisor.requests.post", return_value=mock_resp) as patched:
        yield patched
