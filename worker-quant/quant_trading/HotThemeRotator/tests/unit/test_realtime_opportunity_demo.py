import importlib.util
import io
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_PATH = PROJECT_ROOT / "tools" / "realtime_opportunity_demo.py"


def _load_demo_module():
    spec = importlib.util.spec_from_file_location("realtime_opportunity_demo", DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_realtime_opportunity_demo_builds_v2_sample_panel():
    module = _load_demo_module()
    markdown = module.build_demo_markdown()

    # v2 sectioned markdown — no wide table, one heading per candidate
    assert "# Realtime Opportunity Candidate Panel (v2)" in markdown
    assert "## 1 · 8035.T · ai_semiconductor" in markdown
    assert "## 2 · 7203.T · fx_export" in markdown
    assert "uncalibrated score, not win rate" in markdown
    assert "买入三档" in markdown
    assert "止损" in markdown
    # v1 wide-table header must not be present
    assert "| rank | symbol | trigger_theme" not in markdown


def test_colorize_for_tty_is_a_noop_when_stream_is_not_a_tty():
    module = _load_demo_module()
    plain_stream = io.StringIO()  # StringIO has no isatty() truthy by default
    markdown = module.build_demo_markdown()
    out = module.colorize_for_tty(markdown, stream=plain_stream)
    assert out == markdown
    assert "\x1b[" not in out


def test_colorize_for_tty_adds_ansi_for_tty_streams():
    module = _load_demo_module()

    class FakeTTY:
        def isatty(self) -> bool:
            return True

    markdown = module.build_demo_markdown()
    out = module.colorize_for_tty(markdown, stream=FakeTTY())
    assert "\x1b[1m## 1 · 8035.T · ai_semiconductor\x1b[0m" in out
    assert "\x1b[2m- 理由" in out
