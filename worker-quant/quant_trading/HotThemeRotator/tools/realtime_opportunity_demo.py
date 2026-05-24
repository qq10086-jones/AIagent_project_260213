"""Run a sample realtime opportunity candidate panel.

This demo uses fixture-like in-memory data. It does not fetch the internet and
does not create orders. Output uses the v2 per-candidate sectioned markdown
format and adds subtle ANSI coloring when stdout is a TTY.
"""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.schema import NewsItem
from hot_theme_rotator.data.free_web_opportunity_adapter import (
    FreeWebContextSnapshot,
    FreeWebOpportunityAdapter,
    FreeWebQuote,
)
from hot_theme_rotator.opportunity.opportunity_scanner import scan_opportunities
from hot_theme_rotator.opportunity.price_ladder import build_price_ladder
from hot_theme_rotator.reporting.realtime_opportunity_panel import (
    OpportunityPanelRow,
    render_realtime_opportunity_panel_markdown_v2,
)


ASOF = "2026-05-23T09:10:00+09:00"

# ANSI codes. Subtle — bold for section headings, dim for metadata, reset.
_ANSI_BOLD = "\x1b[1m"
_ANSI_DIM = "\x1b[2m"
_ANSI_RESET = "\x1b[0m"


class DemoQuoteClient:
    def fetch_quotes(self, symbols):
        quotes = {
            "8035.T": FreeWebQuote(
                symbol="8035.T",
                available_ts="2026-05-23T09:05:00+09:00",
                open=44100,
                high=46350,
                low=42750,
                close=45000,
                volume=1_000_000,
                previous_close=43650,
                avg_volume_20d=500_000,
            ),
            "7203.T": FreeWebQuote(
                symbol="7203.T",
                available_ts="2026-05-23T09:05:00+09:00",
                open=2940,
                high=3090,
                low=2850,
                close=3000,
                volume=1_000_000,
                previous_close=2910,
                avg_volume_20d=700_000,
            ),
        }
        return [quotes[symbol] for symbol in symbols if symbol in quotes]


class DemoNewsClient:
    def fetch_news(self, symbols, since_ts, until_ts):
        return [
            NewsItem.from_dict(
                {
                    "news_id": "demo-ai",
                    "available_ts": "2026-05-23T09:04:00+09:00",
                    "source": "demo",
                    "headline": "AI semiconductor demand expands",
                    "body": "",
                    "symbols": ["8035.T"],
                }
            ),
            NewsItem.from_dict(
                {
                    "news_id": "demo-fx",
                    "available_ts": "2026-05-23T09:04:00+09:00",
                    "source": "demo",
                    "headline": "Exporters gain from weaker yen",
                    "body": "",
                    "symbols": ["7203.T"],
                }
            ),
        ]


class DemoContextClient:
    def fetch_context(self, symbols):
        return {
            "8035.T": FreeWebContextSnapshot(
                symbol="8035.T",
                available_ts="2026-05-23T09:03:00+09:00",
                market_context_score=0.30,
            ),
            "7203.T": FreeWebContextSnapshot(
                symbol="7203.T",
                available_ts="2026-05-23T09:03:00+09:00",
                market_context_score=0.10,
            ),
        }


def build_demo_markdown() -> str:
    adapter = FreeWebOpportunityAdapter(
        quote_client=DemoQuoteClient(),
        news_client=DemoNewsClient(),
        context_client=DemoContextClient(),
    )
    inputs = adapter.build_opportunity_inputs(
        symbols=["8035.T", "7203.T"],
        decision_cutoff=ASOF,
    )
    scan = scan_opportunities(inputs=inputs, decision_cutoff=ASOF, top_n=2)
    bar_by_symbol = {item.bar.symbol: item.bar for item in inputs}
    rows = tuple(
        OpportunityPanelRow(
            candidate=candidate,
            ladder=build_price_ladder(bar_by_symbol[candidate.symbol]),
        )
        for candidate in scan.candidates
    )
    return render_realtime_opportunity_panel_markdown_v2(asof=ASOF, rows=rows)


def colorize_for_tty(markdown_text: str, *, stream=None) -> str:
    """Wrap section headings + metadata with ANSI escapes if stream is a TTY.

    Returns the input unchanged when stream is not a TTY, so piped/redirected
    output (and test stdout capture) remain ANSI-free.
    """
    target = stream if stream is not None else sys.stdout
    if not getattr(target, "isatty", lambda: False)():
        return markdown_text
    out: list[str] = []
    for line in markdown_text.splitlines():
        if line.startswith("## "):
            out.append(f"{_ANSI_BOLD}{line}{_ANSI_RESET}")
        elif line.startswith("- 数据缺口") or line.startswith("- 理由"):
            out.append(f"{_ANSI_DIM}{line}{_ANSI_RESET}")
        else:
            out.append(line)
    return "\n".join(out)


def main() -> int:
    print(colorize_for_tty(build_demo_markdown()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
