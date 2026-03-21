"""
离线单元测试：_merge_recent_news 降级与合并逻辑

覆盖场景：
1. OpenBB 正常返回足量新闻 → 不触发 fallback
2. OpenBB 返回不足 → 触发 quote_page + google_rss fallback
3. OpenBB 超时 → 完全降级到 fallback
4. OpenBB 抛异常 → 完全降级到 fallback
5. 重复标题去重
6. blocked_publishers 过滤
7. login.yahoo.co.jp URL 过滤
8. 评分排序（trusted publisher > 普通）
9. 空 quote 输入不崩溃
10. max_items 上限截断
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# 将 worker-quant 根目录加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from worker import _merge_recent_news


def _make_news_item(title, url="http://example.com", publisher="unknown", source="test"):
    return {"title": title, "url": url, "publisher": publisher,
            "published_at": "2026-01-01", "published_ts": None, "source": source}


class TestMergeRecentNews(unittest.TestCase):

    # ------------------------------------------------------------------ #
    # 1. OpenBB 正常返回足量 → 不调用 fallback
    # ------------------------------------------------------------------ #
    @patch("worker._fetch_news_from_google_rss", return_value=[])
    @patch("worker._fetch_news_from_quote_page", return_value=[])
    @patch("worker._fetch_news_from_openbb")
    def test_openbb_sufficient_no_fallback(self, mock_obb, mock_qp, mock_rss):
        mock_obb.return_value = [
            _make_news_item(f"News {i}", publisher="reuters") for i in range(6)
        ]
        quote = {"symbol": "AAPL", "recent_news": []}
        result = _merge_recent_news(quote, max_items=3)

        self.assertEqual(len(result), 3)
        mock_qp.assert_not_called()
        mock_rss.assert_not_called()

    # ------------------------------------------------------------------ #
    # 2. OpenBB 返回不足 → 触发 fallback
    # ------------------------------------------------------------------ #
    @patch("worker._fetch_news_from_google_rss")
    @patch("worker._fetch_news_from_quote_page")
    @patch("worker._fetch_news_from_openbb")
    def test_openbb_insufficient_triggers_fallback(self, mock_obb, mock_qp, mock_rss):
        mock_obb.return_value = [_make_news_item("OBB News 1")]  # 1 < max_items=3
        mock_qp.return_value = [_make_news_item("QP News 1")]
        mock_rss.return_value = [_make_news_item("RSS News 1")]

        quote = {"symbol": "7203.T", "recent_news": []}
        result = _merge_recent_news(quote, max_items=3)

        mock_qp.assert_called_once()
        mock_rss.assert_called_once()
        titles = {r["title"] for r in result}
        self.assertIn("OBB News 1", titles)
        self.assertIn("QP News 1", titles)

    # ------------------------------------------------------------------ #
    # 3. OpenBB 超时 → 返回 [] → 完全降级
    # ------------------------------------------------------------------ #
    @patch("worker._fetch_news_from_google_rss")
    @patch("worker._fetch_news_from_quote_page")
    @patch("worker._fetch_news_from_openbb", return_value=[])
    def test_openbb_timeout_full_fallback(self, mock_obb, mock_qp, mock_rss):
        mock_qp.return_value = [_make_news_item("QP Only")]
        mock_rss.return_value = [_make_news_item("RSS Only")]

        quote = {"symbol": "TSLA", "recent_news": []}
        result = _merge_recent_news(quote, max_items=5)

        mock_qp.assert_called_once()
        mock_rss.assert_called_once()
        titles = {r["title"] for r in result}
        self.assertIn("QP Only", titles)
        self.assertIn("RSS Only", titles)

    # ------------------------------------------------------------------ #
    # 4. OpenBB 抛异常 → _fetch_news_from_openbb 内部 catch → 返回 []
    # ------------------------------------------------------------------ #
    @patch("worker._fetch_news_from_google_rss", return_value=[_make_news_item("RSS Fallback")])
    @patch("worker._fetch_news_from_quote_page", return_value=[])
    @patch("worker._fetch_news_from_openbb", side_effect=Exception("network error"))
    def test_openbb_exception_full_fallback(self, mock_obb, mock_qp, mock_rss):
        # _fetch_news_from_openbb 内部已 catch exception 返回 [],
        # 但若 mock side_effect 直接抛出，验证 _merge_recent_news 也不崩溃
        quote = {"symbol": "NVDA", "recent_news": []}
        # 如果内层没有 catch，这里测试 _merge_recent_news 的健壮性
        try:
            result = _merge_recent_news(quote, max_items=5)
        except Exception:
            self.fail("_merge_recent_news should not propagate exceptions from _fetch_news_from_openbb")

    # ------------------------------------------------------------------ #
    # 5. 重复标题去重
    # ------------------------------------------------------------------ #
    @patch("worker._fetch_news_from_google_rss", return_value=[])
    @patch("worker._fetch_news_from_quote_page", return_value=[])
    @patch("worker._fetch_news_from_openbb")
    def test_deduplication(self, mock_obb, mock_qp, mock_rss):
        mock_obb.return_value = [
            _make_news_item("Duplicate Title"),
            _make_news_item("Duplicate Title"),
            _make_news_item("unique news"),
        ]
        quote = {"symbol": "MSFT", "recent_news": []}
        result = _merge_recent_news(quote, max_items=5)

        titles = [r["title"] for r in result]
        self.assertEqual(titles.count("Duplicate Title"), 1)

    # ------------------------------------------------------------------ #
    # 6. blocked_publishers 过滤
    # ------------------------------------------------------------------ #
    @patch("worker._fetch_news_from_google_rss", return_value=[])
    @patch("worker._fetch_news_from_quote_page", return_value=[])
    @patch("worker._fetch_news_from_openbb")
    def test_blocked_publisher_filtered(self, mock_obb, mock_qp, mock_rss):
        mock_obb.return_value = [
            _make_news_item("Blocked News", publisher="Meyka"),
            _make_news_item("Good News", publisher="reuters"),
        ]
        quote = {"symbol": "AAPL", "recent_news": []}
        result = _merge_recent_news(quote, max_items=5)

        titles = {r["title"] for r in result}
        self.assertNotIn("Blocked News", titles)
        self.assertIn("Good News", titles)

    # ------------------------------------------------------------------ #
    # 7. login.yahoo.co.jp URL 过滤
    # ------------------------------------------------------------------ #
    @patch("worker._fetch_news_from_google_rss", return_value=[])
    @patch("worker._fetch_news_from_quote_page", return_value=[])
    @patch("worker._fetch_news_from_openbb")
    def test_yahoo_login_url_filtered(self, mock_obb, mock_qp, mock_rss):
        mock_obb.return_value = [
            _make_news_item("Gated Yahoo", url="https://login.yahoo.co.jp/config/login"),
            _make_news_item("Open Article", url="https://finance.yahoo.co.jp/news/detail/abc"),
        ]
        quote = {"symbol": "6758.T", "recent_news": []}
        result = _merge_recent_news(quote, max_items=5)

        titles = {r["title"] for r in result}
        self.assertNotIn("Gated Yahoo", titles)
        self.assertIn("Open Article", titles)

    # ------------------------------------------------------------------ #
    # 8. 评分排序：trusted publisher 排在前
    # ------------------------------------------------------------------ #
    @patch("worker._fetch_news_from_google_rss", return_value=[])
    @patch("worker._fetch_news_from_quote_page", return_value=[])
    @patch("worker._fetch_news_from_openbb")
    def test_trusted_publisher_ranked_higher(self, mock_obb, mock_qp, mock_rss):
        mock_obb.return_value = [
            _make_news_item("Low Trust", publisher="some_blog"),
            _make_news_item("High Trust", publisher="bloomberg"),
        ]
        quote = {"symbol": "GS", "recent_news": []}
        result = _merge_recent_news(quote, max_items=5)

        self.assertEqual(result[0]["title"], "High Trust")

    # ------------------------------------------------------------------ #
    # 9. 空 quote 输入不崩溃
    # ------------------------------------------------------------------ #
    @patch("worker._fetch_news_from_google_rss", return_value=[])
    @patch("worker._fetch_news_from_quote_page", return_value=[])
    @patch("worker._fetch_news_from_openbb", return_value=[])
    def test_empty_quote_no_crash(self, mock_obb, mock_qp, mock_rss):
        for bad_input in [None, {}, {"symbol": None}]:
            try:
                result = _merge_recent_news(bad_input, max_items=3)
                self.assertIsInstance(result, list)
            except Exception as e:
                self.fail(f"_merge_recent_news crashed on input {bad_input!r}: {e}")

    # ------------------------------------------------------------------ #
    # 10. max_items 截断
    # ------------------------------------------------------------------ #
    @patch("worker._fetch_news_from_google_rss", return_value=[])
    @patch("worker._fetch_news_from_quote_page", return_value=[])
    @patch("worker._fetch_news_from_openbb")
    def test_max_items_truncation(self, mock_obb, mock_qp, mock_rss):
        mock_obb.return_value = [_make_news_item(f"News {i}") for i in range(20)]
        quote = {"symbol": "AMZN", "recent_news": []}
        result = _merge_recent_news(quote, max_items=5)

        self.assertLessEqual(len(result), 5)


if __name__ == "__main__":
    unittest.main()
