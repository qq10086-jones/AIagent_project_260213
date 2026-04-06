import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daily_run import _format_discord_webhook_payload, _is_discord_webhook_url


class TestAlertWebhookFormat(unittest.TestCase):
    def test_detects_discord_webhook_urls(self):
        self.assertTrue(_is_discord_webhook_url("https://discord.com/api/webhooks/1/abc"))
        self.assertTrue(_is_discord_webhook_url("https://discordapp.com/api/webhooks/1/abc"))
        self.assertFalse(_is_discord_webhook_url("https://example.com/hook"))

    def test_formats_discord_embed_payload(self):
        payload = _format_discord_webhook_payload(
            {
                "ts_utc": "2026-04-05T00:00:00+00:00",
                "level": "warning",
                "event": "alert_webhook_self_test",
                "source": "manual_self_test",
                "strategy_id": "sprint",
            }
        )
        self.assertEqual(payload["username"], "worker-quant")
        self.assertIn("embeds", payload)
        self.assertEqual(len(payload["embeds"]), 1)
        embed = payload["embeds"][0]
        self.assertIn("alert_webhook_self_test", embed["title"])
        self.assertIn("strategy_id", embed["description"])
        self.assertEqual(embed["timestamp"], "2026-04-05T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
