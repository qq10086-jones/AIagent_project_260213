from __future__ import annotations

import argparse
import json
from pathlib import Path

from daily_run import configure_alert_env, load_cfg, run_alert_webhook_self_test


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a webhook self-test for runtime alerts.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--reports_dir", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    configure_alert_env(cfg)
    reports_dir = Path(args.reports_dir or str((cfg.get("model", {}) or {}).get("output_dir", "reports")))
    result = run_alert_webhook_self_test(reports_dir, cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
