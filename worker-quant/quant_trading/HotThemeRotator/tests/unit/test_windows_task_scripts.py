"""Static contracts for local Windows Task Scheduler helper scripts."""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_script(name: str) -> str:
    return (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_price_health_task_script_registers_local_observability_job():
    script = _read_script("register_price_health_task.bat")

    assert "HTR_Price_Health_Report" in script
    assert "tools\\write_price_health_report.py" in script
    assert "--symbols" in script
    assert "--base-dir" in script
    assert "/SC MINUTE" in script
    assert "/MO 15" in script
    assert "/F" in script


def test_price_health_task_script_preserves_stage_zero_boundaries():
    script = _read_script("register_price_health_task.bat").lower()

    assert "observability-only" in script
    assert "notification" not in script.replace("no notification", "")
    assert "broker" not in script.replace("no broker", "")
    assert "order" not in script.replace("no order", "")
    assert "telegram" not in script
    assert "email" not in script
