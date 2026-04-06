import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation_clock import SimulationClock


class TestSimulationClock(unittest.TestCase):
    def test_clock_persists_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "simulation_state.json"
            trading_dates = ["2026-02-03", "2026-02-04", "2026-02-05"]

            clock = SimulationClock.load_or_create(
                start_asof="2026-02-03",
                end_asof="2026-02-05",
                trading_dates=trading_dates,
                state_path=state_path,
                resume=False,
            )
            self.assertEqual(clock.current_asof(), "2026-02-03")
            clock.mark_completed()
            self.assertEqual(clock.last_completed_asof, "2026-02-03")

            resumed = SimulationClock.load_or_create(
                start_asof="2026-02-03",
                end_asof="2026-02-05",
                trading_dates=trading_dates,
                state_path=state_path,
                resume=True,
            )
            self.assertEqual(resumed.current_asof(), "2026-02-04")
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["completed_days"], 1)


if __name__ == "__main__":
    unittest.main()
