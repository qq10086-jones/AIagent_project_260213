import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from db_update import TARGET_UNIVERSE


class TestDbUpdateUniverse(unittest.TestCase):
    def test_vix_ticker_in_default_universe(self):
        symbols = {symbol for symbol, _name, _sector in TARGET_UNIVERSE}
        self.assertIn("1552.T", symbols)


if __name__ == "__main__":
    unittest.main()
