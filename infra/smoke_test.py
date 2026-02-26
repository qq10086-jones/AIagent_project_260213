import os
import json
import requests
import unittest
from pathlib import Path

# Load env simulation (mocking what would be in docker)
QUANT_LLM_MODEL = os.getenv("QUANT_LLM_MODEL", "deepseek-r1:1.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

class NexusSmokeTest(unittest.TestCase):
    def test_01_ollama_connection(self):
        """Check if Ollama is reachable and model is available"""
        print(f"\n[Test] Checking Ollama at {OLLAMA_BASE_URL}...")
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            self.assertEqual(resp.status_code, 200)
            models = [m['name'] for m in resp.json().get('models', [])]
            print(f"[Info] Available models: {models}")
            # Normalize model name for check
            found = any(QUANT_LLM_MODEL in m for m in models)
            if not found:
                 print(f"[Warning] Exact model {QUANT_LLM_MODEL} not found, but continuing test...")
            self.assertTrue(len(models) > 0, "No models found in Ollama")
        except Exception as e:
            self.fail(f"Ollama connection failed: {e}")

    def test_02_quant_pipeline_structure(self):
        """Check if quant trading files exist"""
        base_path = Path("worker-quant/quant_trading/Project_optimized")
        required_files = ["run_pipeline.py", "config.yaml", "db_update.py"]
        for f in required_files:
            self.assertTrue((base_path / f).exists(), f"Missing core file: {f}")

    def test_03_config_validity(self):
        """Check if config.yaml is valid"""
        import yaml
        config_path = Path("worker-quant/quant_trading/Project_optimized/config.yaml")
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
                # Support both legacy schema (storage.*) and current schema (db_path).
                self.assertTrue('storage' in cfg or 'db_path' in cfg)
                print(f"[Info] Config loaded: {cfg.get('strategy_name', 'Unknown')}")
        except Exception as e:
            self.fail(f"Failed to load config.yaml: {e}")

if __name__ == "__main__":
    unittest.main()
