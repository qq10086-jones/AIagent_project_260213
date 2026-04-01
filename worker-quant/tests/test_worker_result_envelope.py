import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from worker_result_helpers import (
    build_worker_result_envelope,
    hash_worker_output,
    validate_single_agent_payload_guardrails,
    validate_single_agent_worker_result_guardrails,
)


class TestWorkerResultEnvelope(unittest.TestCase):
    def test_hash_stable_across_key_order(self):
        self.assertEqual(
            hash_worker_output({"b": 2, "a": 1}),
            hash_worker_output({"a": 1, "b": 2}),
        )

    def test_build_worker_result_envelope(self):
        result = build_worker_result_envelope(
            run_id="run-quant-1",
            worker_name="worker-quant",
            ok=True,
            status="succeeded",
            output={"summary": "briefing ok"},
            duration_ms=15,
            tool_calls=1,
            bounded_validation=[{"name": "shape_check", "ok": True}],
            replay_tag="rt-q1",
            evidence_id="ev-q1",
        )
        self.assertEqual(result["run_id"], "run-quant-1")
        self.assertEqual(result["worker_name"], "worker-quant")
        self.assertTrue(result["ok"])
        self.assertEqual(result["metadata"]["duration_ms"], 15)
        self.assertEqual(result["metadata"]["tool_calls"], 1)
        self.assertEqual(len(result["metadata"]["output_hash"]), 64)
        self.assertEqual(result["metadata"]["replay_tag"], "rt-q1")

    def test_single_agent_guardrails(self):
        ok, errors = validate_single_agent_payload_guardrails({
            "task_envelope": {"decision": "single_agent", "evidence_id": "ev-q1", "replay_tag": "rt-q1"},
            "evidence_id": "ev-q1",
            "replay_tag": "rt-q1",
        })
        self.assertTrue(ok)
        self.assertEqual(errors, [])

        ok, errors = validate_single_agent_payload_guardrails({
            "task_envelope": {"decision": "single_agent"},
        })
        self.assertFalse(ok)
        self.assertGreater(len(errors), 0)

        ok, errors = validate_single_agent_worker_result_guardrails({
            "metadata": {
                "evidence_id": "ev-q1",
                "replay_tag": "rt-q1",
                "output_hash": "hash",
                "bounded_validation": [{"name": "shape", "ok": True}],
            }
        })
        self.assertTrue(ok)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
