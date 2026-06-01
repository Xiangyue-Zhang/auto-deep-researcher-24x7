import tempfile
import unittest
from pathlib import Path

from core.ledger import (
    ExperimentLedger,
    best_metric,
    check_phase_gate,
    detect_stagnation,
)


class ExperimentLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name)
        self.ledger = ExperimentLedger(self.workspace)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_record_and_all_roundtrip(self):
        self.ledger.record(cycle=1, hypothesis="try lr=1e-3", status="launched",
                            metrics={"acc": 0.5}, ts=1000.0)
        self.ledger.record(cycle=2, hypothesis="try lr=1e-4", status="launched",
                            metrics={"acc": 0.6}, ts=1001.0)
        entries = self.ledger.all()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["cycle"], 1)
        self.assertEqual(entries[1]["metrics"]["acc"], 0.6)

    def test_all_skips_malformed_lines(self):
        self.ledger.record(cycle=1, metrics={"acc": 0.5}, ts=1.0)
        with open(self.ledger.path, "a") as fh:
            fh.write("not json\n")
            fh.write("\n")
        self.assertEqual(len(self.ledger.all()), 1)

    def test_recent_and_summary(self):
        for i in range(8):
            self.ledger.record(cycle=i, hypothesis=f"exp {i}", status="launched",
                               metrics={"acc": i / 10}, ts=float(i))
        self.assertEqual(len(self.ledger.recent(3)), 3)
        summary = self.ledger.summary(3)
        self.assertIn("cycle 7", summary)
        self.assertIn("acc=0.7", summary)
        self.assertNotIn("exp 0", summary)

    def test_recent_zero_returns_empty(self):
        self.ledger.record(cycle=1, metrics={"acc": 0.5}, ts=1.0)
        self.assertEqual(self.ledger.recent(0), [])
        self.assertEqual(self.ledger.summary(0), "")

    def test_summary_tolerates_non_dict_metrics(self):
        with open(self.ledger.path, "a") as fh:
            fh.write('{"cycle": 1, "metrics": ["a", "b"]}\n')
            fh.write('{"cycle": 2, "metrics": "acc=0.5"}\n')
        summary = self.ledger.summary(5)  # must not raise
        self.assertIn("no metrics", summary)

    def test_stagnation_ignores_non_dict_metrics(self):
        entries = [{"metrics": ["x"]}, {"metrics": "y"}, {"metrics": {"acc": 0.5}}]
        verdict = detect_stagnation(entries, "acc", threshold_cycles=3)
        self.assertEqual(verdict["n_points"], 1)

    def test_record_never_raises_on_bad_metrics(self):
        # non-dict metrics should be tolerated by the loop, but record expects a
        # dict; passing None is the common case and must not raise.
        result = self.ledger.record(cycle=1, metrics=None, ts=1.0)
        self.assertEqual(result["metrics"], {})

    def test_best_metric_direction(self):
        entries = [{"metrics": {"loss": 0.9}}, {"metrics": {"loss": 0.3}}, {"metrics": {"loss": 0.5}}]
        self.assertEqual(best_metric(entries, "loss", "lower_better"), 0.3)
        self.assertEqual(best_metric(entries, "loss", "higher_better"), 0.9)
        self.assertIsNone(best_metric(entries, "missing", "higher_better"))


class StagnationTests(unittest.TestCase):
    def test_no_metric_key(self):
        verdict = detect_stagnation([{"metrics": {"acc": 0.5}}], "")
        self.assertFalse(verdict["stagnating"])
        self.assertIn("no metric_key", verdict["reason"])

    def test_not_enough_points(self):
        entries = [{"metrics": {"acc": 0.5}}, {"metrics": {"acc": 0.6}}]
        verdict = detect_stagnation(entries, "acc", threshold_cycles=3)
        self.assertFalse(verdict["stagnating"])
        self.assertIn("not enough", verdict["reason"])

    def test_improving_is_not_stagnating(self):
        entries = [{"metrics": {"acc": v}} for v in (0.1, 0.2, 0.3, 0.4, 0.5)]
        verdict = detect_stagnation(entries, "acc", direction="higher_better", threshold_cycles=3)
        self.assertFalse(verdict["stagnating"])
        self.assertEqual(verdict["best"], 0.5)
        self.assertEqual(verdict["cycles_since_improvement"], 0)

    def test_flat_trajectory_is_stagnating(self):
        entries = [{"metrics": {"acc": v}} for v in (0.8, 0.81, 0.81, 0.81, 0.81)]
        verdict = detect_stagnation(entries, "acc", direction="higher_better",
                                    threshold_cycles=3, min_delta=0.0)
        self.assertTrue(verdict["stagnating"])
        self.assertGreaterEqual(verdict["cycles_since_improvement"], 3)

    def test_min_delta_suppresses_tiny_improvements(self):
        entries = [{"metrics": {"acc": v}} for v in (0.80, 0.801, 0.802, 0.803, 0.804)]
        verdict = detect_stagnation(entries, "acc", direction="higher_better",
                                    threshold_cycles=3, min_delta=0.01)
        self.assertTrue(verdict["stagnating"])


class PhaseGateTests(unittest.TestCase):
    def test_gate_met_higher_better(self):
        entries = [{"metrics": {"acc": 0.85}}]
        gate = check_phase_gate(entries, "acc", threshold=0.8, direction="higher_better")
        self.assertTrue(gate["gate_met"])

    def test_gate_not_met(self):
        entries = [{"metrics": {"acc": 0.7}}]
        gate = check_phase_gate(entries, "acc", threshold=0.8, direction="higher_better")
        self.assertFalse(gate["gate_met"])
        self.assertIn("0.8", gate["blocker_reason"])

    def test_gate_no_metric_yet(self):
        gate = check_phase_gate([], "acc", threshold=0.8)
        self.assertFalse(gate["gate_met"])
        self.assertIn("no metric", gate["blocker_reason"])


if __name__ == "__main__":
    unittest.main()
