import unittest

from core.safety import prune_timestamps, scan_violations, seconds_until_allowed


class ScanViolationsTests(unittest.TestCase):
    def test_high_fail_count_flagged(self):
        viols = scan_violations({}, fail_count=3, now=1000.0, fail_threshold=3)
        self.assertEqual(len(viols), 1)
        self.assertIn("consecutive no-progress", viols[0])

    def test_below_threshold_not_flagged(self):
        self.assertEqual(scan_violations({}, fail_count=2, now=1000.0, fail_threshold=3), [])

    def test_stale_running_state_flagged(self):
        state = {"status": "running", "updated_at": 0.0}
        now = 7 * 3600  # 7 hours later
        viols = scan_violations(state, fail_count=0, now=now, stale_state_hours=6)
        self.assertEqual(len(viols), 1)
        self.assertIn("running", viols[0])

    def test_fresh_running_state_not_flagged(self):
        state = {"status": "running", "updated_at": 1000.0}
        viols = scan_violations(state, fail_count=0, now=1000.0 + 60, stale_state_hours=6)
        self.assertEqual(viols, [])

    def test_completed_state_not_flagged_for_staleness(self):
        state = {"status": "completed", "updated_at": 0.0}
        viols = scan_violations(state, fail_count=0, now=99 * 3600, stale_state_hours=6)
        self.assertEqual(viols, [])

    def test_non_dict_state_does_not_raise(self):
        for bad in ([1, 2, 3], "running", 42):
            self.assertEqual(scan_violations(bad, fail_count=0, now=1000.0), [])


class RateLimitTests(unittest.TestCase):
    def test_disabled_returns_zero(self):
        self.assertEqual(seconds_until_allowed([1, 2, 3], now=100, max_per_hour=0), 0.0)

    def test_under_budget_returns_zero(self):
        ts = [100.0, 200.0]
        self.assertEqual(seconds_until_allowed(ts, now=300.0, max_per_hour=6), 0.0)

    def test_over_budget_waits_until_oldest_rolls_off(self):
        # 6 cycles all within the last hour, oldest at now-3000s -> wait 600s.
        now = 10000.0
        ts = [now - 3000, now - 2500, now - 2000, now - 1500, now - 1000, now - 500]
        wait = seconds_until_allowed(ts, now=now, max_per_hour=6, window=3600)
        self.assertAlmostEqual(wait, 600.0)

    def test_over_budget_with_excess_waits_enough_to_get_under_cap(self):
        now = 10000.0
        # 8 in-window starts, cap 6: must wait until the count drops below 6.
        ts = [now - 3500, now - 3400, now - 3300, now - 1000, now - 800, now - 600, now - 400, now - 200]
        wait = seconds_until_allowed(ts, now=now, max_per_hour=6, window=3600)
        # After sleeping `wait`, recompute the in-window count; it must be < 6.
        future = now + wait
        remaining = [t for t in ts if (future - t) < 3600]
        self.assertLess(len(remaining), 6)

    def test_old_timestamps_do_not_count(self):
        now = 10000.0
        ts = [now - 5000, now - 4000]  # both older than 1h window
        self.assertEqual(seconds_until_allowed(ts, now=now, max_per_hour=1, window=3600), 0.0)

    def test_prune_timestamps(self):
        now = 10000.0
        ts = [now - 5000, now - 100, now - 50]
        self.assertEqual(prune_timestamps(ts, now=now, window=3600), [now - 100, now - 50])


if __name__ == "__main__":
    unittest.main()
