import tempfile
import unittest
from pathlib import Path

from core.journal import ResearchJournal


class ResearchJournalTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_append_and_tail(self):
        journal = ResearchJournal(self.workspace)
        journal.append_dead_end("ResNet-50 overfits badly here", ts="2026-06-01 10:00")
        journal.append_insight("lr warmup stabilizes the first 500 steps", ts="2026-06-01 10:05")

        de = journal.dead_ends_tail(1500)
        ins = journal.insights_tail(1500)
        self.assertIn("ResNet-50 overfits", de)
        self.assertIn("- [2026-06-01 10:00]", de)
        self.assertIn("lr warmup stabilizes", ins)

    def test_empty_entries_ignored(self):
        journal = ResearchJournal(self.workspace)
        journal.append_dead_end("   ")
        self.assertNotIn("- [", journal.dead_ends_tail(1500))

    def test_tail_tolerates_string_max_chars(self):
        journal = ResearchJournal(self.workspace)
        journal.append_dead_end("something failed")
        # A YAML-quoted config value arrives as a str; tail must not raise.
        self.assertIn("something failed", journal.dead_ends.tail("1500"))

    def test_tail_on_unreadable_path_returns_empty(self):
        journal = ResearchJournal(self.workspace)
        # Replace the file with a directory so read_text raises IsADirectoryError.
        path = self.workspace / "INSIGHTS.md"
        path.unlink()
        path.mkdir()
        self.assertEqual(journal.insights.tail(1500), "")

    def test_rotation_creates_backup_and_keeps_header(self):
        journal = ResearchJournal(self.workspace, max_chars=200)
        for i in range(40):
            journal.append_dead_end(f"failed approach number {i} with a long description", ts="2026-06-01 10:00")

        dead_ends_path = self.workspace / "DEAD_ENDS.md"
        content = dead_ends_path.read_text()
        self.assertTrue(content.startswith("# Dead Ends"))
        backups = list(self.workspace.glob("DEAD_ENDS.*.bak"))
        self.assertTrue(backups, "rotation should have produced a .bak archive")
        # Most recent entries survive in the live file.
        self.assertIn("number 39", content)


if __name__ == "__main__":
    unittest.main()
