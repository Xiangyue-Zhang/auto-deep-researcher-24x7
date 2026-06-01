import json
import os
import tempfile
import unittest
from pathlib import Path

from core.execution import LocalExecutionBackend
from core.tools import ToolRegistry


class RepoReadingToolTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name) / "workspace"
        self.workspace.mkdir()
        (self.workspace / "src").mkdir()
        (self.workspace / "src" / "train.py").write_text(
            "import torch\n"
            "def main():\n"
            "    lr = 1e-3\n"
            "    return lr\n"
        )
        (self.workspace / "README.md").write_text("# Demo\nuses learning rate\n")
        (self.workspace / "__pycache__").mkdir()
        (self.workspace / "__pycache__" / "junk.txt").write_text("def main(): pass\n")
        self.registry = ToolRegistry(LocalExecutionBackend(self.workspace))

    def tearDown(self):
        self.tempdir.cleanup()

    def test_search_code_finds_match_with_file_and_line(self):
        result = json.loads(self.registry.execute_tool("search_code", {"pattern": r"def main"}))
        self.assertEqual(result["count"], 1)
        hit = result["matches"][0]
        self.assertEqual(hit["file"], "src/train.py")
        self.assertEqual(hit["line"], 2)

    def test_search_code_skips_pycache(self):
        result = json.loads(self.registry.execute_tool("search_code", {"pattern": r"def main"}))
        files = {m["file"] for m in result["matches"]}
        self.assertNotIn("__pycache__/junk.txt", files)

    def test_search_code_ignore_case(self):
        result = json.loads(
            self.registry.execute_tool("search_code", {"pattern": "LEARNING", "ignore_case": True})
        )
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["matches"][0]["file"], "README.md")

    def test_search_code_invalid_regex_returns_error(self):
        result = json.loads(self.registry.execute_tool("search_code", {"pattern": "("}))
        self.assertIn("error", result)
        self.assertIn("Invalid search pattern", result["error"])

    def test_search_code_rejects_path_traversal(self):
        result = json.loads(
            self.registry.execute_tool("search_code", {"pattern": "x", "path": "../"})
        )
        self.assertIn("error", result)
        self.assertIn("escapes workspace", result["error"])

    def test_list_tree_is_recursive_and_marks_dirs(self):
        result = json.loads(self.registry.execute_tool("list_tree", {}))
        tree = result["tree"]
        self.assertIn("src/", tree)
        self.assertIn("src/train.py", tree)
        self.assertNotIn("__pycache__/", tree)

    def test_list_tree_depth_limit(self):
        result = json.loads(self.registry.execute_tool("list_tree", {"max_depth": 1}))
        tree = result["tree"]
        self.assertIn("src/", tree)
        self.assertNotIn("src/train.py", tree)

    def test_read_file_range_returns_numbered_slice(self):
        out = self.registry.execute_tool("read_file", {"path": "src/train.py", "start_line": 2, "end_line": 3})
        self.assertIn("2\tdef main():", out)
        self.assertIn("3\t    lr = 1e-3", out)
        self.assertNotIn("import torch", out)

    def test_read_file_without_range_unchanged(self):
        out = self.registry.execute_tool("read_file", {"path": "README.md"})
        self.assertEqual(out, "# Demo\nuses learning rate\n")

    def test_list_tree_rejects_path_traversal(self):
        result = json.loads(self.registry.execute_tool("list_tree", {"path": ".."}))
        self.assertIn("error", result)
        self.assertIn("escapes workspace", result["error"])

    def test_list_tree_does_not_follow_symlink_outside_workspace(self):
        outside = Path(self.tempdir.name) / "outside"
        (outside / "sub").mkdir(parents=True)
        (outside / "sub" / "secret.txt").write_text("TOPSECRET\n")
        os.symlink(outside, self.workspace / "leak")

        result = json.loads(self.registry.execute_tool("list_tree", {}))
        tree = result["tree"]
        self.assertNotIn("leak/", tree)
        self.assertFalse(any(entry.startswith("leak/") for entry in tree))

    def test_search_code_does_not_read_symlinked_external_file(self):
        outside = Path(self.tempdir.name) / "outside"
        outside.mkdir(exist_ok=True)
        (outside / "creds.txt").write_text("TOPSECRET token\n")
        os.symlink(outside / "creds.txt", self.workspace / "leak.txt")

        result = json.loads(self.registry.execute_tool("search_code", {"pattern": "TOPSECRET"}))
        files = {m["file"] for m in result["matches"]}
        self.assertNotIn("leak.txt", files)
        self.assertEqual(result["count"], 0)


if __name__ == "__main__":
    unittest.main()
