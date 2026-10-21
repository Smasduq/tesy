"""Tests for the read-only file tools."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tesy.computer import ComputerAccess  # noqa: E402


class ComputerAccessTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name).resolve()

        (self.root / "notes.txt").write_text("hello world\nsecond line\nthird line\n")
        (self.root / "docs").mkdir()
        (self.root / "docs" / "todo.md").write_text("buy milk\ncall bob\n")
        (self.root / ".env").write_text("SECRET=nope\n")
        (self.root / ".ssh").mkdir()
        (self.root / ".ssh" / "id_rsa").write_text("PRIVATE KEY\n")
        (self.root / "binary.bin").write_bytes(b"\x00\x01\x02")
        (self.root / "big.txt").write_text("x" * (200 * 1024))

        self.actions: list[str] = []
        self.computer = ComputerAccess(root=self.root, on_action=self.actions.append)

    def run_tool(self, name, **arguments):
        return self.computer.run(name, json.dumps(arguments))

    def test_reads_a_file_with_line_numbers(self):
        result = self.run_tool("read_file", path="notes.txt")
        self.assertIn("~/notes.txt (lines 1-3 of 3)", result)
        self.assertIn("1: hello world", result)
        self.assertIn("3: third line", result)

    def test_reads_a_window(self):
        result = self.run_tool("read_file", path="notes.txt", start_line=2, max_lines=1)
        self.assertIn("lines 2-2 of 3", result)
        self.assertIn("2: second line", result)
        self.assertNotIn("hello world", result)

    def test_relative_paths_start_at_home(self):
        result = self.run_tool("read_file", path="docs/todo.md")
        self.assertIn("1: buy milk", result)

    def test_missing_file_is_explained(self):
        self.assertIn("does not exist", self.run_tool("read_file", path="nope.txt"))

    def test_directory_needs_list_directory(self):
        self.assertIn("is a directory", self.run_tool("read_file", path="docs"))

    def test_outside_home_is_refused(self):
        result = self.run_tool("read_file", path="/etc/passwd")
        self.assertIn("outside", result)

    def test_traversal_is_refused(self):
        result = self.run_tool("read_file", path="../../etc/passwd")
        self.assertIn("outside", result)

    def test_env_file_is_refused(self):
        result = self.run_tool("read_file", path=".env")
        self.assertIn("credential", result)

    def test_ssh_key_is_refused(self):
        result = self.run_tool("read_file", path=".ssh/id_rsa")
        self.assertIn("credential", result)

    def test_binary_file_is_refused(self):
        self.assertIn("binary", self.run_tool("read_file", path="binary.bin"))

    def test_huge_file_is_refused(self):
        self.assertIn("larger than", self.run_tool("read_file", path="big.txt"))

    def test_line_range_past_the_end_is_explained(self):
        self.assertIn("only has 3 lines", self.run_tool("read_file", path="notes.txt", start_line=9))

    def test_lists_a_directory(self):
        result = self.run_tool("list_directory", path=".")
        self.assertIn("docs/", result)
        self.assertIn("notes.txt", result)
        self.assertNotIn(".env", result)
        self.assertNotIn(".ssh", result)

    def test_listing_a_file_points_at_read_file(self):
        self.assertIn("use read_file", self.run_tool("list_directory", path="notes.txt"))

    def test_search_finds_text(self):
        result = self.run_tool("search_files", query="milk")
        self.assertIn("~/docs/todo.md:1: buy milk", result)

    def test_search_skips_secrets(self):
        result = self.run_tool("search_files", query="SECRET")
        self.assertIn("No matches", result)

    def test_search_is_case_insensitive(self):
        result = self.run_tool("search_files", query="HELLO")
        self.assertIn("notes.txt:1:", result)

    def test_search_reports_nothing_found(self):
        self.assertIn("No matches", self.run_tool("search_files", query="zebra"))

    def test_invalid_json_is_reported(self):
        self.assertIn("valid JSON", self.computer.run("read_file", "{oops"))

    def test_unknown_tool_is_reported(self):
        self.assertIn("no tool", self.run_tool("delete_everything"))

    def test_wrong_argument_is_reported(self):
        self.assertIn("wrong arguments", self.run_tool("read_file", nonsense="x"))

    def test_actions_are_reported(self):
        self.run_tool("read_file", path="notes.txt")
        self.assertTrue(any("read ~/notes.txt" in action for action in self.actions))

    def test_schemas_name_the_tools_and_the_root(self):
        schemas = self.computer.schemas()
        names = [schema["function"]["name"] for schema in schemas]
        self.assertEqual(names, ["read_file", "list_directory", "search_files"])
        self.assertIn(str(self.root), json.dumps(schemas))

    def test_symlink_out_of_home_is_not_followed(self):
        link = self.root / "outside"
        try:
            link.symlink_to("/etc")
        except OSError:
            self.skipTest("symlinks are not available here")
        listing = self.run_tool("list_directory", path=".")
        self.assertIn("outside (link outside home", listing)
        self.assertIn("outside", self.run_tool("read_file", path="outside/passwd"))


if __name__ == "__main__":
    unittest.main()
