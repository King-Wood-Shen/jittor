"""Contracts for repository links rendered by the Sphinx source hook."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs"))
from _repo_links import rewrite_repo_links  # noqa: E402


class RepoLinksTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.docs = self.repo / "docs"
        self.source = self.docs / "results" / "report.md"
        self.source.parent.mkdir(parents=True)
        (self.docs / "index.md").write_text("# Index\n", encoding="utf-8")
        skill = self.repo / "agent" / "skills" / "example" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text("# Skill\n", encoding="utf-8")

    def rewrite(self, markdown):
        return rewrite_repo_links(markdown, self.source, self.docs, self.repo, "a" * 40)

    def test_repository_file_becomes_revision_pinned_external_link(self):
        source = "[skill](../../agent/skills/example/SKILL.md#setup)\n"
        expected = (
            "[skill](https://github.com/Jittor/jittor/blob/"
            + "a" * 40
            + "/agent/skills/example/SKILL.md#setup)\n"
        )
        self.assertEqual(self.rewrite(source), expected)

    def test_docs_links_stay_under_myst_validation(self):
        source = "[valid](../index.md) [missing](../absent.md)\n"
        self.assertEqual(self.rewrite(source), source)

    def test_missing_repository_file_fails_instead_of_becoming_external(self):
        with self.assertRaisesRegex(FileNotFoundError, "agent/skills/absent/SKILL.md"):
            self.rewrite("[missing](../../agent/skills/absent/SKILL.md)\n")

    def test_code_and_actual_external_links_are_unchanged(self):
        source = (
            "`[code](../../agent/skills/example/SKILL.md)`\n"
            "```md\n[example](../../agent/skills/example/SKILL.md)\n```\n"
            "[external](https://example.com/file.md)\n"
        )
        self.assertEqual(self.rewrite(source), source)

    def test_repository_escape_is_not_mapped_to_this_repository(self):
        source = "[outside](../../../outside.md)\n"
        self.assertEqual(self.rewrite(source), source)

    def test_fork_build_links_to_its_own_commit(self):
        result = rewrite_repo_links(
            "[skill](../../agent/skills/example/SKILL.md)",
            self.source, self.docs, self.repo, "b" * 40, "King-Wood-Shen/jittor",
        )
        self.assertIn("github.com/King-Wood-Shen/jittor/blob/" + "b" * 40, result)


if __name__ == "__main__":
    unittest.main()
