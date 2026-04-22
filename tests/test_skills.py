import re
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWED_FRONTMATTER_KEYS = {"name", "description", "license", "allowed-tools", "metadata"}


class SkillValidationTests(unittest.TestCase):
    def test_all_repo_skills_use_codex_compatible_frontmatter(self):
        failures = []
        skills_dir = REPO_ROOT / "skills"

        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                failures.append(f"{skill_dir.name}: missing SKILL.md")
                continue

            text = skill_md.read_text()
            match = re.match(r"^---\n(.*?)\n---\n?", text, re.DOTALL)
            if not match:
                failures.append(f"{skill_dir.name}: invalid or missing YAML frontmatter")
                continue

            frontmatter = yaml.safe_load(match.group(1)) or {}
            if not isinstance(frontmatter, dict):
                failures.append(f"{skill_dir.name}: frontmatter is not a YAML dictionary")
                continue

            unexpected = sorted(set(frontmatter.keys()) - ALLOWED_FRONTMATTER_KEYS)
            if unexpected:
                failures.append(
                    f"{skill_dir.name}: unexpected keys {', '.join(unexpected)}"
                )

        self.assertEqual(
            failures,
            [],
            msg="Repo skills must keep Codex-compatible frontmatter:\n" + "\n".join(failures),
        )


if __name__ == "__main__":
    unittest.main()
