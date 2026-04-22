"""
Install Deep Researcher Agent integrations into Claude Code and Codex.

One-command setup:
    python install.py

After installation:
    - Claude Code slash commands are copied into ~/.claude/commands
    - Codex local skills are copied into ~/.codex/skills
"""

import re
import shutil
import sys
from pathlib import Path

import yaml


CLAUDE_DIR = Path.home() / ".claude"
CODEX_DIR = Path.home() / ".codex"
REPO_DIR = Path(__file__).parent
SKILLS_SOURCE = REPO_DIR / "skills"
CORE_SOURCE = REPO_DIR / "core"
GPU_SOURCE = REPO_DIR / "gpu"
CODEX_ALLOWED_FRONTMATTER = {"name", "description", "license", "allowed-tools", "metadata"}
CODEX_INSTALL_MARKER = ".deep-researcher-installed"


def _iter_skill_dirs(skills_source: Path):
    for skill_dir in sorted(skills_source.iterdir()):
        if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
            yield skill_dir


def _sync_python_modules(source_dir: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    if source_dir.exists():
        for py_file in source_dir.glob("*.py"):
            shutil.copy2(py_file, dest_dir / py_file.name)


def _install_runtime_bundle(home_dir: Path, repo_dir: Path):
    bundle_dir = home_dir / "deep-researcher"
    _sync_python_modules(repo_dir / "core", bundle_dir / "core")
    _sync_python_modules(repo_dir / "gpu", bundle_dir / "gpu")

    config_src = repo_dir / "config.yaml"
    config_dest = bundle_dir / "config.yaml"
    if config_src.exists() and not config_dest.exists():
        config_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_src, config_dest)


def _check_codex_conflicts(skills_source: Path, codex_dir: Path):
    codex_skills_dir = codex_dir / "skills"
    for skill_dir in _iter_skill_dirs(skills_source):
        dest_dir = codex_skills_dir / skill_dir.name
        if dest_dir.exists():
            marker = dest_dir / CODEX_INSTALL_MARKER
            if not marker.exists():
                raise RuntimeError(
                    f"Refusing to overwrite existing Codex skill '{skill_dir.name}' "
                    f"at {dest_dir}; marker file not found."
                )


def _parse_frontmatter(skill_text: str):
    match = re.match(r"^---\n(.*?)\n---\n?(.*)$", skill_text, re.DOTALL)
    if not match:
        raise ValueError("Skill file must start with YAML frontmatter")
    frontmatter = yaml.safe_load(match.group(1)) or {}
    if not isinstance(frontmatter, dict):
        raise ValueError("Skill frontmatter must be a YAML dictionary")
    body = match.group(2)
    return frontmatter, body


def _build_codex_skill_text(skill_text: str) -> str:
    frontmatter, body = _parse_frontmatter(skill_text)
    filtered_frontmatter = {
        key: value
        for key, value in frontmatter.items()
        if key in CODEX_ALLOWED_FRONTMATTER
    }
    skill_name = str(filtered_frontmatter.get("name", "")).strip()
    codex_note = (
        f"> Codex note: invoke explicitly as `${skill_name}` when needed. "
        f"The original repo docs may also show `/{skill_name}` because the same "
        "source skill powers Claude Code slash commands.\n\n"
    )
    rendered_frontmatter = yaml.safe_dump(
        filtered_frontmatter,
        sort_keys=False,
        allow_unicode=True,
    ).strip()
    return f"---\n{rendered_frontmatter}\n---\n\n{codex_note}{body.lstrip()}"


def _install_claude_commands(skills_source: Path, claude_dir: Path) -> int:
    claude_commands = claude_dir / "commands"
    claude_commands.mkdir(parents=True, exist_ok=True)

    installed = 0
    for skill_dir in _iter_skill_dirs(skills_source):
        dest = claude_commands / f"{skill_dir.name}.md"
        shutil.copy2(skill_dir / "SKILL.md", dest)
        print(f"    ✓ Claude /{skill_dir.name}")
        installed += 1
    return installed


def _install_codex_skills(skills_source: Path, codex_dir: Path) -> int:
    codex_skills_dir = codex_dir / "skills"
    codex_skills_dir.mkdir(parents=True, exist_ok=True)

    installed = 0
    for skill_dir in _iter_skill_dirs(skills_source):
        dest_dir = codex_skills_dir / skill_dir.name
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(skill_dir, dest_dir)
        skill_text = (skill_dir / "SKILL.md").read_text()
        (dest_dir / "SKILL.md").write_text(_build_codex_skill_text(skill_text))
        (dest_dir / CODEX_INSTALL_MARKER).write_text("installed by Deep Researcher Agent\n")
        print(f"    ✓ Codex ${skill_dir.name}")
        installed += 1
    return installed


def install(
    claude_dir: Path = CLAUDE_DIR,
    codex_dir: Path = CODEX_DIR,
    repo_dir: Path = REPO_DIR,
):
    print()
    print("  Deep Researcher Agent — Installer")
    print("  " + "=" * 40)
    print()

    skills_source = repo_dir / "skills"
    _check_codex_conflicts(skills_source, codex_dir)
    claude_count = _install_claude_commands(skills_source, claude_dir)
    codex_count = _install_codex_skills(skills_source, codex_dir)
    _install_runtime_bundle(claude_dir, repo_dir)
    _install_runtime_bundle(codex_dir, repo_dir)

    print()
    print(
        "  Done! "
        f"{claude_count} Claude commands and {codex_count} Codex skills installed."
    )
    print()
    print("  Available in Claude Code:")
    print("  ─────────────────────────────────────")
    print("    /auto-experiment     Launch 24/7 experiment loop")
    print("    /experiment-status   Check experiment progress")
    print("    /gpu-monitor         GPU status & availability")
    print("    /daily-papers        arXiv paper recommendations")
    print("    /paper-analyze       Deep paper analysis")
    print("    /conf-search         Conference paper search")
    print("    /progress-report     Generate progress report")
    print("    /obsidian-sync       Refresh Obsidian notes")
    print()
    print("  Available in Codex:")
    print("  ─────────────────────────────────────")
    print("    $auto-experiment     Launch 24/7 experiment loop")
    print("    $experiment-status   Check experiment progress")
    print("    $gpu-monitor         GPU status & availability")
    print("    $daily-papers        arXiv paper recommendations")
    print("    $paper-analyze       Deep paper analysis")
    print("    $conf-search         Conference paper search")
    print("    $progress-report     Generate progress report")
    print("    $obsidian-sync       Refresh Obsidian notes")
    print()
    print("  Quick start:")
    print("    1. Create a project with PROJECT_BRIEF.md")
    print("    2. Claude: /auto-experiment --project <path> --gpu 0")
    print("    3. Codex: use $auto-experiment for the same workflow")
    print()
    print("  Restart Codex to pick up newly installed local skills.")
    print()


def uninstall(
    claude_dir: Path = CLAUDE_DIR,
    codex_dir: Path = CODEX_DIR,
    repo_dir: Path = REPO_DIR,
):
    """Remove all installed skills."""
    removed_claude = 0
    claude_commands = claude_dir / "commands"
    for skill_dir in _iter_skill_dirs(repo_dir / "skills"):
        dest = claude_commands / f"{skill_dir.name}.md"
        if dest.exists():
            dest.unlink()
            print(f"    ✗ Claude /{skill_dir.name}")
            removed_claude += 1

    removed_codex = 0
    codex_skills = codex_dir / "skills"
    for skill_dir in _iter_skill_dirs(repo_dir / "skills"):
        dest_dir = codex_skills / skill_dir.name
        if dest_dir.exists():
            marker = dest_dir / CODEX_INSTALL_MARKER
            if marker.exists():
                shutil.rmtree(dest_dir)
                print(f"    ✗ Codex ${skill_dir.name}")
                removed_codex += 1

    for home_dir, label in ((claude_dir, "Claude"), (codex_dir, "Codex")):
        deep_dir = home_dir / "deep-researcher"
        if deep_dir.exists():
            shutil.rmtree(deep_dir)
            print(f"    ✗ {label} runtime bundle")

    print(
        f"\n  Removed {removed_claude} Claude commands and "
        f"{removed_codex} Codex skills."
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--uninstall":
        uninstall()
    else:
        install()
