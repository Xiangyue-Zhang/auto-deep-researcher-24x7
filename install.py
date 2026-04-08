"""
Install Deep Researcher Agent skills into Claude Code.

One-command setup:
    python install.py

After installation, these slash commands are available in Claude Code:
    /auto-experiment     — Launch 24/7 autonomous experiment loop
    /experiment-status   — Check running experiment status
    /gpu-monitor         — GPU status and availability
    /daily-papers        — Daily arXiv paper recommendations
    /paper-analyze       — Deep paper analysis with figure extraction
    /conf-search         — Search top conference papers
    /progress-report     — Generate structured progress report
    /obsidian-sync       — Refresh Obsidian dashboard and daily notes
"""

import shutil
import sys
from pathlib import Path


CLAUDE_DIR = Path.home() / ".claude"
REPO_DIR = Path(__file__).parent
SKILLS_SOURCE = REPO_DIR / "skills"
CORE_SOURCE = REPO_DIR / "core"
GPU_SOURCE = REPO_DIR / "gpu"


def install():
    print()
    print("  Deep Researcher Agent — Installer")
    print("  " + "=" * 40)
    print()

    # 1. Install skills as Claude Code slash commands
    claude_commands = CLAUDE_DIR / "commands"
    claude_commands.mkdir(parents=True, exist_ok=True)

    installed = 0
    for skill_dir in sorted(SKILLS_SOURCE.iterdir()):
        if skill_dir.is_dir():
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                dest = claude_commands / f"{skill_dir.name}.md"
                shutil.copy2(skill_file, dest)
                print(f"    ✓ /{skill_dir.name}")
                installed += 1

    # 2. Install core module (for programmatic use)
    core_dest = CLAUDE_DIR / "deep-researcher" / "core"
    core_dest.mkdir(parents=True, exist_ok=True)
    if CORE_SOURCE.exists():
        for py_file in CORE_SOURCE.glob("*.py"):
            shutil.copy2(py_file, core_dest / py_file.name)

    gpu_dest = CLAUDE_DIR / "deep-researcher" / "gpu"
    gpu_dest.mkdir(parents=True, exist_ok=True)
    if GPU_SOURCE.exists():
        for py_file in GPU_SOURCE.glob("*.py"):
            shutil.copy2(py_file, gpu_dest / py_file.name)

    # 3. Copy default config
    config_src = REPO_DIR / "config.yaml"
    config_dest = CLAUDE_DIR / "deep-researcher" / "config.yaml"
    if config_src.exists() and not config_dest.exists():
        shutil.copy2(config_src, config_dest)

    # Summary
    print()
    print(f"  Done! {installed} skills installed.")
    print()
    print("  Available commands in Claude Code:")
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
    print("  Quick start:")
    print("    1. Create a project with PROJECT_BRIEF.md")
    print("    2. Run: /auto-experiment --project <path> --gpu 0")
    print()


def uninstall():
    """Remove all installed skills."""
    claude_commands = CLAUDE_DIR / "commands"
    removed = 0
    for skill_dir in sorted(SKILLS_SOURCE.iterdir()):
        if skill_dir.is_dir():
            dest = claude_commands / f"{skill_dir.name}.md"
            if dest.exists():
                dest.unlink()
                print(f"    ✗ /{skill_dir.name}")
                removed += 1

    deep_dir = CLAUDE_DIR / "deep-researcher"
    if deep_dir.exists():
        shutil.rmtree(deep_dir)
        print("    ✗ core modules")

    print(f"\n  Removed {removed} skills.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--uninstall":
        uninstall()
    else:
        install()
