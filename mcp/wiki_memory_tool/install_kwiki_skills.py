# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Install wiki-memory-tool skills into a Claude Code skills directory.

Creates symlinks (default) or copies skill directories from
mcp/wiki_memory_tool/skills/ into the target .claude/skills/ directory.

Usage:
    # Symlink into a project's .claude/skills/
    uv run python install_skills.py /path/to/project/.claude/skills

    # Symlink into ~/.claude/skills/ (user-global)
    uv run python install_skills.py --global

    # Copy instead of symlink
    uv run python install_skills.py --copy /path/to/project/.claude/skills
    uv run python install_skills.py --global --copy

    # Remove previously installed skills
    uv run python install_skills.py --uninstall /path/to/project/.claude/skills
"""

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
SKILLS_SOURCE = SCRIPT_DIR / "skills"

SKILL_NAMES = [
    "wiki-discovery-api",
    "wiki-discovery-cli",
    "wiki-discovery-mcp",
    "wiki-query-api",
    "wiki-query-cli",
    "wiki-query-mcp",
]

SKILL_NAME_MAP = {
    "wiki-discovery-api": "kwiki:discover-api",
    "wiki-discovery-cli": "kwiki:discover-cli",
    "wiki-discovery-mcp": "kwiki:discover-mcp",
    "wiki-query-api": "kwiki:query-api",
    "wiki-query-cli": "kwiki:query-cli",
    "wiki-query-mcp": "kwiki:query-mcp",
}


def install_skills(target_dir: Path, copy: bool = False):
    if not target_dir.exists():
        print(f"ERROR: Target directory does not exist: {target_dir}", file=sys.stderr)
        print("Create it first or check the path.", file=sys.stderr)
        sys.exit(1)

    if not SKILLS_SOURCE.exists():
        print(f"ERROR: Source skills directory not found: {SKILLS_SOURCE}", file=sys.stderr)
        sys.exit(1)

    installed = []
    for src_name in SKILL_NAMES:
        src = SKILLS_SOURCE / src_name
        if not src.exists():
            print(f"  SKIP: {src_name} (source not found)")
            continue

        dest_name = SKILL_NAME_MAP.get(src_name, src_name)
        dest = target_dir / dest_name

        if dest.exists() or dest.is_symlink():
            if dest.is_symlink():
                dest.unlink()
            else:
                shutil.rmtree(dest)

        if copy:
            shutil.copytree(src, dest)
            print(f"  COPY: {dest_name}/ <- {src}")
        else:
            dest.symlink_to(src)
            print(f"  LINK: {dest_name}/ -> {src}")

        installed.append(dest_name)

    print(f"\nInstalled {len(installed)} skills into {target_dir}")
    print("Available as: " + ", ".join(f"/{name}" for name in installed))


def uninstall_skills(target_dir: Path):
    if not target_dir.exists():
        print(f"ERROR: Target directory does not exist: {target_dir}", file=sys.stderr)
        sys.exit(1)

    removed = []
    for src_name in SKILL_NAMES:
        dest_name = SKILL_NAME_MAP.get(src_name, src_name)
        dest = target_dir / dest_name

        if dest.exists() or dest.is_symlink():
            if dest.is_symlink():
                dest.unlink()
            else:
                shutil.rmtree(dest)
            print(f"  REMOVED: {dest_name}/")
            removed.append(dest_name)

    if removed:
        print(f"\nRemoved {len(removed)} skills from {target_dir}")
    else:
        print("No wiki skills found to remove.")


def main():
    parser = argparse.ArgumentParser(description="Install wiki-memory-tool skills into a Claude Code skills directory")
    parser.add_argument(
        "target",
        nargs="?",
        help="Target .claude/skills/ directory path",
    )
    parser.add_argument(
        "--global",
        dest="global_install",
        action="store_true",
        help="Install into ~/.claude/skills/",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove previously installed wiki skills",
    )

    args = parser.parse_args()

    if args.global_install:
        target = Path.home() / ".claude" / "skills"
    elif args.target:
        target = Path(args.target).resolve()
    else:
        parser.error("Provide a target directory or use --global")

    if not target.exists():
        print(f"ERROR: Skills directory does not exist: {target}", file=sys.stderr)
        print("Ensure Claude Code is initialized in the target project.", file=sys.stderr)
        sys.exit(1)

    if args.uninstall:
        uninstall_skills(target)
    else:
        mode = "copy" if args.copy else "symlink"
        print(f"Installing wiki skills ({mode}) into: {target}\n")
        install_skills(target, copy=args.copy)


if __name__ == "__main__":
    main()
