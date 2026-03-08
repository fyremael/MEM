"""Build and validate generated docs, then optionally build the MkDocs site."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        joined = " ".join(cmd)
        raise SystemExit(f"command failed ({result.returncode}): {joined}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root path.")
    parser.add_argument("--check", action="store_true", help="Run deterministic stale-file checks only.")
    parser.add_argument(
        "--site-dir",
        default="site",
        help="MkDocs output directory relative to repo root (default: site).",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build MkDocs with --strict when site build is enabled (default: true).",
    )
    parser.add_argument(
        "--site-build",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build MkDocs site after generators/checks (default: true).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    python = sys.executable

    if args.check:
        _run([python, "scripts/build_api_reference.py", "--check"], cwd=repo_root)
        _run([python, "scripts/generate_context_docs.py", "--check"], cwd=repo_root)
    else:
        _run([python, "scripts/build_api_reference.py"], cwd=repo_root)
        _run([python, "scripts/generate_context_docs.py"], cwd=repo_root)

    if args.site_build:
        mkdocs_cmd = [python, "-m", "mkdocs", "build", "--config-file", "mkdocs.yml", "--site-dir", str(args.site_dir)]
        if args.strict:
            mkdocs_cmd.insert(4, "--strict")
        _run(mkdocs_cmd, cwd=repo_root)

    mode = "check" if args.check else "build"
    site_note = "with site build" if args.site_build else "without site build"
    print(f"docs {mode} completed ({site_note})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
