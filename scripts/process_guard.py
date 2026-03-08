from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_FILES = [
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    "CONTRIBUTING.md",
    ".github/pull_request_template.md",
    ".github/CODEOWNERS",
    ".github/ISSUE_TEMPLATE/bug_report.md",
    ".github/ISSUE_TEMPLATE/experiment_request.md",
    ".github/workflows/ci.yml",
    ".github/workflows/reference-docs.yml",
    ".github/workflows/docs-site.yml",
    "mkdocs.yml",
    "requirements-docs.txt",
    "scripts/build_docs.py",
    "scripts/build_api_reference.py",
    "scripts/generate_context_docs.py",
    "docs/index.md",
    "docs/changelog.md",
    "docs/context_snapshot.md",
    "docs/process/README.md",
    "docs/process/DEVELOPMENT_WORKFLOW.md",
    "docs/process/CHANGE_CONTROL.md",
    "docs/process/MEETING_CADENCE.md",
    "docs/process/REPOSITORY_HARDENING.md",
    "docs/reference/README.md",
    "docs/reference/API_INDEX.md",
    "docs/engineering/STATUS_DASHBOARD.md",
    "docs/engineering/DECISION_LOG.md",
    "docs/engineering/EXECUTIVE_OVERVIEW.md",
]

REVISION_HISTORY_FILES = [
    "CONTRIBUTING.md",
    "docs/process/README.md",
    "docs/process/DEVELOPMENT_WORKFLOW.md",
    "docs/process/CHANGE_CONTROL.md",
    "docs/process/MEETING_CADENCE.md",
    "docs/process/REPOSITORY_HARDENING.md",
    "docs/reference/README.md",
    "docs/engineering/README.md",
    "docs/engineering/ARCHITECTURE.md",
    "docs/engineering/PROCESS_FLOW.md",
    "docs/engineering/EXPERIMENT_PLAYBOOK.md",
    "docs/engineering/DECISION_LOG.md",
    "docs/engineering/STATUS_DASHBOARD.md",
    "docs/engineering/EXECUTIVE_OVERVIEW.md",
    "docs/changelog.md",
]


def _check_exists(repo_root: Path, errors: list[str]) -> None:
    for rel_path in REQUIRED_FILES:
        if not (repo_root / rel_path).exists():
            errors.append(f"missing required file: {rel_path}")


def _check_revision_history(repo_root: Path, errors: list[str]) -> None:
    for rel_path in REVISION_HISTORY_FILES:
        path = repo_root / rel_path
        if not path.exists():
            errors.append(f"missing revision-policy file: {rel_path}")
            continue
        content = path.read_text(encoding="utf-8")
        if "## Revision History" not in content:
            errors.append(f"missing '## Revision History' section: {rel_path}")


def _check_ci_gate(repo_root: Path, errors: list[str]) -> None:
    ci_path = repo_root / ".github" / "workflows" / "ci.yml"
    if not ci_path.exists():
        return
    content = ci_path.read_text(encoding="utf-8")
    if "python -m pytest -q tests" not in content:
        errors.append("ci.yml missing required test gate command: python -m pytest -q tests")
    if "python scripts/process_guard.py" not in content:
        errors.append("ci.yml missing process guard command: python scripts/process_guard.py")
    if "python scripts/build_docs.py --check" not in content:
        errors.append("ci.yml missing docs pipeline check command: python scripts/build_docs.py --check")


def _check_gitignore(repo_root: Path, errors: list[str]) -> None:
    gitignore_path = repo_root / ".gitignore"
    if not gitignore_path.exists():
        return
    content = gitignore_path.read_text(encoding="utf-8")
    for required_entry in ("demo_runs/", "test_artifacts/"):
        if required_entry not in content:
            errors.append(f".gitignore missing required entry: {required_entry}")


def run_checks(repo_root: Path) -> list[str]:
    errors: list[str] = []
    _check_exists(repo_root, errors)
    _check_revision_history(repo_root, errors)
    _check_ci_gate(repo_root, errors)
    _check_gitignore(repo_root, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate repository process guardrails.")
    parser.add_argument("--repo-root", default=".", help="Repository root path.")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    errors = run_checks(repo_root)
    if errors:
        print("process guard FAILED")
        for err in errors:
            print(f"- {err}")
        return 1

    print("process guard PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
