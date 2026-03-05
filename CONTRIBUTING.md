# Contributing

## Scope
This repository uses evidence-driven development for model, diagnostics, and decision artifacts.

## Revision History
- 2026-03-04: Added process guard command and pre-PR policy check.
- 2026-03-04: Initial contribution process and Git workflow baseline.

## Branching
1. Create a branch from `main`:
   - `feature/<short-topic>`
   - `fix/<short-topic>`
   - `docs/<short-topic>`
2. Keep branches single-purpose and short-lived.

## Commit Discipline
1. Commit messages use:
   - `feat: ...`
   - `fix: ...`
   - `docs: ...`
   - `test: ...`
   - `chore: ...`
2. Include evidence paths in commit body when changing metrics or decisions.

## Development Loop
1. Install dev dependencies:
```bash
python -m pip install -e .[dev]
```
2. Run tests before opening PR:
```bash
python -m pytest -q tests
```
3. Run process guard before opening PR:
```bash
python scripts/process_guard.py
```
4. Update docs when metrics or decision posture changes:
   - `docs/engineering/STATUS_DASHBOARD.md`
   - `docs/engineering/DECISION_LOG.md`
   - `docs/engineering/EXECUTIVE_OVERVIEW.md`

## Pull Request Requirements
1. Reference issue or experiment request.
2. Summarize behavior change and risk.
3. Include verification evidence (tests, report paths, or command outputs).
4. If go/no-go posture changes, include updated rationale and artifact links.
