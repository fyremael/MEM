# Process Docs

## Revision History
- 2026-03-08: Added docs-site deployment workflow reference for GitHub Pages publishing.
- 2026-03-08: Switched docs enforcement gate to unified docs pipeline (`build_docs.py --check`) with MkDocs validation.
- 2026-03-06: Added API/reference docs automation gate to process enforcement.
- 2026-03-04: Added repository hardening guide and process guard enforcement reference.
- 2026-03-04: Initial process-doc set for methodical development.

## Purpose
This folder defines how work enters, is executed, is evidenced, and is decisioned.

## Documents
- `DEVELOPMENT_WORKFLOW.md`: stage-by-stage delivery workflow.
- `CHANGE_CONTROL.md`: intake, review, approval, and rollout controls.
- `MEETING_CADENCE.md`: operating rhythm and decision checkpoints.
- `REPOSITORY_HARDENING.md`: GitHub branch protection and repository governance setup.

## Enforced Gate
- CI workflow: `.github/workflows/ci.yml`
- Docs publish workflow: `.github/workflows/docs-site.yml`
- Current required verification command: `python -m pytest -q tests`
- Process policy check: `python scripts/process_guard.py`
- Docs pipeline check: `python scripts/build_docs.py --check`
