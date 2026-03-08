# Docs Changelog

## Revision History
- 2026-03-08: Added GitHub Pages deployment workflow for MkDocs site publishing.
- 2026-03-08: Added MkDocs documentation hub, nav, and unified docs build/check pipeline (`build_docs.py` + `generate_context_docs.py`).

## Entries

### 2026-03-08
- Added automated Pages deployment workflow at `.github/workflows/docs-site.yml`.
- Added `mkdocs.yml` with BNRPE-style documentation navigation.
- Added `docs/index.md` as the docs hub page.
- Added auto-generated context snapshot at `docs/context_snapshot.md`.
- Added unified docs orchestration script at `scripts/build_docs.py`.
- Updated CI/process policy checks to enforce `python scripts/build_docs.py --check`.
