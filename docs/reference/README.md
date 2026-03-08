# API and Reference Docs

## Revision History
- 2026-03-08: Added unified docs build/check integration (`build_docs.py`) and auto-generated context snapshot linkage.
- 2026-03-06: Initial automated API/reference documentation pipeline added.

## Scope
This section contains generated API reference pages for all modules in `src/modulus_memory_channels`.

## Generation Commands
Write/update generated pages:
```bash
python scripts/build_docs.py --no-site-build
```

Check that generated pages are up to date:
```bash
python scripts/build_docs.py --check --no-site-build
```

## Generated Files
- `docs/reference/API_INDEX.md`
- `docs/reference/api/*.md`
- `docs/context_snapshot.md`

Do not edit generated API files directly; re-run the generator instead.
