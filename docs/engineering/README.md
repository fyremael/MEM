# Engineering Docs (Living)

This folder is the updateable technical source of truth for the project.

## Revision History
- 2026-03-04: Added revision-history section for process guard compatibility.

## Documents
- `EXECUTIVE_OVERVIEW.md`: decision-ready history, rationale, and current go/no-go status.
- `ARCHITECTURE.md`: model architecture, invariants, and interfaces.
- `PROCESS_FLOW.md`: train/probe/report/sweep/stress process flow and data flow.
- `EXPERIMENT_PLAYBOOK.md`: reproducible command runbooks and interpretation guidance.
- `DECISION_LOG.md`: go/no-go criteria, gate status, and formal decisions.
- `STATUS_DASHBOARD.md`: current snapshot of key metrics and frontier status.
- `REVISION_TEMPLATE.md`: copy/paste template for future revision entries.

## Update Protocol
1. Update `STATUS_DASHBOARD.md` after any new stress/sweep run.
2. Update `EXECUTIVE_OVERVIEW.md` when a new multi-seed reliability snapshot is available.
3. If a design or metric interpretation changes, update `ARCHITECTURE.md` or `PROCESS_FLOW.md`.
4. Record decision-impacting changes in `DECISION_LOG.md`.
5. Add a revision entry at the top of every file changed.

## Revision Policy
- Date format: `YYYY-MM-DD`.
- Keep entries append-only (most recent first).
- Include: scope, reason, impact, links to artifacts.

## Related Process Docs
- `docs/process/README.md`
