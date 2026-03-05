# Development Workflow

## Revision History
- 2026-03-04: Initial stage-gated workflow for methodical process development.

## Stage Gates
1. Intake
   - Create issue using template.
   - Define objective, scope, and success metric.
2. Design
   - Document assumptions and constraints.
   - Identify impacted modules and decision risk.
3. Build
   - Implement on feature branch.
   - Keep commits atomic.
4. Verify
   - Run tests.
   - For experiment changes, regenerate affected report artifacts.
5. Review
   - Open PR with evidence and risk notes.
   - Resolve review feedback and re-run verification.
6. Decide
   - Update `docs/engineering/DECISION_LOG.md` if gate posture changed.
7. Merge and Track
   - Merge to `main`.
   - Update status docs and next actions.

## Required Evidence by Change Type
1. Code behavior changes:
   - test results (`python -m pytest -q tests`)
2. Metrics/threshold changes:
   - updated report JSON/CSV/SVG paths
3. Decision posture changes:
   - updated executive overview and decision log entries

## Definition of Done
1. Branch merged with review.
2. Tests pass for impacted area.
3. Documentation and decision artifacts updated.
4. Follow-up tasks captured as issues.
