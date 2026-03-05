# Change Control

## Revision History
- 2026-03-04: Initial change-control policy for controlled iteration.

## Change Classes
1. `Low`
   - docs-only, non-functional refactors.
2. `Medium`
   - model/training/probe logic changes with local validation.
3. `High`
   - threshold changes, go/no-go posture changes, or architecture changes.

## Approval Policy
1. Low: one reviewer.
2. Medium: one reviewer plus passing tests.
3. High: reviewer plus decision-note update in:
   - `docs/engineering/DECISION_LOG.md`
   - `docs/engineering/EXECUTIVE_OVERVIEW.md`

## Mandatory Change Record
Every PR must include:
1. Purpose and scope.
2. Risk level and rollback path.
3. Verification evidence.
4. Affected docs and artifacts.

## Rollback Rule
If post-merge evidence regresses required criteria:
1. Revert or hotfix immediately.
2. Document incident and impact in decision log.
3. Re-open the change with corrected plan.
