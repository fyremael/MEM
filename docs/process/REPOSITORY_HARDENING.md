# Repository Hardening

## Revision History
- 2026-03-04: Initial branch protection and repository governance checklist.

## GitHub Settings Checklist
1. Set default branch to `main`.
2. Enable branch protection on `main`:
   - require pull request before merge
   - require approvals (>=1)
   - require status checks to pass before merge
   - require conversation resolution before merge
3. Set required checks:
   - `test`
4. Restrict direct pushes to `main` (admins optional).
5. Enable dependency and secret scanning alerts if available.

## Review Ownership
- CODEOWNERS file: `.github/CODEOWNERS`
- Keep ownership entries current with maintainers.

## Merge Strategy
1. Prefer squash merge for single-topic PRs.
2. Require PR template completion.
3. Block merge if process guard fails.
