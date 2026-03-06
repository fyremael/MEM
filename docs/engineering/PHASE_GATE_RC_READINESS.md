# Phase Gate RC Readiness

## Revision History
- 2026-03-05: Initial phase-gate report published with RC execution recommendation.

## Scope
This document records formal gate outcomes for:
1. Expanded frontier reliability
2. Adversarial robustness
3. Reproducibility

## Inputs
- Frontier campaign:
  - `demo_runs/phase_gate_v1/report/phase_gate_report.json`
  - `demo_runs/corridor_reliability_levelup_v1/report/reliability_summary.json`
- Adversarial campaign:
  - `demo_runs/corridor_reliability_levelup_v1_adversarial/report/reliability_summary.json`
- Repro campaign:
  - `demo_runs/corridor_repro_check_v1/report/reliability_summary.json`

## Gate Thresholds
- Frontier gate:
  - overall pass rate >= `0.80`
  - min case pass rate >= `0.67`
  - seeds >= `10`
- Adversarial gate:
  - overall pass rate >= `0.80`
  - min case pass rate >= `0.67`
  - seeds >= `5`
- Repro gate:
  - overall pass-rate delta <= `0.05`
  - max case pass-rate delta <= `0.10`

## Outcomes
- Frontier gate: **PASS**
  - overall pass rate: `1.0` (`60/60`)
  - min case pass rate: `1.0`
  - seeds: `10`
- Adversarial gate: **PASS**
  - overall pass rate: `1.0` (`30/30`)
  - min case pass rate: `1.0`
  - seeds: `5`
- Repro gate: **PASS**
  - overall pass-rate delta: `0.0`
  - max case pass-rate delta: `0.0`

## Decision
**GO for RC execution** on the locked level-up profile.

Scope constraints:
1. This GO applies only to the validated level-up profile/configuration.
2. Legacy baseline profile remains NO-GO for release.
3. Any profile change requires re-running phase gates.

## Execution Checklist
1. Freeze profile config and seed policy in release docs.
2. Build RC artifact pack with linked JSON/CSV/SVG evidence.
3. Keep baseline-vs-level-up comparison artifacts for regression monitoring.
