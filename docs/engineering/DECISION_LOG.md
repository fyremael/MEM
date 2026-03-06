# Decision Log

## Revision History
- 2026-03-05: Confirmed level-up campaign at 10 seeds (`60/60`) and moved to conditional RC-planning GO for that profile.
- 2026-03-05: Added preliminary level-up campaign evidence (`18/18` on 3 seeds) and confirmation gate requirements.
- 2026-03-05: Added GPU validation evidence; confirmed compute path is not the current production blocker.
- 2026-03-05: Added 10-seed (60-trial) reliability evidence; maintained NO-GO for production.
- 2026-03-04: Updated decision gate with 3-seed reliability matrix evidence and explicit production acceptance thresholds.
- 2026-03-04: Initial decision log created with current project gate status.

## Decision Framework
- GO for next phase if:
  - corridor stability criteria pass in target regime
  - retrieval threshold passes with acceptable consistency
- NO-GO for production if:
  - retrieval threshold not robust at required operating frontier

## Current Decision (2026-03-05)
- Decision: **GO (conditional RC planning for level-up profile)**, **NO-GO (legacy baseline profile)**.
- Rationale:
  - corridor stability is consistently strong in stress frontiers.
  - baseline reliability remains far below production threshold.
  - level-up profile now exceeds reliability threshold on the current frontier.

## Evidence Snapshot
- Study: `demo_runs/corridor_stress_v5`
- Cases: 6
- Corridor successes: 2
- Hardest success:
  - `stress_layers16_writes1_dist64_mem6_noise32_pairs96`
- Easiest failure:
  - `stress_layers16_writes1_dist56_mem6_noise32_pairs96`
- Failure mode:
  - retrieval accuracy threshold miss; corridor metrics still pass.

Reliability extension:
- Study: `demo_runs/corridor_reliability_v1/report/reliability_summary.json`
- Seeds: 10
- Trials: 60
- Overall pass rate: 23.3%
- Case pass-rate range: 0.0% to 50.0%

Compute extension:
- Study: `demo_runs/gpu_validation_v1/report/backend_comparison.json`
- GPU backend validated (`cuda:0`) for project benchmark runs.
- Throughput depends on utilization:
  - low-utilization profile: CPU faster
  - high-utilization profile: GPU ~2.58x faster

Level-up extension (preliminary):
- Study: `demo_runs/corridor_reliability_levelup_v1/report/reliability_summary.json`
- Seeds: 10
- Trials: 60
- Overall pass rate: 100.0%
- Interpretation: profile passes the reliability gate on this frontier.

## Pending Gate Items
1. Expand level-up profile validation beyond current frontier difficulty.
2. Maintain side-by-side baseline control for comparison integrity.
3. Hit production acceptance thresholds in expanded scope:
   - overall pass rate >= 80%
   - no frontier case pass rate < 67%
   - no keep-layer instability signal

## Next Formal Review Trigger
- Trigger when:
  - multi-seed frontier summary is complete,
  - and pass-rate criterion is computed for executive sign-off.
