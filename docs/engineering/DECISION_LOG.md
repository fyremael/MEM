# Decision Log

## Revision History
- 2026-03-04: Updated decision gate with 3-seed reliability matrix evidence and explicit production acceptance thresholds.
- 2026-03-04: Initial decision log created with current project gate status.

## Decision Framework
- GO for next phase if:
  - corridor stability criteria pass in target regime
  - retrieval threshold passes with acceptable consistency
- NO-GO for production if:
  - retrieval threshold not robust at required operating frontier

## Current Decision (2026-03-04)
- Decision: **GO (next validation phase)**, **NO-GO (production)**.
- Rationale:
  - corridor stability is consistently strong in stress frontiers.
  - retrieval reliability remains below production threshold at high difficulty boundary.

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
- Seeds: 3
- Trials: 18
- Overall pass rate: 38.9%
- Case pass-rate range: 0.0% to 66.7%

## Pending Gate Items
1. Expand repeatability from 3 seeds to >=10 seeds on the 16-layer frontier.
2. Increase eval sample counts for tighter confidence intervals.
3. Hit production acceptance thresholds:
   - overall pass rate >= 80%
   - no frontier case pass rate < 67%
   - no keep-layer instability signal

## Next Formal Review Trigger
- Trigger when:
  - multi-seed frontier summary is complete,
  - and pass-rate criterion is computed for executive sign-off.
