# Level-Up Campaign

## Revision History
- 2026-03-05: Initial GPU-first level-up campaign report with baseline comparison.

## Objective
Test whether a higher-capacity, GPU-oriented training profile can break the reliability wall observed in `corridor_reliability_v1`.

## Candidate Profile
- Runner: `scripts/reliability_matrix.py`
- Root: `demo_runs/corridor_reliability_levelup_v1`
- Seeds: `1601`, `1703`, `1805` (3 seeds)
- Core settings:
  - `steps=128`
  - `learning_rate=0.0025`
  - `batch_size=32`
  - `eval_batches=12`
  - `d_model=96`, `mlp_dim=256`, `memory_dim=32`
  - `num_layers_grid=16`, `memory_write_layers_grid={1,2,3}`
  - `distance_grid={56,64}`, `num_memories=6`, `num_distractors=32`, `num_pairs=96`
  - thresholds unchanged (`min_eval_accuracy >= 0.999`, keep-layer limits unchanged)

## Results
Source:
- `demo_runs/corridor_reliability_levelup_v1/report/reliability_summary.json`

Headline:
- `overall_pass_rate = 1.0` (`18/18`)
- every case passed in all 3 seeds
- `mean_min_eval_accuracy = 1.0` for all six cases

## Baseline Comparison
Source:
- `demo_runs/corridor_reliability_levelup_v1/report/compare_vs_v1/comparison.json`

Compared against `corridor_reliability_v1` (10 seeds, 60 trials):
- baseline overall pass rate: `23.3%`
- level-up overall pass rate: `100%` (3-seed preliminary)
- overall delta: `+76.7pp`

Case pass-rate deltas:
- `writes1_dist56`: `+0.8`
- `writes1_dist64`: `+0.5`
- `writes2_dist56`: `+0.6`
- `writes2_dist64`: `+0.8`
- `writes3_dist56`: `+1.0`
- `writes3_dist64`: `+0.9`

Visuals:
- `demo_runs/corridor_reliability_levelup_v1/report/compare_vs_v1/passrate_compare.svg`
- `demo_runs/corridor_reliability_levelup_v1/report/compare_vs_v1/mean_min_eval_accuracy_compare.svg`

## Interpretation
1. This is a strong breakout signal that the prior wall was recipe/capacity-bound, not a hard architecture limit.
2. The result is still preliminary because seed count is smaller (`3` vs `10`) and the profile changed.
3. The correct next step is confirmation at scale, not immediate production GO.

## Next Validation Gate
Promote this profile only if:
1. It sustains >=10 seeds on the same matrix.
2. Pass-rate remains >=80% overall with no case below 67%.
3. Corridor stability remains within existing limits.
