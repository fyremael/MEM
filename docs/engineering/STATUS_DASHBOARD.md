# Status Dashboard

## Revision History
- 2026-03-05: Added GPU validation snapshot with CPU-vs-GPU benchmark evidence.
- 2026-03-05: Expanded reliability matrix to 10 seeds and refreshed KPI snapshot and priorities.
- 2026-03-04: Added multi-seed reliability matrix status (`corridor_reliability_v1`) and updated gate posture.
- 2026-03-04: Initial dashboard populated from `corridor_stress_v5`.

## Current Program Status
- Architecture: Dual-stream memory corridor with explicit write/keep scheduling.
- Tooling: Train/probe/report/sweep/stress with resume support.
- Technical status: Corridor stable at high scale; retrieval reliability is current limiter.

## Latest Frontier (v5)
- Source:
  - `demo_runs/corridor_stress_v5/sweep_summary.json`
  - `demo_runs/corridor_stress_v5/sweep_summary.csv`
- Cases: `6`
- Corridor achieved: `2`
- Hardest success:
  - `stress_layers16_writes1_dist64_mem6_noise32_pairs96`
- Easiest failure:
  - `stress_layers16_writes1_dist56_mem6_noise32_pairs96`

## Reliability Matrix (v1)
- Source:
  - `demo_runs/corridor_reliability_v1/report/reliability_summary.json`
  - `demo_runs/corridor_reliability_v1/report/reliability_case_summary.csv`
- Seeds: `10`
- Cases per seed: `6`
- Trials: `60`
- Overall pass rate: `23.3%` (`14/60`)
- Seed success rates:
  - `seed_0509`: `33.3%`
  - `seed_0607`: `33.3%`
  - `seed_0709`: `50.0%`
  - `seed_0811`: `0.0%`
  - `seed_0913`: `50.0%`
  - `seed_1015`: `16.7%`
  - `seed_1117`: `0.0%`
  - `seed_1219`: `0.0%`
  - `seed_1321`: `0.0%`
  - `seed_1423`: `50.0%`

## Compute Validation
- Source:
  - `demo_runs/gpu_validation_v1/report/backend_comparison.json`
  - `demo_runs/gpu_validation_v1_heavy_b64/report/backend_comparison.json`
- Backend proof: GPU confirmed as `cuda:0` in WSL benchmark runs.
- Throughput evidence:
  - Low-utilization profile (`batch=10`): GPU/CPU = `0.53x` (CPU faster)
  - High-utilization profile (`batch=64`): GPU/CPU = `2.58x` (GPU faster)

## KPI Table
| Metric | Current |
|---|---:|
| Max tested layers | 16 |
| Max tested memories | 6 |
| Max tested distance | 64 (eval to 72) |
| Max tested distractors | 32 |
| Max tested pair inventory | 96 |
| Current success rate in v5 frontier | 33.3% |
| Current multi-seed reliability pass rate | 23.3% |
| GPU status | validated (`cuda:0`) |
| Dominant failure type | Retrieval threshold miss |
| Corridor stability in failures | Still passing |

## Key Visuals
- `demo_runs/corridor_stress_v5/corridor_scores.svg`
- `demo_runs/corridor_stress_v5/min_eval_accuracy.svg`
- `demo_runs/corridor_stress_v5/report/compare_train/eval_accuracy_compare.svg`
- `demo_runs/corridor_reliability_v1/report/pass_rate_by_case.svg`
- `demo_runs/corridor_reliability_v1/report/mean_min_eval_accuracy_by_case.svg`
- `demo_runs/corridor_reliability_v1/report/success_rate_by_seed.svg`

## Immediate Next Steps
1. Execute retrieval-robustness interventions and rerun this exact 10-seed matrix.
2. Increase eval batch counts to tighten confidence bounds for threshold decisions.
3. Shift reliability runs toward GPU-favored utilization settings where feasible.
