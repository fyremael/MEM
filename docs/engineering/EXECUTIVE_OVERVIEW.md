# Executive Overview

## Revision History
- 2026-03-05: Added GPU execution validation and throughput findings.
- 2026-03-05: Expanded reliability matrix to 10 seeds (60 trials) and updated go/no-go evidence.
- 2026-03-04: Initial executive brief added with multi-seed reliability analysis and gate decision framing.

## Program History
1. Baseline corridor architecture established with explicit write/keep scheduling and probe instrumentation.
2. Frontier stress campaign (`corridor_stress_v5`) expanded operating envelope to 16 layers, 6 memories, and long distances.
3. Reliability campaign (`corridor_reliability_v1`) repeated the six frontier cases across 10 seeds (60 total trials).
4. Decision process shifted from single-run point outcomes to multi-seed pass-rate evidence.

## Current Status (As of 2026-03-05)
- Decision: **GO for next validation phase**, **NO-GO for production release**.
- Why:
  - Corridor stability is consistently strong, including failed retrieval-threshold cases.
  - Retrieval pass-rate is materially below production requirements across the frontier matrix.
  - GPU execution is validated, but performance depends on workload utilization profile.

## Compute Validation Snapshot
- Source:
  - `demo_runs/gpu_validation_v1/report/backend_comparison.json`
  - `demo_runs/gpu_validation_v1_heavy_b64/report/backend_comparison.json`
- Findings:
  - GPU backend is confirmed (`cuda:0`) in benchmark runs.
  - Low-utilization profile (`batch=10`) is CPU-favored (`GPU/CPU throughput: 0.53x`).
  - Higher-utilization profile (`batch=64`) is GPU-favored (`GPU/CPU throughput: 2.58x`).

## Key Metrics
Source artifacts:
- `demo_runs/corridor_stress_v5/sweep_summary.json`
- `demo_runs/corridor_reliability_v1/report/reliability_summary.json`

| Metric | Value |
|---|---:|
| Frontier cases | 6 |
| Reliability seeds | 10 |
| Total reliability trials | 60 |
| Total passes | 14 |
| Overall pass rate | 23.3% |
| Best case pass rate | 50.0% |
| Worst case pass rate | 0.0% |
| Dominant failure mode | `min_eval_accuracy` threshold miss |
| Corridor stability signal | strong and consistent |

## Case-Level Reliability (10 Seeds)
| Case | Passes | Pass Rate |
|---|---:|---:|
| `stress_layers16_writes1_dist64_mem6_noise32_pairs96` | 5 / 10 | 50.0% |
| `stress_layers16_writes2_dist64_mem6_noise32_pairs96` | 2 / 10 | 20.0% |
| `stress_layers16_writes3_dist64_mem6_noise32_pairs96` | 1 / 10 | 10.0% |
| `stress_layers16_writes1_dist56_mem6_noise32_pairs96` | 2 / 10 | 20.0% |
| `stress_layers16_writes2_dist56_mem6_noise32_pairs96` | 4 / 10 | 40.0% |
| `stress_layers16_writes3_dist56_mem6_noise32_pairs96` | 0 / 10 | 0.0% |

## Decision Reasoning
1. Architectural hypothesis is validated:
   - keep layers preserve memory-channel stability under high difficulty.
   - failures do not show corridor collapse signatures.
2. Product readiness is not yet validated:
   - pass-rate variance is high across seeds and includes multiple zero-success seeds.
   - no frontier case currently meets production reliability requirements.
3. Executive implication:
   - prioritize retrieval robustness optimization before any release gating.
   - do not commit to production launch gate yet.

## Architecture Diagram
```mermaid
flowchart LR
  T[Token Stream] --> A[Token Attention + MLP]
  M[Memory Stream] --> K[Keep Path]
  A --> W[Write Path]
  K --> U[Memory Update]
  W --> U
  U --> R[Read Path]
  R --> T
```

## Process Flow
```mermaid
flowchart TD
  C[Config] --> TR[Train]
  TR --> PR[Probe]
  PR --> RP[Report]
  RP --> ST[Sweep/Stress]
  ST --> RL[Reliability Aggregate]
  RL --> DG[Go/No-Go Gate]
  DG -->|No-Go| IT[Iterate Architecture and Training]
  DG -->|Go| NX[Next Phase / Release Gate]
```

## Visual Evidence Pack
- `demo_runs/corridor_reliability_v1/report/pass_rate_by_case.svg`
- `demo_runs/corridor_reliability_v1/report/mean_min_eval_accuracy_by_case.svg`
- `demo_runs/corridor_reliability_v1/report/success_rate_by_seed.svg`
- `demo_runs/corridor_stress_v5/corridor_scores.svg`
- `demo_runs/corridor_stress_v5/min_eval_accuracy.svg`

## Next Gate Criteria
Production gate should require all:
1. Overall pass rate >= 0.80 on frontier matrix.
2. No frontier case below 0.67 pass rate.
3. No instability pattern in keep-layer criteria.
