# Context Snapshot

This page is auto-generated from tracked benchmark/governance artifacts and engineering status docs.

## Reliability Summaries

| Source | Seeds | Trials | Passes | Overall Pass Rate |
|---|---:|---:|---:|---:|
| _No tracked reliability summaries found_ | n/a | n/a | n/a | n/a |

## Stress Frontier Snapshot

| Source | Cases | Corridor Successes | Hardest Success | Easiest Failure |
|---|---:|---:|---|---|
| _No tracked stress summaries found_ | n/a | n/a | n/a | n/a |

## Phase Gate Reports

| Source | Decision | Frontier | Adversarial | Repro |
|---|---|---|---|---|
| _No tracked phase-gate reports found_ | n/a | n/a | n/a | n/a |

## Backend Comparison

| Source | CPU Backend | GPU Backend | GPU/CPU Throughput |
|---|---|---|---:|
| _No tracked backend comparison files found_ | n/a | n/a | n/a |

## Engineering Status Signals

| Signal | Value | Source |
|---|---|---|
| Current status date | 2026-03-05 | `docs/engineering/EXECUTIVE_OVERVIEW.md` |
| Decision | **GO (RC execution for locked level-up profile)**, **NO-GO (legacy baseline profile)**. | `docs/engineering/EXECUTIVE_OVERVIEW.md` |
| Architecture | Dual-stream memory corridor with explicit write/keep scheduling. | `docs/engineering/STATUS_DASHBOARD.md` |
| Technical status | Corridor stable at high scale; retrieval reliability is current limiter. | `docs/engineering/STATUS_DASHBOARD.md` |
| Overall phase-gate decision | `GO` (RC execution for locked level-up profile) | `docs/engineering/STATUS_DASHBOARD.md` |

## How To Refresh

```bash
python scripts/generate_context_docs.py
```
