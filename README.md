# MEM: Modulus Memory Channels

MEM is a JAX-based project for testing explicit memory-channel behavior in transformer-style models.
The point is simple: run real experiments, inspect failure modes, and keep a paper trail for decisions.

## Why this repo exists
- Memory behavior is easy to overclaim and hard to validate.
- We wanted one workflow for train/probe/sweep/report instead of scattered notebooks.
- We also wanted go/no-go decisions backed by artifacts, not vibes.

## What is already working
- Reliability wall detection and recovery via profile tuning.
- GPU-backed execution runs with throughput analysis.
- Phase-gate evidence for frontier, adversarial, and reproducibility checks.

## Core capabilities
- Dual-stream memory model with explicit write/keep scheduling.
- Probe suite for Jacobian, eigengap, drift, and leakage metrics.
- Corridor scoring and ranked frontier search.
- Multi-seed reliability matrix generation and comparison.
- Automated API/reference doc generation with CI checks.

## Quick start
```bash
python -m pip install -e .[dev]
python -m pytest -q tests
python scripts/process_guard.py
python scripts/build_api_reference.py --check
```

## Useful entrypoints
- CLI: `python -m modulus_memory_channels --help`
- Reliability runner: `python scripts/reliability_matrix.py --help`
- Reliability comparison: `python scripts/reliability_compare.py --help`
- Phase-gate report: `python scripts/phase_gate_report.py --help`
- API docs generator: `python scripts/build_api_reference.py --help`

## Docs map
- Engineering: `docs/engineering/`
- Process/governance: `docs/process/`
- API/reference: `docs/reference/`

Good starting docs:
- `docs/engineering/EXECUTIVE_OVERVIEW.md`
- `docs/engineering/PHASE_GATE_RC_READINESS.md`
- `docs/engineering/LEVEL_UP_CAMPAIGN.md`
