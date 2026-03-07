# MEM: Modulus Memory Channels

Memory systems are easy to hand-wave and hard to prove.  
This repository exists to prove them.

MEM is a disciplined JAX harness for building, stressing, and validating explicit memory-channel behavior in transformer-style models. It combines architecture-level diagnostics, reproducible experiments, and executive-grade decision reporting in one workflow.

## Why MEM
- **Evidence over optimism**: every decision is tied to generated artifacts, metrics, and gate criteria.
- **Stress-ready by design**: train/probe/report/sweep/stress commands are built in.
- **Reproducibility as policy**: process guards, CI gates, and reference docs are automated.
- **Production-minded governance**: go/no-go is explicit, revisioned, and measurable.

## What We Demonstrated
- Reliability wall detection and recovery through profile-level optimization.
- GPU-backed execution validation with workload-aware throughput analysis.
- Full phase-gate closure (frontier, adversarial, reproducibility) with formal RC readiness documentation.

This is not a toy notebook stack. It is an operational research-to-release system.

## Core Capabilities
- Dual-stream memory model with explicit write/keep scheduling.
- Probe suite for Jacobian, eigengap, drift, and leakage metrics.
- Corridor scoring and ranked frontier search.
- Multi-seed reliability matrix aggregation and comparison tooling.
- Automated API/reference docs generation and CI enforcement.

## Fast Start
```bash
python -m pip install -e .[dev]
python -m pytest -q tests
python scripts/process_guard.py
python scripts/build_api_reference.py --check
```

## Key Entrypoints
- CLI: `python -m modulus_memory_channels --help`
- Reliability runner: `python scripts/reliability_matrix.py --help`
- Reliability comparison: `python scripts/reliability_compare.py --help`
- Phase-gate report: `python scripts/phase_gate_report.py --help`
- API docs generator: `python scripts/build_api_reference.py --help`

## Documentation Map
- Engineering source of truth: `docs/engineering/`
- Process and governance: `docs/process/`
- API/reference docs: `docs/reference/`

Start with:
- `docs/engineering/EXECUTIVE_OVERVIEW.md`
- `docs/engineering/PHASE_GATE_RC_READINESS.md`
- `docs/engineering/LEVEL_UP_CAMPAIGN.md`

## Philosophy
Great systems do not hide behind averages.  
They survive hard regimes, explain their behavior, and earn trust under scrutiny.

MEM is built for exactly that.
