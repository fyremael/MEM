# Experiment Playbook

## Revision History
- 2026-03-04: Initial playbook added with reproducible commands and interpretation guidance.

## Baseline Commands

### Single Run
```bash
python -m modulus_memory_channels train --output-dir runs/base --steps 96
python -m modulus_memory_channels probe --run-dir runs/base --output-dir runs/base_probe --probe-layers 0 1 2 --operator sensitivity --state-view memory --perturb
python -m modulus_memory_channels report --run-dir runs/base --probe-dir runs/base_probe --output-dir runs/base_report
```

### Grid Sweep
```bash
python -m modulus_memory_channels sweep \
  --output-dir runs/sweep \
  --num-layers-grid 4 8 \
  --memory-write-layers-grid 1 2 3 \
  --distance-grid 16 32 \
  --num-memories-grid 2 4 \
  --num-distractors-grid 8 16 \
  --num-pairs-grid 32 64 \
  --operator sensitivity --state-view memory --perturb
```

### Stress Frontier (Hardest-First)
```bash
python -m modulus_memory_channels stress \
  --output-dir runs/stress \
  --resume \
  --num-layers-grid 16 \
  --memory-write-layers-grid 1 2 3 \
  --distance-grid 56 64 \
  --num-memories-grid 6 \
  --num-distractors-grid 32 \
  --num-pairs-grid 96 \
  --min-eval-accuracy 0.999 \
  --stop-after-failures 2 \
  --operator sensitivity --state-view memory --perturb
```

## Interpretation Rules
1. Corridor pass requires all criteria pass:
- min eval accuracy threshold
- max keep Jacobian deviation
- max keep state drift
- max keep state delta norm
- max keep leakage

2. Failure typing:
- If only `min_eval_accuracy` fails, classify as retrieval-boundary failure.
- If keep-layer stability criteria fail, classify as corridor failure.

3. Prioritize:
- hardest success case
- easiest failure case
- confidence with multiple seeds when near threshold

## Required Artifacts for Review
- `sweep_summary.json`
- `sweep_summary.csv`
- `sweep_summary.txt`
- `corridor_scores.svg`
- `min_eval_accuracy.svg`
- success-vs-failure comparison plots
