# Experiment Playbook

## Revision History
- 2026-03-05: Added adversarial and phase-gate report command templates.
- 2026-03-05: Added GPU-first level-up reliability campaign command.
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

### Level-Up Reliability Campaign (GPU-First)
```bash
wsl -d Ubuntu -- bash -lc "cd /mnt/f/_codex/MEM && python3 scripts/reliability_matrix.py run-and-aggregate \
  --reliability-root demo_runs/corridor_reliability_levelup_v1 \
  --report-dir demo_runs/corridor_reliability_levelup_v1/report \
  --no-baseline \
  --seeds 1601 1703 1805 1907 2009 2111 2213 2315 2417 2519 \
  --platform gpu \
  --steps 128 \
  --learning-rate 0.0025 \
  --d-model 96 \
  --mlp-dim 256 \
  --memory-dim 32 \
  --batch-size 32 \
  --eval-batches 12"
```

### Adversarial Gate Campaign (GPU-First)
```bash
wsl -d Ubuntu -- bash -lc "cd /mnt/f/_codex/MEM && python3 scripts/reliability_matrix.py run-and-aggregate \
  --reliability-root demo_runs/corridor_reliability_levelup_v1_adversarial \
  --report-dir demo_runs/corridor_reliability_levelup_v1_adversarial/report \
  --no-baseline \
  --seeds 2601 2703 2805 2907 3009 \
  --platform gpu \
  --steps 128 \
  --learning-rate 0.0025 \
  --d-model 96 \
  --mlp-dim 256 \
  --memory-dim 32 \
  --batch-size 32 \
  --distance-grid 64 72 \
  --num-memories-grid 8 \
  --num-distractors-grid 48 \
  --eval-distances 64 72 80 \
  --perturb-distractors 8"
```

### Phase Gate Report
```bash
python scripts/phase_gate_report.py \
  --frontier-summary demo_runs/corridor_reliability_levelup_v1/report/reliability_summary.json \
  --adversarial-summary demo_runs/corridor_reliability_levelup_v1_adversarial/report/reliability_summary.json \
  --repro-summary demo_runs/corridor_repro_check_v1/report/reliability_summary.json \
  --output-dir demo_runs/phase_gate_v1/report
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
