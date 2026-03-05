# GPU Validation

## Revision History
- 2026-03-05: Initial GPU backend validation and CPU-vs-GPU benchmark evidence.

## Objective
Demonstrate whether project runs can execute on GPU and quantify CPU-vs-GPU tradeoffs for this workload.

## Backend Proof
Validated in WSL Ubuntu:
- `nvidia-smi` shows `NVIDIA GeForce RTX 2080` with CUDA `13.1`.
- JAX runtime reports:
  - CPU mode: `backend_actual=cpu`, `devices=[TFRT_CPU_0]`
  - GPU mode: `backend_actual=gpu`, `devices=[cuda:0]`

Reference artifacts:
- `demo_runs/gpu_validation_v1/cpu_benchmark.json`
- `demo_runs/gpu_validation_v1/gpu_benchmark.json`

## Benchmark Method
Script:
- `scripts/backend_benchmark.py`

Compared identical training configs across CPU vs GPU.

## Results
### Profile A (frontier-like, low utilization)
- Output: `demo_runs/gpu_validation_v1/report/backend_comparison.json`
- Config highlight: `batch_size=10`, `steps=64`, `layers=16`, `d_model=64`
- Result:
  - CPU wall time: `56.39s`
  - GPU wall time: `105.56s`
  - Throughput ratio (GPU/CPU): `0.53x`

Interpretation:
- This regime is compile/launch overhead dominated and under-utilizes the GPU.

### Profile B (higher utilization)
- Output: `demo_runs/gpu_validation_v1_heavy_b64/report/backend_comparison.json`
- Config highlight: `batch_size=64`, `steps=64`, `layers=20`, `d_model=96`
- Result:
  - CPU wall time: `307.96s`
  - GPU wall time: `119.19s`
  - Throughput ratio (GPU/CPU): `2.58x`

Interpretation:
- GPU provides clear acceleration once workload size is sufficient.

## Accuracy Consistency
Both profiles showed matching eval outcomes between CPU and GPU in these runs.
- Profile A: same `eval_min_accuracy` (`0.9125`)
- Profile B: same `eval_min_accuracy` (`1.0`)

## Decision Impact
1. GPU execution is validated and usable for project demos and training.
2. Performance benefit depends on operating point; low-batch runs can be CPU-favored.
3. Reliability campaigns should use GPU-oriented batch/model settings if throughput is a goal.
