from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _run_single(
    *,
    backend: str,
    output_json: Path,
    seed: int,
    steps: int,
    eval_batches: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    mlp_dim: int,
    memory_dim: int,
    num_memory_heads: int,
    memory_write_layers: int,
    batch_size: int,
) -> int:
    if backend == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        os.environ.pop("JAX_PLATFORMS", None)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax

    from modulus_memory_channels.bench_write_keep_read import WriteKeepReadConfig
    from modulus_memory_channels.config import MemoryModelConfig
    from modulus_memory_channels.io import save_json
    from modulus_memory_channels.model import MemoryChannelsModel
    from modulus_memory_channels.training import TrainConfig, train_write_keep_read_model

    model_config = MemoryModelConfig(
        vocab_size=512,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        memory_dim=memory_dim,
        num_memory_heads=num_memory_heads,
        memory_write_interval=1,
        memory_write_layers=memory_write_layers,
        keep_write_scale=0.0,
        keep_read_scale=0.25,
        keep_token_mix_scale=0.1,
    )
    task_config = WriteKeepReadConfig(
        vocab_size=512,
        num_pairs=96,
        batch_size=batch_size,
        distance=64,
        num_distractors=32,
        num_memories=6,
    )
    train_config = TrainConfig(
        steps=steps,
        learning_rate=0.0035,
        eval_distances=(56, 64, 72),
        eval_batches=eval_batches,
    )

    model = MemoryChannelsModel(model_config)
    init_key = jax.random.PRNGKey(seed)
    params = model.init(init_key)
    train_key = jax.random.PRNGKey(seed + 1)

    start = time.perf_counter()
    result = train_write_keep_read_model(
        model,
        params,
        task_config,
        train_config,
        key=train_key,
    )
    wall_seconds = time.perf_counter() - start

    eval_values = list(result.eval_accuracy_by_distance.values())
    payload = {
        "backend_requested": backend,
        "backend_actual": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "seed": seed,
        "model_config": {
            "d_model": model_config.d_model,
            "num_layers": model_config.num_layers,
            "num_heads": model_config.num_heads,
            "mlp_dim": model_config.mlp_dim,
            "memory_dim": model_config.memory_dim,
            "num_memory_heads": model_config.num_memory_heads,
            "memory_write_interval": model_config.memory_write_interval,
            "memory_write_layers": model_config.memory_write_layers,
        },
        "task_config": {
            "num_pairs": task_config.num_pairs,
            "batch_size": task_config.batch_size,
            "distance": task_config.distance,
            "num_distractors": task_config.num_distractors,
            "num_memories": task_config.num_memories,
        },
        "train_config": {
            "steps": train_config.steps,
            "learning_rate": train_config.learning_rate,
            "eval_distances": list(train_config.eval_distances),
            "eval_batches": train_config.eval_batches,
        },
        "metrics": {
            "wall_seconds": wall_seconds,
            "steps_per_second": train_config.steps / wall_seconds if wall_seconds > 0 else 0.0,
            "final_train_loss": result.history["loss"][-1],
            "final_train_accuracy": result.history["accuracy"][-1],
            "eval_by_distance": result.eval_accuracy_by_distance,
            "eval_min_accuracy": min(eval_values),
            "eval_mean_accuracy": sum(eval_values) / len(eval_values),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    save_json(payload, output_json)
    print(f"saved benchmark result to {output_json}")
    return 0


def _run_compare(*, cpu_json: Path, gpu_json: Path, output_dir: Path) -> int:
    from modulus_memory_channels.io import load_json, save_json
    from modulus_memory_channels.visualization import write_bar_svg, write_csv

    cpu = load_json(cpu_json)
    gpu = load_json(gpu_json)

    cpu_wall = float(cpu["metrics"]["wall_seconds"])
    gpu_wall = float(gpu["metrics"]["wall_seconds"])
    cpu_sps = float(cpu["metrics"]["steps_per_second"])
    gpu_sps = float(gpu["metrics"]["steps_per_second"])
    cpu_eval_min = float(cpu["metrics"]["eval_min_accuracy"])
    gpu_eval_min = float(gpu["metrics"]["eval_min_accuracy"])

    comparison = {
        "cpu": cpu,
        "gpu": gpu,
        "summary": {
            "wall_seconds_cpu": cpu_wall,
            "wall_seconds_gpu": gpu_wall,
            "speedup_wall_time_cpu_over_gpu": (cpu_wall / gpu_wall) if gpu_wall > 0 else 0.0,
            "steps_per_second_cpu": cpu_sps,
            "steps_per_second_gpu": gpu_sps,
            "throughput_speedup_gpu_over_cpu": (gpu_sps / cpu_sps) if cpu_sps > 0 else 0.0,
            "eval_min_accuracy_cpu": cpu_eval_min,
            "eval_min_accuracy_gpu": gpu_eval_min,
            "eval_min_accuracy_delta_gpu_minus_cpu": gpu_eval_min - cpu_eval_min,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(comparison, output_dir / "backend_comparison.json")
    write_csv(
        [
            {"backend": "cpu", "wall_seconds": cpu_wall, "steps_per_second": cpu_sps, "eval_min_accuracy": cpu_eval_min},
            {"backend": "gpu", "wall_seconds": gpu_wall, "steps_per_second": gpu_sps, "eval_min_accuracy": gpu_eval_min},
        ],
        output_dir / "backend_comparison.csv",
        fieldnames=["backend", "wall_seconds", "steps_per_second", "eval_min_accuracy"],
    )
    write_bar_svg(
        ["cpu", "gpu"],
        [cpu_wall, gpu_wall],
        output_dir / "wall_time_compare.svg",
        title="Wall Time (Lower is Better)",
        y_label="Seconds",
    )
    write_bar_svg(
        ["cpu", "gpu"],
        [cpu_sps, gpu_sps],
        output_dir / "throughput_compare.svg",
        title="Training Throughput (Higher is Better)",
        y_label="Steps / second",
    )
    write_bar_svg(
        ["cpu", "gpu"],
        [cpu_eval_min, gpu_eval_min],
        output_dir / "eval_min_accuracy_compare.svg",
        title="Eval Min Accuracy",
        y_label="Accuracy",
    )
    print(f"saved backend comparison report to {output_dir}")
    return 0


def _run_all(
    *,
    output_dir: Path,
    seed: int,
    steps: int,
    eval_batches: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    mlp_dim: int,
    memory_dim: int,
    num_memory_heads: int,
    memory_write_layers: int,
    batch_size: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cpu_json = output_dir / "cpu_benchmark.json"
    gpu_json = output_dir / "gpu_benchmark.json"

    for backend, out_path in (("cpu", cpu_json), ("gpu", gpu_json)):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "single",
            "--backend",
            backend,
            "--output-json",
            str(out_path),
            "--seed",
            str(seed),
            "--steps",
            str(steps),
            "--eval-batches",
            str(eval_batches),
            "--d-model",
            str(d_model),
            "--num-layers",
            str(num_layers),
            "--num-heads",
            str(num_heads),
            "--mlp-dim",
            str(mlp_dim),
            "--memory-dim",
            str(memory_dim),
            "--num-memory-heads",
            str(num_memory_heads),
            "--memory-write-layers",
            str(memory_write_layers),
            "--batch-size",
            str(batch_size),
        ]
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    return _run_compare(cpu_json=cpu_json, gpu_json=gpu_json, output_dir=output_dir / "report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU vs GPU benchmark on the memory-channel training workload.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="Run one backend benchmark.")
    single.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    single.add_argument("--output-json", required=True)
    single.add_argument("--seed", type=int, default=2026)
    single.add_argument("--steps", type=int, default=64)
    single.add_argument("--eval-batches", type=int, default=8)
    single.add_argument("--d-model", type=int, default=64)
    single.add_argument("--num-layers", type=int, default=16)
    single.add_argument("--num-heads", type=int, default=8)
    single.add_argument("--mlp-dim", type=int, default=160)
    single.add_argument("--memory-dim", type=int, default=24)
    single.add_argument("--num-memory-heads", type=int, default=4)
    single.add_argument("--memory-write-layers", type=int, default=1)
    single.add_argument("--batch-size", type=int, default=10)

    compare = subparsers.add_parser("compare", help="Compare precomputed CPU and GPU benchmark JSON files.")
    compare.add_argument("--cpu-json", required=True)
    compare.add_argument("--gpu-json", required=True)
    compare.add_argument("--output-dir", required=True)

    all_cmd = subparsers.add_parser("run-all", help="Run CPU and GPU benchmarks, then write comparison artifacts.")
    all_cmd.add_argument("--output-dir", required=True)
    all_cmd.add_argument("--seed", type=int, default=2026)
    all_cmd.add_argument("--steps", type=int, default=64)
    all_cmd.add_argument("--eval-batches", type=int, default=8)
    all_cmd.add_argument("--d-model", type=int, default=64)
    all_cmd.add_argument("--num-layers", type=int, default=16)
    all_cmd.add_argument("--num-heads", type=int, default=8)
    all_cmd.add_argument("--mlp-dim", type=int, default=160)
    all_cmd.add_argument("--memory-dim", type=int, default=24)
    all_cmd.add_argument("--num-memory-heads", type=int, default=4)
    all_cmd.add_argument("--memory-write-layers", type=int, default=1)
    all_cmd.add_argument("--batch-size", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "single":
        return _run_single(
            backend=args.backend,
            output_json=Path(args.output_json),
            seed=args.seed,
            steps=args.steps,
            eval_batches=args.eval_batches,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            memory_dim=args.memory_dim,
            num_memory_heads=args.num_memory_heads,
            memory_write_layers=args.memory_write_layers,
            batch_size=args.batch_size,
        )
    if args.command == "compare":
        return _run_compare(
            cpu_json=Path(args.cpu_json),
            gpu_json=Path(args.gpu_json),
            output_dir=Path(args.output_dir),
        )
    if args.command == "run-all":
        return _run_all(
            output_dir=Path(args.output_dir),
            seed=args.seed,
            steps=args.steps,
            eval_batches=args.eval_batches,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            memory_dim=args.memory_dim,
            num_memory_heads=args.num_memory_heads,
            memory_write_layers=args.memory_write_layers,
            batch_size=args.batch_size,
        )
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
