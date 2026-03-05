from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modulus_memory_channels.io import save_json  # noqa: E402
from modulus_memory_channels.visualization import write_bar_svg, write_csv  # noqa: E402


DEFAULT_RELIABILITY_ROOT = Path("demo_runs/corridor_reliability_v1")
DEFAULT_BASELINE_SUMMARY = Path("demo_runs/corridor_stress_v5/sweep_summary.csv")
DEFAULT_REPORT_DIR = DEFAULT_RELIABILITY_ROOT / "report"
DEFAULT_TARGET_SEEDS = (607, 709, 811, 913, 1015, 1117, 1219, 1321, 1423)


@dataclass(frozen=True)
class StressProfile:
    steps: int
    learning_rate: float
    vocab_size: int
    d_model: int
    num_heads: int
    mlp_dim: int
    memory_dim: int
    num_memory_heads: int
    memory_write_interval: int
    batch_size: int
    num_layers_grid: tuple[int, ...]
    memory_write_layers_grid: tuple[int, ...]
    distance_grid: tuple[int, ...]
    num_memories_grid: tuple[int, ...]
    num_distractors_grid: tuple[int, ...]
    num_pairs_grid: tuple[int, ...]
    eval_distances: tuple[int, ...]
    eval_batches: int
    operator: str
    state_view: str
    perturb: bool
    perturb_distractors: int
    min_eval_accuracy: float
    max_keep_jacobian_deviation: float
    max_keep_state_drift: float
    max_keep_state_delta_norm: float
    max_keep_leakage: float
    stop_after_failures: int
    platform: str
    resume: bool


def _seed_name(seed: int) -> str:
    return f"seed_{seed:04d}"


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _as_float(value: str) -> float:
    return float(value.strip())


def _wilson_interval(pass_count: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = pass_count / n
    z2 = z * z
    denom = 1.0 + (z2 / n)
    center = (phat + (z2 / (2.0 * n))) / denom
    half = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n)))
    return max(0.0, center - half), min(1.0, center + half)


def _read_sweep_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _append_many(cmd: list[str], flag: str, values: tuple[int, ...]) -> None:
    cmd.append(flag)
    cmd.extend(str(value) for value in values)


def _profile_from_args(args: argparse.Namespace) -> StressProfile:
    return StressProfile(
        steps=args.steps,
        learning_rate=args.learning_rate,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        memory_dim=args.memory_dim,
        num_memory_heads=args.num_memory_heads,
        memory_write_interval=args.memory_write_interval,
        batch_size=args.batch_size,
        num_layers_grid=tuple(args.num_layers_grid),
        memory_write_layers_grid=tuple(args.memory_write_layers_grid),
        distance_grid=tuple(args.distance_grid),
        num_memories_grid=tuple(args.num_memories_grid),
        num_distractors_grid=tuple(args.num_distractors_grid),
        num_pairs_grid=tuple(args.num_pairs_grid),
        eval_distances=tuple(args.eval_distances),
        eval_batches=args.eval_batches,
        operator=args.operator,
        state_view=args.state_view,
        perturb=args.perturb,
        perturb_distractors=args.perturb_distractors,
        min_eval_accuracy=args.min_eval_accuracy,
        max_keep_jacobian_deviation=args.max_keep_jacobian_deviation,
        max_keep_state_drift=args.max_keep_state_drift,
        max_keep_state_delta_norm=args.max_keep_state_delta_norm,
        max_keep_leakage=args.max_keep_leakage,
        stop_after_failures=args.stop_after_failures,
        platform=args.platform,
        resume=args.resume,
    )


def _resolve_baseline_summary(args: argparse.Namespace) -> Path | None:
    if args.no_baseline:
        return None
    return Path(args.baseline_summary)


def run_seed(seed: int, output_dir: Path, profile: StressProfile) -> None:
    cmd = [
        sys.executable,
        "-m",
        "modulus_memory_channels",
        "stress",
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--steps",
        str(profile.steps),
        "--learning-rate",
        str(profile.learning_rate),
        "--vocab-size",
        str(profile.vocab_size),
        "--d-model",
        str(profile.d_model),
        "--num-heads",
        str(profile.num_heads),
        "--mlp-dim",
        str(profile.mlp_dim),
        "--memory-dim",
        str(profile.memory_dim),
        "--num-memory-heads",
        str(profile.num_memory_heads),
        "--memory-write-interval",
        str(profile.memory_write_interval),
        "--batch-size",
        str(profile.batch_size),
    ]
    if profile.resume:
        cmd.append("--resume")
    _append_many(cmd, "--num-layers-grid", profile.num_layers_grid)
    _append_many(cmd, "--memory-write-layers-grid", profile.memory_write_layers_grid)
    _append_many(cmd, "--distance-grid", profile.distance_grid)
    _append_many(cmd, "--num-memories-grid", profile.num_memories_grid)
    _append_many(cmd, "--num-distractors-grid", profile.num_distractors_grid)
    _append_many(cmd, "--num-pairs-grid", profile.num_pairs_grid)
    _append_many(cmd, "--eval-distances", profile.eval_distances)
    cmd.extend(
        [
        "--eval-batches",
        str(profile.eval_batches),
        "--operator",
        profile.operator,
        "--state-view",
        profile.state_view,
        "--perturb-distractors",
        str(profile.perturb_distractors),
        "--min-eval-accuracy",
        str(profile.min_eval_accuracy),
        "--max-keep-jacobian-deviation",
        str(profile.max_keep_jacobian_deviation),
        "--max-keep-state-drift",
        str(profile.max_keep_state_drift),
        "--max-keep-state-delta-norm",
        str(profile.max_keep_state_delta_norm),
        "--max-keep-leakage",
        str(profile.max_keep_leakage),
        "--stop-after-failures",
        str(profile.stop_after_failures),
        ]
    )
    if profile.perturb:
        cmd.append("--perturb")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{SRC_ROOT}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(SRC_ROOT)
    if profile.platform == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    elif profile.platform == "gpu":
        env["JAX_PLATFORMS"] = "cuda"
    else:
        env.pop("JAX_PLATFORMS", None)
    print(f"running seed {seed} -> {output_dir}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def aggregate(
    reliability_root: Path,
    baseline_summary: Path | None,
    report_dir: Path,
) -> None:
    seed_runs: dict[str, str] = {}
    all_rows: list[dict[str, object]] = []

    if baseline_summary is not None and baseline_summary.exists():
        seed_runs["seed_0509"] = str(baseline_summary)
        for row in _read_sweep_csv(baseline_summary):
            row = dict(row)
            row["seed"] = "seed_0509"
            all_rows.append(row)

    for seed_dir in sorted(reliability_root.glob("seed_*")):
        summary_path = seed_dir / "sweep_summary.csv"
        if not summary_path.exists():
            continue
        seed_name = seed_dir.name
        seed_runs[seed_name] = str(summary_path)
        for row in _read_sweep_csv(summary_path):
            row = dict(row)
            row["seed"] = seed_name
            all_rows.append(row)

    if not all_rows:
        raise FileNotFoundError("No sweep summaries found for aggregation.")

    by_case: dict[str, list[dict[str, object]]] = {}
    by_seed: dict[str, list[dict[str, object]]] = {}
    for row in all_rows:
        case = str(row["case"])
        by_case.setdefault(case, []).append(row)
        seed = str(row["seed"])
        by_seed.setdefault(seed, []).append(row)

    case_summaries: list[dict[str, object]] = []
    total_passes = 0
    total_trials = 0
    for case, rows in by_case.items():
        n = len(rows)
        pass_count = sum(1 for row in rows if _as_bool(str(row["corridor_achieved"])))
        total_passes += pass_count
        total_trials += n
        wilson_low, wilson_high = _wilson_interval(pass_count, n)
        case_summaries.append(
            {
                "case": case,
                "difficulty_score": sum(_as_float(str(r["difficulty_score"])) for r in rows) / n,
                "seeds_tested": n,
                "pass_count": pass_count,
                "pass_rate": pass_count / n,
                "wilson_low_95": wilson_low,
                "wilson_high_95": wilson_high,
                "mean_min_eval_accuracy": sum(_as_float(str(r["min_eval_accuracy"])) for r in rows) / n,
                "min_min_eval_accuracy": min(_as_float(str(r["min_eval_accuracy"])) for r in rows),
                "max_min_eval_accuracy": max(_as_float(str(r["min_eval_accuracy"])) for r in rows),
                "mean_keep_jacobian_deviation": sum(
                    _as_float(str(r["max_keep_jacobian_deviation"])) for r in rows
                )
                / n,
                "mean_keep_state_drift": sum(_as_float(str(r["max_keep_state_drift"])) for r in rows) / n,
                "mean_keep_state_delta_norm": sum(
                    _as_float(str(r["max_keep_state_delta_norm"])) for r in rows
                )
                / n,
                "mean_keep_leakage": sum(_as_float(str(r["max_keep_leakage"])) for r in rows) / n,
            }
        )

    case_summaries.sort(key=lambda row: (float(row["difficulty_score"]), str(row["case"])), reverse=True)

    seed_summaries: list[dict[str, object]] = []
    for seed, rows in sorted(by_seed.items()):
        n = len(rows)
        pass_count = sum(1 for row in rows if _as_bool(str(row["corridor_achieved"])))
        seed_summaries.append(
            {
                "seed": seed,
                "num_cases": n,
                "num_successes": pass_count,
                "success_rate": (pass_count / n) if n else 0.0,
            }
        )

    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        case_summaries,
        report_dir / "reliability_case_summary.csv",
        fieldnames=[
            "case",
            "difficulty_score",
            "seeds_tested",
            "pass_count",
            "pass_rate",
            "wilson_low_95",
            "wilson_high_95",
            "mean_min_eval_accuracy",
            "min_min_eval_accuracy",
            "max_min_eval_accuracy",
            "mean_keep_jacobian_deviation",
            "mean_keep_state_drift",
            "mean_keep_state_delta_norm",
            "mean_keep_leakage",
        ],
    )
    write_csv(
        seed_summaries,
        report_dir / "reliability_seed_summary.csv",
        fieldnames=["seed", "num_cases", "num_successes", "success_rate"],
    )

    summary = {
        "seed_runs": seed_runs,
        "num_seeds": len(seed_summaries),
        "num_cases": len(case_summaries),
        "overall_trials": total_trials,
        "overall_passes": total_passes,
        "overall_pass_rate": (total_passes / total_trials) if total_trials else 0.0,
        "cases": case_summaries,
        "per_seed": seed_summaries,
    }
    save_json(summary, report_dir / "reliability_summary.json")
    report_lines = [
        "# Reliability Matrix Summary",
        "",
        f"- num_seeds: {summary['num_seeds']}",
        f"- num_cases: {summary['num_cases']}",
        f"- overall_trials: {summary['overall_trials']}",
        f"- overall_passes: {summary['overall_passes']}",
        f"- overall_pass_rate: {summary['overall_pass_rate']:.4f}",
        "",
        "## Case Pass Rates",
    ]
    for row in case_summaries:
        report_lines.append(
            f"- {row['case']}: {row['pass_count']}/{row['seeds_tested']} ({row['pass_rate']:.3f})"
        )
    (report_dir / "reliability_summary.md").write_text("\n".join(report_lines) + "\n", encoding="ascii")

    write_bar_svg(
        [str(row["case"]) for row in case_summaries],
        [float(row["pass_rate"]) for row in case_summaries],
        report_dir / "pass_rate_by_case.svg",
        title="Pass Rate by Case",
        y_label="Pass Rate",
    )
    write_bar_svg(
        [str(row["case"]) for row in case_summaries],
        [float(row["mean_min_eval_accuracy"]) for row in case_summaries],
        report_dir / "mean_min_eval_accuracy_by_case.svg",
        title="Mean Min Eval Accuracy by Case",
        y_label="Accuracy",
    )
    write_bar_svg(
        [str(row["seed"]) for row in seed_summaries],
        [float(row["success_rate"]) for row in seed_summaries],
        report_dir / "success_rate_by_seed.svg",
        title="Success Rate by Seed",
        y_label="Success Rate",
    )
    print(f"wrote aggregated reliability report to {report_dir}")


def _cmd_run(args: argparse.Namespace) -> int:
    reliability_root = Path(args.reliability_root)
    reliability_root.mkdir(parents=True, exist_ok=True)
    profile = _profile_from_args(args)
    for seed in args.seeds:
        run_seed(seed, reliability_root / _seed_name(seed), profile)
    return 0


def _cmd_aggregate(args: argparse.Namespace) -> int:
    aggregate(
        reliability_root=Path(args.reliability_root),
        baseline_summary=_resolve_baseline_summary(args),
        report_dir=Path(args.report_dir),
    )
    return 0


def _cmd_run_and_aggregate(args: argparse.Namespace) -> int:
    reliability_root = Path(args.reliability_root)
    reliability_root.mkdir(parents=True, exist_ok=True)
    profile = _profile_from_args(args)
    for seed in args.seeds:
        run_seed(seed, reliability_root / _seed_name(seed), profile)
    aggregate(
        reliability_root=reliability_root,
        baseline_summary=_resolve_baseline_summary(args),
        report_dir=Path(args.report_dir),
    )
    return 0


def _add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--platform", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.0035)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-dim", type=int, default=160)
    parser.add_argument("--memory-dim", type=int, default=24)
    parser.add_argument("--num-memory-heads", type=int, default=4)
    parser.add_argument("--memory-write-interval", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-layers-grid", type=int, nargs="+", default=[16])
    parser.add_argument("--memory-write-layers-grid", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--distance-grid", type=int, nargs="+", default=[56, 64])
    parser.add_argument("--num-memories-grid", type=int, nargs="+", default=[6])
    parser.add_argument("--num-distractors-grid", type=int, nargs="+", default=[32])
    parser.add_argument("--num-pairs-grid", type=int, nargs="+", default=[96])
    parser.add_argument("--eval-distances", type=int, nargs="+", default=[56, 64, 72])
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--operator", choices=["covariance", "sensitivity"], default="sensitivity")
    parser.add_argument("--state-view", choices=["full", "token", "memory"], default="memory")
    parser.add_argument("--perturb", action="store_true", default=True)
    parser.add_argument("--no-perturb", action="store_false", dest="perturb")
    parser.add_argument("--perturb-distractors", type=int, default=2)
    parser.add_argument("--min-eval-accuracy", type=float, default=0.999)
    parser.add_argument("--max-keep-jacobian-deviation", type=float, default=0.05)
    parser.add_argument("--max-keep-state-drift", type=float, default=0.01)
    parser.add_argument("--max-keep-state-delta-norm", type=float, default=0.001)
    parser.add_argument("--max-keep-leakage", type=float, default=0.001)
    parser.add_argument("--stop-after-failures", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and aggregate frontier reliability matrix.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run stress frontier for one or more seeds.")
    run_parser.add_argument("--reliability-root", default=str(DEFAULT_RELIABILITY_ROOT))
    run_parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_TARGET_SEEDS))
    _add_profile_arguments(run_parser)
    run_parser.set_defaults(handler=_cmd_run)

    agg_parser = subparsers.add_parser("aggregate", help="Aggregate reliability summary from seed sweeps.")
    agg_parser.add_argument("--reliability-root", default=str(DEFAULT_RELIABILITY_ROOT))
    agg_parser.add_argument("--baseline-summary", default=str(DEFAULT_BASELINE_SUMMARY))
    agg_parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    agg_parser.add_argument("--no-baseline", action="store_true")
    agg_parser.set_defaults(handler=_cmd_aggregate)

    both_parser = subparsers.add_parser(
        "run-and-aggregate",
        help="Run missing seeds then aggregate reliability report artifacts.",
    )
    both_parser.add_argument("--reliability-root", default=str(DEFAULT_RELIABILITY_ROOT))
    both_parser.add_argument("--baseline-summary", default=str(DEFAULT_BASELINE_SUMMARY))
    both_parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    both_parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_TARGET_SEEDS))
    both_parser.add_argument("--no-baseline", action="store_true")
    _add_profile_arguments(both_parser)
    both_parser.set_defaults(handler=_cmd_run_and_aggregate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
