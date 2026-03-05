from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
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


def run_seed(seed: int, output_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "modulus_memory_channels",
        "stress",
        "--output-dir",
        str(output_dir),
        "--resume",
        "--seed",
        str(seed),
        "--steps",
        "96",
        "--learning-rate",
        "0.0035",
        "--vocab-size",
        "512",
        "--d-model",
        "64",
        "--num-heads",
        "8",
        "--mlp-dim",
        "160",
        "--memory-dim",
        "24",
        "--num-memory-heads",
        "4",
        "--memory-write-interval",
        "1",
        "--batch-size",
        "10",
        "--num-layers-grid",
        "16",
        "--memory-write-layers-grid",
        "1",
        "2",
        "3",
        "--distance-grid",
        "56",
        "64",
        "--num-memories-grid",
        "6",
        "--num-distractors-grid",
        "32",
        "--num-pairs-grid",
        "96",
        "--eval-distances",
        "56",
        "64",
        "72",
        "--eval-batches",
        "4",
        "--operator",
        "sensitivity",
        "--state-view",
        "memory",
        "--perturb",
        "--perturb-distractors",
        "2",
        "--min-eval-accuracy",
        "0.999",
        "--max-keep-jacobian-deviation",
        "0.05",
        "--max-keep-state-drift",
        "0.01",
        "--max-keep-state-delta-norm",
        "0.001",
        "--max-keep-leakage",
        "0.001",
        "--stop-after-failures",
        "0",
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{SRC_ROOT}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(SRC_ROOT)
    print(f"running seed {seed} -> {output_dir}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def aggregate(
    reliability_root: Path,
    baseline_summary: Path,
    report_dir: Path,
) -> None:
    seed_runs: dict[str, str] = {}
    all_rows: list[dict[str, object]] = []

    if baseline_summary.exists():
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
    for seed in args.seeds:
        run_seed(seed, reliability_root / _seed_name(seed))
    return 0


def _cmd_aggregate(args: argparse.Namespace) -> int:
    aggregate(
        reliability_root=Path(args.reliability_root),
        baseline_summary=Path(args.baseline_summary),
        report_dir=Path(args.report_dir),
    )
    return 0


def _cmd_run_and_aggregate(args: argparse.Namespace) -> int:
    reliability_root = Path(args.reliability_root)
    reliability_root.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        run_seed(seed, reliability_root / _seed_name(seed))
    aggregate(
        reliability_root=reliability_root,
        baseline_summary=Path(args.baseline_summary),
        report_dir=Path(args.report_dir),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and aggregate frontier reliability matrix.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run stress frontier for one or more seeds.")
    run_parser.add_argument("--reliability-root", default=str(DEFAULT_RELIABILITY_ROOT))
    run_parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_TARGET_SEEDS))
    run_parser.set_defaults(handler=_cmd_run)

    agg_parser = subparsers.add_parser("aggregate", help="Aggregate reliability summary from seed sweeps.")
    agg_parser.add_argument("--reliability-root", default=str(DEFAULT_RELIABILITY_ROOT))
    agg_parser.add_argument("--baseline-summary", default=str(DEFAULT_BASELINE_SUMMARY))
    agg_parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    agg_parser.set_defaults(handler=_cmd_aggregate)

    both_parser = subparsers.add_parser(
        "run-and-aggregate",
        help="Run missing seeds then aggregate reliability report artifacts.",
    )
    both_parser.add_argument("--reliability-root", default=str(DEFAULT_RELIABILITY_ROOT))
    both_parser.add_argument("--baseline-summary", default=str(DEFAULT_BASELINE_SUMMARY))
    both_parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    both_parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_TARGET_SEEDS))
    both_parser.set_defaults(handler=_cmd_run_and_aggregate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
