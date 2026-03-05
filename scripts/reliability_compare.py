from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modulus_memory_channels.io import load_json, save_json  # noqa: E402
from modulus_memory_channels.visualization import write_csv, write_grouped_bar_svg  # noqa: E402


def compare_reliability(
    *,
    baseline_summary_path: Path,
    candidate_summary_path: Path,
    output_dir: Path,
    baseline_label: str,
    candidate_label: str,
) -> None:
    baseline = load_json(baseline_summary_path)
    candidate = load_json(candidate_summary_path)

    baseline_cases = {row["case"]: row for row in baseline["cases"]}
    candidate_cases = {row["case"]: row for row in candidate["cases"]}
    common_cases = sorted(set(baseline_cases) & set(candidate_cases))

    rows = []
    for case in common_cases:
        b = baseline_cases[case]
        c = candidate_cases[case]
        rows.append(
            {
                "case": case,
                f"{baseline_label}_pass_rate": b["pass_rate"],
                f"{candidate_label}_pass_rate": c["pass_rate"],
                "pass_rate_delta": c["pass_rate"] - b["pass_rate"],
                f"{baseline_label}_mean_min_eval_accuracy": b["mean_min_eval_accuracy"],
                f"{candidate_label}_mean_min_eval_accuracy": c["mean_min_eval_accuracy"],
                "mean_min_eval_accuracy_delta": c["mean_min_eval_accuracy"] - b["mean_min_eval_accuracy"],
            }
        )

    summary = {
        "baseline": {
            "label": baseline_label,
            "num_seeds": baseline["num_seeds"],
            "overall_pass_rate": baseline["overall_pass_rate"],
            "overall_trials": baseline["overall_trials"],
            "overall_passes": baseline["overall_passes"],
        },
        "candidate": {
            "label": candidate_label,
            "num_seeds": candidate["num_seeds"],
            "overall_pass_rate": candidate["overall_pass_rate"],
            "overall_trials": candidate["overall_trials"],
            "overall_passes": candidate["overall_passes"],
        },
        "delta": {
            "overall_pass_rate_delta": candidate["overall_pass_rate"] - baseline["overall_pass_rate"],
            "common_case_count": len(common_cases),
        },
        "case_rows": rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, output_dir / "comparison.json")
    write_csv(
        rows,
        output_dir / "case_passrate_comparison.csv",
        fieldnames=[
            "case",
            f"{baseline_label}_pass_rate",
            f"{candidate_label}_pass_rate",
            "pass_rate_delta",
            f"{baseline_label}_mean_min_eval_accuracy",
            f"{candidate_label}_mean_min_eval_accuracy",
            "mean_min_eval_accuracy_delta",
        ],
    )
    write_grouped_bar_svg(
        [row["case"] for row in rows],
        [row[f"{baseline_label}_pass_rate"] for row in rows],
        [row[f"{candidate_label}_pass_rate"] for row in rows],
        output_dir / "passrate_compare.svg",
        title="Case Pass Rate Comparison",
        y_label="Pass Rate",
        left_label=baseline_label,
        right_label=candidate_label,
    )
    write_grouped_bar_svg(
        [row["case"] for row in rows],
        [row[f"{baseline_label}_mean_min_eval_accuracy"] for row in rows],
        [row[f"{candidate_label}_mean_min_eval_accuracy"] for row in rows],
        output_dir / "mean_min_eval_accuracy_compare.svg",
        title="Case Mean Min Eval Accuracy Comparison",
        y_label="Accuracy",
        left_label=baseline_label,
        right_label=candidate_label,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two reliability summary files.")
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    compare_reliability(
        baseline_summary_path=Path(args.baseline_summary),
        candidate_summary_path=Path(args.candidate_summary),
        output_dir=Path(args.output_dir),
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )
    print(f"saved reliability comparison to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
