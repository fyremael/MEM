from __future__ import annotations

from pathlib import Path

from .io import save_json, save_tree
from .probe_runner import ProbeRun
from .training import AdamState, TrainResult
from .visualization import (
    write_bar_svg,
    write_csv,
    write_grouped_bar_svg,
    write_line_svg,
    write_multi_line_svg,
)


def write_training_artifacts(
    output_dir: str | Path,
    *,
    result: TrainResult,
    optimizer_state: AdamState,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_tree(result.params, output_dir / "params")
    save_tree(optimizer_state, output_dir / "optimizer_state")
    save_json(result.history, output_dir / "train_history.json")
    save_json(result.eval_accuracy_by_distance, output_dir / "eval_accuracy.json")

    history_rows = [
        {"step": index, "loss": result.history["loss"][index], "accuracy": result.history["accuracy"][index]}
        for index in range(len(result.history["loss"]))
    ]
    write_csv(history_rows, output_dir / "train_history.csv", fieldnames=["step", "loss", "accuracy"])
    write_line_svg(
        result.history["loss"],
        output_dir / "loss.svg",
        title="Training Loss",
        x_label="Step",
        y_label="Loss",
    )
    write_line_svg(
        result.history["accuracy"],
        output_dir / "accuracy.svg",
        title="Training Accuracy",
        x_label="Step",
        y_label="Accuracy",
    )

    ordered_distances = sorted(result.eval_accuracy_by_distance)
    eval_rows = [
        {"distance": distance, "accuracy": result.eval_accuracy_by_distance[distance]}
        for distance in ordered_distances
    ]
    write_csv(eval_rows, output_dir / "eval_accuracy.csv", fieldnames=["distance", "accuracy"])
    write_bar_svg(
        [str(distance) for distance in ordered_distances],
        [result.eval_accuracy_by_distance[distance] for distance in ordered_distances],
        output_dir / "eval_accuracy.svg",
        title="Eval Accuracy by Distance",
        y_label="Accuracy",
    )


def write_probe_artifacts(output_dir: str | Path, result: ProbeRun) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.save_json(output_dir / "probe_metrics.json")
    result.save_npz(output_dir / "probe_metrics.npz")

    rows = []
    for layer in result.probe_layers:
        metrics = result.layer_metrics[layer]
        rows.append(
            {
                "layer": layer,
                "operator_type": metrics.subspace.operator_type,
                "jacobian_norm_mean": float(metrics.jacobian_norms.mean()),
                "eigengap": float(metrics.gap.eigengap),
                "drift_to_next": ""
                if metrics.drift_to_next is None
                else float(metrics.drift_to_next),
                "state_drift_to_next": ""
                if metrics.state_drift_to_next is None
                else float(metrics.state_drift_to_next),
                "state_delta_norm": ""
                if metrics.state_delta_norm is None
                else float(metrics.state_delta_norm),
                "leakage": "" if metrics.leakage is None else float(metrics.leakage),
                "dk_predicted_drift": ""
                if metrics.dk_result is None
                else float(metrics.dk_result.predicted_drift),
                "dk_observed_drift": ""
                if metrics.dk_result is None
                else float(metrics.dk_result.observed_drift),
            }
        )
    write_csv(
        rows,
        output_dir / "probe_metrics.csv",
        fieldnames=[
            "layer",
            "operator_type",
            "jacobian_norm_mean",
            "eigengap",
            "drift_to_next",
            "state_drift_to_next",
            "state_delta_norm",
            "leakage",
            "dk_predicted_drift",
            "dk_observed_drift",
        ],
    )

    layer_labels = [str(layer) for layer in result.probe_layers]
    jacobian_values = [float(result.layer_metrics[layer].jacobian_norms.mean()) for layer in result.probe_layers]
    gap_values = [float(result.layer_metrics[layer].gap.eigengap) for layer in result.probe_layers]
    write_bar_svg(
        layer_labels,
        jacobian_values,
        output_dir / "jacobian_norms.svg",
        title="Jacobian Norm by Layer",
        y_label="Norm",
    )
    write_bar_svg(
        layer_labels,
        gap_values,
        output_dir / "eigengaps.svg",
        title="Eigengap by Layer",
        y_label="Gap",
    )


def write_compare_artifacts(
    output_dir: str | Path,
    *,
    left_label: str,
    right_label: str,
    comparison: dict[str, object],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(comparison, output_dir / "comparison.json")

    if "training" in comparison:
        training = comparison["training"]
        history_rows = training["history_rows"]
        eval_rows = training["eval_rows"]
        write_csv(
            history_rows,
            output_dir / "training_history_compare.csv",
            fieldnames=["step", f"{left_label}_loss", f"{right_label}_loss", f"{left_label}_accuracy", f"{right_label}_accuracy"],
        )
        write_csv(
            eval_rows,
            output_dir / "eval_accuracy_compare.csv",
            fieldnames=["distance", left_label, right_label, "delta"],
        )
        write_multi_line_svg(
            [
                (left_label, [row[f"{left_label}_loss"] for row in history_rows], "#0b6e4f"),
                (right_label, [row[f"{right_label}_loss"] for row in history_rows], "#c84c09"),
            ],
            output_dir / "loss_compare.svg",
            title="Training Loss Compare",
            x_label="Step",
            y_label="Loss",
        )
        write_multi_line_svg(
            [
                (left_label, [row[f"{left_label}_accuracy"] for row in history_rows], "#0b6e4f"),
                (right_label, [row[f"{right_label}_accuracy"] for row in history_rows], "#c84c09"),
            ],
            output_dir / "accuracy_compare.svg",
            title="Training Accuracy Compare",
            x_label="Step",
            y_label="Accuracy",
        )
        write_grouped_bar_svg(
            [str(row["distance"]) for row in eval_rows],
            [row[left_label] for row in eval_rows],
            [row[right_label] for row in eval_rows],
            output_dir / "eval_accuracy_compare.svg",
            title="Eval Accuracy Compare",
            y_label="Accuracy",
            left_label=left_label,
            right_label=right_label,
        )

    if "probe" in comparison:
        probe = comparison["probe"]
        probe_rows = probe["rows"]
        write_csv(
            probe_rows,
            output_dir / "probe_compare.csv",
            fieldnames=[
                "layer",
                f"{left_label}_jacobian_norm_mean",
                f"{right_label}_jacobian_norm_mean",
                f"{left_label}_eigengap",
                f"{right_label}_eigengap",
            ],
        )
        write_grouped_bar_svg(
            [str(row["layer"]) for row in probe_rows],
            [row[f"{left_label}_jacobian_norm_mean"] for row in probe_rows],
            [row[f"{right_label}_jacobian_norm_mean"] for row in probe_rows],
            output_dir / "probe_jacobian_compare.svg",
            title="Probe Jacobian Compare",
            y_label="Norm",
            left_label=left_label,
            right_label=right_label,
        )
        write_grouped_bar_svg(
            [str(row["layer"]) for row in probe_rows],
            [row[f"{left_label}_eigengap"] for row in probe_rows],
            [row[f"{right_label}_eigengap"] for row in probe_rows],
            output_dir / "probe_eigengap_compare.svg",
            title="Probe Eigengap Compare",
            y_label="Gap",
            left_label=left_label,
            right_label=right_label,
        )


def write_report_artifacts(output_dir: str | Path, report: dict[str, object]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from .reporting import report_text

    save_json(report, output_dir / "report.json")
    (output_dir / "report.txt").write_text(report_text(report), encoding="ascii")

    rows = []
    for name, criterion in report["criteria"].items():
        rows.append(
            {
                "criterion": name,
                "passed": criterion["passed"],
                "value": criterion["value"],
                "threshold": criterion["threshold"],
                "mode": criterion["mode"],
            }
        )
    write_csv(
        rows,
        output_dir / "criteria.csv",
        fieldnames=["criterion", "passed", "value", "threshold", "mode"],
    )
    write_bar_svg(
        [row["criterion"] for row in rows],
        [1.0 if row["passed"] else 0.0 for row in rows],
        output_dir / "criteria_pass.svg",
        title="Corridor Criteria Pass",
        y_label="Pass",
    )


def write_sweep_artifacts(output_dir: str | Path, summary: dict[str, object]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, output_dir / "sweep_summary.json")
    hardest_success = summary.get("hardest_success")
    easiest_failure = summary.get("easiest_failure")
    lines = [
        f"num_cases: {summary['num_cases']}",
        f"num_corridor_successes: {summary['num_corridor_successes']}",
        f"hardest_success: {'' if hardest_success is None else hardest_success.get('case_name', Path(hardest_success['run_dir']).parent.name)}",
        f"easiest_failure: {'' if easiest_failure is None else easiest_failure.get('case_name', Path(easiest_failure['run_dir']).parent.name)}",
    ]
    (output_dir / "sweep_summary.txt").write_text("\n".join(lines) + "\n", encoding="ascii")

    rows = []
    for report in summary["cases"]:
        case_name = report.get("case_name", Path(report["run_dir"]).parent.name)
        rows.append(
            {
                "case": case_name,
                "corridor_achieved": report["corridor_achieved"],
                "corridor_score": report["corridor_score"],
                "difficulty_score": report["difficulty_score"],
                "min_eval_accuracy": report["eval"]["min_accuracy"],
                "mean_eval_accuracy": report["eval"]["mean_accuracy"],
                "max_keep_jacobian_deviation": report["keep_layers"]["max_jacobian_deviation"],
                "max_keep_state_drift": report["keep_layers"]["max_state_drift"],
                "max_keep_state_delta_norm": report["keep_layers"]["max_state_delta_norm"],
                "max_keep_leakage": report["keep_layers"]["max_leakage"],
            }
        )
    write_csv(
        rows,
        output_dir / "sweep_summary.csv",
        fieldnames=[
            "case",
            "corridor_achieved",
            "corridor_score",
            "difficulty_score",
            "min_eval_accuracy",
            "mean_eval_accuracy",
            "max_keep_jacobian_deviation",
            "max_keep_state_drift",
            "max_keep_state_delta_norm",
            "max_keep_leakage",
        ],
    )
    write_bar_svg(
        [row["case"] for row in rows],
        [row["corridor_score"] for row in rows],
        output_dir / "corridor_scores.svg",
        title="Corridor Score by Case",
        y_label="Score",
    )
    write_bar_svg(
        [row["case"] for row in rows],
        [row["min_eval_accuracy"] for row in rows],
        output_dir / "min_eval_accuracy.svg",
        title="Min Eval Accuracy by Case",
        y_label="Accuracy",
    )
