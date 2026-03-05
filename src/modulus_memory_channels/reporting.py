from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import load_json


@dataclass(frozen=True)
class CorridorThresholds:
    min_eval_accuracy: float = 0.9
    max_keep_jacobian_deviation: float = 0.05
    max_keep_state_drift: float = 1e-2
    max_keep_state_delta_norm: float = 1e-3
    max_keep_leakage: float = 1e-3


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _is_write_layer(layer_index: int, *, memory_write_interval: int, memory_write_layers: int) -> bool:
    return (
        (layer_index % memory_write_interval) == 0
        and (layer_index // memory_write_interval) < memory_write_layers
    )


def _keep_transition_layers(model_config: dict[str, Any], probe_layers: list[int]) -> list[int]:
    num_layers = int(model_config["num_layers"])
    memory_write_interval = int(model_config["memory_write_interval"])
    memory_write_layers = int(model_config.get("memory_write_layers", 1))
    keep_layers = []
    for layer in probe_layers:
        if layer >= num_layers:
            continue
        if not _is_write_layer(
            layer,
            memory_write_interval=memory_write_interval,
            memory_write_layers=memory_write_layers,
        ):
            keep_layers.append(layer)
    return keep_layers


def _criterion(value: float, threshold: float, *, mode: str) -> dict[str, Any]:
    if mode == "min":
        passed = value >= threshold
    elif mode == "max":
        passed = value <= threshold
    else:
        raise ValueError(f"Unsupported criterion mode: {mode}")
    return {"value": value, "threshold": threshold, "passed": passed, "mode": mode}


def _difficulty_score(task_config: dict[str, Any], model_config: dict[str, Any]) -> float:
    return float(
        (4 * int(task_config["num_memories"]))
        + (2 * int(task_config["distance"]))
        + int(task_config["num_distractors"])
        + (0.25 * int(task_config["num_pairs"]))
        + (0.5 * int(model_config["num_layers"]))
        + (0.5 * int(model_config.get("memory_write_layers", 1)))
    )


def build_corridor_report(
    run_dir: str | Path,
    probe_dir: str | Path,
    *,
    thresholds: CorridorThresholds | None = None,
) -> dict[str, Any]:
    thresholds = CorridorThresholds() if thresholds is None else thresholds
    run_dir = Path(run_dir)
    probe_dir = Path(probe_dir)

    run_config = load_json(run_dir / "run_config.json")
    probe_metrics = load_json(probe_dir / "probe_metrics.json")
    eval_accuracy = load_json(run_dir / "eval_accuracy.json")

    model_config = run_config["model_config"]
    task_config = run_config["task_config"]
    eval_items = sorted((int(distance), float(value)) for distance, value in eval_accuracy.items())
    eval_values = [value for _, value in eval_items]
    min_eval_accuracy = min(eval_values) if eval_values else 0.0
    mean_eval_accuracy = sum(eval_values) / len(eval_values) if eval_values else 0.0

    probe_layers = [int(layer) for layer in probe_metrics["layers"].keys()]
    keep_layers = _keep_transition_layers(model_config, sorted(probe_layers))
    keep_rows = [probe_metrics["layers"][str(layer)] for layer in keep_layers]

    if keep_rows:
        max_keep_jacobian_deviation = max(
            abs(float(row["jacobian_norms"][0]) - 1.0)
            if len(row["jacobian_norms"]) == 1
            else abs(sum(float(value) for value in row["jacobian_norms"]) / len(row["jacobian_norms"]) - 1.0)
            for row in keep_rows
        )
        max_keep_state_drift = max(
            _float_or_none(row.get("state_drift_to_next")) or 0.0 for row in keep_rows
        )
        max_keep_state_delta_norm = max(
            _float_or_none(row.get("state_delta_norm")) or 0.0 for row in keep_rows
        )
        max_keep_leakage = max(_float_or_none(row.get("leakage")) or 0.0 for row in keep_rows)
    else:
        max_keep_jacobian_deviation = 1.0
        max_keep_state_drift = 1.0
        max_keep_state_delta_norm = 1.0
        max_keep_leakage = 1.0

    criteria = {
        "min_eval_accuracy": _criterion(
            min_eval_accuracy,
            thresholds.min_eval_accuracy,
            mode="min",
        ),
        "max_keep_jacobian_deviation": _criterion(
            max_keep_jacobian_deviation,
            thresholds.max_keep_jacobian_deviation,
            mode="max",
        ),
        "max_keep_state_drift": _criterion(
            max_keep_state_drift,
            thresholds.max_keep_state_drift,
            mode="max",
        ),
        "max_keep_state_delta_norm": _criterion(
            max_keep_state_delta_norm,
            thresholds.max_keep_state_delta_norm,
            mode="max",
        ),
        "max_keep_leakage": _criterion(
            max_keep_leakage,
            thresholds.max_keep_leakage,
            mode="max",
        ),
    }
    passed_count = sum(1 for item in criteria.values() if item["passed"])
    corridor_score = passed_count / max(len(criteria), 1)

    return {
        "run_dir": str(run_dir),
        "probe_dir": str(probe_dir),
        "state_view": probe_metrics.get("state_view", "memory"),
        "task_config": task_config,
        "memory_schedule": {
            "num_layers": int(model_config["num_layers"]),
            "memory_write_interval": int(model_config["memory_write_interval"]),
            "memory_write_layers": int(model_config.get("memory_write_layers", 1)),
        },
        "difficulty_score": _difficulty_score(task_config, model_config),
        "eval": {
            "by_distance": [{"distance": distance, "accuracy": value} for distance, value in eval_items],
            "min_accuracy": min_eval_accuracy,
            "mean_accuracy": mean_eval_accuracy,
        },
        "keep_layers": {
            "indices": keep_layers,
            "count": len(keep_layers),
            "max_jacobian_deviation": max_keep_jacobian_deviation,
            "max_state_drift": max_keep_state_drift,
            "max_state_delta_norm": max_keep_state_delta_norm,
            "max_leakage": max_keep_leakage,
        },
        "criteria": criteria,
        "corridor_score": corridor_score,
        "corridor_achieved": bool(passed_count == len(criteria)),
    }


def report_text(report: dict[str, Any]) -> str:
    lines = [
        f"corridor_achieved: {report['corridor_achieved']}",
        f"corridor_score: {report['corridor_score']:.3f}",
        f"state_view: {report['state_view']}",
        f"difficulty_score: {report['difficulty_score']:.3f}",
        f"min_eval_accuracy: {report['eval']['min_accuracy']:.4f}",
        f"mean_eval_accuracy: {report['eval']['mean_accuracy']:.4f}",
        f"task: memories={report['task_config']['num_memories']} distance={report['task_config']['distance']} distractors={report['task_config']['num_distractors']} pairs={report['task_config']['num_pairs']}",
        f"keep_layers: {report['keep_layers']['indices']}",
        f"max_keep_jacobian_deviation: {report['keep_layers']['max_jacobian_deviation']:.6f}",
        f"max_keep_state_drift: {report['keep_layers']['max_state_drift']:.6f}",
        f"max_keep_state_delta_norm: {report['keep_layers']['max_state_delta_norm']:.6f}",
        f"max_keep_leakage: {report['keep_layers']['max_leakage']:.6f}",
        "criteria:",
    ]
    for name, criterion in report["criteria"].items():
        lines.append(
            f"  {name}: passed={criterion['passed']} value={criterion['value']:.6f} threshold={criterion['threshold']:.6f} mode={criterion['mode']}"
        )
    return "\n".join(lines) + "\n"


def build_sweep_summary(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        case_reports,
        key=lambda report: (
            report["corridor_achieved"],
            report["corridor_score"],
            report["difficulty_score"],
            report["eval"]["min_accuracy"],
            report["eval"]["mean_accuracy"],
        ),
        reverse=True,
    )
    successes = [report for report in ranked if report["corridor_achieved"]]
    failures = [report for report in ranked if not report["corridor_achieved"]]
    return {
        "num_cases": len(case_reports),
        "num_corridor_successes": len(successes),
        "hardest_success": successes[0] if successes else None,
        "easiest_failure": failures[-1] if failures else None,
        "cases": ranked,
    }
