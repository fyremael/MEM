from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modulus_memory_channels.io import load_json, save_json  # noqa: E402


@dataclass(frozen=True)
class ReliabilityGateThresholds:
    min_overall_pass_rate: float
    min_case_pass_rate: float
    min_num_seeds: int


@dataclass(frozen=True)
class ReproGateThresholds:
    max_overall_pass_rate_delta: float
    max_case_pass_rate_delta: float


def _case_pass_map(summary: dict[str, object]) -> dict[str, float]:
    rows = summary["cases"]
    return {str(row["case"]): float(row["pass_rate"]) for row in rows}


def _eval_reliability_gate(
    *,
    summary: dict[str, object],
    thresholds: ReliabilityGateThresholds,
) -> dict[str, object]:
    case_rates = _case_pass_map(summary)
    overall = float(summary["overall_pass_rate"])
    min_case = min(case_rates.values()) if case_rates else 0.0
    num_seeds = int(summary["num_seeds"])
    checks = {
        "overall_pass_rate": {
            "value": overall,
            "threshold": thresholds.min_overall_pass_rate,
            "passed": overall >= thresholds.min_overall_pass_rate,
        },
        "min_case_pass_rate": {
            "value": min_case,
            "threshold": thresholds.min_case_pass_rate,
            "passed": min_case >= thresholds.min_case_pass_rate,
        },
        "num_seeds": {
            "value": num_seeds,
            "threshold": thresholds.min_num_seeds,
            "passed": num_seeds >= thresholds.min_num_seeds,
        },
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def _eval_repro_gate(
    *,
    reference_summary: dict[str, object],
    repro_summary: dict[str, object],
    thresholds: ReproGateThresholds,
) -> dict[str, object]:
    ref_overall = float(reference_summary["overall_pass_rate"])
    repro_overall = float(repro_summary["overall_pass_rate"])
    overall_delta = abs(repro_overall - ref_overall)

    ref_cases = _case_pass_map(reference_summary)
    repro_cases = _case_pass_map(repro_summary)
    common_cases = sorted(set(ref_cases) & set(repro_cases))
    case_deltas = {case: abs(repro_cases[case] - ref_cases[case]) for case in common_cases}
    max_case_delta = max(case_deltas.values()) if case_deltas else 0.0

    checks = {
        "overall_pass_rate_delta": {
            "value": overall_delta,
            "threshold": thresholds.max_overall_pass_rate_delta,
            "passed": overall_delta <= thresholds.max_overall_pass_rate_delta,
        },
        "max_case_pass_rate_delta": {
            "value": max_case_delta,
            "threshold": thresholds.max_case_pass_rate_delta,
            "passed": max_case_delta <= thresholds.max_case_pass_rate_delta,
        },
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "case_deltas": case_deltas,
    }


def _render_markdown(report: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append("# Phase Gate Report")
    lines.append("")
    lines.append(f"- Decision: **{report['decision']}**")
    lines.append("")

    for gate_name in ("frontier_gate", "adversarial_gate", "repro_gate"):
        gate = report[gate_name]
        lines.append(f"## {gate_name}")
        lines.append(f"- passed: `{gate['passed']}`")
        checks = gate["checks"]
        for check_name, check in checks.items():
            lines.append(
                f"- {check_name}: value={check['value']}, threshold={check['threshold']}, passed={check['passed']}"
            )
        if gate_name == "repro_gate":
            case_deltas = gate.get("case_deltas", {})
            lines.append("- case_deltas:")
            for case, delta in sorted(case_deltas.items()):
                lines.append(f"  - {case}: {delta}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build phase-gate decision report from campaign summaries.")
    parser.add_argument("--frontier-summary", required=True)
    parser.add_argument("--adversarial-summary", required=True)
    parser.add_argument("--repro-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frontier-min-overall", type=float, default=0.8)
    parser.add_argument("--frontier-min-case", type=float, default=0.67)
    parser.add_argument("--frontier-min-seeds", type=int, default=10)
    parser.add_argument("--adversarial-min-overall", type=float, default=0.8)
    parser.add_argument("--adversarial-min-case", type=float, default=0.67)
    parser.add_argument("--adversarial-min-seeds", type=int, default=5)
    parser.add_argument("--repro-max-overall-delta", type=float, default=0.05)
    parser.add_argument("--repro-max-case-delta", type=float, default=0.1)
    args = parser.parse_args(argv)

    frontier = load_json(args.frontier_summary)
    adversarial = load_json(args.adversarial_summary)
    repro = load_json(args.repro_summary)

    frontier_gate = _eval_reliability_gate(
        summary=frontier,
        thresholds=ReliabilityGateThresholds(
            min_overall_pass_rate=args.frontier_min_overall,
            min_case_pass_rate=args.frontier_min_case,
            min_num_seeds=args.frontier_min_seeds,
        ),
    )
    adversarial_gate = _eval_reliability_gate(
        summary=adversarial,
        thresholds=ReliabilityGateThresholds(
            min_overall_pass_rate=args.adversarial_min_overall,
            min_case_pass_rate=args.adversarial_min_case,
            min_num_seeds=args.adversarial_min_seeds,
        ),
    )
    repro_gate = _eval_repro_gate(
        reference_summary=frontier,
        repro_summary=repro,
        thresholds=ReproGateThresholds(
            max_overall_pass_rate_delta=args.repro_max_overall_delta,
            max_case_pass_rate_delta=args.repro_max_case_delta,
        ),
    )

    decision = "GO" if (frontier_gate["passed"] and adversarial_gate["passed"] and repro_gate["passed"]) else "NO-GO"
    report = {
        "decision": decision,
        "frontier_gate": frontier_gate,
        "adversarial_gate": adversarial_gate,
        "repro_gate": repro_gate,
        "inputs": {
            "frontier_summary": args.frontier_summary,
            "adversarial_summary": args.adversarial_summary,
            "repro_summary": args.repro_summary,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(report, output_dir / "phase_gate_report.json")
    (output_dir / "phase_gate_report.md").write_text(_render_markdown(report), encoding="ascii")
    print(f"saved phase gate report to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
