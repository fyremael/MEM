"""Generate a docs context snapshot from tracked artifact JSON files."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "docs" / "context_snapshot.md"

RELIABILITY_SUMMARIES = [
    "demo_runs/corridor_reliability_v1/report/reliability_summary.json",
    "demo_runs/corridor_reliability_levelup_v1/report/reliability_summary.json",
]

PHASE_GATE_REPORTS = [
    "demo_runs/phase_gate_v1/report/phase_gate_report.json",
]

GPU_COMPARISONS = [
    "demo_runs/gpu_validation_v1/report/backend_comparison.json",
    "demo_runs/gpu_validation_v1_heavy_b64/report/backend_comparison.json",
]

STRESS_SUMMARIES = [
    "demo_runs/corridor_stress_v5/sweep_summary.json",
]

ENGINEERING_SIGNALS = {
    "docs/engineering/EXECUTIVE_OVERVIEW.md": [
        ("Current status date", r"^## Current Status \(As of ([^)]+)\)"),
        ("Decision", r"^- Decision:\s*(.+)$"),
    ],
    "docs/engineering/STATUS_DASHBOARD.md": [
        ("Architecture", r"^- Architecture:\s*(.+)$"),
        ("Technical status", r"^- Technical status:\s*(.+)$"),
        ("Overall phase-gate decision", r"^- Overall phase-gate decision:\s*(.+)$"),
    ],
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _is_tracked(repo_root: Path, path: Path) -> bool:
    if not path.exists():
        return False
    rel = path.relative_to(repo_root)
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", rel.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100.0:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _extract_gate_status(gate: Any) -> str:
    if not isinstance(gate, dict):
        return "n/a"
    passed = gate.get("passed")
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    return "n/a"


def _extract_first_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        return "n/a"
    if match.lastindex:
        return match.group(1).strip()
    return match.group(0).strip()


def _render_context(repo_root: Path) -> str:
    lines = [
        "# Context Snapshot",
        "",
        "This page is auto-generated from tracked benchmark/governance artifacts and engineering status docs.",
        "",
        "## Reliability Summaries",
        "",
        "| Source | Seeds | Trials | Passes | Overall Pass Rate |",
        "|---|---:|---:|---:|---:|",
    ]

    found_reliability = False
    for rel_path in RELIABILITY_SUMMARIES:
        artifact_path = repo_root / rel_path
        if not _is_tracked(repo_root, artifact_path):
            continue
        payload = _load_json(artifact_path)
        if payload is None:
            continue
        found_reliability = True
        lines.append(
            "| `{source}` | {seeds} | {trials} | {passes} | {pass_rate} |".format(
                source=rel_path,
                seeds=payload.get("num_seeds", "n/a"),
                trials=payload.get("overall_trials", "n/a"),
                passes=payload.get("overall_passes", "n/a"),
                pass_rate=_pct(payload.get("overall_pass_rate")),
            )
        )
    if not found_reliability:
        lines.append("| _No tracked reliability summaries found_ | n/a | n/a | n/a | n/a |")

    lines.extend(["", "## Stress Frontier Snapshot", "", "| Source | Cases | Corridor Successes | Hardest Success | Easiest Failure |", "|---|---:|---:|---|---|"])
    found_stress = False
    for rel_path in STRESS_SUMMARIES:
        artifact_path = repo_root / rel_path
        if not _is_tracked(repo_root, artifact_path):
            continue
        payload = _load_json(artifact_path)
        if payload is None:
            continue
        hardest = payload.get("hardest_success")
        easiest = payload.get("easiest_failure")
        hardest_case = hardest.get("case_name", "n/a") if isinstance(hardest, dict) else "n/a"
        easiest_case = easiest.get("case_name", "n/a") if isinstance(easiest, dict) else "n/a"
        found_stress = True
        lines.append(
            "| `{source}` | {cases} | {successes} | `{hardest}` | `{easiest}` |".format(
                source=rel_path,
                cases=payload.get("num_cases", "n/a"),
                successes=payload.get("num_corridor_successes", "n/a"),
                hardest=hardest_case,
                easiest=easiest_case,
            )
        )
    if not found_stress:
        lines.append("| _No tracked stress summaries found_ | n/a | n/a | n/a | n/a |")

    lines.extend(["", "## Phase Gate Reports", "", "| Source | Decision | Frontier | Adversarial | Repro |", "|---|---|---|---|---|"])
    found_gate = False
    for rel_path in PHASE_GATE_REPORTS:
        artifact_path = repo_root / rel_path
        if not _is_tracked(repo_root, artifact_path):
            continue
        payload = _load_json(artifact_path)
        if payload is None:
            continue
        found_gate = True
        lines.append(
            "| `{source}` | `{decision}` | `{frontier}` | `{adversarial}` | `{repro}` |".format(
                source=rel_path,
                decision=payload.get("decision", "n/a"),
                frontier=_extract_gate_status(payload.get("frontier_gate")),
                adversarial=_extract_gate_status(payload.get("adversarial_gate")),
                repro=_extract_gate_status(payload.get("repro_gate")),
            )
        )
    if not found_gate:
        lines.append("| _No tracked phase-gate reports found_ | n/a | n/a | n/a | n/a |")

    lines.extend(["", "## Backend Comparison", "", "| Source | CPU Backend | GPU Backend | GPU/CPU Throughput |", "|---|---|---|---:|"])
    found_backend = False
    for rel_path in GPU_COMPARISONS:
        artifact_path = repo_root / rel_path
        if not _is_tracked(repo_root, artifact_path):
            continue
        payload = _load_json(artifact_path)
        if payload is None:
            continue
        cpu_payload = payload.get("cpu")
        gpu_payload = payload.get("gpu")
        summary_payload = payload.get("summary")
        cpu_backend = cpu_payload.get("backend_actual", "n/a") if isinstance(cpu_payload, dict) else "n/a"
        gpu_backend = gpu_payload.get("backend_actual", "n/a") if isinstance(gpu_payload, dict) else "n/a"
        speedup = (
            summary_payload.get("throughput_speedup_gpu_over_cpu")
            if isinstance(summary_payload, dict)
            else None
        )
        speedup_text = "n/a"
        if speedup is not None:
            try:
                speedup_text = f"{float(speedup):.2f}x"
            except (TypeError, ValueError):
                speedup_text = "n/a"
        found_backend = True
        lines.append(f"| `{rel_path}` | `{cpu_backend}` | `{gpu_backend}` | {speedup_text} |")
    if not found_backend:
        lines.append("| _No tracked backend comparison files found_ | n/a | n/a | n/a |")

    lines.extend(["", "## Engineering Status Signals", "", "| Signal | Value | Source |", "|---|---|---|"])
    found_signal = False
    for rel_path, extract_rules in ENGINEERING_SIGNALS.items():
        path = repo_root / rel_path
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        for label, pattern in extract_rules:
            value = _extract_first_match(content, pattern)
            lines.append(f"| {label} | {value} | `{rel_path}` |")
            found_signal = True
    if not found_signal:
        lines.append("| _No engineering signal files found_ | n/a | n/a |")

    lines.extend(
        [
            "",
            "## How To Refresh",
            "",
            "```bash",
            "python scripts/generate_context_docs.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def generate(repo_root: Path, output_path: Path) -> str:
    output = _render_context(repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    return output


def check(repo_root: Path, output_path: Path) -> list[str]:
    expected = _render_context(repo_root)
    if not output_path.exists():
        return [f"missing generated file: {output_path.relative_to(repo_root).as_posix()}"]
    actual = output_path.read_text(encoding="utf-8")
    if actual != expected:
        return [f"stale generated file: {output_path.relative_to(repo_root).as_posix()}"]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root path.")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="Context snapshot markdown output path.")
    parser.add_argument("--check", action="store_true", help="Fail if output file is missing or stale.")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output_path).resolve()
    if args.check:
        errors = check(repo_root, output_path)
        if errors:
            print("context snapshot check FAILED")
            for err in errors:
                print(f"- {err}")
            return 1
        print("context snapshot check PASSED")
        return 0

    generate(repo_root, output_path)
    print(f"generated context snapshot at {output_path.relative_to(repo_root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
