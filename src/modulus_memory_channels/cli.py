from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jnp

from .artifacts import (
    write_compare_artifacts,
    write_probe_artifacts,
    write_report_artifacts,
    write_sweep_artifacts,
    write_training_artifacts,
)
from .bench_write_keep_read import WriteKeepReadConfig, generate_write_keep_read_batch
from .config import MemoryModelConfig, ProbeConfig
from .io import load_json, load_tree, save_json, save_tree
from .model import MemoryChannelsModel
from .probe_runner import run_probe_suite
from .reporting import CorridorThresholds, build_corridor_report, build_sweep_summary
from .training import AdamState, TrainConfig, train_write_keep_read_model


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-dim", type=int, default=64)
    parser.add_argument("--memory-dim", type=int, default=8)
    parser.add_argument("--num-memory-heads", type=int, default=2)
    parser.add_argument("--memory-write-interval", type=int, default=2)
    parser.add_argument("--memory-write-layers", type=int, default=1)


def _model_config_from_args(args: argparse.Namespace) -> MemoryModelConfig:
    return MemoryModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        memory_dim=args.memory_dim,
        num_memory_heads=args.num_memory_heads,
        memory_write_interval=args.memory_write_interval,
        memory_write_layers=args.memory_write_layers,
    )


def _task_config_from_args(args: argparse.Namespace) -> WriteKeepReadConfig:
    return WriteKeepReadConfig(
        vocab_size=args.vocab_size,
        num_pairs=args.num_pairs,
        batch_size=args.batch_size,
        distance=args.distance,
        num_distractors=args.num_distractors,
        num_memories=args.num_memories,
    )


def _add_report_threshold_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-eval-accuracy", type=float, default=0.9)
    parser.add_argument("--max-keep-jacobian-deviation", type=float, default=0.05)
    parser.add_argument("--max-keep-state-drift", type=float, default=1e-2)
    parser.add_argument("--max-keep-state-delta-norm", type=float, default=1e-3)
    parser.add_argument("--max-keep-leakage", type=float, default=1e-3)


def _thresholds_from_args(args: argparse.Namespace) -> CorridorThresholds:
    return CorridorThresholds(
        min_eval_accuracy=args.min_eval_accuracy,
        max_keep_jacobian_deviation=args.max_keep_jacobian_deviation,
        max_keep_state_drift=args.max_keep_state_drift,
        max_keep_state_delta_norm=args.max_keep_state_delta_norm,
        max_keep_leakage=args.max_keep_leakage,
    )


def _write_run_metadata(
    output_dir: Path,
    *,
    model_config: MemoryModelConfig,
    task_config: WriteKeepReadConfig | None = None,
    train_config: TrainConfig | None = None,
) -> None:
    payload = {
        "model_config": {
            "vocab_size": model_config.vocab_size,
            "d_model": model_config.d_model,
            "num_layers": model_config.num_layers,
            "num_heads": model_config.num_heads,
            "mlp_dim": model_config.mlp_dim,
            "memory_dim": model_config.memory_dim,
            "num_memory_heads": model_config.num_memory_heads,
            "memory_write_interval": model_config.memory_write_interval,
            "memory_write_layers": model_config.memory_write_layers,
            "keep_write_scale": model_config.keep_write_scale,
            "keep_read_scale": model_config.keep_read_scale,
            "keep_token_mix_scale": model_config.keep_token_mix_scale,
            "alpha_init": model_config.alpha_init,
            "beta_init": model_config.beta_init,
            "memory_write_init": model_config.memory_write_init,
            "memory_keep_init": model_config.memory_keep_init,
            "memory_read_init": model_config.memory_read_init,
            "rms_norm_eps": model_config.rms_norm_eps,
            "max_seq_len": model_config.max_seq_len,
            "causal": model_config.causal,
        }
    }
    if task_config is not None:
        payload["task_config"] = {
            "vocab_size": task_config.vocab_size,
            "num_pairs": task_config.num_pairs,
            "batch_size": task_config.batch_size,
            "distance": task_config.distance,
            "num_distractors": task_config.num_distractors,
            "num_memories": task_config.num_memories,
            "pad_token_id": task_config.pad_token_id,
            "bos_token_id": task_config.bos_token_id,
            "query_token_id": task_config.query_token_id,
            "distractor_token_low": task_config.distractor_token_low,
        }
    if train_config is not None:
        payload["train_config"] = {
            "steps": train_config.steps,
            "learning_rate": train_config.learning_rate,
            "beta1": train_config.beta1,
            "beta2": train_config.beta2,
            "eps": train_config.eps,
            "weight_decay": train_config.weight_decay,
            "eval_distances": list(train_config.eval_distances),
            "eval_batches": train_config.eval_batches,
        }
    save_json(payload, output_dir / "run_config.json")


def _load_model_from_run_dir(run_dir: Path) -> tuple[MemoryChannelsModel, dict]:
    config_data = load_json(run_dir / "run_config.json")
    model_config = MemoryModelConfig(**config_data["model_config"])
    model = MemoryChannelsModel(model_config)
    params = load_tree(run_dir / "params")
    return model, params


def _make_perturbed_tokens(
    tokens,
    task_config: WriteKeepReadConfig,
    *,
    seed: int,
    perturb_distractors: int,
):
    perturbed_tokens = tokens
    middle_start = 1 + (2 * task_config.num_memories)
    middle_len = max(task_config.distance + task_config.num_distractors, 1)
    position_key = jax.random.PRNGKey(seed + 1)
    value_key = jax.random.PRNGKey(seed + 2)
    positions = jax.random.randint(
        position_key,
        (task_config.batch_size, perturb_distractors),
        0,
        middle_len,
    )
    positions = positions + middle_start
    replacement_tokens = jax.random.randint(
        value_key,
        (task_config.batch_size, perturb_distractors),
        task_config.distractor_token_low + (2 * task_config.num_pairs),
        task_config.vocab_size,
    )
    row_indices = jnp.arange(task_config.batch_size)[:, None]
    return perturbed_tokens.at[row_indices, positions].set(replacement_tokens)


def _ambient_dim_for_state_view(model_config: MemoryModelConfig, state_view: str) -> int:
    if state_view == "memory":
        return model_config.memory_dim
    if state_view == "token":
        return model_config.d_model
    return model_config.d_model + model_config.memory_dim


def _resolved_k_eigs(requested_k: int, model_config: MemoryModelConfig, state_view: str) -> int:
    ambient_dim = _ambient_dim_for_state_view(model_config, state_view)
    return min(requested_k, max(ambient_dim - 1, 1))


def _train_command(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_dir:
        resume_dir = Path(args.resume_dir)
        config_data = load_json(resume_dir / "run_config.json")
        model_config = MemoryModelConfig(**config_data["model_config"])
        task_config = WriteKeepReadConfig(**config_data["task_config"])
        model = MemoryChannelsModel(model_config)
        params = load_tree(resume_dir / "params")
        optimizer_state_data = load_tree(resume_dir / "optimizer_state")
        optimizer_state = AdamState(
            step=int(optimizer_state_data["step"]),
            first_moment=optimizer_state_data["first_moment"],
            second_moment=optimizer_state_data["second_moment"],
        )
        history = load_json(resume_dir / "train_history.json")
    else:
        model_config = _model_config_from_args(args)
        task_config = _task_config_from_args(args)
        model = MemoryChannelsModel(model_config)
        init_key = jax.random.PRNGKey(args.seed)
        params = model.init(init_key)
        optimizer_state = None
        history = None

    train_config = TrainConfig(
        steps=args.steps,
        learning_rate=args.learning_rate,
        eval_distances=tuple(args.eval_distances),
        eval_batches=args.eval_batches,
    )

    def checkpoint_callback(step: int, current_params, current_optimizer_state, current_history) -> None:
        if args.checkpoint_every < 1 or step % args.checkpoint_every != 0:
            return
        checkpoint_dir = _checkpoint_snapshot(
            output_dir,
            step=step,
            params=current_params,
            optimizer_state=current_optimizer_state,
            history=current_history,
            eval_accuracy_by_distance={},
        )
        _write_run_metadata(
            checkpoint_dir,
            model_config=model_config,
            task_config=task_config,
            train_config=train_config,
        )

    train_key = jax.random.PRNGKey(args.seed + 1)
    result = train_write_keep_read_model(
        model,
        params,
        task_config,
        train_config,
        key=train_key,
        optimizer_state=optimizer_state,
        history=history,
        step_callback=checkpoint_callback,
    )

    write_training_artifacts(
        output_dir,
        result=result,
        optimizer_state=result.optimizer_state,
    )
    _write_run_metadata(
        output_dir,
        model_config=model_config,
        task_config=task_config,
        train_config=train_config,
    )
    print(f"saved training artifacts to {output_dir}")
    return 0


def _checkpoint_snapshot(
    output_dir: Path,
    *,
    step: int,
    params,
    optimizer_state,
    history,
    eval_accuracy_by_distance: dict[int, float],
) -> Path:
    checkpoint_dir = output_dir / "checkpoints" / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    from .training import TrainResult

    write_training_artifacts(
        checkpoint_dir,
        result=TrainResult(
            params=params,
            history={"loss": list(history["loss"]), "accuracy": list(history["accuracy"])},
            eval_accuracy_by_distance=eval_accuracy_by_distance,
            optimizer_state=optimizer_state,
        ),
        optimizer_state=optimizer_state,
    )
    return checkpoint_dir


def _probe_command(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, params = _load_model_from_run_dir(run_dir)
    config_data = load_json(run_dir / "run_config.json")
    task_config = WriteKeepReadConfig(**config_data["task_config"])
    task_config = WriteKeepReadConfig(
        vocab_size=task_config.vocab_size,
        num_pairs=task_config.num_pairs,
        batch_size=args.batch_size or task_config.batch_size,
        distance=args.distance if args.distance is not None else task_config.distance,
        num_distractors=args.num_distractors
        if args.num_distractors is not None
        else task_config.num_distractors,
        num_memories=task_config.num_memories,
        pad_token_id=task_config.pad_token_id,
        bos_token_id=task_config.bos_token_id,
        query_token_id=task_config.query_token_id,
        distractor_token_low=task_config.distractor_token_low,
    )

    batch_key = jax.random.PRNGKey(args.seed)
    batch = generate_write_keep_read_batch(batch_key, task_config)
    perturbed_tokens = None
    if args.perturb:
        perturbed_tokens = _make_perturbed_tokens(
            batch.tokens,
            task_config,
            seed=args.seed,
            perturb_distractors=args.perturb_distractors,
        )

    probe_config = ProbeConfig(
        probe_layers=tuple(args.probe_layers),
        k_eigs=_resolved_k_eigs(args.k_eigs, model.config, args.state_view),
        power_iters=args.power_iters,
        subspace_iters=args.subspace_iters,
        subspace_oversample=args.subspace_oversample,
        seed=args.seed,
        operator=args.operator,
        state_view=args.state_view,
    )
    result = run_probe_suite(
        model,
        params,
        batch.tokens,
        probe_config,
        perturbed_tokens=perturbed_tokens,
    )
    write_probe_artifacts(output_dir, result)
    save_json(
        {
            "probe_layers": list(args.probe_layers),
            "operator": args.operator,
            "k_eigs": args.k_eigs,
            "power_iters": args.power_iters,
            "subspace_iters": args.subspace_iters,
            "subspace_oversample": args.subspace_oversample,
            "seed": args.seed,
            "state_view": args.state_view,
        },
        output_dir / "probe_config.json",
    )
    print(f"saved probe artifacts to {output_dir}")
    return 0


def _report_command(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_corridor_report(
        args.run_dir,
        args.probe_dir,
        thresholds=_thresholds_from_args(args),
    )
    write_report_artifacts(output_dir, report)
    print(f"saved corridor report to {output_dir}")
    return 0


def _run_case_pipeline(
    *,
    case_dir: Path,
    case_name: str,
    case_seed: int,
    model_config: MemoryModelConfig,
    task_config: WriteKeepReadConfig,
    train_config: TrainConfig,
    thresholds: CorridorThresholds,
    operator: str,
    state_view: str,
    k_eigs: int,
    power_iters: int,
    subspace_iters: int,
    subspace_oversample: int,
    perturb: bool,
    perturb_distractors: int,
) -> dict[str, object]:
    train_dir = case_dir / "train"
    probe_dir = case_dir / "probe"
    report_dir = case_dir / "report"

    model = MemoryChannelsModel(model_config)
    params = model.init(jax.random.PRNGKey(case_seed))
    result = train_write_keep_read_model(
        model,
        params,
        task_config,
        train_config,
        key=jax.random.PRNGKey(case_seed + 1),
    )
    write_training_artifacts(
        train_dir,
        result=result,
        optimizer_state=result.optimizer_state,
    )
    _write_run_metadata(
        train_dir,
        model_config=model_config,
        task_config=task_config,
        train_config=train_config,
    )

    batch = generate_write_keep_read_batch(jax.random.PRNGKey(case_seed + 2), task_config)
    perturbed_tokens = None
    if perturb:
        perturbed_tokens = _make_perturbed_tokens(
            batch.tokens,
            task_config,
            seed=case_seed + 2,
            perturb_distractors=perturb_distractors,
        )
    probe_config = ProbeConfig(
        probe_layers=tuple(range(model_config.num_layers + 1)),
        k_eigs=_resolved_k_eigs(k_eigs, model_config, state_view),
        power_iters=power_iters,
        subspace_iters=subspace_iters,
        subspace_oversample=subspace_oversample,
        seed=case_seed + 3,
        operator=operator,
        state_view=state_view,
    )
    probe_result = run_probe_suite(
        model,
        result.params,
        batch.tokens,
        probe_config,
        perturbed_tokens=perturbed_tokens,
    )
    write_probe_artifacts(probe_dir, probe_result)
    save_json(
        {
            "probe_layers": list(probe_config.probe_layers),
            "operator": probe_config.operator,
            "k_eigs": probe_config.k_eigs,
            "power_iters": probe_config.power_iters,
            "subspace_iters": probe_config.subspace_iters,
            "subspace_oversample": probe_config.subspace_oversample,
            "seed": probe_config.seed,
            "state_view": probe_config.state_view,
        },
        probe_dir / "probe_config.json",
    )

    report = build_corridor_report(
        train_dir,
        probe_dir,
        thresholds=thresholds,
    )
    report["case_name"] = case_name
    write_report_artifacts(report_dir, report)
    return report


def _load_existing_case_report(case_dir: Path) -> dict[str, object] | None:
    report_path = case_dir / "report" / "report.json"
    if not report_path.exists():
        return None
    return load_json(report_path)


def _sweep_command(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = _thresholds_from_args(args)
    case_reports = []
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    combinations = list(
        product(
            args.num_layers_grid,
            args.memory_write_layers_grid,
            args.distance_grid,
            args.num_memories_grid,
            args.num_distractors_grid,
            args.num_pairs_grid,
        )
    )
    for case_index, (
        num_layers,
        memory_write_layers,
        distance,
        num_memories,
        num_distractors,
        num_pairs,
    ) in enumerate(combinations):
        case_name = (
            f"layers{num_layers}_writes{memory_write_layers}"
            f"_dist{distance}_mem{num_memories}"
            f"_noise{num_distractors}_pairs{num_pairs}"
        )
        case_dir = cases_dir / case_name
        if args.resume:
            existing_report = _load_existing_case_report(case_dir)
            if existing_report is not None:
                case_reports.append(existing_report)
                continue

        model_config = MemoryModelConfig(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=num_layers,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            memory_dim=args.memory_dim,
            num_memory_heads=args.num_memory_heads,
            memory_write_interval=args.memory_write_interval,
            memory_write_layers=memory_write_layers,
        )
        task_config = WriteKeepReadConfig(
            vocab_size=args.vocab_size,
            num_pairs=num_pairs,
            batch_size=args.batch_size,
            distance=distance,
            num_distractors=num_distractors,
            num_memories=num_memories,
        )
        train_config = TrainConfig(
            steps=args.steps,
            learning_rate=args.learning_rate,
            eval_distances=tuple(args.eval_distances),
            eval_batches=args.eval_batches,
        )
        case_seed = args.seed + (case_index * 17)
        report = _run_case_pipeline(
            case_dir=case_dir,
            case_name=case_name,
            case_seed=case_seed,
            model_config=model_config,
            task_config=task_config,
            train_config=train_config,
            thresholds=thresholds,
            operator=args.operator,
            state_view=args.state_view,
            k_eigs=args.k_eigs,
            power_iters=args.power_iters,
            subspace_iters=args.subspace_iters,
            subspace_oversample=args.subspace_oversample,
            perturb=args.perturb,
            perturb_distractors=args.perturb_distractors,
        )
        case_reports.append(report)
        write_sweep_artifacts(output_dir, build_sweep_summary(case_reports))

    summary = build_sweep_summary(case_reports)
    write_sweep_artifacts(output_dir, summary)
    print(f"saved sweep artifacts to {output_dir}")
    return 0


def _stress_command(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = _thresholds_from_args(args)
    case_reports = []
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    combinations = list(
        product(
            args.num_layers_grid,
            args.memory_write_layers_grid,
            args.distance_grid,
            args.num_memories_grid,
            args.num_distractors_grid,
            args.num_pairs_grid,
        )
    )
    combinations.sort(
        key=lambda item: (
            item[3],
            item[2],
            item[4],
            item[5],
            item[0],
            item[1],
        ),
        reverse=True,
    )

    for case_index, (
        num_layers,
        memory_write_layers,
        distance,
        num_memories,
        num_distractors,
        num_pairs,
    ) in enumerate(combinations):
        case_name = (
            f"stress_layers{num_layers}_writes{memory_write_layers}"
            f"_dist{distance}_mem{num_memories}"
            f"_noise{num_distractors}_pairs{num_pairs}"
        )
        case_dir = cases_dir / case_name
        if args.resume:
            existing_report = _load_existing_case_report(case_dir)
            if existing_report is not None:
                case_reports.append(existing_report)
                if args.stop_after_failures > 0:
                    recent = case_reports[-args.stop_after_failures :]
                    if len(recent) == args.stop_after_failures and all(
                        not case["corridor_achieved"] for case in recent
                    ):
                        break
                continue
        model_config = MemoryModelConfig(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=num_layers,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            memory_dim=args.memory_dim,
            num_memory_heads=args.num_memory_heads,
            memory_write_interval=args.memory_write_interval,
            memory_write_layers=memory_write_layers,
        )
        task_config = WriteKeepReadConfig(
            vocab_size=args.vocab_size,
            num_pairs=num_pairs,
            batch_size=args.batch_size,
            distance=distance,
            num_distractors=num_distractors,
            num_memories=num_memories,
        )
        train_config = TrainConfig(
            steps=args.steps,
            learning_rate=args.learning_rate,
            eval_distances=tuple(args.eval_distances),
            eval_batches=args.eval_batches,
        )
        case_seed = args.seed + (case_index * 29)
        report = _run_case_pipeline(
            case_dir=case_dir,
            case_name=case_name,
            case_seed=case_seed,
            model_config=model_config,
            task_config=task_config,
            train_config=train_config,
            thresholds=thresholds,
            operator=args.operator,
            state_view=args.state_view,
            k_eigs=args.k_eigs,
            power_iters=args.power_iters,
            subspace_iters=args.subspace_iters,
            subspace_oversample=args.subspace_oversample,
            perturb=args.perturb,
            perturb_distractors=args.perturb_distractors,
        )
        case_reports.append(report)
        summary = build_sweep_summary(case_reports)
        summary["search_mode"] = "stress"
        summary["stop_after_failures"] = args.stop_after_failures
        write_sweep_artifacts(output_dir, summary)
        if args.stop_after_failures > 0:
            recent = case_reports[-args.stop_after_failures :]
            if len(recent) == args.stop_after_failures and all(
                not case["corridor_achieved"] for case in recent
            ):
                break

    summary = build_sweep_summary(case_reports)
    summary["search_mode"] = "stress"
    summary["stop_after_failures"] = args.stop_after_failures
    write_sweep_artifacts(output_dir, summary)
    print(f"saved stress artifacts to {output_dir}")
    return 0


def _compare_training(left_dir: Path, right_dir: Path, left_label: str, right_label: str) -> dict[str, object] | None:
    left_history_path = left_dir / "train_history.json"
    right_history_path = right_dir / "train_history.json"
    left_eval_path = left_dir / "eval_accuracy.json"
    right_eval_path = right_dir / "eval_accuracy.json"
    if not (left_history_path.exists() and right_history_path.exists() and left_eval_path.exists() and right_eval_path.exists()):
        return None

    left_history = load_json(left_history_path)
    right_history = load_json(right_history_path)
    max_steps = max(len(left_history["loss"]), len(right_history["loss"]))
    history_rows = []
    for step in range(max_steps):
        left_loss = left_history["loss"][step] if step < len(left_history["loss"]) else left_history["loss"][-1]
        right_loss = right_history["loss"][step] if step < len(right_history["loss"]) else right_history["loss"][-1]
        left_acc = left_history["accuracy"][step] if step < len(left_history["accuracy"]) else left_history["accuracy"][-1]
        right_acc = right_history["accuracy"][step] if step < len(right_history["accuracy"]) else right_history["accuracy"][-1]
        history_rows.append(
            {
                "step": step,
                f"{left_label}_loss": left_loss,
                f"{right_label}_loss": right_loss,
                f"{left_label}_accuracy": left_acc,
                f"{right_label}_accuracy": right_acc,
            }
        )

    left_eval = {int(key): value for key, value in load_json(left_eval_path).items()}
    right_eval = {int(key): value for key, value in load_json(right_eval_path).items()}
    distances = sorted(set(left_eval) | set(right_eval))
    eval_rows = []
    for distance in distances:
        left_value = left_eval.get(distance, 0.0)
        right_value = right_eval.get(distance, 0.0)
        eval_rows.append(
            {
                "distance": distance,
                left_label: left_value,
                right_label: right_value,
                "delta": right_value - left_value,
            }
        )
    return {"history_rows": history_rows, "eval_rows": eval_rows}


def _compare_probe(left_dir: Path, right_dir: Path, left_label: str, right_label: str) -> dict[str, object] | None:
    left_probe_path = left_dir / "probe_metrics.json"
    right_probe_path = right_dir / "probe_metrics.json"
    if not (left_probe_path.exists() and right_probe_path.exists()):
        return None

    left_probe = load_json(left_probe_path)["layers"]
    right_probe = load_json(right_probe_path)["layers"]
    layers = sorted(set(left_probe) & set(right_probe), key=int)
    rows = []
    for layer in layers:
        rows.append(
            {
                "layer": int(layer),
                f"{left_label}_jacobian_norm_mean": sum(left_probe[layer]["jacobian_norms"]) / len(left_probe[layer]["jacobian_norms"]),
                f"{right_label}_jacobian_norm_mean": sum(right_probe[layer]["jacobian_norms"]) / len(right_probe[layer]["jacobian_norms"]),
                f"{left_label}_eigengap": left_probe[layer]["eigengap"],
                f"{right_label}_eigengap": right_probe[layer]["eigengap"],
            }
        )
    return {"rows": rows}


def _compare_command(args: argparse.Namespace) -> int:
    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison: dict[str, object] = {}
    training = _compare_training(left_dir, right_dir, args.left_label, args.right_label)
    if training is not None:
        comparison["training"] = training
    probe = _compare_probe(left_dir, right_dir, args.left_label, args.right_label)
    if probe is not None:
        comparison["probe"] = probe
    if not comparison:
        raise FileNotFoundError("No comparable training or probe artifacts found in the provided directories.")

    write_compare_artifacts(
        output_dir,
        left_label=args.left_label,
        right_label=args.right_label,
        comparison=comparison,
    )
    print(f"saved comparison artifacts to {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modulus-memory-channels")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train on the synthetic write-keep-read task.")
    _add_model_arguments(train_parser)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--resume-dir")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--steps", type=int, default=200)
    train_parser.add_argument("--learning-rate", type=float, default=3e-3)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--distance", type=int, default=8)
    train_parser.add_argument("--num-distractors", type=int, default=8)
    train_parser.add_argument("--num-pairs", type=int, default=16)
    train_parser.add_argument("--num-memories", type=int, default=1)
    train_parser.add_argument("--eval-distances", type=int, nargs="+", default=[4, 8, 16, 32])
    train_parser.add_argument("--eval-batches", type=int, default=8)
    train_parser.add_argument("--checkpoint-every", type=int, default=0)
    train_parser.set_defaults(handler=_train_command)

    probe_parser = subparsers.add_parser("probe", help="Run probe diagnostics from a saved training run.")
    probe_parser.add_argument("--run-dir", required=True)
    probe_parser.add_argument("--output-dir", required=True)
    probe_parser.add_argument("--seed", type=int, default=0)
    probe_parser.add_argument("--batch-size", type=int)
    probe_parser.add_argument("--distance", type=int)
    probe_parser.add_argument("--num-distractors", type=int)
    probe_parser.add_argument("--probe-layers", type=int, nargs="+", required=True)
    probe_parser.add_argument("--operator", choices=["covariance", "sensitivity"], default="covariance")
    probe_parser.add_argument("--state-view", choices=["full", "token", "memory"], default="memory")
    probe_parser.add_argument("--k-eigs", type=int, default=4)
    probe_parser.add_argument("--power-iters", type=int, default=8)
    probe_parser.add_argument("--subspace-iters", type=int, default=8)
    probe_parser.add_argument("--subspace-oversample", type=int, default=4)
    probe_parser.add_argument("--perturb", action="store_true")
    probe_parser.add_argument("--perturb-distractors", type=int, default=2)
    probe_parser.set_defaults(handler=_probe_command)

    compare_parser = subparsers.add_parser("compare", help="Compare two training or probe artifact directories.")
    compare_parser.add_argument("--left-dir", required=True)
    compare_parser.add_argument("--right-dir", required=True)
    compare_parser.add_argument("--output-dir", required=True)
    compare_parser.add_argument("--left-label", default="left")
    compare_parser.add_argument("--right-label", default="right")
    compare_parser.set_defaults(handler=_compare_command)

    report_parser = subparsers.add_parser("report", help="Score a saved run against corridor criteria.")
    report_parser.add_argument("--run-dir", required=True)
    report_parser.add_argument("--probe-dir", required=True)
    report_parser.add_argument("--output-dir", required=True)
    _add_report_threshold_arguments(report_parser)
    report_parser.set_defaults(handler=_report_command)

    sweep_parser = subparsers.add_parser("sweep", help="Run train/probe/report ablations over a small grid.")
    _add_model_arguments(sweep_parser)
    sweep_parser.add_argument("--output-dir", required=True)
    sweep_parser.add_argument("--resume", action="store_true")
    sweep_parser.add_argument("--seed", type=int, default=0)
    sweep_parser.add_argument("--steps", type=int, default=32)
    sweep_parser.add_argument("--learning-rate", type=float, default=3e-3)
    sweep_parser.add_argument("--batch-size", type=int, default=16)
    sweep_parser.add_argument("--num-distractors", type=int, default=2)
    sweep_parser.add_argument("--num-pairs", type=int, default=16)
    sweep_parser.add_argument("--num-memories", type=int, default=1)
    sweep_parser.add_argument("--eval-distances", type=int, nargs="+", default=[4, 8, 12])
    sweep_parser.add_argument("--eval-batches", type=int, default=4)
    sweep_parser.add_argument("--num-layers-grid", type=int, nargs="+", default=[2, 4])
    sweep_parser.add_argument("--memory-write-layers-grid", type=int, nargs="+", default=[1, 2])
    sweep_parser.add_argument("--distance-grid", type=int, nargs="+", default=[4, 8])
    sweep_parser.add_argument("--num-memories-grid", type=int, nargs="+", default=[1, 2])
    sweep_parser.add_argument("--num-distractors-grid", type=int, nargs="+", default=[2, 6])
    sweep_parser.add_argument("--num-pairs-grid", type=int, nargs="+", default=[16, 24])
    sweep_parser.add_argument("--operator", choices=["covariance", "sensitivity"], default="sensitivity")
    sweep_parser.add_argument("--state-view", choices=["full", "token", "memory"], default="memory")
    sweep_parser.add_argument("--k-eigs", type=int, default=4)
    sweep_parser.add_argument("--power-iters", type=int, default=8)
    sweep_parser.add_argument("--subspace-iters", type=int, default=8)
    sweep_parser.add_argument("--subspace-oversample", type=int, default=4)
    sweep_parser.add_argument("--perturb", action="store_true")
    sweep_parser.add_argument("--perturb-distractors", type=int, default=2)
    _add_report_threshold_arguments(sweep_parser)
    sweep_parser.set_defaults(handler=_sweep_command)

    stress_parser = subparsers.add_parser("stress", help="Search for the hardest corridor-preserving task regime.")
    _add_model_arguments(stress_parser)
    stress_parser.add_argument("--output-dir", required=True)
    stress_parser.add_argument("--resume", action="store_true")
    stress_parser.add_argument("--seed", type=int, default=0)
    stress_parser.add_argument("--steps", type=int, default=64)
    stress_parser.add_argument("--learning-rate", type=float, default=3e-3)
    stress_parser.add_argument("--batch-size", type=int, default=16)
    stress_parser.add_argument("--num-distractors", type=int, default=2)
    stress_parser.add_argument("--num-pairs", type=int, default=16)
    stress_parser.add_argument("--num-memories", type=int, default=1)
    stress_parser.add_argument("--eval-distances", type=int, nargs="+", default=[8, 12, 16])
    stress_parser.add_argument("--eval-batches", type=int, default=4)
    stress_parser.add_argument("--num-layers-grid", type=int, nargs="+", default=[4, 6])
    stress_parser.add_argument("--memory-write-layers-grid", type=int, nargs="+", default=[1, 2, 3])
    stress_parser.add_argument("--distance-grid", type=int, nargs="+", default=[8, 12, 16])
    stress_parser.add_argument("--num-memories-grid", type=int, nargs="+", default=[1, 2, 3])
    stress_parser.add_argument("--num-distractors-grid", type=int, nargs="+", default=[4, 8, 12])
    stress_parser.add_argument("--num-pairs-grid", type=int, nargs="+", default=[16, 24, 32])
    stress_parser.add_argument("--operator", choices=["covariance", "sensitivity"], default="sensitivity")
    stress_parser.add_argument("--state-view", choices=["full", "token", "memory"], default="memory")
    stress_parser.add_argument("--k-eigs", type=int, default=4)
    stress_parser.add_argument("--power-iters", type=int, default=8)
    stress_parser.add_argument("--subspace-iters", type=int, default=8)
    stress_parser.add_argument("--subspace-oversample", type=int, default=4)
    stress_parser.add_argument("--perturb", action="store_true")
    stress_parser.add_argument("--perturb-distractors", type=int, default=2)
    stress_parser.add_argument("--stop-after-failures", type=int, default=3)
    _add_report_threshold_arguments(stress_parser)
    stress_parser.set_defaults(handler=_stress_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
