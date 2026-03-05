from pathlib import Path
import shutil
import uuid

import jax
import jax.numpy as jnp

from modulus_memory_channels.bench_write_keep_read import (
    WriteKeepReadConfig,
    compute_retrieval_accuracy,
    generate_write_keep_read_batch,
)
from modulus_memory_channels.cli import main as cli_main
from modulus_memory_channels.config import MemoryModelConfig, ProbeConfig
from modulus_memory_channels.dk_predictor import davis_kahan_bound
from modulus_memory_channels.gap import eigengap
from modulus_memory_channels.io import load_json, load_tree, save_tree
from modulus_memory_channels.jacobian_norm import jacobian_spectral_norm
from modulus_memory_channels.model import MemoryChannelsModel
from modulus_memory_channels.probe_runner import run_probe_suite
from modulus_memory_channels.subspace import covariance_operator, principal_angle_drift, top_eigenspace
from modulus_memory_channels.training import TrainConfig, train_write_keep_read_model


def test_jacobian_norm_matches_linear_map():
    diag = jnp.array([3.0, 2.0, 1.0], dtype=jnp.float32)
    matrix = jnp.diag(diag)
    x = jnp.ones((3,), dtype=jnp.float32)

    estimate = jacobian_spectral_norm(lambda value: matrix @ value, x, key=jax.random.PRNGKey(0))
    assert jnp.isclose(estimate, 3.0, atol=1e-3)


def test_covariance_gap_and_drift():
    samples = jnp.array(
        [
            [[4.0, 0.0], [3.0, 0.0]],
            [[-4.0, 0.0], [-3.0, 0.0]],
            [[2.0, 0.1], [1.0, -0.1]],
        ],
        dtype=jnp.float32,
    )
    operator = covariance_operator(samples)
    subspace = top_eigenspace(operator, 1)
    assert eigengap(subspace.eigenvalues, 1) > 0.0
    assert principal_angle_drift(subspace.basis, subspace.basis) < 1e-6


def test_davis_kahan_bound_is_finite():
    base = jnp.diag(jnp.array([5.0, 1.0], dtype=jnp.float32))
    perturbed = jnp.array([[5.0, 0.2], [0.2, 1.0]], dtype=jnp.float32)
    result = davis_kahan_bound(base, perturbed, k=1)
    assert result.gap > 0.0
    assert result.predicted_drift >= 0.0
    assert result.observed_drift >= 0.0


def test_write_keep_read_batch_and_accuracy():
    config = WriteKeepReadConfig(vocab_size=64, batch_size=4, distance=5, num_distractors=3)
    batch = generate_write_keep_read_batch(jax.random.PRNGKey(1), config)
    assert batch.tokens.shape[0] == 4
    assert jnp.all(batch.readout_positions == (batch.tokens.shape[1] - 1))

    logits = jnp.zeros((4, batch.tokens.shape[1], 64), dtype=jnp.float32)
    logits = logits.at[jnp.arange(4), batch.readout_positions, batch.targets].set(10.0)
    accuracy = compute_retrieval_accuracy(logits, batch.readout_positions, batch.targets)
    assert jnp.isclose(accuracy, 1.0)


def test_probe_runner_returns_metrics():
    model = MemoryChannelsModel(
        MemoryModelConfig(
            vocab_size=64,
            d_model=16,
            num_layers=2,
            num_heads=4,
            mlp_dim=32,
            memory_dim=8,
            num_memory_heads=2,
        )
    )
    params = model.init(jax.random.PRNGKey(2))
    tokens = jnp.array([[1, 5, 6, 2, 5], [1, 7, 8, 2, 7]], dtype=jnp.int32)
    probe = ProbeConfig(probe_layers=(0, 1, 2), k_eigs=2, power_iters=4)

    run = run_probe_suite(model, params, tokens, probe)
    assert set(run.layer_metrics.keys()) == {0, 1, 2}
    assert run.layer_metrics[0].jacobian_norms.shape == (2,)


def test_probe_runner_supports_sensitivity_operator():
    model = MemoryChannelsModel(
        MemoryModelConfig(
            vocab_size=48,
            d_model=8,
            num_layers=1,
            num_heads=2,
            mlp_dim=16,
            memory_dim=4,
            num_memory_heads=1,
        )
    )
    params = model.init(jax.random.PRNGKey(3))
    tokens = jnp.array([[1, 5, 6, 7], [1, 8, 9, 10]], dtype=jnp.int32)
    perturbed_tokens = jnp.array([[1, 5, 11, 7], [1, 8, 12, 10]], dtype=jnp.int32)
    probe = ProbeConfig(
        probe_layers=(1,),
        k_eigs=2,
        power_iters=4,
        subspace_iters=4,
        subspace_oversample=2,
        operator="sensitivity",
    )

    run = run_probe_suite(model, params, tokens, probe, perturbed_tokens=perturbed_tokens)
    metrics = run.layer_metrics[1]
    assert metrics.subspace.operator_type == "sensitivity"
    assert metrics.subspace.basis.shape[0] == tokens.shape[1] * model.config.memory_dim
    assert metrics.dk_result is not None
    assert metrics.dk_result.predicted_drift >= 0.0


def test_training_loop_improves_accuracy():
    model = MemoryChannelsModel(
        MemoryModelConfig(
            vocab_size=64,
            d_model=16,
            num_layers=2,
            num_heads=4,
            mlp_dim=32,
            memory_dim=8,
            num_memory_heads=2,
        )
    )
    params = model.init(jax.random.PRNGKey(4))
    task = WriteKeepReadConfig(
        vocab_size=64,
        batch_size=16,
        distance=0,
        num_distractors=0,
        num_pairs=8,
    )
    train = TrainConfig(
        steps=60,
        learning_rate=1e-2,
        eval_distances=(0, 2),
        eval_batches=4,
    )

    result = train_write_keep_read_model(
        model,
        params,
        task,
        train,
        key=jax.random.PRNGKey(5),
    )
    assert len(result.history["loss"]) == train.steps
    assert result.history["loss"][-1] < result.history["loss"][0]
    assert result.eval_accuracy_by_distance[0] >= 0.5


def _test_dir(name: str) -> Path:
    path = Path("test_artifacts") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_checkpoint_round_trip():
    artifact_dir = _test_dir("checkpoint")
    model = MemoryChannelsModel(
        MemoryModelConfig(
            vocab_size=32,
            d_model=8,
            num_layers=1,
            num_heads=2,
            mlp_dim=16,
            memory_dim=4,
            num_memory_heads=1,
        )
    )
    params = model.init(jax.random.PRNGKey(6))
    try:
        stem = artifact_dir / "params"
        save_tree(params, stem)
        loaded = load_tree(stem)

        flat_original, _ = jax.tree_util.tree_flatten(params)
        flat_loaded, _ = jax.tree_util.tree_flatten(loaded)
        assert len(flat_original) == len(flat_loaded)
        for original, restored in zip(flat_original, flat_loaded, strict=True):
            assert jnp.allclose(original, restored)
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


def test_cli_train_and_probe_outputs():
    artifact_dir = _test_dir("cli")
    train_dir = artifact_dir / "train_run"
    probe_dir = artifact_dir / "probe_run"
    report_dir = artifact_dir / "report_run"
    compare_train_dir = artifact_dir / "compare_train"
    compare_probe_dir = artifact_dir / "compare_probe"
    sweep_dir = artifact_dir / "sweep_run"
    stress_dir = artifact_dir / "stress_run"

    try:
        train_exit = cli_main(
            [
                "train",
                "--output-dir",
                str(train_dir),
                "--seed",
                "7",
                "--steps",
                "10",
                "--learning-rate",
                "0.01",
                "--vocab-size",
                "48",
                "--d-model",
                "8",
                "--num-layers",
                "1",
                "--num-heads",
                "2",
                "--mlp-dim",
                "16",
                "--memory-dim",
                "4",
                "--num-memory-heads",
                "1",
                "--batch-size",
                "8",
                "--distance",
                "2",
                "--num-distractors",
                "1",
                "--num-pairs",
                "8",
                "--eval-distances",
                "2",
                "4",
                "--eval-batches",
                "2",
                "--checkpoint-every",
                "5",
            ]
        )
        assert train_exit == 0
        assert (train_dir / "params.npz").exists()
        assert (train_dir / "params_manifest.json").exists()
        assert (train_dir / "optimizer_state.npz").exists()
        assert (train_dir / "optimizer_state_manifest.json").exists()
        assert (train_dir / "train_history.json").exists()
        assert (train_dir / "train_history.csv").exists()
        assert (train_dir / "eval_accuracy.json").exists()
        assert (train_dir / "eval_accuracy.csv").exists()
        assert (train_dir / "run_config.json").exists()
        assert (train_dir / "loss.svg").exists()
        assert (train_dir / "accuracy.svg").exists()
        assert (train_dir / "eval_accuracy.svg").exists()
        assert (train_dir / "checkpoints" / "step_000005" / "params.npz").exists()
        assert (train_dir / "checkpoints" / "step_000010" / "run_config.json").exists()

        resume_exit = cli_main(
            [
                "train",
                "--resume-dir",
                str(train_dir),
                "--output-dir",
                str(train_dir),
                "--seed",
                "9",
                "--steps",
                "5",
                "--learning-rate",
                "0.01",
                "--eval-distances",
                "2",
                "4",
                "--eval-batches",
                "2",
            ]
        )
        assert resume_exit == 0
        history = load_json(train_dir / "train_history.json")
        assert len(history["loss"]) == 15

        probe_exit = cli_main(
            [
                "probe",
                "--run-dir",
                str(train_dir),
                "--output-dir",
                str(probe_dir),
                "--seed",
                "8",
                "--probe-layers",
                "0",
                "1",
                "--operator",
                "sensitivity",
                "--k-eigs",
                "2",
                "--power-iters",
                "4",
                "--subspace-iters",
                "4",
                "--subspace-oversample",
                "2",
                "--perturb",
            ]
        )
        assert probe_exit == 0
        assert (probe_dir / "probe_metrics.json").exists()
        assert (probe_dir / "probe_metrics.npz").exists()
        assert (probe_dir / "probe_metrics.csv").exists()
        assert (probe_dir / "probe_config.json").exists()
        assert (probe_dir / "jacobian_norms.svg").exists()
        assert (probe_dir / "eigengaps.svg").exists()

        metrics = load_json(probe_dir / "probe_metrics.json")
        assert metrics["layers"]["1"]["operator_type"] == "sensitivity"

        report_exit = cli_main(
            [
                "report",
                "--run-dir",
                str(train_dir),
                "--probe-dir",
                str(probe_dir),
                "--output-dir",
                str(report_dir),
            ]
        )
        assert report_exit == 0
        assert (report_dir / "report.json").exists()
        assert (report_dir / "report.txt").exists()
        assert (report_dir / "criteria.csv").exists()

        compare_train_exit = cli_main(
            [
                "compare",
                "--left-dir",
                str(train_dir / "checkpoints" / "step_000005"),
                "--right-dir",
                str(train_dir),
                "--output-dir",
                str(compare_train_dir),
                "--left-label",
                "checkpoint",
                "--right-label",
                "final",
            ]
        )
        assert compare_train_exit == 0
        assert (compare_train_dir / "comparison.json").exists()
        assert (compare_train_dir / "training_history_compare.csv").exists()
        assert (compare_train_dir / "loss_compare.svg").exists()
        assert (compare_train_dir / "eval_accuracy_compare.svg").exists()

        compare_probe_exit = cli_main(
            [
                "compare",
                "--left-dir",
                str(probe_dir),
                "--right-dir",
                str(probe_dir),
                "--output-dir",
                str(compare_probe_dir),
                "--left-label",
                "base",
                "--right-label",
                "base_copy",
            ]
        )
        assert compare_probe_exit == 0
        assert (compare_probe_dir / "comparison.json").exists()
        assert (compare_probe_dir / "probe_compare.csv").exists()
        assert (compare_probe_dir / "probe_jacobian_compare.svg").exists()

        sweep_exit = cli_main(
            [
                "sweep",
                "--output-dir",
                str(sweep_dir),
                "--seed",
                "13",
                "--steps",
                "4",
                "--learning-rate",
                "0.01",
                "--vocab-size",
                "48",
                "--d-model",
                "8",
                "--num-heads",
                "2",
                "--mlp-dim",
                "16",
                "--memory-dim",
                "4",
                "--num-memory-heads",
                "1",
                "--batch-size",
                "8",
                "--num-distractors",
                "1",
                "--num-pairs",
                "8",
                "--num-layers-grid",
                "2",
                "--memory-write-layers-grid",
                "1",
                "--distance-grid",
                "2",
                "--num-memories-grid",
                "1",
                "--num-distractors-grid",
                "1",
                "--num-pairs-grid",
                "8",
                "--eval-distances",
                "2",
                "4",
                "--eval-batches",
                "2",
            ]
        )
        assert sweep_exit == 0
        assert (sweep_dir / "sweep_summary.json").exists()
        assert (sweep_dir / "sweep_summary.csv").exists()
        assert (sweep_dir / "corridor_scores.svg").exists()

        stress_exit = cli_main(
            [
                "stress",
                "--output-dir",
                str(stress_dir),
                "--seed",
                "17",
                "--steps",
                "4",
                "--learning-rate",
                "0.01",
                "--vocab-size",
                "48",
                "--d-model",
                "8",
                "--num-heads",
                "2",
                "--mlp-dim",
                "16",
                "--memory-dim",
                "4",
                "--num-memory-heads",
                "1",
                "--batch-size",
                "8",
                "--num-layers-grid",
                "2",
                "--memory-write-layers-grid",
                "1",
                "--distance-grid",
                "2",
                "--num-memories-grid",
                "1",
                "--num-distractors-grid",
                "1",
                "--num-pairs-grid",
                "8",
                "--eval-distances",
                "2",
                "4",
                "--eval-batches",
                "2",
                "--stop-after-failures",
                "1",
            ]
        )
        assert stress_exit == 0
        assert (stress_dir / "sweep_summary.json").exists()
        assert (stress_dir / "sweep_summary.txt").exists()

        stress_resume_exit = cli_main(
            [
                "stress",
                "--output-dir",
                str(stress_dir),
                "--resume",
                "--seed",
                "17",
                "--steps",
                "4",
                "--learning-rate",
                "0.01",
                "--vocab-size",
                "48",
                "--d-model",
                "8",
                "--num-heads",
                "2",
                "--mlp-dim",
                "16",
                "--memory-dim",
                "4",
                "--num-memory-heads",
                "1",
                "--batch-size",
                "8",
                "--num-layers-grid",
                "2",
                "--memory-write-layers-grid",
                "1",
                "--distance-grid",
                "2",
                "--num-memories-grid",
                "1",
                "--num-distractors-grid",
                "1",
                "--num-pairs-grid",
                "8",
                "--eval-distances",
                "2",
                "4",
                "--eval-batches",
                "2",
                "--stop-after-failures",
                "1",
            ]
        )
        assert stress_resume_exit == 0
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)
