# `modulus_memory_channels.probe_runner`

Source: `src/modulus_memory_channels/probe_runner.py`

## Module Summary
No module-level documentation provided.

## Public Constants
No public constants detected.

## Public Classes
### `LayerProbeMetrics`

Signature: `class LayerProbeMetrics`

No documentation provided.

Methods:
- No public methods detected.

### `ProbeRun`

Signature: `class ProbeRun`

No documentation provided.

Methods:
- `def to_serializable(self) -> dict[str, object]`: No documentation provided.
- `def save_json(self, path: str | Path) -> None`: No documentation provided.
- `def save_npz(self, path: str | Path) -> None`: No documentation provided.

## Public Functions
### `run_probe_suite`

Signature: `def run_probe_suite(model: MemoryChannelsModel, params: Params, tokens: Array, probe_config: ProbeConfig, *, attention_mask: Array | None = None, perturbed_tokens: Array | None = None) -> ProbeRun`

No documentation provided.
