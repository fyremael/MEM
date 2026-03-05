from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import json

import jax
import jax.numpy as jnp
import numpy as np


Array = jax.Array


def _as_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _as_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def save_json(data: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(_as_jsonable(data), indent=2), encoding="ascii")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="ascii"))


def _serialize_tree(node: Any, arrays: dict[str, np.ndarray], prefix: str = "root") -> Any:
    if is_dataclass(node):
        return _serialize_tree(asdict(node), arrays, prefix)
    if isinstance(node, dict):
        return {
            "type": "dict",
            "items": {key: _serialize_tree(value, arrays, f"{prefix}.{key}") for key, value in node.items()},
        }
    if isinstance(node, list):
        return {
            "type": "list",
            "items": [_serialize_tree(value, arrays, f"{prefix}[{index}]") for index, value in enumerate(node)],
        }
    array = np.asarray(node)
    arrays[prefix] = array
    return {"type": "array", "key": prefix}


def _deserialize_tree(manifest: Any, arrays: dict[str, np.ndarray]) -> Any:
    node_type = manifest["type"]
    if node_type == "dict":
        return {
            key: _deserialize_tree(value, arrays)
            for key, value in manifest["items"].items()
        }
    if node_type == "list":
        return [_deserialize_tree(value, arrays) for value in manifest["items"]]
    if node_type == "array":
        return jnp.asarray(arrays[manifest["key"]])
    raise ValueError(f"Unsupported manifest node type: {node_type}")


def save_tree(tree: Any, output_stem: str | Path) -> None:
    output_stem = Path(output_stem)
    arrays: dict[str, np.ndarray] = {}
    manifest = _serialize_tree(tree, arrays)
    np.savez(output_stem.with_suffix(".npz"), **arrays)
    save_json(manifest, output_stem.with_name(f"{output_stem.name}_manifest.json"))


def load_tree(input_stem: str | Path) -> Any:
    input_stem = Path(input_stem)
    manifest = load_json(input_stem.with_name(f"{input_stem.name}_manifest.json"))
    with np.load(input_stem.with_suffix(".npz")) as data:
        arrays = {key: data[key] for key in data.files}
    return _deserialize_tree(manifest, arrays)
