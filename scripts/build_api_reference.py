from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_ROOT = REPO_ROOT / "src" / "modulus_memory_channels"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "docs" / "reference"


@dataclass(frozen=True)
class FunctionRef:
    name: str
    signature: str
    doc: str
    is_async: bool


@dataclass(frozen=True)
class ConstantRef:
    name: str
    value: str


@dataclass(frozen=True)
class ClassRef:
    name: str
    signature: str
    doc: str
    methods: tuple[FunctionRef, ...]


@dataclass(frozen=True)
class ModuleRef:
    module_name: str
    source_relpath: str
    doc: str
    classes: tuple[ClassRef, ...]
    functions: tuple[FunctionRef, ...]
    constants: tuple[ConstantRef, ...]


def _doc_or_placeholder(doc: str | None) -> str:
    if doc is None or not doc.strip():
        return "No module-level documentation provided."
    return doc.strip()


def _first_line(doc: str | None) -> str:
    if doc is None or not doc.strip():
        return "No documentation provided."
    return doc.strip().splitlines()[0]


def _render_expr(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<unparseable>"


def _format_arg(arg: ast.arg, default: ast.AST | None = None) -> str:
    rendered = arg.arg
    if arg.annotation is not None:
        rendered += f": {_render_expr(arg.annotation)}"
    if default is not None:
        rendered += f" = {_render_expr(default)}"
    return rendered


def _format_args(args: ast.arguments) -> str:
    rendered: list[str] = []
    positional = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    default_padding = len(positional) - len(defaults)
    padded_defaults: list[ast.AST | None] = [None] * default_padding + defaults

    for idx, arg in enumerate(args.posonlyargs):
        rendered.append(_format_arg(arg, padded_defaults[idx]))
    if args.posonlyargs:
        rendered.append("/")

    for idx, arg in enumerate(args.args, start=len(args.posonlyargs)):
        rendered.append(_format_arg(arg, padded_defaults[idx]))

    if args.vararg is not None:
        rendered.append("*" + _format_arg(args.vararg))
    elif args.kwonlyargs:
        rendered.append("*")

    for kw_arg, kw_default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        rendered.append(_format_arg(kw_arg, kw_default))

    if args.kwarg is not None:
        rendered.append("**" + _format_arg(args.kwarg))
    return ", ".join(rendered)


def _format_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    signature = f"{prefix}{node.name}({_format_args(node.args)})"
    if node.returns is not None:
        signature += f" -> {_render_expr(node.returns)}"
    return signature


def _format_class_signature(node: ast.ClassDef) -> str:
    if not node.bases:
        return f"class {node.name}"
    bases = ", ".join(_render_expr(base) for base in node.bases)
    return f"class {node.name}({bases})"


def _module_name_for_path(package_root: Path, source_path: Path) -> str:
    rel = source_path.relative_to(package_root)
    if rel.name == "__init__.py":
        suffix = rel.parent.as_posix().replace("/", ".")
    else:
        suffix = rel.with_suffix("").as_posix().replace("/", ".")
    if not suffix or suffix == ".":
        return "modulus_memory_channels"
    return f"modulus_memory_channels.{suffix}"


def _is_public_name(name: str) -> bool:
    return not name.startswith("_")


def _extract_module_ref(package_root: Path, source_path: Path) -> ModuleRef:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    module_doc = _doc_or_placeholder(ast.get_docstring(tree))

    classes: list[ClassRef] = []
    functions: list[FunctionRef] = []
    constants: list[ConstantRef] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public_name(node.name):
            functions.append(
                FunctionRef(
                    name=node.name,
                    signature=_format_function_signature(node),
                    doc=_first_line(ast.get_docstring(node)),
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                )
            )
        elif isinstance(node, ast.ClassDef) and _is_public_name(node.name):
            methods: list[FunctionRef] = []
            for class_node in node.body:
                if isinstance(class_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public_name(class_node.name):
                    methods.append(
                        FunctionRef(
                            name=class_node.name,
                            signature=_format_function_signature(class_node),
                            doc=_first_line(ast.get_docstring(class_node)),
                            is_async=isinstance(class_node, ast.AsyncFunctionDef),
                        )
                    )
            classes.append(
                ClassRef(
                    name=node.name,
                    signature=_format_class_signature(node),
                    doc=_first_line(ast.get_docstring(node)),
                    methods=tuple(methods),
                )
            )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper() and _is_public_name(target.id):
                    constants.append(ConstantRef(name=target.id, value=_render_expr(node.value)))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id.isupper() and _is_public_name(node.target.id):
                value = _render_expr(node.value) if node.value is not None else "<no default>"
                constants.append(ConstantRef(name=node.target.id, value=value))

    module_name = _module_name_for_path(package_root, source_path)
    source_relpath = source_path.relative_to(REPO_ROOT).as_posix()
    return ModuleRef(
        module_name=module_name,
        source_relpath=source_relpath,
        doc=module_doc,
        classes=tuple(classes),
        functions=tuple(functions),
        constants=tuple(constants),
    )


def _render_module_markdown(module_ref: ModuleRef) -> str:
    lines: list[str] = []
    lines.append(f"# `{module_ref.module_name}`")
    lines.append("")
    lines.append(f"Source: `{module_ref.source_relpath}`")
    lines.append("")
    lines.append("## Module Summary")
    lines.append(module_ref.doc)
    lines.append("")

    lines.append("## Public Constants")
    if module_ref.constants:
        for constant in module_ref.constants:
            lines.append(f"- `{constant.name}` = `{constant.value}`")
    else:
        lines.append("No public constants detected.")
    lines.append("")

    lines.append("## Public Classes")
    if module_ref.classes:
        for class_ref in module_ref.classes:
            lines.append(f"### `{class_ref.name}`")
            lines.append("")
            lines.append(f"Signature: `{class_ref.signature}`")
            lines.append("")
            lines.append(class_ref.doc)
            lines.append("")
            lines.append("Methods:")
            if class_ref.methods:
                for method in class_ref.methods:
                    lines.append(f"- `{method.signature}`: {method.doc}")
            else:
                lines.append("- No public methods detected.")
            lines.append("")
    else:
        lines.append("No public classes detected.")
        lines.append("")

    lines.append("## Public Functions")
    if module_ref.functions:
        for function_ref in module_ref.functions:
            lines.append(f"### `{function_ref.name}`")
            lines.append("")
            lines.append(f"Signature: `{function_ref.signature}`")
            lines.append("")
            lines.append(function_ref.doc)
            lines.append("")
    else:
        lines.append("No public functions detected.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_index_markdown(module_refs: list[ModuleRef]) -> str:
    lines: list[str] = []
    lines.append("# API Reference Index")
    lines.append("")
    lines.append("This file is generated by `scripts/build_api_reference.py`. Do not edit manually.")
    lines.append("")
    lines.append("| Module | Classes | Functions | Constants | Link |")
    lines.append("|---|---:|---:|---:|---|")
    for module_ref in module_refs:
        module_slug = module_ref.module_name.replace("modulus_memory_channels.", "").replace(".", "_")
        if module_ref.module_name == "modulus_memory_channels":
            module_slug = "modulus_memory_channels"
        link = f"api/{module_slug}.md"
        lines.append(
            f"| `{module_ref.module_name}` | {len(module_ref.classes)} | {len(module_ref.functions)} | {len(module_ref.constants)} | [{module_slug}]({link}) |"
        )
    lines.append("")
    return "\n".join(lines)


def _expected_outputs(package_root: Path, output_root: Path) -> dict[Path, str]:
    module_paths = sorted(package_root.glob("*.py"))
    module_refs = [_extract_module_ref(package_root, source_path) for source_path in module_paths]
    expected: dict[Path, str] = {}

    api_dir = output_root / "api"
    for module_ref in module_refs:
        module_slug = module_ref.module_name.replace("modulus_memory_channels.", "").replace(".", "_")
        if module_ref.module_name == "modulus_memory_channels":
            module_slug = "modulus_memory_channels"
        expected[api_dir / f"{module_slug}.md"] = _render_module_markdown(module_ref)
    expected[output_root / "API_INDEX.md"] = _render_index_markdown(module_refs)
    return expected


def _check_output(expected: dict[Path, str]) -> list[str]:
    errors: list[str] = []
    for path, content in expected.items():
        if not path.exists():
            errors.append(f"missing generated file: {path.relative_to(REPO_ROOT).as_posix()}")
            continue
        existing = path.read_text(encoding="ascii")
        if existing != content:
            errors.append(f"stale generated file: {path.relative_to(REPO_ROOT).as_posix()}")
    return errors


def _write_output(expected: dict[Path, str], output_root: Path) -> None:
    for path, content in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="ascii")

    generated_paths = set(expected.keys())
    api_dir = output_root / "api"
    if api_dir.exists():
        for candidate in api_dir.glob("*.md"):
            if candidate not in generated_paths:
                candidate.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build deterministic API/reference markdown docs from source modules.")
    parser.add_argument("--package-root", default=str(DEFAULT_PACKAGE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--check", action="store_true", help="Fail if generated docs are missing or stale.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    package_root = Path(args.package_root)
    output_root = Path(args.output_root)

    expected = _expected_outputs(package_root, output_root)
    if args.check:
        errors = _check_output(expected)
        if errors:
            print("api reference check FAILED")
            for err in errors:
                print(f"- {err}")
            return 1
        print("api reference check PASSED")
        return 0

    _write_output(expected, output_root)
    print(f"wrote API reference docs to {output_root.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
