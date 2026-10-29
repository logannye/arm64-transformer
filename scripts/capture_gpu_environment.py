"""Authenticate the installed GPU development dependencies without importing them.

Uses only Python's standard library. This records installed bytes, not proof of
an originating wheel archive. No installation, network request or GPU call occurs.
The bounded marker/version evaluator fails closed on unsupported active syntax.
"""

import argparse
import ast
import base64
import csv
import datetime
import hashlib
import importlib.machinery
import importlib.metadata
import io
import json
import operator
import os
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "runtime": ("torch", "numpy", "mlx", "mlx-metal"),
    "validation_tools": ("pytest", "ruff"),
    "report_tools": ("matplotlib", "psutil"),
}
FORBIDDEN_IMPORTS = {"torch", "numpy", "mlx", "matplotlib", "psutil", "pytest", "ruff"}


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def version(value):
    match = re.fullmatch(r"(\d+(?:\.\d+)*)(?:\.post(\d+))?", value)
    if match is None:
        raise ValueError(f"Unsupported version syntax: {value!r}")
    release = tuple(int(part) for part in match[1].split("."))
    return release + (0,) * (8 - len(release)), -1 if match[2] is None else int(match[2])


def marker_environment():
    return {
        "extra": "",
        "os_name": os.name,
        "sys_platform": sys.platform,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "platform_python_implementation": platform.python_implementation(),
        "implementation_name": sys.implementation.name,
        "python_version": ".".join(map(str, sys.version_info[:2])),
        "python_full_version": platform.python_version(),
    }


def marker_applies(expression, environment):
    """Evaluate the installed closure's PEP508 marker subset without eval()."""
    comparisons = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
    }

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in environment:
            return environment[node.id]
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(visit(value) for value in node.values)
            if isinstance(node.op, ast.Or):
                return any(visit(value) for value in node.values)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = visit(node.left), visit(node.comparators[0])
            operation = comparisons.get(type(node.ops[0]))
            if operation is None:
                raise ValueError(f"Unsupported marker operator: {expression}")
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            if names & {"python_version", "python_full_version"} and not isinstance(
                node.ops[0], (ast.In, ast.NotIn)
            ):
                left, right = version(left), version(right)
            return operation(left, right)
        raise ValueError(f"Unsupported dependency marker: {expression}")

    result = visit(ast.parse(expression, mode="eval"))
    if type(result) is not bool:
        raise ValueError(f"Dependency marker must produce a boolean: {expression}")
    return result


def requirement(raw, environment):
    value, separator, marker = raw.partition(";")
    if separator and not marker_applies(marker.strip(), environment):
        return None
    match = re.fullmatch(r"\s*([A-Za-z0-9][A-Za-z0-9_.-]*)\s*(.*?)\s*", value)
    if match is None or match[2].startswith(("[", "@")):
        raise ValueError(f"Unsupported active requirement: {raw}")
    return canonical(match[1]), match[2].strip("() ")


def satisfies(installed, constraints):
    operations = {
        "==": operator.eq,
        "!=": operator.ne,
        ">=": operator.ge,
        "<=": operator.le,
        ">": operator.gt,
        "<": operator.lt,
    }
    for part in constraints.split(",") if constraints else ():
        match = re.fullmatch(r"\s*(==|!=|>=|<=|>|<)\s*([^\s]+)\s*", part)
        if match is None:
            raise ValueError(f"Unsupported active version constraint: {part}")
        if not operations[match[1]](version(installed), version(match[2])):
            return False
    return True


def closure(roots, distributions, environment):
    selected, dependencies = set(), {}
    pending = list(roots)
    while pending:
        name = canonical(pending.pop())
        if name in selected:
            continue
        if name not in distributions:
            raise ValueError(f"Required distribution is not installed: {name}")
        selected.add(name)
        dependencies[name] = []
        for raw in distributions[name].requires or ():
            parsed = requirement(raw, environment)
            if parsed is None:
                continue
            child, constraints = parsed
            if child not in distributions or not satisfies(
                distributions[child].version, constraints
            ):
                raise ValueError(f"Unsatisfied dependency of {name}: {raw}")
            dependencies[name].append({"name": child, "requirement": raw})
            pending.append(child)
    return sorted(selected), dependencies


def file_identity(path, expected_hash=None, expected_size=None):
    algorithms = {"sha256"}
    if expected_hash:
        algorithm, _ = expected_hash.split("=", 1)
        algorithms.add(algorithm)
    hashes = {name: hashlib.new(name) for name in algorithms}
    before = path.stat()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            for digest in hashes.values():
                digest.update(chunk)
    after = path.stat()
    identity = {"bytes": size, "sha256": hashes["sha256"].hexdigest()}
    identity["unchanged_while_reading"] = (before.st_size, before.st_mtime_ns, before.st_ino) == (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ) and size == after.st_size
    if expected_hash:
        algorithm, encoded = expected_hash.split("=", 1)
        actual = base64.urlsafe_b64encode(hashes[algorithm].digest()).rstrip(b"=").decode()
        identity.update(
            record_hash=expected_hash, record_hash_matches=actual == encoded.rstrip("=")
        )
    if expected_size:
        identity.update(
            record_bytes=int(expected_size), record_size_matches=size == int(expected_size)
        )
    return identity


def aliaser(roots):
    roots = sorted(
        {
            (candidate, label)
            for label, path in roots.items()
            for candidate in (path.absolute(), path.resolve())
        },
        key=lambda item: len(item[0].parts),
        reverse=True,
    )

    def alias(path):
        path = Path(path).absolute()
        for candidate in (path, path.resolve()):
            for root, label in roots:
                if candidate.is_relative_to(root):
                    suffix = str(candidate.relative_to(root))
                    return label if suffix == "." else f"{label}/{suffix}"
        return str(path)

    return alias


def package_identity(distribution, allowed_roots, alias):
    metadata_path = Path(distribution._path)
    record_path = metadata_path / "RECORD"
    record_bytes = record_path.read_bytes()
    record = record_bytes.decode("utf-8")
    files, errors = {}, []
    skipped_bytecode = []
    for relative, expected_hash, expected_size in csv.reader(io.StringIO(record)):
        path = Path(distribution.locate_file(relative)).absolute()
        if path.suffix == ".pyc" or "__pycache__" in path.parts:
            skipped_bytecode.append(relative)
            continue
        resolved = path.resolve()
        if not any(resolved.is_relative_to(root) for root in allowed_roots):
            raise ValueError(f"Distribution RECORD escapes allowed installation roots: {relative}")
        key = alias(resolved)
        if key in files:
            raise ValueError(f"Duplicate installed file in RECORD: {relative}")
        try:
            identity = file_identity(path, expected_hash, expected_size)
            if path.is_symlink():
                identity["symlink_target"] = alias(resolved)
            files[key] = identity
            if not all(
                identity.get(field, True)
                for field in (
                    "record_hash_matches",
                    "record_size_matches",
                    "unchanged_while_reading",
                )
            ):
                errors.append(key)
        except OSError as exc:
            files[key] = {"error": f"{type(exc).__name__}: {exc.strerror}"}
            errors.append(key)
    metadata = {
        name: file_identity(metadata_path / name) for name in ("METADATA", "WHEEL", "RECORD")
    }
    if metadata["RECORD"]["sha256"] != hashlib.sha256(record_bytes).hexdigest():
        errors.append(alias(record_path))
    errors.extend(
        alias(metadata_path / name)
        for name, identity in metadata.items()
        if not identity["unchanged_while_reading"]
    )
    wheel = distribution.read_text("WHEEL") or ""
    return {
        "name": distribution.metadata["Name"],
        "version": distribution.version,
        "metadata_path": alias(metadata_path),
        "wheel_tags": [line[5:] for line in wheel.splitlines() if line.startswith("Tag: ")],
        "wheel_metadata": wheel,
        "metadata_files": metadata,
        "requires_dist": distribution.requires or [],
        "direct_url": distribution.read_text("direct_url.json"),
        "installed_files": files,
        "file_count": len(files),
        "bytes": sum(value.get("bytes", 0) for value in files.values()),
        "record_hashes_verified": sum(
            value.get("record_hash_matches") is True for value in files.values()
        ),
        "record_hashes_absent": sum("record_hash" not in value for value in files.values()),
        "skipped_generated_bytecode": skipped_bytecode,
        "record_integrity_passed": not errors,
        "record_integrity_failures": errors,
    }


def capture(root):
    if (
        sys.version_info[:2] != (3, 12)
        or platform.system() != "Darwin"
        or platform.machine() != "arm64"
    ):
        raise ValueError("This development snapshot targets CPython3.12 on macOS arm64")
    venv = (root / ".venv").resolve()
    target = (root / "work/gpu-v1/python-site").resolve()
    site = venv / "lib/python3.12/site-packages"
    roots = {
        ".venv": root / ".venv",
        "work/gpu-v1/python-site": root / "work/gpu-v1/python-site",
        "$PYTHON_BASE": Path(sys.base_prefix).resolve(),
        "$REPO": root,
    }
    alias = aliaser(roots)
    distributions, shadowed = {}, {}
    for location in (target, site):
        for distribution in importlib.metadata.distributions(path=[str(location)]):
            name = canonical(distribution.metadata["Name"])
            if name in distributions:
                shadowed.setdefault(name, []).append(alias(distribution._path))
            else:
                distributions[name] = distribution
    environment = marker_environment()
    groups, dependencies, names = {}, {}, set()
    for group, group_roots in GROUPS.items():
        groups[group], edges = closure(group_roots, distributions, environment)
        names.update(groups[group])
        dependencies.update(edges)
    requirements = root / "requirements-gpu-development.txt"
    pins = {}
    for line in requirements.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, separator, pinned = line.partition("==")
        if not separator or canonical(name) in pins:
            raise ValueError("Expected unique exact development requirement pins")
        pins[canonical(name)] = pinned
    if set(pins) != names or any(distributions[name].version != pins[name] for name in names):
        raise ValueError("Observed dependency closure does not match development requirement pins")
    if names & shadowed.keys():
        raise ValueError(f"Shadowed selected distributions: {sorted(names & shadowed.keys())}")
    packages = {}
    for name in sorted(names):
        packages[name] = package_identity(distributions[name], (venv, target), alias)
        print(
            f"Captured {name}=={packages[name]['version']}: {packages[name]['file_count']} files",
            flush=True,
        )
    executable = Path(sys.executable)
    intended_path = [str(target), str(root), *sys.path]
    resolutions = {}
    for name in ("torch", "numpy", "mlx", "pytest", "matplotlib", "psutil", "ruff"):
        spec = importlib.machinery.PathFinder.find_spec(name, intended_path)
        resolutions[name] = (
            None
            if spec is None
            else {
                "origin": None if spec.origin is None else alias(spec.origin),
                "namespace_paths": [alias(path) for path in spec.submodule_search_locations or ()],
            }
        )
    forbidden = sorted({name.split(".")[0] for name in sys.modules} & FORBIDDEN_IMPORTS)
    if forbidden:
        raise RuntimeError(f"Numerical/tool modules were unexpectedly imported: {forbidden}")
    return {
        "schema": 1,
        "scope": (
            "Installed development dependency identity; "
            "not frozen formal admission or wheel provenance"
        ),
        "captured_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "record_integrity_passed": all(
            item["record_integrity_passed"] for item in packages.values()
        ),
        "marker_environment": environment,
        "groups": groups,
        "dependencies": dependencies,
        "packages": packages,
        "package_count": len(packages),
        "installed_file_count": sum(item["file_count"] for item in packages.values()),
        "installed_bytes": sum(item["bytes"] for item in packages.values()),
        "python": {
            "version": sys.version,
            "executable": alias(executable),
            "resolved_executable": alias(executable.resolve()),
            "executable_identity": file_identity(executable),
            "base_prefix": "$PYTHON_BASE",
            "prefix": alias(sys.prefix),
        },
        "capture_sys_path": [alias(path or Path.cwd()) for path in sys.path],
        "metadata_search_paths": [alias(target), alias(site)],
        "intended_gpu_pythonpath": [alias(target), "$REPO"],
        "module_resolution_without_import_under_intended_path": resolutions,
        "no_numerical_or_gpu_imports": True,
        "requirements_identity": file_identity(requirements),
        "capture_script_identity": file_identity(Path(__file__)),
        "limits": [
            "Wheel tags and installed RECORDs do not establish the original archive URL or hash.",
            "Only the mandatory runtime and validation/report dependency closure is inventoried.",
            "Files are owned by distribution RECORDs; unrecorded generated caches are excluded.",
            "Python standard-library files and OS GPU drivers are outside this package inventory.",
            "PathFinder inspects resolution without importing; run paths need separate receipts.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output must be fresh; preserve earlier identity receipts")
    result = capture(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "package_count",
                    "installed_file_count",
                    "installed_bytes",
                    "record_integrity_passed",
                )
            }
        )
    )
    if not result["record_integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
