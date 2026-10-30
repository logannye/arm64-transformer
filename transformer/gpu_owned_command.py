"""Public owned-study entrypoint with explicit authenticated import roots.

Invoke by absolute file path under canonical CPython with -P -S -B, a fresh
-X pycache_prefix, and PYTHONHASHSEED=0. The plan/storage commands need no runtime
or data. Every executing command authenticates the selected runtime first, then
uses the existing producer, runner, auditor or reporter with unchanged gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

PROTOCOL = "gpu-v4-owned-five-engine-v1"
BOOTSTRAP_FILES = frozenset(
    "transformer/" + name + ".py"
    for name in (
        "__init__",
        "gpu_environment",
        "gpu_owned_backend",
        "gpu_owned_origins",
        "gpu_owned_command",
        "gpu_owned_study",
        "gpu_study",
        "gpu_study_context",
        "gpu_study_runtime",
    )
)
ROUTES = {
    "plan": ("gpu_study", ["plan", "--context", "owned"]),
    "storage": ("gpu_study", ["storage"]),
    "freeze": ("gpu_study", ["freeze", "--context", "owned"]),
    "admit": ("gpu_study_admission", ["run"]),
    "seal": ("gpu_study_admission", ["seal"]),
    "verify": ("gpu_study", ["verify"]),
    "run": ("gpu_study", ["run"]),
    "audit": ("gpu_study_audit", []),
    "report": ("gpu_study_report", []),
    "repeatability": ("gpu_owned_repeatability", []),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate authority key")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )


def launch_controls():
    require(
        sys.flags.safe_path
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
        and sys.flags.hash_randomization == 0
        and not sys.flags.optimize,
        "Use canonical Python -P -S -B with PYTHONHASHSEED=0",
    )
    require(
        str(Path(sys.executable).resolve()) == sys.executable, "Use the canonical interpreter path"
    )
    require(
        sys.pycache_prefix is not None and not Path(sys.pycache_prefix).exists(),
        "Set a fresh nonexistent bytecode prefix before any project import",
    )
    allowed = {"PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE"}
    require(
        not any(name.startswith("PYTHON") and name not in allowed for name in os.environ),
        "Ambient Python override is forbidden",
    )
    require(
        os.environ.get("PYTHONHASHSEED") == "0"
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        "Owned Python environment must retain fixed hash seed and disabled bytecode writes",
    )
    require(
        not any(name.startswith("DYLD_") for name in os.environ),
        "Native library override is forbidden",
    )


def route(command, arguments):
    require(command in ROUTES, "Unknown owned command")
    require(
        not any(arg == "--context" or arg.startswith("--context=") for arg in arguments),
        "Owned command cannot switch namespaces",
    )
    for index, argument in enumerate(arguments):
        if argument == "--manifest":
            require(index + 1 < len(arguments), "Manifest path missing")
            require(
                read(arguments[index + 1]).get("protocol") == PROTOCOL,
                "Owned command rejects standard manifest",
            )
        elif argument.startswith("--manifest="):
            require(
                read(argument.split("=", 1)[1]).get("protocol") == PROTOCOL,
                "Owned command rejects standard manifest",
            )
    module, prefix = ROUTES[command]
    return "transformer." + module, prefix + list(arguments)


def runtime_authority(root, inventory_path, receipt_path, expected_sha256):
    require(
        all((root, inventory_path, receipt_path, expected_sha256)),
        "Executing owned commands require runtime inventory and externally pinned receipt",
    )
    require(digest(receipt_path) == expected_sha256, "Owned runtime receipt trust anchor differs")
    receipt, inventory = read(receipt_path), read(inventory_path)
    require(
        receipt.get("schema") == "gpu-owned-runtime-identity-v1"
        and receipt.get("protocol") == PROTOCOL,
        "Wrong owned runtime identity",
    )
    project = receipt["project"]
    source = Path(project["root"])
    require(
        source.resolve(strict=True) == Path(__file__).resolve().parents[1], "Project root differs"
    )
    require(
        isinstance(project.get("files"), dict) and BOOTSTRAP_FILES <= set(project["files"]),
        "Runtime receipt omits a bootstrap source file",
    )
    for relative, sha in project["files"].items():
        require(
            not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            "Unsafe project source inventory",
        )
        path = source / relative
        require(
            path.resolve(strict=True).is_relative_to(source) and digest(path) == sha,
            "Project source changed before import",
        )
    require(
        project["files"].get("transformer/gpu_owned_command.py") == digest(__file__),
        "Owned command dispatcher is unbound",
    )
    # Do not activate any inventory-supplied runtime path until full discovery,
    # payload hashes and package resolution authenticate that inventory. The
    # authority modules are stdlib-only; append bound project source after stdlib.
    sys.path.append(str(source))
    from transformer.gpu_environment import configure_environment
    from transformer.gpu_owned_backend import authenticate_runtime
    from transformer.gpu_owned_study import FRAMEWORKS
    from transformer.gpu_study import ENVIRONMENT

    configure_environment(ENVIRONMENT)
    authority = authenticate_runtime(root, inventory, FRAMEWORKS, receipt_path, expected_sha256)
    paths = [str(source), *[str(Path(root).resolve() / name) for name in inventory["python_paths"]]]
    sys.path[:0] = paths
    return authority


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", default="/")
    parser.add_argument("--runtime-inventory")
    parser.add_argument("--runtime-receipt")
    parser.add_argument("--runtime-receipt-sha256")
    parser.add_argument("command", choices=ROUTES)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    launch_controls()
    module, arguments = route(args.command, args.arguments)
    authority = None
    if args.command in {"plan", "storage"}:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    else:
        authority = runtime_authority(
            args.runtime_root,
            args.runtime_inventory,
            args.runtime_receipt,
            args.runtime_receipt_sha256,
        )
    try:
        sys.argv = [module, *arguments]
        result = importlib.import_module(module).main()
    finally:
        require(not Path(sys.pycache_prefix).exists(), "Owned command wrote a bytecode cache")
        if authority is not None:
            authority.verify()
    return 0 if result is None else result


if __name__ == "__main__":
    raise SystemExit(main())
