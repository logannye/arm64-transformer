"""Standalone provenance launcher for unchanged frozen CPU Python entrypoints.

This file deliberately imports no transformer modules. The selected frozen CPU
package stays authoritative. The parent authenticates admission; this child
rechecks its hash-bound authority, inputs, source/runtime and actual Torch origins
before and after runpy executes the original entrypoint. Proof work is outside
all timing recorded by that entrypoint. No optimizer or operator is substituted.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import runpy
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROTOCOL = "gpu-v4-owned-five-engine-v1"
SCHEMA = "gpu-owned-cpu-execution-v1"
HELPER = "transformer/gpu_owned_origins.py"
LAUNCHER = "transformer/gpu_owned_cpu.py"
ENTRYPOINTS = {
    "fit": ("cpu", "transformer/python_train.py"),
    "tie": ("cpu", "transformer/python_train.py"),
    "admission": ("source", "transformer/gpu_study_cpu_admission.py"),
    "replay": ("source", "transformer/gpu_study_replay.py"),
}
REJECTIONS = frozenset(
    {"Out-of-range CPU tokens", "Malformed CPU token batch", "Malformed CPU parameters"}
)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read(path):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "Duplicate authority key")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )


def file_at(root, name, expected):
    root = Path(root).resolve(strict=True)
    path = root / name
    require(
        isinstance(name, str)
        and name
        and not Path(name).is_absolute()
        and Path(name).as_posix() == name
        and all(part not in (".", "..") for part in Path(name).parts),
        "Unsafe authority file",
    )
    require(
        path.is_file() and path.resolve(strict=True).is_relative_to(root), "Missing authority file"
    )
    require(digest(path) == expected, "Changed authority file: " + name)
    return path.resolve(strict=True)


def validate_authority(value):
    require(
        value.get("schema") == "gpu-owned-cpu-authority-v1" and value.get("protocol") == PROTOCOL,
        "Wrong CPU authority namespace",
    )
    manifest = value["manifest"]
    require(
        manifest.get("protocol") == PROTOCOL
        and manifest.get("phase") == "frozen"
        and manifest.get("admission_seed") == 20261108,
        "CPU execution needs the frozen owned contract",
    )
    require(
        value.get("manifest_identity_sha256") == identity(manifest), "CPU manifest identity differs"
    )
    kind, engine = value["kind"], value["engine"]
    require(
        kind in ENTRYPOINTS and engine in ("cpu_torch", "oracle"), "Unknown CPU launch operation"
    )
    require(engine != "oracle" or kind == "replay", "Oracle is replay-only")
    require(
        value.get("expected_rejection") is None
        or (kind == "admission" and value["expected_rejection"] in REJECTIONS),
        "Unexpected CPU rejection contract",
    )
    role, relative = ENTRYPOINTS[kind]
    require(
        value.get("entrypoint")
        == {"role": role, "path": relative, "sha256": manifest["bindings"][role].get(relative)},
        "Wrong frozen CPU entrypoint",
    )
    require(value["entrypoint"]["sha256"] is not None, "Unbound CPU entrypoint")
    require(
        manifest["bindings"]["source"].get(LAUNCHER) is not None
        and manifest["bindings"]["source"].get(HELPER) is not None,
        "Missing CPU provenance source",
    )
    require(
        value.get("runtime_identity_sha256")
        == identity(manifest["bindings"]["runtime"])
        == manifest["bindings"]["torch_distribution_sha256"],
        "CPU/MPS runtime distribution differs",
    )
    require(
        value.get("expected_steps") == (1024 if kind == "fit" else 2 if kind == "tie" else None),
        "Wrong CPU optimizer count contract",
    )
    if kind in ("fit", "replay"):
        require(
            isinstance(manifest.get("admission"), dict)
            and set(manifest["admission"]) == {"sha256"},
            "CPU measurement requires sealed admission",
        )
    else:
        require(
            manifest.get("admission") is None,
            "CPU admission bridge requires fresh pre-admission contract",
        )
    require(
        isinstance(value["arguments"], list)
        and all(isinstance(x, str) and x for x in value["arguments"]),
        "Malformed exact CPU argv",
    )
    output = str(Path(value["output"]).resolve())
    args = value["arguments"]
    if kind in ("fit", "tie"):
        require(args == [value["fixture"], output], "Training input/output argv differs")
    elif kind == "admission":
        require(
            args == ["--engine", engine, "--input", value["operands"], "--output", output],
            "CPU admission argv differs",
        )
    else:
        require(
            args
            == [
                "--engine",
                engine,
                "--fixture",
                value["fixture"],
                "--checkpoint",
                value["checkpoint"],
                "--output",
                output,
            ],
            "CPU replay argv differs",
        )
    return value


def authenticate(path, expected_sha256):
    require(digest(path) == expected_sha256, "CPU authority trust anchor differs")
    value = validate_authority(read(path))
    require(
        sys.flags.safe_path
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
        and sys.flags.hash_randomization == 0
        and not sys.flags.optimize,
        "Owned CPU child requires -P -S -B and fixed hash seed",
    )
    manifest, roots = value["manifest"], value["roots"]
    for role in ("source", "cpu", "runtime"):
        for name, sha in manifest["bindings"][role].items():
            file_at(roots[role], name, sha)
    require(
        file_at(roots["source"], LAUNCHER, manifest["bindings"]["source"][LAUNCHER])
        == Path(__file__).resolve(),
        "Wrong executing CPU launcher",
    )
    runtime = Path(roots["runtime"]).resolve()
    python = Path(sys.executable).resolve()
    require(
        python.is_relative_to(runtime)
        and manifest["bindings"]["runtime"].get(python.relative_to(runtime).as_posix())
        == digest(python),
        "CPU interpreter is outside selected runtime",
    )
    require(
        "PYTHONPATH" not in os.environ and "PYTHONHOME" not in os.environ,
        "Ambient CPU interpreter path is forbidden",
    )
    require(
        sys.pycache_prefix == str(Path(value["output"] + ".unused-python-cache"))
        and not Path(sys.pycache_prefix).exists(),
        "CPU bytecode prefix changed or appeared",
    )
    expected_paths = value["pythonpath"].split(os.pathsep)
    require(
        sys.path[: len(expected_paths)] == expected_paths, "CPU bootstrap import precedence differs"
    )
    for key, expected in manifest["environment"].items():
        require(
            os.environ.get(key) == expected if expected is not None else key not in os.environ,
            "CPU numerical environment differs: " + key,
        )
    if manifest.get("admission") is not None:
        receipt = Path(value["admission_receipt"])
        require(digest(receipt) == manifest["admission"]["sha256"], "CPU admission receipt changed")
        admitted = read(receipt)
        fields = (
            "protocol",
            "workload",
            "timing",
            "environment",
            "cells",
            "bindings",
            "gate_inventory_sha256",
            "admission_seed",
            "owned_frameworks",
        )
        require(
            admitted.get("protocol") == PROTOCOL
            and admitted.get("seed") == 20261108
            and admitted.get("status") == "passed"
            and admitted.get("binding_sha256") == identity({key: manifest[key] for key in fields}),
            "CPU admission receipt is stale or failed",
        )
    # A separate control-plane process evaluates the complete raw authority.
    # The frozen CPU process never imports the current transformer package.
    env = dict(os.environ)
    cache = Path(value["output"] + ".guard-cache")
    require(not cache.exists() and not cache.is_symlink(), "CPU guard cache must be absent")
    bootstrap = (
        "import hashlib,pathlib,runpy,sys;script=pathlib.Path(sys.argv[1]);"
        "expected=" + repr(manifest["bindings"]["source"][LAUNCHER]) + ";"
        "actual=hashlib.sha256(script.read_bytes()).hexdigest();"
        "assert actual==expected,'CPU guard source changed';"
        "sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    subprocess.run(
        [
            str(Path(sys.executable).resolve()),
            "-P",
            "-S",
            "-B",
            "-X",
            "pycache_prefix=" + str(cache),
            "-c",
            bootstrap,
            str(Path(__file__).resolve()),
            "--guard",
            "--authority",
            str(Path(path).resolve()),
            "--authority-sha256",
            expected_sha256,
        ],
        cwd=roots["source"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=14400,
    )
    for absolute, sha in value["input_sha256"].items():
        require(Path(absolute).is_absolute() and digest(absolute) == sha, "CPU input changed")
    return value


def guard_controls():
    require(
        sys.flags.safe_path
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
        and sys.flags.hash_randomization == 0
        and not sys.flags.optimize
        and sys.pycache_prefix is not None
        and not Path(sys.pycache_prefix).exists(),
        "CPU guard requires isolated imports, fixed hash seed and absent bytecode cache",
    )
    require(
        os.environ.get("PYTHONHASHSEED") == "0"
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        "CPU guard environment differs",
    )


def guard(path, expected_sha256):
    """Read-only complete authority validation in its own current-source process."""
    guard_controls()
    require(digest(path) == expected_sha256, "CPU guard authority changed")
    value = validate_authority(read(path))
    for role in ("source", "cpu", "runtime"):
        for name, sha in value["manifest"]["bindings"][role].items():
            file_at(value["roots"][role], name, sha)
    paths = [
        value["roots"]["source"],
        *[
            str(Path(value["roots"]["runtime"]) / name)
            for name in value["manifest"]["bindings"]["runtime_python_paths"]
        ],
    ]
    sys.path[:0] = paths
    from transformer.gpu_study import validate_manifest, verify_admission, verify_bindings
    from transformer.gpu_study_admission import verify_baseline_repeatability

    manifest = validate_manifest(value["manifest"])
    if manifest["admission"] is not None:
        verify_admission(manifest, value["roots"], value["admission_receipt"])
    else:
        verify_bindings(manifest, value["roots"])
        verify_baseline_repeatability(manifest, value["roots"])
    return True


def native_libraries(payload, runtime, files):
    require(sys.platform == "darwin", "Owned CPU anchor requires the matched macOS runtime")
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.argtypes = []
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    result = {}
    for index in range(dyld._dyld_image_count()):
        raw = dyld._dyld_get_image_name(index)
        if not raw:
            continue
        path = Path(os.fsdecode(raw)).resolve(strict=True)
        if not path.name.startswith(("libtorch", "libc10", "libomp", "libshm")):
            continue
        require(
            str(path) in payload and path.is_relative_to(runtime), "CPU loaded foreign native image"
        )
        relative = path.relative_to(runtime).as_posix()
        require(
            files.get(relative) == payload[str(path)] == digest(path),
            "CPU native image bytes differ",
        )
        result[relative] = digest(path)
    require(
        any(name.endswith("libtorch_cpu.dylib") for name in result),
        "CPU Torch native library absent",
    )
    return result


def torch_proof(value):
    import torch

    manifest, roots = value["manifest"], value["roots"]
    baseline = Path(roots["baseline"]) / manifest["owned_frameworks"]["runtime_receipt"]
    require(
        digest(baseline)
        == manifest["bindings"]["baseline"][manifest["owned_frameworks"]["runtime_receipt"]],
        "CPU owned runtime receipt changed",
    )
    receipt = read(baseline)
    selected = receipt["frameworks"]["pytorch"]
    require(
        selected["selection"] == manifest["owned_frameworks"]["frameworks"]["pytorch"],
        "CPU loaded an unselected Torch distribution",
    )
    native_manifest = Path(selected["manifest"]["path"])
    require(
        digest(native_manifest)
        == selected["manifest"]["sha256"]
        == selected["selection"]["runtime_manifest_sha256"],
        "CPU native manifest differs",
    )
    files = read(native_manifest)["files"]
    payload = {
        str(Path(selected["payload_root"]) / name): record["sha256"]
        for name, record in files.items()
    }
    helper = file_at(roots["source"], HELPER, manifest["bindings"]["source"][HELPER])
    verify = runpy.run_path(str(helper))["verify_loaded_framework_modules"]
    origins = verify(
        "pytorch", payload, {"files": manifest["bindings"]["runtime"]}, roots["runtime"]
    )
    require(
        torch.version.git_version == selected["selection"]["base"]
        and not torch.ops.loaded_libraries,
        "CPU Torch source or custom-operator state differs",
    )
    project = {}
    for name, module in dict(sys.modules).items():
        if name != "transformer" and not name.startswith("transformer."):
            continue
        path = Path(getattr(module, "__file__", "")).resolve(strict=True)
        require(
            path.is_relative_to(Path(roots["cpu"]).resolve()),
            "Current project shadowed frozen CPU model",
        )
        relative = path.relative_to(Path(roots["cpu"]).resolve()).as_posix()
        require(
            manifest["bindings"]["cpu"].get(relative) == digest(path), "Unbound frozen CPU module"
        )
        project[name] = {"path": relative, "sha256": digest(path)}
    libraries = native_libraries(
        payload, Path(roots["runtime"]).resolve(), manifest["bindings"]["runtime"]
    )
    return {
        "framework": origins,
        "native_libraries": libraries,
        "cpu_modules": project,
        "source_revision": torch.version.git_version,
        "core": str(Path(torch._C.__file__).resolve()),
        "operator_libraries": sorted(torch.ops.loaded_libraries),
        "host_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
    }


def execute(path, expected_sha256):
    value = authenticate(path, expected_sha256)
    proof_path = Path(value["proof"])
    require(not proof_path.exists() and not proof_path.is_symlink(), "CPU proof collision")
    proof = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "kind": value["kind"],
        "engine": value["engine"],
        "manifest_identity_sha256": value["manifest_identity_sha256"],
        "runtime_identity_sha256": value["runtime_identity_sha256"],
        "authority_sha256": expected_sha256,
        "complete": False,
        "integrity_after": False,
        "input_sha256": value["input_sha256"],
        "expected_steps": value["expected_steps"],
        "expected_rejection": value.get("expected_rejection"),
        "arguments": value["arguments"],
        "runtime_root": value["roots"]["runtime"],
        "pid": os.getpid(),
        "started_unix": time.time(),
        "entrypoint": value["entrypoint"],
        "source_sha256": value["manifest"]["bindings"]["source"],
        "cpu_source_sha256": value["manifest"]["bindings"]["cpu"],
        "timing_scope": "Outside frozen CPU timing",
    }
    try:
        require(not Path(value["output"]).exists(), "CPU output collision")
        proof["origins_before"] = torch_proof(value)
        entry = value["entrypoint"]
        script = file_at(value["roots"][entry["role"]], entry["path"], entry["sha256"])
        sys.argv = [str(script), *value["arguments"]]
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as error:
            require(error.code in (0, None), "Frozen CPU entrypoint failed")
        except ValueError as error:
            if value.get("expected_rejection") != str(error):
                raise
            proof["rejection"] = {"type": "ValueError", "message": str(error)}
        require(
            ("rejection" in proof) == (value.get("expected_rejection") is not None),
            "CPU operation did not produce its prescribed rejection",
        )
        proof["origins_after"] = torch_proof(value)
        authenticate(path, expected_sha256)
        output = Path(value["output"])
        if "rejection" in proof:
            require(not output.exists(), "Rejected CPU operation emitted a result")
            proof["output_sha256"] = {}
        elif value["kind"] in ("fit", "tie"):
            proof["output_sha256"] = {
                p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()
            }
            performance = struct.unpack("<8d", (output / "performance.f64").read_bytes())
            require(performance[6] == value["expected_steps"], "CPU optimizer step count differs")
            proof["actual_steps"] = performance[6]
        else:
            proof["output_sha256"] = {output.name: digest(output)}
        proof.update(complete=True, integrity_after=True)
    except BaseException:
        proof["error"] = traceback.format_exc()
        raise
    finally:
        proof["ended_unix"] = time.time()
        proof_path.parent.mkdir(parents=True, exist_ok=True)
        with proof_path.open("x") as stream:
            json.dump(proof, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return proof


def prepare_command(
    manifest,
    roots,
    engine,
    kind,
    arguments,
    output,
    *,
    admission_receipt=None,
    expected_rejection=None,
):
    """Parent-only adapter; original CPU executable and all numerical files stay unchanged."""
    from transformer.gpu_study import python_path, verify_bindings, write_new
    from transformer.gpu_study_context import python_command

    require(manifest["protocol"] == PROTOCOL, "Owned CPU launcher needs owned manifest")
    verify_bindings(manifest, roots)
    output = Path(output).resolve()
    role, relative = ENTRYPOINTS[kind]
    value = {
        "schema": "gpu-owned-cpu-authority-v1",
        "protocol": PROTOCOL,
        "manifest": manifest,
        "manifest_identity_sha256": identity(manifest),
        "roots": {key: str(Path(path).resolve()) for key, path in roots.items()},
        "kind": kind,
        "engine": engine,
        "arguments": list(arguments),
        "output": str(output),
        "runtime_identity_sha256": identity(manifest["bindings"]["runtime"]),
        "entrypoint": {
            "role": role,
            "path": relative,
            "sha256": manifest["bindings"][role][relative],
        },
        "pythonpath": python_path(manifest, roots, cpu=True),
        "expected_steps": 1024 if kind == "fit" else 2 if kind == "tie" else None,
        "expected_rejection": expected_rejection,
        "admission_receipt": str(Path(admission_receipt).resolve())
        if admission_receipt is not None
        else None,
    }
    if kind in ("fit", "tie"):
        value["fixture"] = arguments[0]
        inputs = list(Path(value["fixture"]).iterdir())
    elif kind == "admission":
        value["operands"] = arguments[3]
        inputs = [Path(value["operands"])]
    else:
        value["fixture"], value["checkpoint"] = arguments[3], arguments[5]
        inputs = [*Path(value["fixture"]).iterdir(), Path(value["checkpoint"])]
    value["input_sha256"] = {
        str(path.resolve()): digest(path) for path in sorted(inputs) if path.is_file()
    }
    value["proof"] = str(
        output / "owned-cpu-proof.json"
        if kind in ("fit", "tie")
        else Path(str(output) + ".owned-cpu-proof.json")
    )
    validate_authority(value)
    authority_path = Path(str(output) + ".owned-authority.json")
    write_new(authority_path, value)
    return python_command(manifest, output, roots=roots, cpu=True) + [
        str(Path(roots["source"]).resolve() / LAUNCHER),
        "--authority",
        str(authority_path),
        "--authority-sha256",
        digest(authority_path),
    ]


def verify_proof(
    path,
    manifest,
    *,
    engine,
    kind,
    output_hashes,
    input_hashes=None,
    expected_rejection=None,
):
    """Archive-safe checks; equality is recomputed against authenticated artifacts."""
    proof = read(path)
    require(
        proof.get("schema") == SCHEMA
        and proof.get("protocol") == PROTOCOL
        and proof.get("engine") == engine
        and proof.get("kind") == kind,
        "Wrong owned CPU execution proof",
    )
    require(
        proof.get("manifest_identity_sha256") == identity(manifest)
        and proof.get("runtime_identity_sha256") == identity(manifest["bindings"]["runtime"])
        and proof.get("source_sha256") == manifest["bindings"]["source"]
        and proof.get("cpu_source_sha256") == manifest["bindings"]["cpu"],
        "Stale owned CPU proof",
    )
    role, relative = ENTRYPOINTS[kind]
    require(
        proof.get("entrypoint")
        == {"role": role, "path": relative, "sha256": manifest["bindings"][role][relative]},
        "CPU proof entrypoint changed",
    )
    require(
        proof.get("complete") is True
        and proof.get("integrity_after") is True
        and "error" not in proof
        and proof.get("output_sha256") == output_hashes,
        "CPU execution or artifacts changed",
    )
    require(
        proof.get("expected_rejection") == expected_rejection,
        "CPU rejection contract changed",
    )
    if expected_rejection is not None:
        require(
            kind == "admission"
            and expected_rejection in REJECTIONS
            and output_hashes == {}
            and proof.get("rejection") == {"type": "ValueError", "message": expected_rejection},
            "CPU expected rejection was not authenticated",
        )
    else:
        require("rejection" not in proof, "CPU successful operation unexpectedly rejected")
    if input_hashes is not None:
        inputs = proof.get("input_sha256", {})
        require(
            len({Path(path).name for path in inputs}) == len(inputs)
            and {Path(path).name: sha for path, sha in inputs.items()} == input_hashes,
            "CPU checkpoint or fixture inputs changed",
        )
    expected_steps = 1024 if kind == "fit" else 2 if kind == "tie" else None
    require(proof.get("expected_steps") == expected_steps, "CPU step contract changed")
    if expected_steps is not None:
        require(proof.get("actual_steps") == expected_steps, "CPU optimizer count differs")
    runtime = manifest["bindings"]["runtime"]
    selection = manifest["owned_frameworks"]["frameworks"]["pytorch"]
    for key in ("origins_before", "origins_after"):
        row = proof[key]
        require(
            row.get("source_revision") == selection["base"] and row.get("operator_libraries") == [],
            "CPU Torch source/override state differs",
        )
        core = Path(row["core"]).relative_to(proof["runtime_root"]).as_posix()
        packages = manifest["bindings"]["runtime_layout"]["packages"]["torch"]
        package_roots = [
            Path(item).parent if item.endswith(".py") else Path(item) for item in packages
        ]
        require(
            core in runtime
            and core.endswith(".so")
            and "/torch/_C." in core
            and any(Path(row["core"]).is_relative_to(root) for root in package_roots),
            "CPU proof core is outside selected shared Torch package",
        )
        framework = row.get("framework", {})
        module_origins = framework.get("module_origins", {})
        require(
            set(framework)
            == {
                "schema",
                "framework",
                "module_origins",
                "synthetic_origins",
                "native_aliases",
                "namespace_packages",
                "module_count",
            }
            and framework.get("schema") == "gpu-owned-loaded-framework-origins-v1"
            and framework.get("framework") == "pytorch"
            and isinstance(module_origins, dict)
            and module_origins.get("torch._C") == row["core"]
            and "torch" in module_origins,
            "CPU proof lacks actual loaded Torch module origins",
        )
        synthetic, aliases, namespaces = (
            framework["synthetic_origins"],
            framework["native_aliases"],
            framework["namespace_packages"],
        )
        require(
            isinstance(synthetic, dict)
            and isinstance(aliases, dict)
            and namespaces == {}
            and set(synthetic) == {"torch.ops", "torch.classes"},
            "CPU proof omits registered Torch synthetic namespaces",
        )
        require(
            type(framework["module_count"]) is int
            and framework["module_count"] == len(module_origins) + len(synthetic) + len(aliases)
            and len(set(module_origins) | set(synthetic) | set(aliases))
            == framework["module_count"],
            "CPU proof framework module inventory is incomplete or duplicated",
        )
        package = Path(module_origins["torch"]).parent
        for name, backing_name, label, cls in (
            ("torch.ops", "torch._ops", "_ops.py", "_Ops"),
            ("torch.classes", "torch._classes", "_classes.py", "_Classes"),
        ):
            absolute = str(package / label)
            relative = Path(absolute).relative_to(proof["runtime_root"]).as_posix()
            require(
                relative in runtime
                and module_origins.get(backing_name) == absolute
                and synthetic[name]
                == {
                    "backing_module": backing_name,
                    "backing_file": absolute,
                    "backing_sha256": runtime[relative],
                    "exact_class": backing_name + "." + cls,
                    "singleton_identity_verified": True,
                },
                "CPU proof synthetic namespace backing differs from selected runtime",
            )
        for name, alias in aliases.items():
            require(
                name.startswith("torch._C.")
                and alias == {"extension": row["core"], "sha256": runtime[core]},
                "CPU proof native alias differs from selected extension",
            )
        for absolute in module_origins.values():
            relative = Path(absolute).relative_to(proof["runtime_root"]).as_posix()
            require(
                relative in runtime
                and any(Path(absolute).is_relative_to(root.parent) for root in package_roots),
                "CPU proof module origin differs from selected runtime",
            )
        libraries = row.get("native_libraries", {})
        require(
            libraries
            and any(name.endswith("libtorch_cpu.dylib") for name in libraries)
            and all(runtime.get(name) == sha for name, sha in libraries.items()),
            "CPU proof native libraries differ",
        )
        modules = row.get("cpu_modules", {})
        require(
            all(
                manifest["bindings"]["cpu"].get(value["path"]) == value["sha256"]
                for value in modules.values()
            ),
            "CPU proof used changed numerical modules",
        )
    after = proof["origins_after"]
    require(
        {"transformer", "transformer.reference", "transformer.spec"} <= set(after["cpu_modules"]),
        "CPU proof omits frozen numerical modules",
    )
    require(
        after.get("deterministic") is True
        and after.get("warn_only") is False
        and after.get("host_threads") == 1,
        "CPU execution lost strict controls",
    )
    if kind in ("fit", "tie"):
        require(after.get("interop_threads") == 1, "CPU training interop threads differ")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard", action="store_true")
    parser.add_argument("--authority", required=True)
    parser.add_argument("--authority-sha256", required=True)
    args = parser.parse_args()
    if args.guard:
        guard(args.authority, args.authority_sha256)
    else:
        execute(args.authority, args.authority_sha256)
