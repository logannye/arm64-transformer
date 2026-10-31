"""Fresh complete MLX suites for an authenticated maintained joint runtime.

Preparation authenticates files only. The separate run command launches two
serial fresh workers; no framework installation, compilation, or override occurs.
"""

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import json
import math
import os
import re
import runpy
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

SCHEMA = "gpu-owned-suite-execution-v1"
PLAN_SCHEMA = "gpu-owned-mlx-suite-plan-v1"
PROTOCOL = "gpu-v4-owned-five-engine-v1"
PRODUCER = "integration/gpu-owned/owned_mlx_suites.py"
INVENTORY_SHA = "cff557d2b1d68857bc488e77e6672fb0fb386409e88ad6ca4e7fa31bd862d852"
COMPAT_INVENTORY_SHA = "7e57d93ce0969f7c3be7ca7c7d137b7f0fdfeeda0b21547d7e7b1302d9fbd41a"
SUITE_CONTRACTS = ("original", "torch-dev-compat", "torch-dev-compat-numpy2")
SUITES = ("mlx_cpp", "mlx_python")
SUBPROCESS_AST_SHA = "90a6406e042b09e83ed671864ad0cf30cefa2e6c4e148ca60d24ac26cbb6b433"
BOOTSTRAP_FILES = {
    "transformer/__init__.py",
    "transformer/gpu_owned_backend.py",
    "transformer/gpu_owned_origins.py",
    "transformer/gpu_study.py",
    "transformer/gpu_study_runtime.py",
    "transformer/gpu_study_context.py",
    "transformer/gpu_environment.py",
    "transformer/gpu_mlx_compile.py",
    PRODUCER,
}
LIMITS = {
    "mlx_cpp_seconds": 1800,
    "mlx_python_seconds": 3600,
    "wrapper_grace_seconds": 600,
    "output_bytes": 2 * 1024**3,
    "minimum_free_bytes": 5 * 1024**3,
}
CONTROLS = {
    "MLX_ENABLE_TF32": "0",
    "MLX_SDPA_VJP_FALLBACK": "0",
    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    "PYTORCH_MPS_FAST_MATH": "0",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "METAL_DEBUG_ERROR_MODE": "0",
    "METAL_DEVICE_WRAPPER_TYPE": "1",
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "SKBUILD_EDITABLE_REBUILD": "false",
}
ABSENT = (
    "MLX_DISABLE_COMPILE",
    "MLX_METALLIB_PATH",
    "DEVICE",
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTORCH_MPS_PREFER_METAL",
    "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH",
    "DYLD_INSERT_LIBRARIES",
    "PYTEST_ADDOPTS",
    "PYTEST_PLUGINS",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    "PYTORCH_OPINFO_SAMPLE_INPUT_INDEX",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
    )


def binding(path, expected=None):
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path and not path.is_symlink(), "Noncanonical file")
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), "Nonregular file")
    result = {"path": str(path), "sha256": sha256(path), "bytes": before.st_size}
    after = path.stat()
    require(
        all(
            getattr(before, key) == getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mode", "st_mtime_ns", "st_ctime_ns")
        ),
        "File changed during hashing",
    )
    if expected is not None:
        require(result == expected, "File binding changed")
    return result


def pinned(path, expected):
    value = binding(path)
    require(value["sha256"] == expected, "External SHA256 differs")
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def controls(base):
    allowed = {
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "USER",
        "LOGNAME",
        "SHELL",
        "SYSTEMROOT",
    }
    result = {name: value for name, value in base.items() if name in allowed}
    result.update(CONTROLS)
    return result


def check_controls(*, descendant=False):
    allowed = {name for name in CONTROLS if name.startswith("PYTHON")} | {"PYTHONPYCACHEPREFIX"}
    require(
        not any(name.startswith("PYTHON") and name not in allowed for name in os.environ),
        "Unrecognized Python interpreter override",
    )
    if "PYTHONPYCACHEPREFIX" in os.environ:
        require(
            os.environ["PYTHONPYCACHEPREFIX"] == sys.pycache_prefix,
            "Inherited bytecode prefix differs",
        )
    require(all(os.environ.get(k) == v for k, v in CONTROLS.items()), "Runtime controls differ")
    require(
        all(k not in os.environ for k in ABSENT if not (descendant and k == "DEVICE")),
        "Runtime or test override present",
    )
    if descendant:
        require(os.environ.get("DEVICE") == "gpu", "Unchanged descendant must request GPU")


def suite_inventory(suite_contract="original"):
    require(suite_contract in SUITE_CONTRACTS, "Unknown MLX suite contract")
    filename, expected_sha = (
        ("owned-suite-inventory-v1.json", INVENTORY_SHA)
        if suite_contract == "original"
        else ("owned-suite-inventory-mlx-compat-v1.json", COMPAT_INVENTORY_SHA)
    )
    path = Path(__file__).with_name(filename).resolve()
    row = pinned(path, expected_sha)
    value = read_json(path)
    require(
        value["schema"] == "gpu-owned-suite-inventory-v1"
        and value["protocol"] == PROTOCOL
        and value["formal_eligible"] is False,
        "Wrong frozen suite inventory",
    )
    return value["mlx"], row


def numpy_version(plan):
    contract = plan.get("suite_contract", "original")
    require(contract in SUITE_CONTRACTS, "Unknown MLX suite contract")
    return "2.2.6" if contract == "torch-dev-compat-numpy2" else "1.26.4"


def python_test_driver(runtime, bundle, paths=None):
    """Bind the upstream config and complete installed test dependencies before collection."""
    paths = sys.path if paths is None else paths
    dependencies = {}
    for module, distribution, version in (
        ("pytest", "pytest", "9.1.1"),
        ("tensorflow", "tensorflow", "2.20.0"),
        ("ml_dtypes", "ml_dtypes", "0.5.3"),
    ):
        spec = importlib.machinery.PathFinder.find_spec(module, paths)
        require(
            spec is not None and isinstance(spec.origin, str), "Missing test dependency: " + module
        )
        source = binding(Path(spec.origin).resolve(strict=True))
        require(
            runtime["files"].get(source["path"].lstrip("/")) == source["sha256"],
            "Unbound test dependency: " + module,
        )
        root = Path(source["path"]).parent.parent
        matches = [
            d
            for d in importlib.metadata.distributions(path=[str(root)])
            if d.metadata["Name"].replace("-", "_").lower() == distribution
        ]
        require(
            len(matches) == 1 and matches[0].version == version,
            "Test dependency version differs: " + module,
        )
        metadata = binding(Path(matches[0]._path) / "METADATA")
        require(
            runtime["files"].get(metadata["path"].lstrip("/")) == metadata["sha256"],
            "Unbound test dependency metadata: " + module,
        )
        dependencies[module] = dict(version=version, source=source, metadata=metadata)
    config = binding(bundle / "source/pyproject.toml")
    require(
        runtime["files"].get(config["path"].lstrip("/")) == config["sha256"],
        "Unbound upstream pytest configuration",
    )
    return dict(root=str(bundle / "source"), config=config, dependencies=dependencies)


def python_arguments(bundle, output):
    return [
        str(bundle / "source/python/tests"),
        "--rootdir=" + str(bundle / "source"),
        "-c",
        str(bundle / "source/pyproject.toml"),
        "-q",
        "-ra",
        "--durations=20",
        "--junitxml=" + str(output / "python.xml"),
        "-o",
        "cache_dir=" + str(output / "pytest-cache"),
    ]


def verify_project_before_import(receipt):
    project = receipt["project"]
    require(set(project) == {"root", "files"}, "Missing complete project source authority")
    root = Path(project["root"])
    require(root.resolve(strict=True) == root, "Noncanonical project root")
    require(
        project["files"].get(PRODUCER) == sha256(Path(__file__).resolve()),
        "This producer is not bound by the runtime receipt",
    )
    require(BOOTSTRAP_FILES <= set(project["files"]), "Unbound bootstrap module")
    for relative, expected in project["files"].items():
        require(
            not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            "Unsafe project source path",
        )
        pinned(root / relative, expected)
    require((root / PRODUCER).resolve() == Path(__file__).resolve(), "Producer root differs")
    for name, module in tuple(sys.modules.items()):
        if name == "transformer" or name.startswith("transformer."):
            path = Path(getattr(module, "__file__", "")).resolve(strict=True)
            require(
                path.is_relative_to(root)
                and project["files"].get(str(path.relative_to(root))) == sha256(path),
                "Foreign preloaded project module",
            )
    return root


def output_path(output, project):
    require(
        output.is_absolute()
        and output.resolve(strict=False) == output
        and not output.exists()
        and output.is_relative_to(Path(project["root"]).parents[1] / "work/gpu-v4-owned"),
        "Require a fresh canonical owned-development output path",
    )
    return output


def verify_import_resolution(runtime, paths):
    for name in ("numpy", "torch", "mlx"):
        spec = importlib.machinery.PathFinder.find_spec(name, paths)
        require(spec is not None, "Required framework import is absent")
        locations = ([spec.origin] if spec.origin else []) + list(
            spec.submodule_search_locations or []
        )
        resolved = [str(Path(path).resolve(strict=True)) for path in locations]
        require(
            resolved == runtime["layout"]["packages"][name],
            "Actual import resolution differs from authenticated runtime: " + name,
        )


def verify_loaded_files(runtime, authority, project, modules=None):
    result = {}
    root = Path(authority.root)
    project_root = Path(project["root"])
    for name, module in tuple((sys.modules if modules is None else modules).items()):
        try:
            source = vars(module).get("__file__")
        except TypeError:
            source = None
        if not isinstance(source, str):
            continue
        path = Path(source).resolve(strict=True)
        relative = str(path.relative_to(root)) if path.is_relative_to(root) else None
        expected = runtime["files"].get(relative)
        if expected is None and path.is_relative_to(project_root):
            expected = project["files"].get(str(path.relative_to(project_root)))
        require(
            expected is not None and sha256(path) == expected,
            "Loaded Python dependency is outside the frozen runtime/project: " + name,
        )
        result[name] = {"path": str(path), "sha256": expected}
    return result


def bootstrap(plan):
    require(
        sys.flags.safe_path
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
        and not sys.flags.isolated
        and not sys.flags.optimize
        and sys.flags.hash_randomization == 0
        and os.environ.get("PYTHONHASHSEED") == "0",
        "Use canonical CPython -P -S -B with fixed PYTHONHASHSEED=0",
    )
    require(
        isinstance(sys.pycache_prefix, str)
        and Path(sys.pycache_prefix).is_absolute()
        and not Path(sys.pycache_prefix).exists(),
        "Fresh nonexistent bytecode prefix required",
    )
    require(
        not any(
            name.split(".")[0] in {"torch", "torchgen", "functorch", "mlx", "numpy"}
            for name in sys.modules
        ),
        "Framework was imported before authentication",
    )
    require("_virtualenv" not in sys.modules, "Editable/site hook executed")
    for value in plan["authority_inputs"].values():
        binding(value["path"], value)
    receipt = read_json(plan["authority_inputs"]["receipt"]["path"])
    project = verify_project_before_import(receipt)
    inventory = read_json(plan["authority_inputs"]["inventory"]["path"])
    selection = read_json(plan["authority_inputs"]["selection"]["path"])
    runtime_root = Path(plan["runtime_root"])
    require(
        Path(sys.executable).resolve(strict=True)
        == Path(inventory["layout"]["interpreter_resolved"]),
        "Selected CPython executable differs",
    )
    # No site processing, executable .pth file or editable finder is invoked.
    require(
        all(
            not Path(name).is_absolute() and ".." not in Path(name).parts
            for name in inventory["python_paths"]
        ),
        "Unsafe supplemental import root",
    )
    roots = [str(runtime_root / name) for name in inventory["python_paths"]]
    require(all(Path(name).resolve(strict=True).is_dir() for name in roots), "Missing import root")
    initial = list(sys.path)
    require(
        not any("site-packages" in path or "dist-packages" in path for path in initial),
        "Unexpected initial site directory",
    )
    # Authenticate the dependency bytes before making their import roots visible.
    # The authority itself imports only bound project code and the base stdlib;
    # an unverified dependency directory must not shadow those stdlib imports.
    sys.path[:] = list(dict.fromkeys([str(project), *initial]))
    backend = importlib.import_module("transformer.gpu_owned_backend")
    authority = backend.authenticate_runtime(
        runtime_root,
        inventory,
        selection,
        plan["authority_inputs"]["receipt"]["path"],
        plan["authority_inputs"]["receipt"]["sha256"],
    )
    backend.verify_loaded_project_modules(receipt["project"])
    sys.path[:] = list(dict.fromkeys([str(project), *roots, *initial]))
    verify_import_resolution(inventory, sys.path)
    return authority, backend, inventory, selection


def test_source(authority, inventory):
    receipt, _ = authority.verify()
    row = receipt["frameworks"]["mlx"]
    bundle = Path(row["payload_root"])
    manifest = read_json(row["manifest"]["path"])
    actual = {
        name: value["sha256"]
        for name, value in manifest["files"].items()
        if name.startswith(("source/python/tests/", "source/tests/"))
    }
    require(actual == inventory["test_source_sha256"], "Complete MLX test-source inventory differs")
    for name, expected in actual.items():
        pinned(bundle / name, expected)
    required = []
    for filename in ("test_gather_grad.py", "test_fast_sdpa_training.py"):
        tree = ast.parse((bundle / "source/python/tests" / filename).read_text())
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                required.extend(
                    [node.name, method.name]
                    for method in node.body
                    if isinstance(method, ast.FunctionDef) and method.name.startswith("test_")
                )
    require(
        required == inventory["required_gpu_regressions"] and len(required) == 27,
        "Required GPU regression source inventory changed",
    )
    return bundle, actual


def normalized_ast(node):
    if isinstance(node, ast.AST):
        return {
            "node": type(node).__name__,
            "fields": {
                name: normalized_ast(value)
                for name, value in ast.iter_fields(node)
                if value is not None and value != []
            },
        }
    if isinstance(node, list):
        return [normalized_ast(value) for value in node]
    return node


def subprocess_contract(bundle, expected):
    rows = []
    for name in sorted(expected["test_source_sha256"]):
        if not name.endswith(".py"):
            continue
        tree = ast.parse((bundle / name).read_text())
        for node in ast.walk(tree):
            require(
                not (isinstance(node, ast.ImportFrom) and node.module == "subprocess"),
                "Undeclared subprocess import alias",
            )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"
            ):
                rows.append(
                    {
                        "source": name,
                        "line": node.lineno,
                        "operation": node.func.attr,
                        "ast": normalized_ast(node),
                    }
                )
    require(
        hashlib.sha256(canonical(rows)).hexdigest() == SUBPROCESS_AST_SHA,
        "Frozen whole-suite subprocess call shapes changed",
    )
    return {
        "mode": "explicit_authenticated_interpreter_launcher_conformance_only",
        "calls": rows,
        "calls_sha256": SUBPROCESS_AST_SHA,
        "python_invocation": ["source/python/tests/test_gather_grad.py", "--repeatability-worker"],
        "required_python_children": 2,
        "upstream_child_timeout_seconds": 120,
        "system_query": "sysctl -n machdep.cpu.brand_string",
        "subprocess_function_monkeypatch": False,
        "test_source_modified": False,
        "measured_baseline_launcher": False,
    }


def launcher_text(plan, plan_path, plan_sha, parent_output, parent_pid, interpreter):
    command = [
        interpreter,
        "-P",
        "-S",
        "-B",
        str(Path(__file__).resolve()),
        "descendant",
        "--plan",
        str(plan_path),
        "--plan-sha256",
        plan_sha,
        "--parent-output",
        str(parent_output),
        "--parent-pid",
        str(parent_pid),
        "--",
    ]
    return "#!/bin/sh\nexec " + " ".join(shlex.quote(value) for value in command) + ' "$@"\n'


def descendant_invocation(plan, upstream):
    invocation = [
        str(Path(plan["bundle"]) / plan["descendant_contract"]["python_invocation"][0]),
        plan["descendant_contract"]["python_invocation"][1],
    ]
    require(upstream == invocation, "Undeclared upstream Python subprocess invocation")
    return invocation


def descendant(plan_path, plan_sha, parent_output, parent_pid, upstream):
    # The unchanged test spawns this launcher; stdout belongs entirely to that test.
    require(os.getppid() == parent_pid, "Unexpected descendant parent PID")
    require(
        parent_output.is_dir() and parent_output.resolve(strict=True) == parent_output,
        "Invalid parent suite output",
    )
    require(
        parent_output.is_relative_to(Path(__file__).resolve().parents[4] / "work/gpu-v4-owned"),
        "Descendant output is outside the owned workspace",
    )
    children = parent_output / "descendants"
    require(
        children.is_dir()
        and children.resolve(strict=True) == children
        and not children.is_symlink(),
        "Parent did not declare canonical descendant routing",
    )
    output = Path(tempfile.mkdtemp(prefix="child-" + str(os.getpid()) + "-", dir=children))
    sys.pycache_prefix = str(output / "unused-bytecode-cache")
    os.environ["PYTHONPYCACHEPREFIX"] = sys.pycache_prefix
    record = {
        "schema": "gpu-owned-mlx-suite-descendant-v1",
        "protocol": PROTOCOL,
        "producer": PRODUCER,
        "plan_sha256": plan_sha,
        "plan_path": str(plan_path),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "parent_output": str(parent_output),
        "bytecode_prefix": sys.pycache_prefix,
        "started_unix": time.time(),
        "formal_admission_opened": False,
        "formal_fits": 0,
        "complete": False,
        "passed": False,
        "returncode": 2,
    }
    authority = backend = runtime = expected = None
    try:
        check_controls(descendant=True)
        plan, authority, backend, runtime, selection, expected = authenticate_plan(
            plan_path, plan_sha
        )
        invocation = descendant_invocation(plan, upstream)
        launcher = parent_output / "archive-python"
        text = launcher_text(
            plan, plan_path, plan_sha, parent_output, parent_pid, runtime["layout"]["interpreter"]
        )
        require(
            launcher.read_text() == text and launcher.stat().st_mode & 0o111,
            "Authenticated interpreter launcher changed",
        )
        record.update(
            invocation=invocation,
            launcher=binding(launcher),
            runtime_receipt_sha256=authority.receipt_sha256,
            framework_manifest_sha256=selection["mlx"]["runtime_manifest_sha256"],
            runtime_root=str(authority.root),
        )
        verified = authority.verify()
        receipt, payloads = verified
        record["runtime_identity_sha256_before"] = receipt["runtime_identity_sha256"]
        record["source_sha256_before"] = receipt["project"]["files"]
        import numpy as np

        require(
            np.__version__ == numpy_version(plan)
            and str(Path(np.__file__).resolve()) in runtime["layout"]["packages"]["numpy"],
            "Descendant NumPy version/origin differs",
        )
        record["numpy"] = {"version": np.__version__, "path": str(Path(np.__file__).resolve())}
        import mlx.core as mx

        require(
            mx.default_device() == mx.gpu and mx.metal.is_available(), "Descendant GPU required"
        )
        compiler = importlib.import_module("transformer.gpu_mlx_compile")
        record["compilation_before"] = backend.validate_compilation_probe(
            compiler.verify_compilation(mx)
        )
        record["native_origins_before"] = backend._loaded_identity("mlx", authority, verified)
        record["loaded_python_files_before"] = verify_loaded_files(
            runtime, authority, receipt["project"]
        )
        sys.path.append(str(Path(invocation[0]).parent))
        verify_import_resolution(runtime, sys.path)
        old_argv = list(sys.argv)
        sys.argv[:] = invocation
        try:
            runpy.run_path(invocation[0], run_name="__main__")
            code = 0
        except SystemExit as error:
            code = error.code if isinstance(error.code, int) else 0 if error.code is None else 1
        finally:
            sys.argv[:] = old_argv
        record.update(returncode=code, complete=True, passed=code == 0)
    except BaseException:
        record["error"] = traceback.format_exc()
    finally:
        if authority is not None:
            try:
                check_controls(descendant=True)
                final = authority.verify()
                record["runtime_identity_sha256_after"] = final[0]["runtime_identity_sha256"]
                record["source_sha256_after"] = final[0]["project"]["files"]
                backend.verify_loaded_project_modules(final[0]["project"])
                verify_import_resolution(runtime, sys.path)
                record["loaded_python_files_after"] = verify_loaded_files(
                    runtime, authority, final[0]["project"]
                )
                record["native_origins_after"] = backend._loaded_identity("mlx", authority, final)
                record["compilation_after"] = backend.validate_compilation_probe(
                    compiler.verify_compilation(mx)
                )
                require(
                    record["runtime_identity_sha256_before"]
                    == record["runtime_identity_sha256_after"]
                    and record["source_sha256_before"] == record["source_sha256_after"]
                    and not Path(sys.pycache_prefix).exists(),
                    "Descendant postflight changed",
                )
                record["integrity_after"] = True
            except BaseException:
                record.update(
                    complete=False,
                    passed=False,
                    returncode=2,
                    integrity_after=False,
                    integrity_error=traceback.format_exc(),
                )
        record["ended_unix"] = time.time()
        write_json(output / "receipt.json", record)
    return int(record["returncode"])


def validate_descendant_proofs(
    value, runtime, runtime_root, project, bundle, *, expected_numpy="1.26.4"
):
    """Check retained origin maps without opening live paths (archive-safe).

    The caller authenticates runtime/project maps and raw receipt bytes first.
    Returned absolute file bindings can additionally be rehashed by a live audit.
    """
    root, project_root, bundle = Path(runtime_root), Path(project["root"]), Path(bundle)
    require(value.get("runtime_root") == str(root), "Descendant runtime root differs")
    files = {}

    def bound(path, expected=None, *, native=False):
        require(isinstance(path, str), "Missing retained import path")
        source = Path(path)
        require(
            source.is_absolute() and ".." not in source.parts and str(source) == path,
            "Noncanonical retained import path",
        )
        sha = None
        if source.is_relative_to(root):
            sha = runtime["files"].get(str(source.relative_to(root)))
        if sha is None and source.is_relative_to(project_root) and not native:
            sha = project["files"].get(str(source.relative_to(project_root)))
        require(
            isinstance(sha, str)
            and re.fullmatch(r"[0-9a-f]{64}", sha)
            and (expected is None or sha == expected),
            "Retained import is outside the authenticated runtime/project",
        )
        if native:
            require(source.is_relative_to(bundle / "runtime"), "Foreign descendant native image")
        files[path] = sha
        return sha

    numpy = value.get("numpy", {})
    require(
        numpy.get("version") == expected_numpy
        and numpy.get("path") in runtime["layout"]["packages"]["numpy"],
        "Descendant NumPy dependency differs",
    )
    bound(numpy["path"])
    for phase in ("before", "after"):
        imported = value["loaded_python_files_" + phase]
        require(isinstance(imported, dict) and imported, "Missing descendant import map")
        for module, row in imported.items():
            require(
                isinstance(module, str) and set(row) == {"path", "sha256"}, "Malformed import map"
            )
            bound(row["path"], row["sha256"])
        require(
            imported.get("numpy", {}).get("path") == numpy["path"],
            "Descendant NumPy loaded-file proof differs",
        )
        native = value["native_origins_" + phase]
        require(
            set(native) == {"module_origins", "framework_module_origins", "native_libraries"}
            and set(native["framework_module_origins"]) == {"mlx"},
            "Malformed descendant native-origin proof",
        )
        origins = native["framework_module_origins"]["mlx"]
        require(
            set(origins)
            == {
                "schema",
                "framework",
                "module_origins",
                "synthetic_origins",
                "native_aliases",
                "namespace_packages",
                "module_count",
            }
            and origins["schema"] == "gpu-owned-loaded-framework-origins-v1"
            and origins["framework"] == "mlx"
            and origins["synthetic_origins"] == {},
            "Malformed descendant framework module schema",
        )
        maps = [
            origins[key]
            for key in (
                "module_origins",
                "synthetic_origins",
                "native_aliases",
                "namespace_packages",
            )
        ]
        require(
            all(isinstance(item, dict) for item in maps)
            and type(origins["module_count"]) is int
            and origins["module_count"] == sum(map(len, maps))
            and len(set().union(*maps)) == origins["module_count"],
            "Descendant framework module counts overlap or differ",
        )
        core = origins["module_origins"].get("mlx.core")
        require(
            isinstance(core, str)
            and Path(core).parent == bundle / "runtime/python/mlx"
            and Path(core).name.startswith("core.")
            and core.endswith(".so")
            and native["module_origins"] == [core]
            and imported.get("mlx.core", {}).get("path") == core,
            "Selected MLX core origin is absent",
        )
        core_sha = bound(core, native=True)
        for module, path in origins["module_origins"].items():
            require(module == "mlx" or module.startswith("mlx."), "Foreign framework module")
            bound(path, native=True)
        for module, row in origins["native_aliases"].items():
            require(
                module.startswith("mlx.core.") and row == {"extension": core, "sha256": core_sha},
                "Unbound native extension alias",
            )
        namespaces = origins["namespace_packages"]
        require(set(namespaces) <= {"mlx"}, "Unknown MLX namespace")
        if "mlx" in namespaces:
            require(
                namespaces["mlx"]
                == {"locations": runtime["layout"]["packages"]["mlx"], "extension": core}
                and runtime["layout"]["packages"]["mlx"] == [str(bundle / "runtime/python/mlx")],
                "MLX namespace origin differs",
            )
        else:
            require("mlx" in origins["module_origins"], "Missing top-level MLX origin")
        libraries = native["native_libraries"]
        require(
            isinstance(libraries, list)
            and libraries
            and len(set(libraries)) == len(libraries)
            and str(bundle / "runtime/python/mlx/lib/libmlx.dylib") in libraries,
            "Selected libmlx is absent",
        )
        for path in libraries:
            bound(path, native=True)
    return dict(sorted(files.items()))


def audit_descendants(
    parent_output,
    parent_pid,
    plan,
    plan_sha,
    runtime,
    selected_receipt,
    validator,
    *,
    plan_path,
    parent_interval,
):
    root = parent_output / "descendants"
    require(root.is_dir(), "Missing declared descendant directory")
    paths = sorted(root.glob("child-*/receipt.json"))
    require(
        len(paths) == plan["descendant_contract"]["required_python_children"],
        "Missing or extra upstream fresh-process descendants",
    )
    rows, intervals = [], []
    for path in paths:
        row = raw_binding(path, parent_output)
        value = read_json(path)
        require(
            value["schema"] == "gpu-owned-mlx-suite-descendant-v1"
            and value["protocol"] == PROTOCOL
            and value["producer"] == PRODUCER
            and value["plan_sha256"] == plan_sha
            and value["plan_path"] == str(plan_path)
            and value["parent_pid"] == parent_pid
            and value["parent_output"] == str(parent_output)
            and type(value["pid"]) is int
            and value["pid"] > 0
            and value["pid"] != parent_pid
            and path.parent.name.startswith("child-" + str(value["pid"]) + "-")
            and value["complete"] is True
            and value["integrity_after"] is True
            and value["returncode"] == 0
            and value["formal_admission_opened"] is False
            and value["formal_fits"] == 0
            and "error" not in value
            and "integrity_error" not in value,
            "Failed, stale or incomplete upstream descendant",
        )
        require(
            value["runtime_identity_sha256_before"]
            == value["runtime_identity_sha256_after"]
            == selected_receipt["runtime_identity_sha256"]
            and value["runtime_receipt_sha256"] == plan["authority_inputs"]["receipt"]["sha256"]
            and value["framework_manifest_sha256"]
            == selected_receipt["frameworks"]["mlx"]["manifest"]["sha256"]
            and value["source_sha256_before"]
            == value["source_sha256_after"]
            == selected_receipt["project"]["files"],
            "Descendant runtime/source identity differs",
        )
        require(
            value["bytecode_prefix"] == str(path.parent / "unused-bytecode-cache")
            and not Path(value["bytecode_prefix"]).exists(),
            "Descendant cache changed",
        )
        binding(value["launcher"]["path"], value["launcher"])
        require(
            value["launcher"]["path"] == str(parent_output / "archive-python"),
            "Descendant used another launcher",
        )
        require(
            (parent_output / "archive-python").read_text()
            == launcher_text(
                plan,
                plan_path,
                plan_sha,
                parent_output,
                parent_pid,
                runtime["layout"]["interpreter"],
            ),
            "Descendant launcher invocation or contents changed",
        )
        interval = (value["started_unix"], value["ended_unix"])
        require(
            all(type(x) in (int, float) and math.isfinite(x) for x in interval)
            and parent_interval[0] <= interval[0] <= interval[1] <= parent_interval[1],
            "Descendant execution interval differs",
        )
        intervals.append(interval)
        invocation = [
            str(Path(plan["bundle"]) / plan["descendant_contract"]["python_invocation"][0]),
            plan["descendant_contract"]["python_invocation"][1],
        ]
        require(value["invocation"] == invocation, "Descendant invocation changed")
        for phase in ("before", "after"):
            validator(value["compilation_" + phase])
        files = validate_descendant_proofs(
            value,
            runtime,
            plan["runtime_root"],
            selected_receipt["project"],
            plan["bundle"],
            expected_numpy=numpy_version(plan),
        )
        for path, expected in files.items():
            pinned(path, expected)
        rows.append(row)
    intervals.sort()
    require(
        all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:])),
        "Upstream fresh-process descendants overlapped",
    )
    return rows


def freeze(
    runtime_root,
    runtime_inventory,
    selection,
    receipt,
    receipt_sha,
    output,
    *,
    suite_contract="original",
):
    sys.pycache_prefix = str(output.with_name(output.name + ".unused-python-cache"))
    check_controls()
    require(not output.exists(), "Preparation output must be new")
    inputs = {
        "inventory": binding(runtime_inventory),
        "selection": binding(selection),
        "receipt": pinned(receipt, receipt_sha),
    }
    plan = {
        "schema": PLAN_SCHEMA,
        "protocol": PROTOCOL,
        "runtime_root": str(runtime_root),
        "authority_inputs": inputs,
        "producer": binding(Path(__file__).resolve()),
        "limits": copy.deepcopy(LIMITS),
        "suite_order": list(SUITES),
        "formal_admission_opened": False,
        "formal_fits": 0,
        "suite_contract": suite_contract,
    }
    authority, _, runtime, selected = bootstrap(plan)
    require(
        numpy_version(plan) == selected["pytorch"].get("numpy_abi_version", "1.26.4"),
        "MLX suite contract differs from selected Torch NumPy ABI",
    )
    output_path(output, authority.verify()[0]["project"])
    expected, inventory_binding = suite_inventory(suite_contract)
    bundle, sources = test_source(authority, expected)
    plan.update(
        suite_inventory=inventory_binding,
        bundle=str(bundle),
        test_source_sha256=sources,
        descendant_contract=subprocess_contract(bundle, expected),
        python_test_driver=python_test_driver(runtime, bundle),
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    return binding(output / "plan.json")


def authenticate_plan(path, expected_sha):
    pinned(path, expected_sha)
    plan = read_json(path)
    require(
        plan["schema"] == PLAN_SCHEMA
        and plan["protocol"] == PROTOCOL
        and plan["limits"] == LIMITS
        and plan["suite_order"] == list(SUITES)
        and plan["formal_admission_opened"] is False
        and plan["formal_fits"] == 0,
        "Wrong or weakened frozen suite plan",
    )
    binding(plan["producer"]["path"], plan["producer"])
    require(plan["producer"]["path"] == str(Path(__file__).resolve()), "Foreign suite producer")
    expected, identity = suite_inventory(plan.get("suite_contract", "original"))
    require(plan["suite_inventory"] == identity, "Suite inventory differs")
    authority, backend, runtime, selection = bootstrap(plan)
    require(
        numpy_version(plan) == selection["pytorch"].get("numpy_abi_version", "1.26.4"),
        "MLX suite contract differs from selected Torch NumPy ABI",
    )
    bundle, sources = test_source(authority, expected)
    require(
        plan["bundle"] == str(bundle) and plan["test_source_sha256"] == sources,
        "Frozen source inventory differs",
    )
    require(
        plan["descendant_contract"] == subprocess_contract(bundle, expected),
        "Prospective descendant routing changed",
    )
    require(
        plan["python_test_driver"] == python_test_driver(runtime, bundle),
        "Prospective Python test driver changed",
    )
    return plan, authority, backend, runtime, selection, expected


def python_inventory(xml, log, expected):
    tree = ET.parse(xml)
    rows, failures = [], []
    for case in tree.iter("testcase"):
        item = {
            "class": case.attrib.get("classname", ""),
            "name": case.attrib.get("name", ""),
            "status": "skipped" if case.find("skipped") is not None else "passed",
        }
        if item["status"] == "skipped":
            item["reason"] = case.find("skipped").attrib.get("message", "")
        for tag in ("error", "failure"):
            for error in case.findall(tag):
                failures.append(
                    {
                        "class": item["class"],
                        "name": item["name"],
                        "kind": tag,
                        "message": error.attrib.get("message", ""),
                        "traceback": error.text or "",
                    }
                )
        rows.append(item)
    rows.sort(key=lambda x: (x["class"], x["name"]))
    require(rows == expected["python_cases"], "Full Python IDs or unchanged eligibility differs")
    text = log.read_text(errors="strict")
    summaries = re.findall(
        r"^([0-9]+) passed, ([0-9]+) skipped, ([0-9]+) subtests passed in "
        r"[0-9.]+s(?: \([^\n]*\))?$",
        text,
        re.M,
    )
    success = not failures and len(summaries) == 1
    if success:
        passed, skipped, subtests = map(int, summaries[0])
        require(
            passed == 1046 and skipped == 7 and subtests == expected["python_subtests"],
            "Complete Python or subtest count differs",
        )
    return {
        "cases": len(rows),
        "skipped": 7,
        "failures": failures,
        "passed": success,
        "subtests_passed": int(summaries[0][2]) if len(summaries) == 1 else None,
    }


def cpp_inventory(log, expected):
    text = log.read_text(errors="strict")
    cases = re.findall(
        r"\[doctest\] test cases:\s*(\d+)\s*\|\s*(\d+) passed"
        r"\s*\|\s*(\d+) failed\s*\|\s*(\d+) skipped",
        text,
    )
    assertions = re.findall(
        r"\[doctest\] assertions:\s*(\d+)\s*\|\s*(\d+) passed\s*\|\s*(\d+) failed", text
    )
    require(len(cases) == len(assertions) == 1, "Missing or ambiguous C++ totals")
    count, passed, failed, skipped = map(int, cases[0])
    total, good, bad = map(int, assertions[0])
    require(
        count == expected["cpp_case_count"]
        and total == expected["cpp_assertions"]
        and passed + failed == count
        and skipped == 0
        and good + bad == total,
        "Complete C++ test or assertion inventory differs",
    )
    return {
        "cases": count,
        "passed_cases": passed,
        "failed_cases": failed,
        "skipped": skipped,
        "assertions": total,
        "assertions_passed": good,
        "assertions_failed": bad,
        "passed": failed == bad == 0,
    }


def raw_binding(path, root):
    row = binding(path)
    row["path"] = str(Path(row["path"]).relative_to(root))
    return row


def completed_receipt(receipt):
    """Full coverage/integrity is separate from a numerical pass."""
    return (
        receipt.get("complete") is True
        and receipt.get("integrity_after") is True
        and "error" not in receipt
        and "integrity_error" not in receipt
    )


def post_pytest_stream(mx):
    """The source-bound pytest_unconfigure deliberately clears all streams."""
    require(mx.default_device() == mx.gpu and mx.metal.is_available(), "GPU default changed")
    stream = mx.new_stream(mx.gpu)
    require(stream.device == mx.gpu, "Post-suite stream is not GPU")
    mx.set_default_stream(stream)
    require(mx.default_stream(mx.gpu) == stream, "Post-suite default stream differs")
    return {
        "reason": "upstream_pytest_unconfigure_clears_streams",
        "new_gpu_stream": True,
        "stream_index": stream.index,
        "default_stream_verified": True,
    }


def stop(child, own_group=True):
    if child.poll() is not None:
        return
    if own_group:
        os.killpg(child.pid, signal.SIGTERM)
    else:
        child.terminate()
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if own_group:
            os.killpg(child.pid, signal.SIGKILL)
        else:
            child.kill()
        child.wait()


def bounded_process(command, *, cwd, log, env, timeout, output, new_session=True):
    import shutil

    record = {
        "command": list(map(str, command)),
        "pid": None,
        "returncode": None,
        "timed_out": False,
        "started_unix": time.time(),
        "ended_unix": None,
    }
    with log.open("x") as stream:
        child = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=new_session,
        )
        record["pid"] = child.pid
        try:
            while child.poll() is None:
                if time.time() - record["started_unix"] > timeout:
                    record["timed_out"] = True
                    raise TimeoutError("Suite process exceeded frozen timeout")
                require(
                    shutil.disk_usage(output).free >= LIMITS["minimum_free_bytes"],
                    "Suite disk reserve exhausted",
                )
                size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                require(size <= LIMITS["output_bytes"], "Suite output exceeded bound")
                time.sleep(1)
        except BaseException:
            record["error"] = traceback.format_exc()
            stop(child, own_group=new_session)
        finally:
            record.update(returncode=child.returncode, ended_unix=time.time())
    return record


def linkage(executable, library, output, label):
    proof = {}
    suffixes = {"-L": "libraries", "-l": "load-commands"}
    for name, path in (("executable", executable), ("library", library)):
        parts = {}
        for flag in ("-L", "-l"):
            result = subprocess.run(
                ["/usr/bin/otool", flag, str(path)],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            target = output / (label + "-" + name + "-" + suffixes[flag] + ".txt")
            # -L and -l are distinct otool modes, but those spellings collide
            # as filenames on the usual case-insensitive macOS filesystem.
            with target.open("x") as stream:
                stream.write(result.stdout)
            parts[flag] = result.stdout
        proof[name] = {
            "file": binding(path),
            "otool": parts,
            "raw": {
                flag: raw_binding(
                    output / (label + "-" + name + "-" + suffixes[flag] + ".txt"), output
                )
                for flag in ("-L", "-l")
            },
        }
    installs = re.findall(
        r"^\s+(.+?) \(compatibility version", proof["executable"]["otool"]["-L"], re.M
    )
    mlx_names = [name for name in installs if Path(name).name == library.name]
    require(len(mlx_names) == 1, "C++ executable must link selected libmlx exactly once")
    rpaths = re.findall(
        r"cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (.+?) \(offset", proof["executable"]["otool"]["-l"]
    )

    def resolve(value):
        return Path(
            value.replace("@executable_path", str(executable.parent)).replace(
                "@loader_path", str(executable.parent)
            )
        )

    name = mlx_names[0]
    if name.startswith("@rpath/"):
        candidates = [resolve(path) / name.removeprefix("@rpath/") for path in rpaths]
        resolved = [path.resolve(strict=True) for path in candidates if path.exists()]
        require(resolved and resolved[0] == library, "C++ libmlx resolution differs")
    else:
        require(resolve(name).resolve(strict=True) == library, "C++ direct libmlx differs")
    for name in installs:
        require(
            name == mlx_names[0] or name.startswith(("/usr/lib/", "/System/Library/")),
            "Unbound non-system C++ dependency",
        )
    library_installs = re.findall(
        r"^\s+(.+?) \(compatibility version", proof["library"]["otool"]["-L"], re.M
    )
    require(
        library_installs and library_installs[0] == "@rpath/" + library.name,
        "Unexpected selected library install identity",
    )
    require(
        all(name.startswith(("/usr/lib/", "/System/Library/")) for name in library_installs[1:]),
        "Unbound non-system libmlx dependency",
    )
    proof["selected_libmlx"] = str(library)
    return proof


def worker(plan_path, plan_sha, suite, output):
    require(suite in SUITES and output.is_dir(), "Unknown suite or absent fresh parent output")
    receipt_path = output / "receipt.json"
    require(not receipt_path.exists(), "Refusing existing suite receipt")
    record = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "name": suite,
        "producer": PRODUCER,
        "formal_admission_opened": False,
        "formal_fits": 0,
        "formal_eligible": False,
        "execution_scope": "full_gpu_default_suite",
        "no_filters": True,
        "no_test_source_modifications": True,
        "worker_pid": os.getpid(),
        "plan_sha256": plan_sha,
        "plan_path": str(plan_path),
        "bytecode_prefix": sys.pycache_prefix,
        "complete": False,
        "passed": False,
        "raw": {},
    }
    authority = backend = runtime = expected = None
    started = time.time()
    code = 2
    try:
        check_controls()
        require(
            sys.pycache_prefix == str(output / "unused-bytecode-cache")
            and not Path(sys.pycache_prefix).exists(),
            "Require a fresh unused bytecode prefix",
        )
        plan, authority, backend, runtime, selection, expected = authenticate_plan(
            plan_path, plan_sha
        )
        verified = authority.verify()
        receipt, payloads = verified
        bundle = Path(plan["bundle"])
        import numpy as np

        require(np.__version__ == numpy_version(plan), "Joint runtime NumPy version differs")
        numpy_path = str(Path(np.__file__).resolve(strict=True))
        require(numpy_path in runtime["layout"]["packages"]["numpy"], "NumPy origin differs")
        import mlx.core as mx

        require(mx.default_device() == mx.gpu and mx.metal.is_available(), "Natural GPU required")
        compiler = importlib.import_module("transformer.gpu_mlx_compile")
        compilation = backend.validate_compilation_probe(compiler.verify_compilation(mx))
        before = backend._loaded_identity("mlx", authority, verified)
        record["loaded_python_files_before"] = verify_loaded_files(
            runtime, authority, receipt["project"]
        )
        identity = hashlib.sha256(canonical(runtime["files"])).hexdigest()
        record.update(
            runtime_identity_sha256_before=identity,
            runtime_root=str(authority.root),
            runtime_receipt_sha256=authority.receipt_sha256,
            framework_manifest_sha256=selection["mlx"]["runtime_manifest_sha256"],
            source_sha256_before=receipt["project"]["files"],
            test_source_sha256=plan["test_source_sha256"],
            numpy={"version": np.__version__, "path": numpy_path},
            controls=dict(CONTROLS),
            natural_gpu_default=True,
            compilation_before=compilation,
        )
        native = {
            str(Path(path).relative_to(authority.root)): sha
            for path, sha in payloads["mlx"].items()
            if path.endswith((".dylib", ".metallib", ".so"))
        }
        record["native_artifacts"] = native
        os.environ["PYTHONPYCACHEPREFIX"] = sys.pycache_prefix
        if suite == "mlx_cpp":
            executable = bundle / "runtime/bin/tests"
            libraries = [
                Path(path) for path in payloads["mlx"] if Path(path).name == "libmlx.dylib"
            ]
            require(len(libraries) == 1, "Ambiguous libmlx payload")
            library = libraries[0]
            before_linkage = linkage(executable, library, output, "before")
            before_linkage["files"] = copy.deepcopy(native)
            record["process"] = bounded_process(
                [str(executable)],
                cwd=bundle,
                log=output / "cpp.log",
                env=controls(os.environ),
                timeout=LIMITS["mlx_cpp_seconds"],
                output=output,
                new_session=False,
            )
            record["native_origin_verification"] = {
                "method": "direct_pinned_executable_and_linkage",
                "before": before_linkage,
                "after": linkage(executable, library, output, "after"),
                "supervisor_loaded_images": before,
            }
            record["native_origin_verification"]["after"]["files"] = copy.deepcopy(native)
            code = record["process"]["returncode"]
            record["raw"]["log"] = raw_binding(output / "cpp.log", output)
            observed = cpp_inventory(output / "cpp.log", expected)
        else:
            executable = Path(runtime["layout"]["interpreter_resolved"])
            pytest_spec = importlib.machinery.PathFinder.find_spec("pytest", sys.path)
            require(
                pytest_spec is not None and isinstance(pytest_spec.origin, str),
                "Missing ordinary pytest package",
            )
            pytest_source = Path(pytest_spec.origin).resolve(strict=True)
            require(
                runtime["files"].get(str(pytest_source.relative_to(authority.root)))
                == sha256(pytest_source),
                "Unbound pytest import",
            )
            import pytest

            require(
                pytest.__version__ == "9.1.1"
                and str(pytest_source)
                == plan["python_test_driver"]["dependencies"]["pytest"]["source"]["path"],
                "Loaded pytest version/origin differs",
            )
            record["python_test_driver"] = python_test_driver(runtime, bundle)
            arguments = python_arguments(bundle, output)
            record["suite_arguments"] = arguments
            os.chdir(bundle / "source")
            launcher = output / "archive-python"
            launcher.write_text(
                launcher_text(
                    plan, plan_path, plan_sha, output, os.getpid(), runtime["layout"]["interpreter"]
                )
            )
            launcher.chmod(0o700)
            (output / "descendants").mkdir()
            record["descendant_routing"] = {
                "contract": plan["descendant_contract"],
                "launcher": raw_binding(launcher, output),
                "producer_sha256": plan["producer"]["sha256"],
                "children": [],
            }
            actual_executable = sys.executable
            try:
                sys.executable = str(launcher)
                code = int(pytest.main(arguments))
            finally:
                sys.executable = actual_executable
                # The unchanged upstream conftest clears streams even on a
                # collection error. Recreate only the postflight probe's stream;
                # retain the actual test outcome and require GPU first.
                record["post_pytest_stream"] = post_pytest_stream(mx)
            record["descendant_routing"]["children"] = audit_descendants(
                output,
                os.getpid(),
                plan,
                plan_sha,
                runtime,
                authority.verify()[0],
                backend.validate_compilation_probe,
                plan_path=plan_path,
                parent_interval=(started, time.time()),
            )
            sys.stdout.flush()
            sys.stderr.flush()
            record["process"] = {
                "pid": os.getpid(),
                "returncode": code,
                "timed_out": False,
                "started_unix": started,
                "ended_unix": time.time(),
                "command": [sys.executable, *sys.orig_argv[1:]],
            }
            record["raw"]["log"] = raw_binding(output / "worker.log", output)
            if (output / "python.xml").exists():
                record["raw"]["xml"] = raw_binding(output / "python.xml", output)
            observed = python_inventory(output / "python.xml", output / "worker.log", expected)
            record["native_origin_verification"] = {
                "method": "loaded_images",
                "before": {"files": copy.deepcopy(native), "loaded_images": before},
            }
        executable_relative = str(executable.relative_to(authority.root))
        require(
            runtime["files"].get(executable_relative) == sha256(executable), "Unbound executable"
        )
        record["executable"] = {
            "runtime_relative": executable_relative,
            "sha256": sha256(executable),
        }
        record["inventory_result"] = observed
        record["passed"] = code == 0 and observed["passed"]
        record["complete"] = True
    except BaseException:
        record["error"] = traceback.format_exc()
        code = 2
    finally:
        if authority is not None:
            try:
                check_controls()
                require(
                    "_virtualenv" not in sys.modules and not Path(sys.pycache_prefix).exists(),
                    "A site hook or mutable bytecode cache appeared",
                )
                final = authority.verify()
                record["source_sha256_after"] = final[0]["project"]["files"]
                record["runtime_identity_sha256_after"] = final[0]["runtime_identity_sha256"]
                backend.verify_loaded_project_modules(final[0]["project"])
                verify_import_resolution(runtime, sys.path)
                record["loaded_python_files_after"] = verify_loaded_files(
                    runtime, authority, final[0]["project"]
                )
                test_source(authority, expected)
                import mlx.core as mx

                require(
                    mx.default_device() == mx.gpu and mx.metal.is_available(), "GPU default changed"
                )
                record["compilation_after"] = backend.validate_compilation_probe(
                    compiler.verify_compilation(mx)
                )
                after = backend._loaded_identity("mlx", authority, final)
                if suite == "mlx_python":
                    record.setdefault("native_origin_verification", {})["after"] = {
                        "files": copy.deepcopy(record["native_artifacts"]),
                        "loaded_images": after,
                    }
                else:
                    record["supervisor_native_origins_after"] = after
                require(
                    record["runtime_identity_sha256_before"]
                    == record["runtime_identity_sha256_after"]
                    and record["source_sha256_before"] == record["source_sha256_after"],
                    "Runtime or project source changed",
                )
                record["integrity_after"] = True
            except BaseException:
                record.update(
                    complete=False,
                    passed=False,
                    integrity_after=False,
                    integrity_error=traceback.format_exc(),
                )
                code = 2
        sys.stdout.flush()
        sys.stderr.flush()
        if suite == "mlx_python":
            record.setdefault(
                "process",
                {
                    "pid": os.getpid(),
                    "returncode": code,
                    "timed_out": False,
                    "started_unix": started,
                    "ended_unix": time.time(),
                    "command": [sys.executable, *sys.orig_argv[1:]],
                },
            )
            if (output / "python.xml").exists():
                record["raw"]["xml"] = raw_binding(output / "python.xml", output)
        if (output / "worker.log").exists():
            record["worker_log"] = raw_binding(output / "worker.log", output)
            if suite == "mlx_python":
                record["raw"]["log"] = record["worker_log"]
        if (output / "descendants").is_dir():
            record.setdefault("descendant_routing", {})["retained_child_receipts"] = [
                raw_binding(path, output)
                for path in sorted((output / "descendants").glob("child-*/receipt.json"))
            ]
        record["retained_files"] = {
            path.name: raw_binding(path, output)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name not in {"receipt.json", "receipt.json.writing"}
        }
        write_json(receipt_path, record)
    return int(code if code is not None and code >= 0 else 2)


def audit_execution(row, plan, plan_sha, expected, runtime, selected_receipt, probe_validator):
    require(row["name"] in SUITES, "Unknown retained suite")
    if "receipt" not in row:
        return False
    binding(row["receipt"]["path"], row["receipt"])
    receipt = read_json(row["receipt"]["path"])
    root = Path(row["receipt"]["path"]).parent
    execution = row["launcher"]
    require(
        receipt.get("bytecode_prefix") == str(root / "unused-bytecode-cache")
        and not (root / "unused-bytecode-cache").exists(),
        "Retained bytecode prefix changed or became populated",
    )
    require(
        receipt["schema"] == SCHEMA
        and receipt["protocol"] == PROTOCOL
        and receipt["producer"] == PRODUCER
        and receipt["worker_pid"] == execution["pid"]
        and receipt["plan_sha256"] == plan_sha
        and receipt["name"] == row["name"],
        "Stale or foreign suite receipt",
    )
    for raw in [
        *receipt["raw"].values(),
        receipt["worker_log"],
        *receipt["retained_files"].values(),
    ]:
        relative = Path(raw["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe raw output path")
        require(raw_binding(root / relative, root) == raw, "Earlier raw suite output changed")
    for label in ("before", "after"):
        proof = receipt.get("native_origin_verification", {}).get(label, {})
        for kind in ("executable", "library"):
            native = proof.get(kind)
            if native is None:
                continue
            binding(native["file"]["path"], native["file"])
            for flag, raw in native["raw"].items():
                require(
                    raw_binding(root / raw["path"], root) == raw
                    and (root / raw["path"]).read_text() == native["otool"][flag],
                    "Saved native linkage evidence changed",
                )
    routing = receipt.get("descendant_routing", {})
    for child in routing.get("retained_child_receipts", []):
        relative = Path(child["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe descendant path")
        require(raw_binding(root / relative, root) == child, "Earlier child receipt changed")
    if (
        execution["returncode"] != 0
        or execution["timed_out"]
        or receipt.get("complete") is not True
        or receipt.get("integrity_after") is not True
        or "error" in receipt
        or "integrity_error" in receipt
    ):
        return False
    require(
        receipt["formal_admission_opened"] is False
        and receipt["formal_fits"] == 0
        and receipt["formal_eligible"] is False
        and receipt["execution_scope"] == "full_gpu_default_suite"
        and receipt["no_filters"] is True
        and receipt["no_test_source_modifications"] is True,
        "Suite qualification or execution scope changed",
    )
    runtime_hash = hashlib.sha256(canonical(runtime["files"])).hexdigest()
    require(
        receipt["runtime_root"] == plan["runtime_root"]
        and receipt["runtime_identity_sha256_before"]
        == receipt["runtime_identity_sha256_after"]
        == runtime_hash
        and receipt["runtime_receipt_sha256"] == plan["authority_inputs"]["receipt"]["sha256"]
        and receipt["source_sha256_before"]
        == receipt["source_sha256_after"]
        == selected_receipt["project"]["files"]
        and receipt["framework_manifest_sha256"]
        == selected_receipt["frameworks"]["mlx"]["manifest"]["sha256"]
        and receipt["test_source_sha256"] == plan["test_source_sha256"],
        "Retained suite authority/source identity differs",
    )
    native = receipt["native_artifacts"]
    require(
        native and all(runtime["files"].get(name) == sha for name, sha in native.items()),
        "Unbound native artifacts in retained suite",
    )
    proof = receipt["native_origin_verification"]
    require(
        proof["method"]
        == (
            "loaded_images"
            if row["name"] == "mlx_python"
            else "direct_pinned_executable_and_linkage"
        )
        and proof["before"]["files"] == proof["after"]["files"] == native,
        "Native-origin proof changed or disappeared",
    )
    executable = receipt["executable"]
    require(
        runtime["files"].get(executable["runtime_relative"]) == executable["sha256"],
        "Retained executable is unbound",
    )
    for phase in ("before", "after"):
        probe_validator(receipt["compilation_" + phase])
    process = receipt["process"]
    require(
        type(process["pid"]) is int
        and process["pid"] > 0
        and process["returncode"] == 0
        and process["timed_out"] is False
        and execution["started_unix"]
        <= process["started_unix"]
        <= process["ended_unix"]
        <= execution["ended_unix"],
        "Retained suite process differs or failed",
    )
    raw = receipt["raw"]
    if row["name"] == "mlx_python":
        require(
            receipt["python_test_driver"]
            == plan["python_test_driver"]
            == python_test_driver(runtime, Path(plan["bundle"]))
            and receipt["suite_arguments"] == python_arguments(Path(plan["bundle"]), root),
            "Retained upstream pytest configuration or arguments differ",
        )
        stream = receipt.get("post_pytest_stream", {})
        require(
            stream.get("reason") == "upstream_pytest_unconfigure_clears_streams"
            and stream.get("new_gpu_stream") is True
            and stream.get("default_stream_verified") is True
            and type(stream.get("stream_index")) is int
            and stream["stream_index"] >= 0,
            "Missing post-pytest GPU stream proof",
        )
        require(
            process["pid"] == execution["pid"]
            and process["command"] == execution["command"]
            and set(raw) == {"log", "xml"},
            "Wrong Python launch or evidence",
        )
        children = audit_descendants(
            root,
            execution["pid"],
            plan,
            plan_sha,
            runtime,
            selected_receipt,
            probe_validator,
            plan_path=Path(receipt["plan_path"]),
            parent_interval=(process["started_unix"], process["ended_unix"]),
        )
        require(
            routing["contract"] == plan["descendant_contract"]
            and routing["children"] == children == routing["retained_child_receipts"]
            and raw_binding(root / routing["launcher"]["path"], root) == routing["launcher"],
            "Descendant routing or evidence changed",
        )
        require(
            routing["producer_sha256"] == plan["producer"]["sha256"]
            and process["command"][process["command"].index("--plan") + 1] == receipt["plan_path"],
            "Descendant parent plan or producer differs",
        )
        result = python_inventory(root / raw["xml"]["path"], root / raw["log"]["path"], expected)
    else:
        require(
            process["pid"] != execution["pid"]
            and process["command"] == [str(Path(plan["bundle"]) / "runtime/bin/tests")]
            and set(raw) == {"log"},
            "Wrong C++ launch or evidence",
        )
        result = cpp_inventory(root / raw["log"]["path"], expected)
    return result["passed"] is True


def run(plan_path, plan_sha, output):
    sys.pycache_prefix = str(output.with_name(output.name + ".unused-python-cache"))
    check_controls()
    require(not output.exists(), "Run output must be fresh")
    plan, authority, backend, runtime, _, expected = authenticate_plan(plan_path, plan_sha)
    output_path(output, authority.verify()[0]["project"])
    require(
        not output.is_relative_to(Path(plan["bundle"])), "Output cannot mutate framework bundle"
    )
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema": "gpu-owned-mlx-suites-v1",
        "protocol": PROTOCOL,
        "plan": binding(plan_path),
        "formal_admission_opened": False,
        "formal_fits": 0,
        "complete": False,
        "passed": False,
        "suites": [],
    }
    write_json(output / "summary.json", summary)
    try:
        for suite in SUITES:
            authority.verify()
            destination = output / suite
            destination.mkdir()
            temporary = destination / "tmp"
            temporary.mkdir()
            environment = controls(os.environ)
            environment["TMPDIR"] = str(temporary)
            command = [
                runtime["layout"]["interpreter"],
                "-P",
                "-S",
                "-B",
                "-X",
                "pycache_prefix=" + str(destination / "unused-bytecode-cache"),
                str(Path(__file__).resolve()),
                "worker",
                "--plan",
                str(plan_path),
                "--plan-sha256",
                plan_sha,
                "--suite",
                suite,
                "--output",
                str(destination),
            ]
            execution = bounded_process(
                command,
                cwd=destination,
                log=destination / "worker.log",
                env=environment,
                timeout=LIMITS[suite + "_seconds"] + LIMITS["wrapper_grace_seconds"],
                output=output,
            )
            row = {"name": suite, "launcher": execution, "complete": False, "passed": False}
            summary["suites"].append(row)
            receipt_path = destination / "receipt.json"
            if receipt_path.exists():
                row["receipt"] = binding(receipt_path)
                row["complete"] = completed_receipt(read_json(receipt_path))
                row["passed"] = audit_execution(
                    row,
                    plan,
                    plan_sha,
                    expected,
                    runtime,
                    authority.verify()[0],
                    backend.validate_compilation_probe,
                )
            write_json(output / "summary.json", summary)
        selected_receipt, _ = authority.verify()
        for row in summary["suites"]:
            row["passed"] = audit_execution(
                row,
                plan,
                plan_sha,
                expected,
                runtime,
                selected_receipt,
                backend.validate_compilation_probe,
            )
        summary.update(
            complete=all(row["complete"] for row in summary["suites"]),
            passed=all(row["passed"] for row in summary["suites"]),
        )
    except BaseException:
        summary["error"] = traceback.format_exc()
    finally:
        write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    for name in ("runtime-root", "runtime-inventory", "selection", "receipt", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--receipt-sha256", required=True)
    p.add_argument("--suite-contract", choices=SUITE_CONTRACTS, default="original")
    for name in ("run", "worker"):
        p = subs.add_parser(name)
        p.add_argument("--plan", type=Path, required=True)
        p.add_argument("--plan-sha256", required=True)
        p.add_argument("--output", type=Path, required=True)
        if name == "worker":
            p.add_argument("--suite", choices=SUITES, required=True)
    p = subs.add_parser("descendant")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--parent-output", type=Path, required=True)
    p.add_argument("--parent-pid", type=int, required=True)
    p.add_argument("upstream", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "descendant":
        upstream = args.upstream[1:] if args.upstream[:1] == ["--"] else args.upstream
        return descendant(
            args.plan.absolute(),
            args.plan_sha256,
            args.parent_output.absolute(),
            args.parent_pid,
            upstream,
        )
    if args.command == "prepare":
        result = freeze(
            args.runtime_root.resolve(strict=True),
            args.runtime_inventory.resolve(strict=True),
            args.selection.resolve(strict=True),
            args.receipt.resolve(strict=True),
            args.receipt_sha256,
            args.output.absolute(),
            suite_contract=args.suite_contract,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "worker":
        return worker(args.plan.absolute(), args.plan_sha256, args.suite, args.output.absolute())
    result = run(args.plan.absolute(), args.plan_sha256, args.output.absolute())
    print(json.dumps({k: result[k] for k in ("complete", "passed")}, indent=2))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
