"""Run the unchanged complete MPS suite in a captured owned joint runtime.

Preparation is read-only except one new plan/cache directory. Only the stock
conformance extension may compile during an explicitly invoked run. Framework
builds remain hosted. Suite evidence is eligibility input, never admission.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import stat
import subprocess
import sys
import time
import traceback
import unittest
from pathlib import Path

SCHEMA = "gpu-owned-suite-execution-v1"
PLAN_SCHEMA = "gpu-owned-pytorch-suite-plan-v1"
PROTOCOL = "gpu-v4-owned-five-engine-v1"
PRODUCER = "integration/gpu-owned/owned_pytorch_suites.py"
SHARED = "integration/gpu-owned/owned_mlx_suites.py"
POLICY = "integration/gpu-owned/owned_pytorch_fixture.py"
ARCHIVE = "integration/gpu-owned/owned_pytorch_archive.py"
INVENTORY = "integration/gpu-owned/owned-suite-inventory-v1.json"
INVENTORY_SHA = "cff557d2b1d68857bc488e77e6672fb0fb386409e88ad6ca4e7fa31bd862d852"
HALF_V3_INVENTORY = "integration/gpu-owned/owned-suite-inventory-half-v3.json"
HALF_V3_INVENTORY_SHA = "00d3a5c03b32a91cc28f24355c4a43045f044c77c8d06bcf765c75fbe8b8f639"
FALLBACK_AST_SHA = "3d8e80a2297c719dd126826912efaa594a2104a9f930addb4daf4c955d75439e"
HELPERS = {
    "driver": (
        "run_integrated_v2.py",
        "440cbf36b9bbcf818555c4be488e62207012aab22702419bcd3c1f93708ca982",
    ),
    "driver_policy": (
        "test_extension_policy_v2.py",
        "4582d3a5299ed8f9752297fa52ed47a7d3e5fb030a21c39f4487b86d5fffb583",
    ),
    "reference": ("replay.py", "6b0a5253526d9ac7123247b4695932c394eb8e95a1229e2657e42d5c37880e56"),
}
LIMITS = {
    "timeout_seconds": 21600,
    "cache_bytes": 1024**3,
    "cache_files": 64,
    "minimum_free_bytes": 5 * 1024**3,
    "output_bytes": 2 * 1024**3,
    "jobs": 2,
    "poll_seconds": 1,
}
PINS = {
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    "PYTORCH_MPS_FAST_MATH": "0",
    "PYTORCH_TEST_WITH_SLOW": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "SKBUILD_EDITABLE_REBUILD": "false",
    "MLX_ENABLE_TF32": "0",
    "MLX_SDPA_VJP_FALLBACK": "0",
}
ABSENT = (
    "PYTHONPATH",
    "PYTHONHOME",
    "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH",
    "DYLD_INSERT_LIBRARIES",
    "PYTORCH_MPS_PREFER_METAL",
    "MLX_DISABLE_COMPILE",
    "PYTORCH_OPINFO_SAMPLE_INPUT_INDEX",
    "PYTEST_ADDOPTS",
    "PYTEST_PLUGINS",
    "PYTORCH_TEST_SKIP_FAST",
    "PYTORCH_TEST_SKIP_FAST_TESTS",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )


def binding(path, expected=None):
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path and not path.is_symlink(), "Noncanonical input")
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), "Nonregular input")
    row = {"path": str(path), "sha256": digest(path), "bytes": before.st_size}
    after = path.stat()
    require(
        all(
            getattr(before, k) == getattr(after, k)
            for k in ("st_dev", "st_ino", "st_size", "st_mode", "st_mtime_ns", "st_ctime_ns")
        ),
        "Input changed while hashing",
    )
    if expected is not None:
        require(row == expected, "Input binding differs")
    return row


def pinned(path, expected):
    row = binding(path)
    require(row["sha256"] == expected, "External SHA256 differs")
    return row


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def inventory_spec(plan):
    """Only explicit, source-pinned prospective inventories are executable."""
    choice = plan.get("suite_contract", "original")
    require(choice in {"original", "half-v3"}, "Unknown prospective suite contract")
    if choice == "half-v3":
        return HALF_V3_INVENTORY, HALF_V3_INVENTORY_SHA
    return INVENTORY, INVENTORY_SHA


def suite_inventory(plan, selected=None):
    relative, expected_sha = inventory_spec(plan)
    path = Path(__file__).resolve().parents[2] / relative
    pinned(path, expected_sha)
    value = read(path)
    if plan.get("suite_contract", "original") == "half-v3":
        require(selected is not None, "Half-v3 needs an authenticated framework selection")
        wanted = value["prospective_selection"]
        require(
            all(
                selected["pytorch"].get(key) == wanted[key]
                for key in ("base", "patches", "combined_patch_sha256", "build_run")
            ),
            "Half-v3 suite cannot use an earlier or unselected framework",
        )
    return value


def helpers(plan):
    """Authenticate this entrypoint and shared bootstrap before any project import."""
    raw = plan["authority_inputs"]["receipt"]
    binding(raw["path"], raw)
    receipt = read(raw["path"])
    project = receipt["project"]
    root = Path(project["root"])
    require(root.resolve(strict=True) == root, "Noncanonical project")
    require((root / PRODUCER) == Path(__file__).resolve(), "Foreign producer path")
    inventory, inventory_sha = inventory_spec(plan)
    for name in (
        PRODUCER,
        SHARED,
        POLICY,
        ARCHIVE,
        inventory,
        "integration/gpu-owned/stock_mps_fixture_fragments.py.txt",
    ):
        pinned(root / name, project["files"][name])
    pinned(root / inventory, inventory_sha)
    shared = load(root / SHARED, "owned_mps_shared_bootstrap")
    policy = load(root / POLICY, "owned_mps_fixture_policy")
    return shared, policy


def controls(base, policy, cache):
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
        "DEVELOPER_DIR",
        "SDKROOT",
        "MACOSX_DEPLOYMENT_TARGET",
    }
    result = {key: value for key, value in base.items() if key in allowed}
    result.update(PINS)
    policy.configure_environment(result, cache)
    return result


def check_controls(policy, cache, *, fallback=False):
    expected = {**PINS, "PYTORCH_ENABLE_MPS_FALLBACK": "1" if fallback else "0"}
    require(all(os.environ.get(k) == v for k, v in expected.items()), "Runtime controls changed")
    require(not any(k in os.environ for k in ABSENT), "Runtime/test override is forbidden")
    allowed = {k for k in PINS if k.startswith("PYTHON")} | {"PYTHONPYCACHEPREFIX"}
    require(
        not any(k.startswith("PYTHON") and k not in allowed for k in os.environ),
        "Unrecognized Python override",
    )
    if "PYTHONPYCACHEPREFIX" in os.environ:
        require(os.environ["PYTHONPYCACHEPREFIX"] == sys.pycache_prefix, "Bytecode prefix differs")
    policy.require_environment(os.environ, cache)


def safe_relative(name):
    path = Path(name)
    require(
        isinstance(name, str)
        and name
        and not path.is_absolute()
        and path.as_posix() == name
        and all(p not in (".", "..") for p in path.parts),
        "Unsafe source member",
    )
    return path


def source_files(plan, selection, *, verify=True):
    """An external full-source receipt is distinct from the runtime's native manifest."""
    row = plan["source_manifest"]
    binding(row["path"], row)
    value = read(row["path"])
    base = value.get("source_revision", value.get("base"))
    selected = selection["pytorch"]
    combined = selected.get("combined_patch_sha256")
    if combined is None:
        require(len(selected["patches"]) == 1, "Combined patch authority is missing")
        combined = selected["patches"][0]
    require(
        base == selected["base"] and value.get("patch_sha256") == combined,
        "Full source does not match the selected native build",
    )
    require(
        isinstance(value.get("files"), dict) and value["files"], "Missing full source inventory"
    )
    root = Path(plan["source_root"])
    require(root.is_dir() and root.resolve(strict=True) == root, "Noncanonical full source root")
    result = {}
    for name, item in value["files"].items():
        relative = safe_relative(name)
        require(
            isinstance(item, dict)
            and isinstance(item.get("sha256"), str)
            and len(item["sha256"]) == 64
            and type(item.get("bytes")) is int
            and item["bytes"] >= 0,
            "Malformed full source file identity",
        )
        result[name] = item["sha256"]
        if not verify:
            continue
        path = root / relative
        if path.is_symlink():
            require(
                item.get("mode") == "120000" or item.get("type") == "symlink",
                "Undeclared source symlink",
            )
            raw = os.readlink(path).encode()
            require(
                len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"],
                "Source symlink changed",
            )
        else:
            actual = binding(path)
            require(
                actual["sha256"] == item["sha256"] and actual["bytes"] == item["bytes"],
                "Full source bytes changed: " + name,
            )
    if verify:
        actual = set()
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if name != ".git"]
            for name in list(dirs):
                path = Path(current) / name
                if path.is_symlink():
                    actual.add(path.relative_to(root).as_posix())
                    dirs.remove(name)
            for name in files:
                if name != ".git":
                    actual.add((Path(current) / name).relative_to(root).as_posix())
        require(actual == set(result), "Complete source inventory has missing or extra files")
    binding(row["path"], row)
    return result


def subprocess_contract(path):
    tree = ast.parse(Path(path).read_text())
    rows = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TestFallbackWarning"
    ]
    require(
        len(rows) == 1
        and hashlib.sha256(ast.dump(rows[0], include_attributes=False).encode()).hexdigest()
        == FALLBACK_AST_SHA,
        "Upstream subprocess source changed",
    )
    # The class AST is authenticated before extracting either executable string.
    method = next(
        n
        for n in rows[0].body
        if isinstance(n, ast.FunctionDef) and n.name == "test_warn_on_not_implemented_with_fallback"
    )
    expression = next(
        n.value
        for n in method.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "script" for t in n.targets)
    )
    require(isinstance(expression, ast.JoinedStr), "Fallback script shape differs")
    script = "".join(
        n.value
        if isinstance(n, ast.Constant)
        else "torch.linalg.ldl_factor(torch.eye(2, device='mps'))"
        for n in expression.values
    )
    return {
        "ast_sha256": FALLBACK_AST_SHA,
        "invocations": [
            {
                "name": "import",
                "argv": ["-W", "always", "-c", "import torch"],
                "fallback_after": "0",
            },
            {
                "name": "fallback_warning",
                "argv": ["-W", "always", "-c", script],
                "fallback_after": "1",
            },
        ],
        "scope": (
            "Unchanged upstream conformance subprocesses; fallback=1 only inside its warning test"
        ),
    }


def validate_output(output, protected):
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), "Output must be new")
    require(output.resolve() == output, "Output traverses symlinks")
    for value in protected:
        root = Path(value).resolve(strict=True)
        require(
            not output.is_relative_to(root) and not root.is_relative_to(output),
            "Output overlaps immutable input",
        )
    return output


def protected_roots(plan, runtime, receipt):
    root = Path(plan["runtime_root"])
    return list(
        {
            Path(plan["source_root"]),
            Path(receipt["project"]["root"]),
            Path(sys.prefix),
            Path(sys.base_prefix),
            *[root / name for name in runtime["python_paths"]],
            *[(root / name).resolve(strict=True).parent for name in runtime["files"]],
            *[Path(row["payload_root"]) for row in receipt["frameworks"].values()],
        }
    )


def authenticate_plan(path, expected_sha, *, full_source=True):
    pinned(path, expected_sha)
    plan = read(path)
    require(
        plan["schema"] == PLAN_SCHEMA
        and plan["protocol"] == PROTOCOL
        and plan["limits"] == LIMITS
        and plan["formal_fits"] == 0
        and plan["formal_admission_opened"] is False,
        "Wrong or weakened suite plan",
    )
    binding(plan["producer"]["path"], plan["producer"])
    require(plan["producer"]["path"] == str(Path(__file__).resolve()), "Wrong suite producer")
    shared, policy = helpers(plan)
    authority, backend, runtime, selection = shared.bootstrap(plan)
    inventory, _ = inventory_spec(plan)
    inventory_path = Path(__file__).resolve().parents[2] / inventory
    expected = suite_inventory(plan, selection)
    require(
        plan["inventory"] == binding(inventory_path),
        "Suite inventory changed",
    )
    sources = source_files(plan, selection, verify=full_source)
    require(
        sources["test/test_mps.py"] == expected["pytorch"]["test_source_sha256"],
        "Selected test source differs; future candidates require a prospective inventory",
    )
    policy.verify_inputs(
        Path(plan["source_root"]), sources, expected["pytorch"]["test_source_sha256"]
    )
    require(
        plan["subprocess_contract"]
        == subprocess_contract(Path(plan["source_root"]) / "test/test_mps.py"),
        "Subprocess contract changed",
    )
    for name, (filename, sha) in HELPERS.items():
        row = plan["retained_helpers"][name]
        require(
            Path(row["path"]).name == filename and row["sha256"] == sha, "Historical helper differs"
        )
        binding(row["path"], row)
    require(
        Path(plan["retained_helpers"]["driver_policy"]["path"]).parent
        == Path(plan["retained_helpers"]["driver"]["path"]).parent,
        "Driver policy is not adjacent",
    )
    binding(plan["cache_identity"]["path"], plan["cache_identity"])
    check_controls(policy, Path(plan["cache"]))
    receipt, _ = authority.verify()
    fixture = policy.prepare(
        Path(plan["cache"]),
        Path(plan["source_root"]),
        Path(receipt["frameworks"]["pytorch"]["payload_root"]),
        sources,
        expected["pytorch"]["test_source_sha256"],
        selection["pytorch"]["runtime_manifest_sha256"],
    )
    require(
        {k: v for k, v in fixture.items() if k != "before"}
        == {k: v for k, v in plan["fixture_preparation"].items() if k != "before"},
        "Fixture source/toolchain/control identity changed",
    )
    require(
        policy.MAX_CACHE_BYTES == LIMITS["cache_bytes"]
        and policy.MAX_CACHE_FILES == LIMITS["cache_files"]
        and policy.MIN_FREE_BYTES == LIMITS["minimum_free_bytes"]
        and policy.CONTROLS["MAX_JOBS"] == str(LIMITS["jobs"])
        and policy.CONTROLS["CMAKE_BUILD_PARALLEL_LEVEL"] == str(LIMITS["jobs"]),
        "Fixture resource bounds differ",
    )
    require(dict(os.environ) == plan["environment"], "Suite environment differs from frozen plan")
    return plan, shared, policy, authority, backend, runtime, selection, expected, sources


def freeze(args):
    sys.pycache_prefix = str(args.output.with_name(args.output.name + ".unused-python-cache"))
    inputs = {
        "inventory": binding(args.runtime_inventory),
        "selection": binding(args.selection),
        "receipt": pinned(args.receipt, args.receipt_sha256),
    }
    plan = {
        "schema": PLAN_SCHEMA,
        "protocol": PROTOCOL,
        "producer": binding(Path(__file__).resolve()),
        "runtime_root": str(args.runtime_root),
        "authority_inputs": inputs,
        "source_root": str(args.source),
        "source_manifest": pinned(args.source_manifest, args.source_manifest_sha256),
        "formal_admission_opened": False,
        "formal_fits": 0,
        "limits": dict(LIMITS),
        "suite_contract": getattr(args, "suite_contract", "original"),
    }
    shared, policy = helpers(plan)
    authority, _, runtime, selected = shared.bootstrap(plan)
    receipt, _ = authority.verify()
    validate_output(args.output, protected_roots(plan, runtime, receipt))
    inventory, inventory_sha = inventory_spec(plan)
    inventory_path = Path(__file__).resolve().parents[2] / inventory
    plan["inventory"] = pinned(inventory_path, inventory_sha)
    expected = suite_inventory(plan, selected)["pytorch"]
    files = source_files(plan, selected)
    require(
        files["test/test_mps.py"] == expected["test_source_sha256"], "Unselected full suite source"
    )
    policy.verify_inputs(args.source, files, expected["test_source_sha256"])
    plan["subprocess_contract"] = subprocess_contract(args.source / "test/test_mps.py")
    plan["retained_helpers"] = {
        "driver": pinned(args.driver, HELPERS["driver"][1]),
        "driver_policy": pinned(
            args.driver.with_name(HELPERS["driver_policy"][0]), HELPERS["driver_policy"][1]
        ),
        "reference": pinned(args.reference, HELPERS["reference"][1]),
    }
    cache = args.output / policy.CACHE_NAME
    env = controls(os.environ, policy, cache)
    require(
        dict(os.environ) == env,
        "Prepare under the same sanitized explicit environment as execution",
    )
    # All inputs and source selection are authenticated before claiming storage.
    args.output.mkdir(parents=True, exist_ok=False)
    old = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(env)
        proof = policy.prepare(
            cache,
            args.source,
            Path(receipt["frameworks"]["pytorch"]["payload_root"]),
            files,
            expected["test_source_sha256"],
            selected["pytorch"]["runtime_manifest_sha256"],
        )
    finally:
        os.environ.clear()
        os.environ.update(old)
    plan.update(
        cache=str(cache),
        cache_identity=binding(cache / "cache-identity.json"),
        fixture_preparation=proof,
        environment=env,
    )
    write(args.output / "plan.json", plan)
    return binding(args.output / "plan.json")


def origins(backend, authority, runtime):
    verified = authority.verify()
    proof = backend._loaded_identity("mps", authority, verified)
    files = {
        str(Path(p).relative_to(authority.root)): runtime["files"][
            str(Path(p).relative_to(authority.root))
        ]
        for p in proof["native_libraries"]
    }
    require(any(Path(p).name == "libtorch_cpu.dylib" for p in files), "Torch native image missing")
    return {**proof, "files": files}


def loaded_files(runtime, authority, project, source, sources, extra, modules=None):
    roots = [
        (Path(authority.root), runtime["files"]),
        (Path(project["root"]), project["files"]),
        (Path(source), sources),
    ]
    result = {}
    for name, module in tuple((sys.modules if modules is None else modules).items()):
        try:
            value = vars(module).get("__file__")
        except TypeError:
            value = None
        if not isinstance(value, str):
            continue
        path = Path(value).resolve(strict=True)
        expected = extra.get(str(path))
        for root, files in roots:
            if path.is_relative_to(root):
                expected = files.get(str(path.relative_to(root)), expected)
        require(expected is not None and digest(path) == expected, "Unbound loaded module: " + name)
        result[name] = {"path": str(path), "sha256": expected}
    return result


def validate_native_proof(proof, runtime, root):
    require(
        set(proof) == {"module_origins", "framework_module_origins", "native_libraries", "files"},
        "Incomplete native-origin proof",
    )
    root = Path(root)

    def known(absolute):
        path = Path(absolute)
        require(path.is_absolute() and path.is_relative_to(root), "Foreign native/module path")
        name = path.relative_to(root).as_posix()
        require(name in runtime["files"], "Native/module path is absent from runtime")
        return runtime["files"][name]

    libraries = proof["native_libraries"]
    require(
        isinstance(libraries, list) and libraries == sorted(set(libraries)) and libraries,
        "Missing or duplicate native libraries",
    )
    expected = {Path(p).relative_to(root).as_posix(): known(p) for p in libraries}
    require(
        expected == proof["files"] and any(Path(p).name == "libtorch_cpu.dylib" for p in libraries),
        "Native library evidence differs",
    )
    require(
        set(proof["framework_module_origins"]) == {"pytorch"}, "Unexpected framework origin scope"
    )
    framework = proof["framework_module_origins"]["pytorch"]
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
        and framework["schema"] == "gpu-owned-loaded-framework-origins-v1"
        and framework["framework"] == "pytorch",
        "Incomplete Torch origin schema",
    )
    modules = framework["module_origins"]
    synthetic, aliases = framework["synthetic_origins"], framework["native_aliases"]
    require(
        isinstance(modules, dict)
        and isinstance(synthetic, dict)
        and isinstance(aliases, dict)
        and framework["namespace_packages"] == {}
        and set(synthetic) == {"torch.ops", "torch.classes"},
        "Incomplete Torch namespace inventory",
    )
    require(
        type(framework["module_count"]) is int
        and framework["module_count"]
        == len(modules) + len(synthetic) + len(aliases)
        == len(set(modules) | set(synthetic) | set(aliases)),
        "Duplicated or truncated Torch origins",
    )
    require(
        proof["module_origins"] == [modules.get("torch"), modules.get("torch._C")],
        "Top-level/core origin binding differs",
    )
    package = Path(modules["torch"]).parent
    core = modules["torch._C"]
    require(
        Path(core).parent == package and Path(core).name.startswith("_C.") and core.endswith(".so"),
        "Wrong Torch native extension",
    )
    for absolute in modules.values():
        known(absolute)
        require(Path(absolute).is_relative_to(package.parent), "Foreign Torch namespace origin")
    for name, backing, filename, cls in (
        ("torch.ops", "torch._ops", "_ops.py", "_Ops"),
        ("torch.classes", "torch._classes", "_classes.py", "_Classes"),
    ):
        absolute = str(package / filename)
        require(
            modules.get(backing) == absolute
            and synthetic[name]
            == {
                "backing_module": backing,
                "backing_file": absolute,
                "backing_sha256": known(absolute),
                "exact_class": backing + "." + cls,
                "singleton_identity_verified": True,
            },
            "Torch synthetic backing changed",
        )
    for name, row in aliases.items():
        require(
            name.startswith("torch._C.") and row == {"extension": core, "sha256": known(core)},
            "Torch native alias changed",
        )
    return proof


def validate_loaded_proof(
    imported, native, runtime, runtime_root, project, source_root, sources, extra
):
    """Validate retained import identities without accessing their original paths."""
    require(isinstance(imported, dict) and imported, "Missing retained Python import map")
    roots = [
        (Path(runtime_root), runtime["files"]),
        (Path(project["root"]), project["files"]),
        (Path(source_root), sources),
    ]
    for name, row in imported.items():
        require(
            isinstance(name, str)
            and name
            and isinstance(row, dict)
            and set(row) == {"path", "sha256"},
            "Malformed retained import map",
        )
        absolute = row["path"]
        require(isinstance(absolute, str), "Missing retained import path")
        path = Path(absolute)
        require(
            path.is_absolute() and str(path) == absolute and ".." not in path.parts,
            "Noncanonical retained import path",
        )
        hashes = {
            files[str(path.relative_to(root))]
            for root, files in roots
            if path.is_relative_to(root) and str(path.relative_to(root)) in files
        }
        if absolute in extra:
            hashes.add(extra[absolute])
        require(
            len(hashes) == 1
            and isinstance(row["sha256"], str)
            and row["sha256"] in hashes
            and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]),
            "Retained import is outside or differs from its authenticated authority",
        )
    framework = native["framework_module_origins"]["pytorch"]
    for name, path in framework["module_origins"].items():
        require(
            imported.get(name, {}).get("path") == path,
            "Native and Python module origin maps differ",
        )
    numpy_path = imported.get("numpy", {}).get("path")
    require(
        numpy_path in runtime["layout"]["packages"]["numpy"],
        "Retained NumPy origin is absent or differs",
    )
    return imported


def validate_descendant_intervals(proofs, parent_pid, parent_interval):
    """Require unique child processes, parent containment and serial execution."""
    require(type(parent_pid) is int and parent_pid > 0, "Invalid parent process")

    def interval(value):
        require(
            len(value) == 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in value)
            and value[0] <= value[1],
            "Invalid retained process interval",
        )
        return value

    parent_start, parent_end = interval(parent_interval)
    pids, intervals = set(), []
    for proof in proofs:
        pid = proof["pid"]
        require(
            type(pid) is int
            and pid > 0
            and pid != parent_pid
            and pid not in pids
            and type(proof["parent_pid"]) is int
            and proof["parent_pid"] == parent_pid,
            "Duplicate or foreign descendant process",
        )
        pids.add(pid)
        start, end = interval((proof["started_unix"], proof["ended_unix"]))
        require(parent_start <= start <= end <= parent_end, "Descendant interval escaped parent")
        intervals.append((start, end))
    intervals.sort()
    require(
        all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])),
        "Conformance descendants overlapped",
    )
    return intervals


def worker_command(plan, path, sha, output, runtime):
    source = plan["producer"]
    code = (
        "import hashlib,pathlib,runpy,sys;"
        f"p=pathlib.Path({source['path']!r});"
        f"assert hashlib.sha256(p.read_bytes()).hexdigest()=={source['sha256']!r};"
        f"q=pathlib.Path({str(path)!r});"
        f"assert hashlib.sha256(q.read_bytes()).hexdigest()=={sha!r};"
        "sys.argv=[str(p),'worker','--plan',str(q),'--plan-sha256',"
        + repr(sha)
        + ",'--output',"
        + repr(str(output))
        + "];runpy.run_path(str(p),run_name='__main__')"
    )
    return [
        runtime["layout"]["interpreter_resolved"],
        "-P",
        "-S",
        "-B",
        "-X",
        "pycache_prefix=" + str(output / "unused-bytecode-cache"),
        "-c",
        code,
    ]


def child_command(plan, plan_path, plan_sha, output, pid, *, runtime=None):
    if runtime is None:
        runtime = read(plan["authority_inputs"]["inventory"]["path"])
    executable = runtime["layout"]["interpreter_resolved"]
    # The tiny stdlib launcher authenticates both entrypoint and plan before import.
    source = plan["producer"]
    names = {tuple(row["argv"]): row["name"] for row in plan["subprocess_contract"]["invocations"]}
    prefix = (
        "import sys;" + f"names={names!r};"
        "assert tuple(sys.argv[1:]) in names;"
        f"sys.pycache_prefix={str(output)!r}+'/descendant-'+names[tuple(sys.argv[1:])]+'/unused-bytecode-cache';"
    )
    code = prefix + (
        "import hashlib,pathlib,runpy,sys;assert not pathlib.Path(sys.pycache_prefix).exists();"
        f"p=pathlib.Path({source['path']!r});"
        f"assert hashlib.sha256(p.read_bytes()).hexdigest()=={source['sha256']!r};"
        f"q=pathlib.Path({str(plan_path)!r});"
        f"assert hashlib.sha256(q.read_bytes()).hexdigest()=={plan_sha!r};"
        "sys.argv=[str(p),'descendant','--plan',str(q),'--plan-sha256',"
        + repr(plan_sha)
        + ",'--parent-output',"
        + repr(str(output))
        + ",'--parent-pid',"
        + repr(str(pid))
        + ",'--',*sys.argv[1:]];runpy.run_path(str(p),run_name='__main__')"
    )
    return [executable, "-P", "-S", "-B", "-c", code]


def descendant(plan_path, plan_sha, parent_output, parent_pid, upstream):
    require(os.getppid() == parent_pid, "Wrong conformance subprocess parent")
    pinned(plan_path, plan_sha)
    plan = read(plan_path)
    rows = [r for r in plan["subprocess_contract"]["invocations"] if r["argv"] == upstream]
    require(len(rows) == 1, "Undeclared upstream subprocess invocation")
    row = rows[0]
    target = parent_output / ("descendant-" + row["name"])
    require(not target.exists(), "Duplicate conformance subprocess")
    sys.pycache_prefix = str(target / "unused-bytecode-cache")
    plan, shared, policy, authority, backend, runtime, _, _, sources = authenticate_plan(
        plan_path, plan_sha, full_source=False
    )
    require(Path.cwd() == Path(plan["source_root"]) / "test", "Conformance subprocess cwd differs")
    target.mkdir()
    proof = {
        "schema": "gpu-owned-mps-suite-descendant-v1",
        "protocol": PROTOCOL,
        "plan_sha256": plan_sha,
        "producer": PRODUCER,
        "parent_pid": parent_pid,
        "pid": os.getpid(),
        "invocation": row,
        "complete": False,
        "runtime_receipt_sha256": authority.receipt_sha256,
        "formal_fits": 0,
        "formal_admission_opened": False,
        "bytecode_prefix": sys.pycache_prefix,
        "started_unix": time.time(),
    }
    code = 0
    try:
        import warnings

        warnings.simplefilter("always")
        # Execute the exact source-bound string. The import-warning test must see
        # the first Torch import itself; no pre-import probe can hide a warning.
        sys.argv = ["-c"]
        exec(compile(upstream[3], "<string>", "exec"), {"__name__": "__main__"})
    except SystemExit as exc:
        code = exc.code if type(exc.code) is int else 1
    except BaseException:
        proof["error"] = traceback.format_exc()
        code = 2
    try:
        require(code == 0, "Upstream conformance subprocess failed")
        check_controls(policy, Path(plan["cache"]), fallback=row["fallback_after"] == "1")
        proof["native_origin_verification"] = origins(backend, authority, runtime)
        receipt, _ = authority.verify()
        proof["loaded_python_files"] = loaded_files(
            runtime, authority, receipt["project"], plan["source_root"], sources, {}
        )
        binding(plan["source_manifest"]["path"], plan["source_manifest"])
        require(not Path(sys.pycache_prefix).exists(), "Child cache was populated")
        proof["complete"] = True
    except BaseException:
        proof["integrity_error"] = traceback.format_exc()
        code = 2
    proof["returncode"] = code
    proof["ended_unix"] = time.time()
    write(target / "receipt.json", proof)
    return code


def validate_descendant_proof(
    proof, plan, plan_sha, pid, bytecode_prefix, runtime, runtime_receipt, sources
):
    """Check a retained child without consulting its original paths."""
    require(
        proof["schema"] == "gpu-owned-mps-suite-descendant-v1"
        and proof["protocol"] == PROTOCOL
        and proof["producer"] == PRODUCER
        and proof["plan_sha256"] == plan_sha
        and proof["parent_pid"] == pid
        and type(proof["pid"]) is int
        and proof["pid"] > 0
        and proof["pid"] != pid
        and proof["complete"] is True
        and type(proof["returncode"]) is int
        and proof["returncode"] == 0
        and proof["invocation"] in plan["subprocess_contract"]["invocations"]
        and "error" not in proof
        and "integrity_error" not in proof
        and proof["formal_fits"] == 0
        and proof["formal_admission_opened"] is False,
        "Unqualified conformance subprocess proof",
    )
    require(
        proof["runtime_receipt_sha256"] == plan["authority_inputs"]["receipt"]["sha256"],
        "Conformance subprocess runtime differs",
    )
    require(
        proof["bytecode_prefix"] == bytecode_prefix,
        "Child bytecode policy differs",
    )
    validate_native_proof(proof["native_origin_verification"], runtime, plan["runtime_root"])
    validate_loaded_proof(
        proof.get("loaded_python_files"),
        proof["native_origin_verification"],
        runtime,
        plan["runtime_root"],
        runtime_receipt["project"],
        plan["source_root"],
        sources,
        {},
    )


def check_descendants(output, plan, plan_sha, pid, parent_interval):
    runtime = read(plan["authority_inputs"]["inventory"]["path"])
    binding(plan["authority_inputs"]["inventory"]["path"], plan["authority_inputs"]["inventory"])

    runtime_receipt = read(plan["authority_inputs"]["receipt"]["path"])
    binding(plan["authority_inputs"]["receipt"]["path"], plan["authority_inputs"]["receipt"])
    binding(plan["source_manifest"]["path"], plan["source_manifest"])
    sources = {
        name: item["sha256"]
        for name, item in read(plan["source_manifest"]["path"])["files"].items()
    }
    actual = sorted(p.name for p in output.glob("descendant-*"))
    expected = sorted("descendant-" + r["name"] for r in plan["subprocess_contract"]["invocations"])
    require(actual == expected, "Missing, duplicate or foreign conformance subprocess")
    proofs, children = [], []
    for row in plan["subprocess_contract"]["invocations"]:
        path = output / ("descendant-" + row["name"]) / "receipt.json"
        proof = read(path)
        require(proof["invocation"] == row, "Conformance child order differs")
        validate_descendant_proof(
            proof,
            plan,
            plan_sha,
            pid,
            str(path.parent / "unused-bytecode-cache"),
            runtime,
            runtime_receipt,
            sources,
        )
        require(not Path(proof["bytecode_prefix"]).exists(), "Child cache was populated")
        children.append(proof)
        proofs.append(binding(path))
    validate_descendant_intervals(children, pid, parent_interval)
    return proofs


def run_unchanged_suite(runner, suite, launcher):
    original = sys.executable
    try:
        sys.executable = str(launcher)
        return runner.run(suite)
    finally:
        sys.executable = original


def worker(path, sha, output):
    sys.pycache_prefix = str(output / "unused-bytecode-cache")
    record = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "name": "pytorch_mps",
        "producer": PRODUCER,
        "worker_pid": os.getpid(),
        "started_unix": time.time(),
        "plan_sha256": sha,
        "complete": False,
        "passed": False,
        "formal_fits": 0,
        "formal_admission_opened": False,
        "formal_eligible": False,
        "execution_scope": "full_gpu_default_suite",
        "no_filters": True,
        "no_test_source_modifications": True,
        "raw": {},
        "bytecode_prefix": sys.pycache_prefix,
    }
    context = None
    try:
        context = authenticate_plan(path, sha)
        plan, shared, policy, authority, backend, runtime, selected, expected, sources = context
        receipt, _ = authority.verify()
        validate_output(output, protected_roots(plan, runtime, receipt))
        output.mkdir(parents=True)
        record.update(
            runtime_identity_sha256_before=hashlib.sha256(canonical(runtime["files"])).hexdigest(),
            runtime_receipt_sha256=authority.receipt_sha256,
            framework_manifest_sha256=selected["pytorch"]["runtime_manifest_sha256"],
            source_sha256_before=receipt["project"]["files"],
            test_source_sha256={"test/test_mps.py": sources["test/test_mps.py"]},
            source_manifest=plan["source_manifest"],
            fixture_before=policy.snapshot(Path(plan["cache"]), hashes=True),
        )
        import torch

        require(
            torch.backends.mps.is_available() and not torch.ops.loaded_libraries,
            "Real MPS without custom operator override required",
        )
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        retained = plan["retained_helpers"]
        driver = load(retained["driver"]["path"], "owned_mps_retained_driver")
        reference = load(retained["reference"]["path"], "owned_mps_retained_result")
        extras = {r["path"]: r["sha256"] for r in retained.values()}
        # Only source/test is an import root, never the framework source package.
        sys.path.append(str(Path(plan["source_root"]) / "test"))
        shared.verify_import_resolution(runtime, sys.path)
        test_path = Path(plan["source_root"]) / "test/test_mps.py"
        sys.argv = [str(test_path)]
        module = driver.load_module(test_path, "integrated_upstream_test_mps")
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        ids = [test.id() for test in driver.flatten(suite)]
        require(ids == expected["pytorch"]["test_ids"], "Full test collection changed")
        original_executable = sys.executable
        launcher = output / "owned-suite-python"
        launcher.write_text(
            "#!/bin/sh\nexec "
            + shlex.join(child_command(plan, path, sha, output, os.getpid()))
            + ' "$@"\n'
        )
        launcher.chmod(0o700)
        before = origins(backend, authority, runtime)
        record["native_artifacts"] = before["files"]
        record["native_origin_verification"] = {"method": "loaded_images", "before": before}
        record["loaded_python_files_before"] = loaded_files(
            runtime, authority, receipt["project"], plan["source_root"], sources, extras
        )
        validate_loaded_proof(
            record["loaded_python_files_before"],
            before,
            runtime,
            authority.root,
            receipt["project"],
            plan["source_root"],
            sources,
            extras,
        )
        record["conditions_before"] = driver.conditions()
        write(output / "receipt.json", record)
        with (output / "tests.log").open("x") as log:
            runner = unittest.TextTestRunner(
                stream=log, verbosity=2, resultclass=driver.RetainedResult
            )
            result = run_unchanged_suite(runner, suite, launcher)
        tests = reference.suite_record(result)
        write(output / "tests.json", tests)
        torch.mps.synchronize()
        record["conditions_after"] = driver.conditions()
        record["stock_fixture_loaded"] = policy.loaded_extension(module, Path(plan["cache"]))
        extras[record["stock_fixture_loaded"]["path"]] = record["stock_fixture_loaded"]["sha256"]
        record["descendants"] = check_descendants(
            output, plan, sha, os.getpid(), (record["started_unix"], time.time())
        )
        after = origins(backend, authority, runtime)
        require(
            before["files"] == after["files"], "Native framework images changed during full suite"
        )
        record["native_origin_verification"]["after"] = after
        record["loaded_python_files_after"] = loaded_files(
            runtime, authority, receipt["project"], plan["source_root"], sources, extras
        )
        validate_loaded_proof(
            record["loaded_python_files_after"],
            after,
            runtime,
            authority.root,
            receipt["project"],
            plan["source_root"],
            sources,
            extras,
        )
        executable = binding(Path(original_executable).resolve(strict=True))
        relative = str(Path(executable["path"]).relative_to(authority.root))
        require(runtime["files"].get(relative) == executable["sha256"], "Unbound interpreter")
        record["executable"] = {"runtime_relative": relative, "sha256": executable["sha256"]}
        from transformer.gpu_owned_eligibility import validate_mps

        try:
            record["test_inventory"] = validate_mps(tests, expected)
            record["passed"] = True
        except ValueError as exc:
            record["qualification_failure"] = str(exc)
        record["complete"] = True
    except BaseException:
        record["error"] = traceback.format_exc()
    finally:
        if context is not None:
            plan, shared, policy, authority, backend, runtime, selected, expected, _ = context
            try:
                check_controls(policy, Path(plan["cache"]))
                require(
                    not Path(sys.pycache_prefix).exists(), "Worker bytecode prefix was populated"
                )
                source_files(plan, selected)
                receipt, _ = authority.verify()
                backend.verify_loaded_project_modules(receipt["project"])
                record.update(
                    integrity_after=True,
                    source_sha256_after=receipt["project"]["files"],
                    runtime_identity_sha256_after=hashlib.sha256(
                        canonical(runtime["files"])
                    ).hexdigest(),
                    fixture_after=policy.snapshot(Path(plan["cache"]), hashes=True),
                )
            except BaseException:
                record.update(
                    integrity_after=False, integrity_error=traceback.format_exc(), passed=False
                )
        if output.exists():
            for key, filename in (("log", "tests.log"), ("tests", "tests.json")):
                if (output / filename).is_file():
                    row = binding(output / filename)
                    record["raw"][key] = {**row, "path": filename}
            write(output / "receipt.json", record)
    return 0 if record["complete"] and record["passed"] and record.get("integrity_after") else 1


def validate_execution(
    record, worker_output, execution, plan, plan_sha, authority, runtime, selected, expected
):
    receipt, _ = authority.verify()
    require(
        record.get("worker_pid") == execution["pid"] and record.get("plan_sha256") == plan_sha,
        "Stale or foreign suite worker",
    )
    for phase in ("before", "after"):
        validate_native_proof(record["native_origin_verification"][phase], runtime, authority.root)
    # Reuse the production evidence validator against the actual external authority.
    from transformer.gpu_owned_eligibility import validate_suite

    path = authority.receipt_path
    manifest = {
        "bindings": {
            "runtime": runtime["files"],
            "runtime_layout": runtime["layout"],
            "source": receipt["project"]["files"],
            "baseline": {path: authority.receipt_sha256},
        },
        "owned_frameworks": {"frameworks": selected, "runtime_receipt": path},
    }
    validate_suite(record, worker_output, manifest, expected, "pytorch_mps", receipt)
    check_descendants(
        worker_output,
        plan,
        record["plan_sha256"],
        execution["pid"],
        (execution["started_unix"], execution["ended_unix"]),
    )
    require(
        execution["returncode"] == 0 and not execution["timed_out"] and "error" not in execution,
        "Failed full-suite process",
    )


def run(path, sha, output):
    sys.pycache_prefix = str(output / "supervisor-unused-bytecode-cache")
    context = authenticate_plan(path, sha)
    plan, shared, policy, authority, _, runtime, selected, expected, _ = context
    receipt, _ = authority.verify()
    validate_output(output, protected_roots(plan, runtime, receipt))
    output.mkdir(parents=True)
    worker_output = output / "full"
    command = worker_command(plan, path, sha, worker_output, runtime)
    execution = {
        "command": command,
        "pid": None,
        "returncode": None,
        "timed_out": False,
        "started_unix": time.time(),
        "ended_unix": None,
    }
    try:
        with (output / "worker.log").open("x") as log:
            child = subprocess.Popen(
                command,
                cwd=Path(plan["source_root"]) / "test",
                env=plan["environment"],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            execution["pid"] = child.pid
            try:
                while child.poll() is None:
                    if time.time() - execution["started_unix"] > LIMITS["timeout_seconds"]:
                        execution["timed_out"] = True
                        raise TimeoutError("Full MPS suite exceeded its frozen time budget")
                    policy.snapshot(Path(plan["cache"]))
                    total = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                    require(total <= LIMITS["output_bytes"], "Full-suite output exceeded bound")
                    time.sleep(LIMITS["poll_seconds"])
            except BaseException:
                execution["error"] = traceback.format_exc()
                shared.stop(child)
            finally:
                execution.update(returncode=child.wait(), ended_unix=time.time())
    finally:
        write(output / "process.json", execution)
    summary = {
        "schema": "gpu-owned-pytorch-suite-summary-v1",
        "protocol": PROTOCOL,
        "plan": binding(path),
        "process": execution,
        "complete": False,
        "passed": False,
        "formal_fits": 0,
        "formal_admission_opened": False,
        "formal_eligible": False,
    }
    try:
        source_files(plan, selected)
        authority.verify()
        record = read(worker_output / "receipt.json")
        require(
            record["worker_pid"] == execution["pid"] and record["plan_sha256"] == sha,
            "Foreign worker receipt",
        )
        for phase in ("before", "after"):
            validate_native_proof(
                record["native_origin_verification"][phase], runtime, plan["runtime_root"]
            )
        record["process"] = execution
        write(worker_output / "receipt.json", record)
        summary["receipt"] = binding(worker_output / "receipt.json")
        for row in record["raw"].values():
            safe_relative(row["path"])
            require(
                binding(worker_output / row["path"])["sha256"] == row["sha256"]
                and (worker_output / row["path"]).stat().st_size == row["bytes"],
                "Raw suite output changed",
            )
        require(
            record["complete"] is True
            and record["integrity_after"] is True
            and "error" not in record
            and "integrity_error" not in record,
            "Incomplete full suite",
        )
        archive = load(Path(__file__).resolve().parents[2] / ARCHIVE, "owned_mps_archive")
        record["portable_evidence"] = archive.retain(
            plan, path, worker_output, record, binding, LIMITS
        )
        write(worker_output / "receipt.json", record)
        summary["receipt"] = binding(worker_output / "receipt.json")
        summary["complete"] = True
        validate_execution(
            record, worker_output, execution, plan, sha, authority, runtime, selected, expected
        )
        summary["passed"] = True
    except BaseException:
        summary["error"] = traceback.format_exc()
    write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    for name in (
        "runtime-root",
        "runtime-inventory",
        "selection",
        "receipt",
        "source",
        "source-manifest",
        "driver",
        "reference",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--receipt-sha256", required=True)
    p.add_argument("--source-manifest-sha256", required=True)
    p.add_argument("--suite-contract", choices=("original", "half-v3"), default="original")
    for name in ("run", "worker"):
        p = subs.add_parser(name)
        p.add_argument("--plan", type=Path, required=True)
        p.add_argument("--plan-sha256", required=True)
        p.add_argument("--output", type=Path, required=True)
    p = subs.add_parser("descendant")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--parent-output", type=Path, required=True)
    p.add_argument("--parent-pid", type=int, required=True)
    p.add_argument("upstream", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.absolute())
    if args.command == "prepare":
        print(json.dumps(freeze(args), indent=2))
        return 0
    if args.command == "descendant":
        upstream = args.upstream[1:] if args.upstream[:1] == ["--"] else args.upstream
        return descendant(
            args.plan, args.plan_sha256, args.parent_output, args.parent_pid, upstream
        )
    if args.command == "worker":
        return worker(args.plan, args.plan_sha256, args.output)
    result = run(args.plan, args.plan_sha256, args.output)
    print(json.dumps({k: result[k] for k in ("complete", "passed")}, indent=2))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
