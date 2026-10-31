"""Capture maintained-runtime identity under the exact isolated launch policy.

No framework is imported, installed, compiled, or executed. No modeling fixture
or admission stream is opened. The output is identity evidence, never admission.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
import platform
import stat
import subprocess
import sys
import traceback
from pathlib import Path

PRODUCER = "integration/gpu-owned/capture_runtime.py"
SCHEMA = "gpu-owned-runtime-capture-v1"
PROTOCOL = "gpu-v4-owned-five-engine-v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_binding(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path, "Noncanonical input file")
    before = path.stat()
    require(not path.is_symlink() and stat.S_ISREG(before.st_mode), "Nonregular input file")
    value = {"path": str(path), "sha256": sha(path), "bytes": before.st_size}
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(
        all(getattr(before, k) == getattr(after, k) for k in fields), "Input changed during read"
    )
    return value


def directory(path):
    path = Path(path)
    require(
        path.is_absolute() and path.resolve(strict=True) == path and path.is_dir(),
        "Noncanonical directory",
    )
    return path


def git(project, *args):
    result = subprocess.run(
        ["git", "-C", str(project), *args], capture_output=True, check=True, timeout=60
    )
    return result.stdout


def source_inventory(project, expected_head):
    require(
        git(project, "rev-parse", "HEAD").decode().strip() == expected_head, "Source head differs"
    )
    require(not git(project, "status", "--porcelain").strip(), "Source tree is not clean")
    tree = git(project, "ls-tree", "-r", "-z", "--full-tree", "HEAD")
    index = git(project, "ls-files", "--stage", "-z")
    algorithm = git(project, "rev-parse", "--show-object-format").decode().strip()
    require(algorithm in {"sha1", "sha256"}, "Unsupported Git object format")
    expected, staged = {}, {}
    for raw in tree.decode().rstrip("\0").split("\0"):
        metadata, name = raw.split("\t", 1)
        mode, kind, oid = metadata.split()
        require(kind == "blob" and mode in {"100644", "100755"}, "Unsupported tracked file kind")
        require(name not in expected, "Duplicate tracked path")
        expected[name] = (mode, oid)
    for raw in index.decode().rstrip("\0").split("\0"):
        metadata, name = raw.split("\t", 1)
        mode, oid, stage = metadata.split()
        require(stage == "0" and name not in staged, "Unmerged or duplicate index entry")
        staged[name] = (mode, oid)
    require(staged == expected, "Index differs from declared commit")
    names = set(expected)
    require(PRODUCER in names, "Capture producer is not committed")
    metal = "native/metal/build/libtransformer_metal.dylib"
    if (project / metal).is_file():
        names.add(metal)
    result = {}
    for relative in sorted(names):
        path = Path(relative)
        require(
            relative and not path.is_absolute() and ".." not in path.parts, "Unsafe tracked path"
        )
        file = project / path
        proof = file_binding(file)
        if relative in expected:
            mode, oid = expected[relative]
            before = file.stat()
            require(
                bool(before.st_mode & 0o111) == (mode == "100755"),
                "Tracked executable mode differs",
            )
            blob = hashlib.new(algorithm)
            blob.update(f"blob {before.st_size}\0".encode())
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    blob.update(block)
            require(blob.hexdigest() == oid, "Tracked bytes differ from declared commit")
            require(file_binding(file) == proof, "Tracked file changed during commit comparison")
        result[relative] = proof["sha256"]
    require(
        (project / PRODUCER).resolve() == Path(__file__).resolve(), "Capture source origin differs"
    )
    require(
        git(project, "rev-parse", "HEAD").decode().strip() == expected_head, "Source head changed"
    )
    require(
        git(project, "ls-tree", "-r", "-z", "--full-tree", "HEAD") == tree, "Source tree changed"
    )
    require(git(project, "ls-files", "--stage", "-z") == index, "Source index changed")
    require(not git(project, "status", "--porcelain").strip(), "Source became dirty")
    return {"root": str(project), "files": result}


def validate_output(output, roots):
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), "Output must be fresh")
    require(output.parent.resolve(strict=True) == output.parent, "Noncanonical output parent")
    for root in roots:
        require(
            not output.is_relative_to(Path(root).resolve(strict=True)),
            "Output is inside an input tree",
        )
    return output


def verify_actual_import_resolution(inventory, search_path=None):
    for name in ("numpy", "torch", "mlx"):
        spec = importlib.machinery.PathFinder.find_spec(
            name, sys.path if search_path is None else search_path
        )
        require(spec is not None, "Missing actual package resolution")
        locations = ([spec.origin] if spec.origin else []) + list(
            spec.submodule_search_locations or []
        )
        actual = [str(Path(path).resolve(strict=True)) for path in locations]
        require(
            actual == inventory["layout"]["packages"][name],
            "Actual import path shadows the declared package",
        )


def import_roots(
    torch_runtime, numpy_overlay, mlx_bundle, dependencies, *, supplemental_dependencies=()
):
    roots = [
        directory(torch_runtime),
        directory(numpy_overlay),
        directory(mlx_bundle / "runtime/python"),
        directory(mlx_bundle),
        directory(dependencies),
        *(directory(path) for path in supplemental_dependencies),
    ]
    require(len(set(roots)) == len(roots), "Duplicate runtime root")
    return roots


def load_control_plane(project, original_path):
    # Dependency directories must not shadow stdlib imports used to establish
    # their own authority. This path is supplied by the isolated interpreter.
    sys.path[:] = [str(project), *original_path]
    return tuple(
        importlib.import_module("transformer." + name)
        for name in ("gpu_owned_study", "gpu_study_runtime", "gpu_owned_backend")
    )


def require_launch():
    require(
        sys.flags.safe_path
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
        and not sys.flags.optimize
        and sys.version_info[:2] == (3, 12)
        and sys.flags.hash_randomization == 0
        and os.environ.get("PYTHONHASHSEED") == "0",
        "Capture requires canonical CPython 3.12 -P -S -B and fixed hash seed zero",
    )
    require(sys.platform == "darwin" and platform.machine() == "arm64", "macOS ARM64 required")
    require(
        Path(sys.executable).absolute() == Path(sys.executable).resolve(strict=True),
        "Launch the canonical interpreter path",
    )
    require(
        not any(
            key.split(".", 1)[0]
            in {"torch", "torchgen", "functorch", "mlx", "numpy", "transformer"}
            for key in sys.modules
        ),
        "Capture must start before framework or project imports",
    )
    require(
        not any("editable" in type(finder).__module__ for finder in sys.meta_path),
        "Editable import hook active",
    )
    python_environment = {
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    require(
        all(
            key in python_environment and value == python_environment[key]
            for key, value in os.environ.items()
            if key.startswith("PYTHON")
        )
        and not any(key.startswith("DYLD_") for key in os.environ),
        "Undeclared Python or loader environment override",
    )
    require(
        not any(
            key in os.environ
            for key in (
                "PYTHONPATH",
                "PYTHONHOME",
                "DYLD_INSERT_LIBRARIES",
                "DYLD_LIBRARY_PATH",
                "DYLD_FRAMEWORK_PATH",
            )
        ),
        "Import override present",
    )


def half_v3_selection(project, source, original, plan_path, plan_sha, torch_runtime):
    """Authenticate a built development candidate; never change formal selection."""
    require(plan_path is not None and plan_sha is not None, "Both candidate plan bindings required")
    relative = "integration/gpu-owned/candidate_half_v3_validation.py"
    path = project / relative
    require(
        file_binding(path)["sha256"] == source["files"].get(relative), "Unbound candidate helper"
    )
    spec = importlib.util.spec_from_file_location("captured_half_v3_candidate", path)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    plan, _ = candidate.authenticate_plan(plan_path, plan_sha)
    plan_binding = file_binding(plan_path)
    require(plan_binding["sha256"] == plan_sha, "Candidate plan changed after authentication")
    require(Path(plan["runtime"]) == torch_runtime, "Capture runtime differs from candidate plan")
    profile = candidate.candidate_profile(plan.get("candidate", "half-v3"))
    contract, _, _ = candidate.selection_contract(plan.get("candidate", "half-v3"))
    selected = json.loads(json.dumps(original))
    selected["pytorch"] = {
        "kind": "project_maintained_full_source_build",
        "base": contract["source_revision"],
        "patches": [row["sha256"] for row in contract["component_patches"]],
        "combined_patch_sha256": contract["combined_patch_sha256"],
        "build_run": int(profile["run"]),
        "runtime_manifest_sha256": plan["runtime_manifest"]["sha256"],
        "engines": ["cpu_torch", "mps"],
        "same_distribution_required": True,
    }
    if plan.get("candidate") == "numpy2-v1":
        selected["pytorch"]["numpy_abi_version"] = "2.2.6"
    return selected, {
        "plan": plan_binding,
        "helper": file_binding(path),
        "schema": candidate.SCHEMA,
        "formal_eligible": False,
        "scope": "New development runtime identity; historical formal selection unchanged",
    }


def mlx_test_compatibility_selection(project, source, original, mlx_bundle, receipt_sha):
    """Reuse unchanged native bytes after one explicitly declared test-only fix."""
    relative = "integration/gpu-owned/derive_mlx_test_compat.py"
    helper = project / relative
    require(
        file_binding(helper)["sha256"] == source["files"].get(relative),
        "Unbound MLX compatibility helper",
    )
    spec = importlib.util.spec_from_file_location("captured_mlx_test_compat", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    proof = file_binding(mlx_bundle / "TEST-COMPATIBILITY.json")
    manifest = file_binding(mlx_bundle / "MANIFEST.json")
    require(proof["sha256"] == receipt_sha, "MLX compatibility receipt changed")
    _, contract = module.authenticate(mlx_bundle, receipt_sha)
    require(
        file_binding(mlx_bundle / "TEST-COMPATIBILITY.json") == proof
        and file_binding(mlx_bundle / "MANIFEST.json") == manifest,
        "MLX compatibility authority changed during authentication",
    )
    require(
        contract["original_manifest_sha256"] == original["mlx"]["runtime_manifest_sha256"]
        and contract["native_build_run"] == original["mlx"]["build_run"],
        "MLX compatibility native authority differs",
    )
    selected = json.loads(json.dumps(original))
    selected["mlx"]["patches"].append(contract["patch_sha256"])
    selected["mlx"]["runtime_manifest_sha256"] = manifest["sha256"]
    selected["mlx"]["test_source_compatibility_receipt_sha256"] = receipt_sha
    return selected, {
        "receipt": proof,
        "helper": file_binding(helper),
        "native_compilation": False,
        "scope": "Python test version parsing only; native sources and binaries unchanged",
        "formal_eligible": False,
    }


def precision_api_capsule(project, source, framework, capsule, receipt_sha):
    """Authenticate an additive Python library extension and its full native lineage."""
    require(framework in {"pytorch", "mlx"}, "Unknown precision API framework")
    relative = "integration/gpu-owned/derive_precision_api.py"
    helper = project / relative
    require(
        file_binding(helper)["sha256"] == source["files"].get(relative),
        "Unbound precision API derivation helper",
    )
    spec = importlib.util.spec_from_file_location("captured_precision_api", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipt = file_binding(capsule / "PRECISION-API.json")
    require(receipt["sha256"] == receipt_sha, "Precision API receipt changed")
    proof, _ = module.authenticate(capsule, receipt_sha)
    require(proof["framework"] == framework, "Precision API framework differs")
    require(
        file_binding(capsule / "PRECISION-API.json") == receipt,
        "Precision API receipt changed during authentication",
    )
    return proof, dict(
        receipt=receipt,
        helper=file_binding(helper),
        native_compilation=False,
        formal_eligible=False,
    )


def precision_api_selection(original, framework, capsule, proof, binding):
    roots = proof["original"]["roots"]
    old_manifest = (
        Path(roots["runtime"]) / "runtime-manifest.json"
        if framework == "pytorch"
        else Path(roots["bundle"]) / "MANIFEST.json"
    )
    require(
        file_binding(old_manifest)["sha256"] == original[framework]["runtime_manifest_sha256"]
        and proof["original"]["native_build_run"] == original[framework]["build_run"],
        "Precision API native authority differs from selected baseline",
    )
    manifest = capsule / (
        "runtime/runtime-manifest.json" if framework == "pytorch" else "MANIFEST.json"
    )
    selected = json.loads(json.dumps(original))
    selected[framework]["runtime_manifest_sha256"] = file_binding(manifest)["sha256"]
    selected[framework]["precision_api"] = dict(
        module="torch.nn.precise_fp32" if framework == "pytorch" else "mlx.nn.precise_fp32",
        contract_sha256=proof["contract"]["sha256"],
        derivation_receipt_sha256=binding["receipt"]["sha256"],
        stock_operators_unchanged=True,
        formal_eligible=False,
    )
    return selected


def capture(
    project,
    expected_head,
    torch_runtime,
    numpy_overlay,
    mlx_bundle,
    dependencies,
    output,
    *,
    half_v3_plan=None,
    half_v3_plan_sha256=None,
    mlx_test_compatibility_sha256=None,
    supplemental_dependencies=(),
    torch_precision_api_root=None,
    torch_precision_api_sha256=None,
    mlx_precision_api_sha256=None,
):
    require_launch()
    project = directory(project)
    paths = import_roots(
        torch_runtime,
        numpy_overlay,
        mlx_bundle,
        dependencies,
        supplemental_dependencies=supplemental_dependencies,
    )
    output = validate_output(output, [project, *paths, Path(sys.prefix), Path(sys.base_prefix)])
    prefix = output / "unused-bytecode-cache"
    require(not prefix.exists() and not prefix.is_symlink(), "Bytecode prefix must be unused")
    sys.pycache_prefix = str(prefix)
    record = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "status": "started",
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "source_head": expected_head,
        "supplemental_dependency_roots": [
            str(directory(path)) for path in supplemental_dependencies
        ],
        "producer": file_binding(Path(__file__).resolve()),
        "launch": {
            "executable": str(Path(sys.executable).resolve()),
            "flags": ["-P", "-S", "-B"],
            "hash_seed": 0,
            "bytecode_prefix": str(prefix),
        },
    }
    created = False
    try:
        source = source_inventory(project, expected_head)
        original_path = list(sys.path)
        owned, runtime, backend = load_control_plane(project, original_path)
        selection = owned.FRAMEWORKS
        precision = {}
        require(
            (torch_precision_api_root is None) == (torch_precision_api_sha256 is None),
            "Both Torch precision capsule bindings required",
        )
        for name, capsule, digest in (
            ("pytorch", torch_precision_api_root, torch_precision_api_sha256),
            ("mlx", mlx_bundle, mlx_precision_api_sha256),
        ):
            if digest is not None:
                capsule = directory(capsule)
                if name == "pytorch":
                    require(torch_runtime == capsule / "runtime", "Torch precision runtime differs")
                precision[name] = (
                    capsule,
                    *precision_api_capsule(project, source, name, capsule, digest),
                )
        original_torch = (
            Path(precision["pytorch"][1]["original"]["roots"]["runtime"])
            if "pytorch" in precision
            else torch_runtime
        )
        original_mlx = (
            Path(precision["mlx"][1]["original"]["roots"]["bundle"])
            if "mlx" in precision
            else mlx_bundle
        )
        if half_v3_plan is not None or half_v3_plan_sha256 is not None:
            selection, record["development_candidate"] = half_v3_selection(
                project,
                source,
                selection,
                half_v3_plan,
                half_v3_plan_sha256,
                original_torch,
            )
        if mlx_test_compatibility_sha256 is not None:
            selection, record["mlx_test_compatibility"] = mlx_test_compatibility_selection(
                project, source, selection, original_mlx, mlx_test_compatibility_sha256
            )
        for name, (capsule, proof, binding) in precision.items():
            selection = precision_api_selection(selection, name, capsule, proof, binding)
            record.setdefault("precision_api_derivations", {})[name] = binding
        proposed_path = [str(project), *map(str, paths), *original_path]
        inventory = runtime.discover_runtime(Path("/"), paths)
        verify_actual_import_resolution(inventory, proposed_path)
        # This also covers resolved symlink targets discovered by the read-only scan.
        validate_output(output, inventory["layout"]["scan_roots"])
        validate_output(output, {Path("/", name).parent for name in inventory["files"]})
        output.mkdir()
        created = True
        require(
            inventory["layout"]["interpreter"] == inventory["layout"]["interpreter_resolved"]
            and inventory["layout"]["prefix"] == inventory["layout"]["base_prefix"],
            "Isolated runtime layout is inconsistent",
        )
        torch_manifest = file_binding(torch_runtime / "runtime-manifest.json")
        mlx_manifest = file_binding(mlx_bundle / "MANIFEST.json")
        for name, actual in (("pytorch", torch_manifest), ("mlx", mlx_manifest)):
            require(
                actual["sha256"] == selection[name]["runtime_manifest_sha256"],
                "Unselected framework manifest",
            )
        combined = selection["pytorch"].get("combined_patch_sha256")
        if combined is None:
            require(len(selection["pytorch"]["patches"]) == 1, "Missing combined patch identity")
            combined = selection["pytorch"]["patches"][0]
        receipt = {
            "schema": backend.SCHEMA,
            "protocol": PROTOCOL,
            "runtime_identity_sha256": hashlib.sha256(canonical(inventory["files"])).hexdigest(),
            "project": source,
            "bytecode_policy": {"mode": "fresh_nonexistent_prefix", "writes_disabled": True},
            "frameworks": {
                "pytorch": {
                    "selection": selection["pytorch"],
                    "manifest": torch_manifest,
                    "payload_root": str(torch_runtime),
                    "combined_patch_sha256": combined,
                },
                "mlx": {
                    "selection": selection["mlx"],
                    "manifest": mlx_manifest,
                    "payload_root": str(mlx_bundle),
                },
            },
        }
        for name, value in (
            ("runtime-inventory.json", inventory),
            ("owned-runtime.json", receipt),
            ("framework-selection.json", selection),
        ):
            with (output / name).open("x") as stream:
                json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
        authority = backend.authenticate_runtime(
            "/",
            inventory,
            selection,
            output / "owned-runtime.json",
            sha(output / "owned-runtime.json"),
        )
        backend.verify_loaded_project_modules(receipt["project"])
        authority.verify()
        # Activate the now-authenticated roots without executing site hooks.
        sys.path[:] = proposed_path
        verify_actual_import_resolution(inventory)
        require(
            source_inventory(project, expected_head) == source, "Project changed during capture"
        )
        require(
            record["producer"] == file_binding(Path(__file__).resolve()),
            "Producer changed during capture",
        )
        require(
            sys.pycache_prefix == str(prefix) and not prefix.exists(), "Bytecode isolation changed"
        )
        require(
            not any(key.split(".", 1)[0] in {"torch", "mlx", "numpy"} for key in sys.modules),
            "Capture unexpectedly imported a framework",
        )
        record.update(
            status="identity_captured_not_qualified",
            runtime_root="/",
            runtime_inventory=file_binding(output / "runtime-inventory.json"),
            runtime_receipt=file_binding(output / "owned-runtime.json"),
            framework_selection=file_binding(output / "framework-selection.json"),
            runtime_identity_sha256=receipt["runtime_identity_sha256"],
            project_identity_sha256=hashlib.sha256(canonical(source)).hexdigest(),
            runtime_files=len(inventory["files"]),
            project_files=len(source["files"]),
            limits=[
                "Identity capture executes no tensor operation or numerical gate.",
                "Loaded framework origins must be checked in the subsequent real worker.",
                "System shared-cache library bytes are represented by OS build identity.",
            ],
        )
    except BaseException:
        record.update(status="failed", exception=traceback.format_exc())
        raise
    finally:
        if created:
            with (output / "capture.json").open("x") as stream:
                json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "project",
        "torch-runtime",
        "numpy-overlay",
        "mlx-bundle",
        "dependencies",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--half-v3-plan", type=Path)
    parser.add_argument("--half-v3-plan-sha256")
    parser.add_argument("--mlx-test-compatibility-sha256")
    parser.add_argument("--supplemental-dependencies", type=Path, action="append", default=[])
    parser.add_argument("--torch-precision-api-root", type=Path)
    parser.add_argument("--torch-precision-api-sha256")
    parser.add_argument("--mlx-precision-api-sha256")
    args = parser.parse_args()
    result = capture(
        args.project,
        args.expected_head,
        args.torch_runtime,
        args.numpy_overlay,
        args.mlx_bundle,
        args.dependencies,
        args.output,
        half_v3_plan=args.half_v3_plan,
        half_v3_plan_sha256=args.half_v3_plan_sha256,
        mlx_test_compatibility_sha256=args.mlx_test_compatibility_sha256,
        supplemental_dependencies=args.supplemental_dependencies,
        torch_precision_api_root=args.torch_precision_api_root,
        torch_precision_api_sha256=args.torch_precision_api_sha256,
        mlx_precision_api_sha256=args.mlx_precision_api_sha256,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "runtime_files", "project_files", "runtime_identity_sha256")
            }
        )
    )


if __name__ == "__main__":
    main()
