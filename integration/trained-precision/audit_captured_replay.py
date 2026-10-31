"""Authenticate captured-runtime replay provenance separately from raw numerics.

This live audit imports no numerical framework. The ordinary collection auditor
must additionally recompute every numerical comparison from the retained arrays.
Neither audit admits a framework repair or a formal performance experiment.
"""

import argparse
import copy
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SUITE = load("captured_audit_suite", "integration/gpu-owned/owned_mlx_suites.py")
MPS = load("captured_audit_mps", "integration/gpu-owned/owned_pytorch_suites.py")
require = SUITE.require


def known_file(path, runtime, runtime_root, project):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "Noncanonical loaded file")
    expected = set()
    for root, files in (
        (Path(runtime_root), runtime["files"]),
        (Path(project["root"]), project["files"]),
    ):
        if path.is_relative_to(root):
            value = files.get(path.relative_to(root).as_posix())
            if value is not None:
                expected.add(value)
    require(len(expected) == 1, "Loaded file is outside or ambiguous in captured authority")
    SUITE.pinned(path, expected.pop())


def audit_origins(value, engine, capture, runtime, receipt):
    require(
        value["runtime_identity_sha256"] == capture["runtime_identity_sha256"],
        "Loaded runtime identity changed",
    )
    numpy = value["numpy"]
    require(
        numpy["version"]
        == receipt["frameworks"]["pytorch"]["selection"].get("numpy_abi_version", "1.26.4")
        and numpy["path"] in runtime["layout"]["packages"]["numpy"],
        "Captured NumPy differs",
    )
    imported = value["loaded_python_files"]
    require(
        imported and imported.get("numpy", {}).get("path") == numpy["path"],
        "NumPy import proof missing",
    )
    for name, row in imported.items():
        require(isinstance(name, str) and set(row) == {"path", "sha256"}, "Malformed import proof")
        known_file(row["path"], runtime, capture["runtime_root"], receipt["project"])
        SUITE.pinned(row["path"], row["sha256"])
    native = copy.deepcopy(value["pytorch"])
    require(
        set(native) == {"module_origins", "framework_module_origins", "native_libraries"},
        "Malformed native proof",
    )
    scopes = set(native["framework_module_origins"])
    require(
        scopes == {"pytorch"} or (engine == "mlx" and scopes == {"pytorch", "mlx"}),
        "Unexpected framework scope",
    )
    for path in native["native_libraries"]:
        known_file(path, runtime, capture["runtime_root"], receipt["project"])
    native["framework_module_origins"] = {"pytorch": native["framework_module_origins"]["pytorch"]}
    root = Path(capture["runtime_root"])
    native["files"] = {
        str(Path(p).relative_to(root)): runtime["files"][str(Path(p).relative_to(root))]
        for p in native["native_libraries"]
    }
    MPS.validate_native_proof(native, runtime, root)
    # Reuse the complete retained Python/Torch namespace consistency validator.
    MPS.validate_loaded_proof(
        imported,
        native,
        runtime,
        root,
        receipt["project"],
        Path(receipt["project"]["root"]),
        {},
        {},
    )
    if engine == "mlx":
        mlx = copy.deepcopy(value["mlx"])
        require(
            set(mlx["framework_module_origins"]) == {"pytorch", "mlx"}
            and mlx["framework_module_origins"]["pytorch"]
            == value["pytorch"]["framework_module_origins"]["pytorch"],
            "Framework origin maps disagree",
        )
        bundle = Path(receipt["frameworks"]["mlx"]["payload_root"])
        mlx["framework_module_origins"] = {"mlx": mlx["framework_module_origins"]["mlx"]}
        # Both frameworks are loaded for the dual-FP64 audit. The descendant
        # validator expects only MLX images, so validate that subset while the
        # complete native-library list above remains independently authenticated.
        mlx["native_libraries"] = [
            p for p in mlx["native_libraries"] if Path(p).is_relative_to(bundle / "runtime")
        ]
        child = dict(
            runtime_root=str(root),
            numpy=numpy,
            loaded_python_files_before=imported,
            loaded_python_files_after=imported,
            native_origins_before=mlx,
            native_origins_after=mlx,
        )
        SUITE.validate_descendant_proofs(child, runtime, root, receipt["project"], bundle)
        # This validates the changing-input compiled probe, not just a flag.
        previous = list(sys.path)
        try:
            sys.path.insert(0, str(ROOT))
            backend = load("captured_audit_backend", "transformer/gpu_owned_backend.py")
            backend.validate_compilation_probe(value["compilation"])
        finally:
            sys.path[:] = previous


def audit(sidecar_path, sidecar_sha, capture_path, capture_sha):
    sidecar_path, capture_path = Path(sidecar_path), Path(capture_path)
    sidecar_binding = SUITE.pinned(sidecar_path, sidecar_sha)
    capture_binding = SUITE.pinned(capture_path, capture_sha)
    sidecar, capture = SUITE.read_json(sidecar_path), SUITE.read_json(capture_path)
    require(
        capture["schema"] == "gpu-owned-runtime-capture-v1"
        and capture["status"] == "identity_captured_not_qualified"
        and capture["formal_eligible"] is False
        and capture["formal_fits"] == 0
        and capture["admission_stream_opened"] is False,
        "Wrong runtime capture",
    )
    require(
        sidecar["schema"] == "trained-precision-captured-runtime-replay-v1"
        and sidecar["complete"] is True
        and sidecar["postflight_passed"] is True
        and sidecar["formal_eligible"] is False
        and sidecar["formal_fits"] == 0
        and sidecar["admission_stream_opened"] is False
        and not any(k in sidecar for k in ("error", "postflight_error")),
        "Incomplete captured replay",
    )
    require(
        sidecar["capture"] == capture_binding
        and sidecar["runtime_receipt"] == capture["runtime_receipt"]
        and sidecar["runtime_identity_sha256"] == capture["runtime_identity_sha256"],
        "Replay capture differs",
    )
    for key in ("runtime_inventory", "framework_selection", "runtime_receipt"):
        SUITE.binding(capture[key]["path"], capture[key])
    receipt = SUITE.read_json(capture["runtime_receipt"]["path"])
    runtime = SUITE.read_json(capture["runtime_inventory"]["path"])
    require(
        receipt["runtime_identity_sha256"] == capture["runtime_identity_sha256"],
        "Runtime receipt differs",
    )
    for key, relative in (
        ("producer", "integration/trained-precision/captured_replay.py"),
        ("worker_source", "transformer/trained_precision.py"),
    ):
        proof = sidecar[key]
        require(
            proof["path"] == str(Path(receipt["project"]["root"]) / relative)
            and proof["sha256"] == receipt["project"]["files"][relative],
            "Unbound replay source",
        )
        SUITE.binding(proof["path"], proof)
    worker_binding = sidecar["worker_receipt"]
    require(
        Path(worker_binding["path"]) == sidecar_path.with_name("result.json"),
        "Foreign worker receipt",
    )
    SUITE.binding(worker_binding["path"], worker_binding)
    worker = SUITE.read_json(worker_binding["path"])
    require(
        sidecar["engine"] in {"cpu_torch", "mps", "mlx"}
        and sidecar["width"] in {512, 1024}
        and worker["complete"] is True
        and type(worker["passed"]) is bool
        and worker["engine"] == sidecar["engine"]
        and worker["width"] == sidecar["width"]
        and type(sidecar["pid"]) is int
        and sidecar["pid"] > 0
        and worker["pid"] == sidecar["pid"]
        and "owned_runtime" not in worker
        and sidecar["passed"] is worker["passed"]
        and sidecar["worker_returncode"] == int(not worker["passed"]),
        "Worker identity or numerical outcome differs",
    )
    for phase in ("before", "after"):
        audit_origins(sidecar["origins_" + phase], sidecar["engine"], capture, runtime, receipt)
    if "library_model" in worker:
        require("--library-api" in sidecar["command"], "Library API execution flag missing")
        framework = "mlx" if sidecar["engine"] == "mlx" else "pytorch"
        selected = receipt["frameworks"][framework]["selection"].get("precision_api", {})
        proof = worker["library_api"]
        require(
            selected.get("module") == proof["module"]
            and selected.get("derivation_receipt_sha256")
            == capture["precision_api_derivations"][framework]["receipt"]["sha256"],
            "Library API is absent from the captured selection",
        )
        loaded = sidecar["origins_after"]["loaded_python_files"]
        require(
            loaded.get(proof["module"], {}).get("path") == proof["module_path"],
            "Actual loaded precision API differs from model proof",
        )
        prefix = proof["module"].rsplit(".", 1)[0]
        for name in (
            proof["module"],
            prefix + "._fp32_precise_math",
            prefix + "._fp32_precise_dense",
        ):
            require(name in loaded, "Missing loaded generic precision module")
            known_file(loaded[name]["path"], runtime, capture["runtime_root"], receipt["project"])
    else:
        require(
            "--library-api" not in sidecar["command"],
            "Stock/diagnostic model mislabeled as library API",
        )
    SUITE.pinned(sidecar_path, sidecar_sha)
    SUITE.pinned(capture_path, capture_sha)
    return dict(
        schema="trained-precision-captured-runtime-audit-v1",
        evidence_valid=True,
        numerics_reported_passed=worker["passed"],
        numerics_independently_audited=False,
        sidecar=sidecar_binding,
        capture=capture_binding,
        worker=worker_binding,
        runtime_identity_sha256=capture["runtime_identity_sha256"],
        formal_eligible=False,
        admission_stream_opened=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("sidecar", "sidecar-sha256", "capture", "capture-sha256"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    import json

    print(
        json.dumps(
            audit(args.sidecar, args.sidecar_sha256, args.capture, args.capture_sha256), indent=2
        )
    )
