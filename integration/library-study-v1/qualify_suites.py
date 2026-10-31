"""Reaudit complete NumPy 2 suites from raw evidence without executing devices.

Use the captured canonical interpreter with -I -S -B. Neither selected candidate package
paths nor reserved operands are activated. Existing frozen component validators
retain the complete test inventories, stock fixture and descendant requirements.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("library_suite_authority", HERE / "authority.py")
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
C = A.C


def load(role, name):
    path = Path(role["root"]) / A.relative(name)
    C.require(C.binding(path)["sha256"] == role["files"][name], "Unbound suite auditor")
    spec = importlib.util.spec_from_file_location("library_suite_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def imports(saved, inventory, project):
    C.require(isinstance(saved, dict) and saved, "Missing retained import inventory")
    roots = [
        (Path(inventory["root"]), inventory["files"]),
        (Path(project["root"]), project["files"]),
    ]
    for name, row in saved.items():
        C.require(isinstance(name, str) and set(row) == {"path", "sha256"}, "Malformed import")
        path = Path(row["path"])
        C.require(path.is_absolute() and str(path) == row["path"], "Noncanonical import")
        hashes = {
            files[path.relative_to(root).as_posix()]
            for root, files in roots
            if path.is_relative_to(root) and path.relative_to(root).as_posix() in files
        }
        C.require(hashes == {row["sha256"]}, "Retained import differs from captured authority")
        C.require(C.binding(path)["sha256"] == row["sha256"], "Live retained import changed")


def execute(args):
    C.require(
        sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode,
        "Use captured canonical Python -I -S -B",
    )
    C.require(
        not any(n.split(".", 1)[0] in {"transformer", "torch", "mlx"} for n in sys.modules),
        "Fresh independent audit process required",
    )
    control = A.source_role(HERE.parents[1], args.control_revision, control=True)
    math = A.source_role(args.math_root, C.MATH_REVISION)
    capture_binding = C.binding(args.capture)
    C.require(capture_binding["sha256"] == args.capture_sha256, "Capture trust anchor differs")
    capture = C.read(args.capture, args.capture_sha256)
    for name in ("runtime_inventory", "runtime_receipt", "framework_selection"):
        C.bound(capture[name])
    inventory = C.read(capture["runtime_inventory"]["path"], capture["runtime_inventory"]["sha256"])
    receipt = C.read(capture["runtime_receipt"]["path"], capture["runtime_receipt"]["sha256"])
    selection = C.read(
        capture["framework_selection"]["path"], capture["framework_selection"]["sha256"]
    )
    C.require(receipt["project"] == dict(root=math["root"], files=math["files"]), "Wrong math role")
    C.require(
        C.canonical(selection) == C.canonical(C.comparison()["framework_selection"]),
        "Unselected framework packages",
    )
    sys.path.insert(0, math["root"])
    from transformer.gpu_owned_backend import (
        authenticate_runtime,
        validate_compilation_probe,
        validate_loaded_origin_proof,
    )
    from transformer.gpu_study_runtime import verify_runtime

    verify_runtime(capture["runtime_root"], inventory)
    authority = authenticate_runtime(
        capture["runtime_root"],
        inventory,
        selection,
        capture["runtime_receipt"]["path"],
        capture["runtime_receipt"]["sha256"],
    )
    authority.verify()
    mps = load(math, "integration/gpu-owned/owned_pytorch_suites.py")
    mlx = load(math, "integration/gpu-owned/owned_mlx_suites.py")
    summaries = {}
    summary_bindings = []
    for name, path, digest in (
        ("mps", args.mps_summary, args.mps_summary_sha256),
        ("mlx", args.mlx_summary, args.mlx_summary_sha256),
    ):
        value = C.read(path, digest)
        bound = C.binding(path)
        C.require(bound["sha256"] == digest, "Summary changed during authentication")
        summary_bindings.append(bound)
        C.require(
            value["complete"] is True
            and value["passed"] is True
            and value["formal_fits"] == 0
            and value["formal_admission_opened"] is False,
            "Incomplete full-suite summary",
        )
        C.bound(value["plan"])
        plan = C.read(value["plan"]["path"], value["plan"]["sha256"])
        C.require(plan["runtime_root"] == capture["runtime_root"], "Wrong suite runtime")
        for key, capture_key in (
            ("inventory", "runtime_inventory"),
            ("receipt", "runtime_receipt"),
            ("selection", "framework_selection"),
        ):
            C.require(
                plan["authority_inputs"][key] == capture[capture_key], "Wrong suite authority"
            )
        summaries[name] = value
    mps_summary = summaries["mps"]
    mps_plan = C.read(mps_summary["plan"]["path"])
    C.require(mps_plan["suite_contract"] == "numpy2-v1", "Explicit NumPy 2 MPS contract required")
    C.bound(mps_summary["receipt"])
    mps_receipt = C.read(mps_summary["receipt"]["path"], mps_summary["receipt"]["sha256"])
    expected = mps.suite_inventory(mps_plan, selection)
    mps.validate_execution(
        mps_receipt,
        Path(mps_summary["receipt"]["path"]).parent,
        mps_summary["process"],
        mps_plan,
        mps_summary["plan"]["sha256"],
        authority,
        inventory,
        selection,
        expected,
    )
    mlx_summary = summaries["mlx"]
    mlx_plan = C.read(mlx_summary["plan"]["path"])
    C.require(
        mlx_plan["suite_contract"] == "torch-dev-compat-numpy2",
        "Explicit NumPy 2 MLX contract required",
    )
    expected, expected_binding = mlx.suite_inventory(mlx_plan["suite_contract"])
    C.require(mlx_plan["suite_inventory"] == expected_binding, "Wrong MLX inventory")
    bundle, sources = mlx.test_source(authority, expected)
    C.require(
        mlx_plan["bundle"] == str(bundle)
        and mlx_plan["test_source_sha256"] == sources
        and mlx_plan["descendant_contract"] == mlx.subprocess_contract(bundle, expected)
        and mlx_plan["python_test_driver"] == mlx.python_test_driver(inventory, bundle),
        "MLX plan, source or descendant contract differs",
    )
    C.require(
        [r["name"] for r in mlx_summary["suites"]] == ["mlx_cpp", "mlx_python"],
        "Incomplete MLX suite denominator",
    )
    for row in mlx_summary["suites"]:
        C.bound(row["receipt"])
        record = C.read(row["receipt"]["path"], row["receipt"]["sha256"])
        C.require(
            record["natural_gpu_default"] is True
            and record["numpy"]["version"] == selection["pytorch"]["numpy_abi_version"]
            and record["numpy"]["path"] in inventory["layout"]["packages"]["numpy"],
            "MLX suite used a different device or NumPy ABI",
        )
        for phase in ("before", "after"):
            imports(
                record["loaded_python_files_" + phase],
                dict(inventory, root=capture["runtime_root"]),
                receipt["project"],
            )
            proof = record["native_origin_verification"][phase]
            if row["name"] == "mlx_python":
                validate_loaded_origin_proof(
                    proof["loaded_images"],
                    inventory=inventory,
                    runtime_root=capture["runtime_root"],
                    receipt=receipt,
                )
        C.require(
            mlx.audit_execution(
                row,
                mlx_plan,
                mlx_summary["plan"]["sha256"],
                expected,
                inventory,
                receipt,
                validate_compilation_probe,
            ),
            "Raw MLX suite audit failed",
        )
    from_source = load(control, "integration/library-study-v1/audit.py")
    origins = from_source.project_origins(math, dict(sys.modules))
    C.require(
        "torch.nn.precise_fp32" not in sys.modules and "mlx.core" not in sys.modules,
        "Suite audit must not activate the selected numerical candidate",
    )
    A.verify_role(control, control=True)
    A.verify_role(math)
    authority.verify()
    for item in [capture_binding, *summary_bindings]:
        C.bound(item)
    return dict(
        schema="frozen-library-full-suite-audit-v1",
        evidence_valid=True,
        full_suites_passed=True,
        capture=capture_binding,
        control=control,
        math=math,
        summaries=summary_bindings,
        oracle_project_origins=origins,
        formal_eligible=False,
        formal_fits=0,
        admission_stream_opened=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-revision", required=True)
    parser.add_argument("--math-root", type=Path, required=True)
    for name in ("capture", "mps-summary", "mlx-summary"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    C.write(args.output, execute(args))
    print(json.dumps(dict(full_suites_passed=True, output=C.binding(args.output))))


if __name__ == "__main__":
    main()
