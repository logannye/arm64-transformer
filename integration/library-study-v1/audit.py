"""Run immutable numerical auditors with explicitly authenticated oracle imports.

Use an existing independent audit environment. Activate only frozen project
source; this never activates the selected candidate package paths or trains.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("frozen_audit_authority", HERE / "authority.py")
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
C = A.C
AUDITORS = {
    "repeatability": "integration/trained-precision/audit_library_repeatability.py",
    "common_state": "integration/trained-precision/audit_library_common_state.py",
}


def project_origins(role, modules):
    result = {}
    root = Path(role["root"])
    for name, module in modules.items():
        if name != "transformer" and not name.startswith("transformer."):
            continue
        p = C.regular(module.__file__)
        C.require(p.is_relative_to(root), "Independent oracle imported another project checkout")
        relative = p.relative_to(root).as_posix()
        bound = C.binding(p)
        C.require(
            role["files"].get(relative) == bound["sha256"], "Unbound independent oracle source"
        )
        result[name] = bound
    C.require(result, "Independent oracle source proof is empty")
    return result


def validate_origins(proof, roles, capture, inventory, engine):
    """Validate retained live origins without importing the selected runtime."""
    from transformer.gpu_owned_backend import (
        validate_compilation_probe,
        validate_loaded_origin_proof,
    )

    C.require(
        proof["runtime_identity_sha256"] == capture["runtime_identity_sha256"]
        and proof["source_roles_sha256"] == C.identity(roles),
        "Retained runtime/source identity differs",
    )
    receipt = C.read(C.bound(capture["runtime_receipt"]))
    required = {"pytorch"} | ({"mlx"} if engine == "mlx" else set())
    C.require(required <= set(proof), "Missing live framework origin proof")
    for name in ("pytorch", "mlx"):
        if name in proof:
            validate_loaded_origin_proof(
                proof[name],
                inventory=inventory,
                runtime_root=capture["runtime_root"],
                receipt=receipt,
            )
    if engine == "mlx":
        validate_compilation_probe(proof["compilation"])
    spec = importlib.util.spec_from_file_location(
        "library_saved_imports", HERE / "qualify_suites.py"
    )
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    suite.imports(
        proof["loaded_python_files"],
        dict(inventory, root=capture["runtime_root"]),
        A.merged_project(roles),
    )


def execute(args):
    C.require(
        sys.flags.isolated and sys.flags.dont_write_bytecode, "Use independent audit Python -I -B"
    )
    C.require(
        not any(n.split(".", 1)[0] in {"transformer", "torch", "mlx"} for n in sys.modules),
        "Independent auditor requires fresh project/framework imports",
    )
    control = A.source_role(HERE.parents[1], args.control_revision, control=True)
    math = A.source_role(args.math_root, C.MATH_REVISION)
    capture = C.read(args.capture, args.capture_sha256)
    C.bound(capture["runtime_receipt"])
    receipt = C.read(capture["runtime_receipt"]["path"], capture["runtime_receipt"]["sha256"])
    C.require(
        receipt["project"] == dict(root=math["root"], files=math["files"]),
        "Audit source differs from the captured numerical implementation",
    )
    # This path intentionally precedes the editable project in the independent
    # environment; candidate runtime paths are never added.
    sys.path.insert(0, math["root"])
    path = Path(math["root"]) / AUDITORS[args.kind]
    C.require(C.binding(path)["sha256"] == math["files"][AUDITORS[args.kind]], "Unbound auditor")
    spec = importlib.util.spec_from_file_location("selected_frozen_raw_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args.kind == "repeatability":
        C.require(
            all((args.first, args.first_sha256, args.second, args.second_sha256))
            and args.receipt is None
            and args.receipt_sha256 is None,
            "Complete repeatability pair required",
        )
        result = module.audit_pair(
            (args.first, args.first_sha256),
            (args.second, args.second_sha256),
            args.capture,
            args.capture_sha256,
        )
    else:
        C.require(
            args.receipt is not None
            and args.receipt_sha256 is not None
            and all(
                v is None for v in (args.first, args.first_sha256, args.second, args.second_sha256)
            ),
            "One common-state receipt required",
        )
        result = module.audit(args.receipt, args.receipt_sha256, args.capture, args.capture_sha256)
    C.require(
        "torch.nn.precise_fp32" not in sys.modules and "mlx.core" not in sys.modules,
        "Selected candidate must not replace the independent FP64 oracle",
    )
    result["auditor_project_module_origins"] = project_origins(math, dict(sys.modules))
    result["auditor_control_role"] = control
    result["auditor_math_revision"] = math["revision"]
    result["auditor_source"] = C.binding(path)
    A.verify_role(control, control=True)
    A.verify_role(math)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=tuple(AUDITORS), required=True)
    parser.add_argument("--control-revision", required=True)
    parser.add_argument("--math-root", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--capture-sha256", required=True)
    for name in ("first", "second", "receipt"):
        parser.add_argument("--" + name, type=Path)
        parser.add_argument("--" + name + "-sha256")
    value = execute(parser.parse_args())
    print(json.dumps(value, indent=2, allow_nan=False))
    raise SystemExit(int(not (value["numerics_passed"] and value.get("exact_passed", True))))
