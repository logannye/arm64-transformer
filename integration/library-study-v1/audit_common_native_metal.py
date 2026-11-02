"""Recompute native/Metal common-state accuracy against independent FP64 equations."""

import argparse
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        "native_common_audit_" + name, HERE / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D, COMMON, RAW = load("audit_development"), load("common_native_metal"), load("evidence")
A, C = D.A, D.C


def execute(args):
    C.require(
        sys.flags.isolated and sys.flags.dont_write_bytecode, "Use independent audit Python -I -B"
    )
    C.require(
        not any(n.split(".", 1)[0] in {"transformer", "torch", "mlx"} for n in sys.modules),
        "Independent auditor requires fresh source/framework imports",
    )
    control = A.source_role(HERE.parents[1], args.control_revision, control=True)
    math = A.source_role(args.math_root, C.MATH_REVISION)
    roles = dict(control=control, math=math)
    record = C.read(args.receipt, args.receipt_sha256)
    C.bound(record["job"])
    job = C.read(record["job"]["path"], record["job"]["sha256"])
    load("worker").validate_job(job)
    C.require(
        job["kind"] == "native_metal_common_state"
        and job["roles"] == roles
        and job["capture"] == C.binding(args.capture)
        and Path(job["output"]) == Path(args.receipt).parent
        and os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(job["width"]),
        "Foreign common-state worker, capture or audit width",
    )
    capture = C.read(args.capture, args.capture_sha256)
    C.bound(capture["runtime_receipt"])
    receipt = C.read(capture["runtime_receipt"]["path"], capture["runtime_receipt"]["sha256"])
    C.require(
        receipt["project"] == dict(root=math["root"], files=math["files"]), "Wrong math source"
    )
    D.worker_identity(record, job, capture["runtime_identity_sha256"], roles)
    D.N.verify(job["native_assets"], math)
    sys.path.insert(0, math["root"])
    import numpy as np
    import torch

    from transformer.gpu_admission_checks import numpy_model
    from transformer.gpu_owned_backend import validate_loaded_origin_proof
    from transformer.gpu_repeatability import _torch_reference

    torch.set_num_threads(1)
    C.bound(capture["runtime_inventory"])
    inventory = C.read(capture["runtime_inventory"]["path"], capture["runtime_inventory"]["sha256"])
    D.runtime_validation(record, job, capture, inventory, roles)
    saved_imports = load("qualify_suites").imports
    for phase in ("before", "after"):
        proof = record["origins_" + phase]
        validate_loaded_origin_proof(
            proof["pytorch"],
            inventory=inventory,
            runtime_root=capture["runtime_root"],
            receipt=receipt,
        )
        saved_imports(
            proof["loaded_python_files"],
            dict(inventory, root=capture["runtime_root"]),
            A.merged_project(roles),
        )
    common, independent = COMMON.helpers(SimpleNamespace(roles=roles))
    source, fixture = COMMON.trajectory(job, SimpleNamespace(roles=roles))
    count = 8 * job["width"] ** 2 + 301 * job["width"] + 128
    rows = np.fromfile(fixture["training"]["path"], "<i4").reshape(2048, 33)
    gradient = (
        np.random.default_rng(np.random.SeedSequence([20261103, job["width"], 91]))
        .normal(0, 0.01, count)
        .astype(np.float32)
    )
    result = record["outputs"]
    C.require(
        result["schema"] == "frozen-library-native-metal-common-state-v1"
        and result["engine"] == job["engine"]
        and result["width"] == job["width"]
        and result["seed"] == 20261103
        and result["trajectory"] == job["trajectory"]
        and result["raw_store"] == job["raw_store"]
        and result["expected_cases"] == common.cases()
        and [r["name"] for r in result["cases"]] == common.cases(),
        "Incomplete or foreign common-state case inventory",
    )
    checked = []
    for row in result["cases"]:
        family, step_text, shape = row["name"].split("/")
        state = independent.checkpoint(source, int(step_text.split("-")[1]), count)
        raw = row["actual"]
        C.require(
            type(raw["bytes"]) is int and 0 < raw["bytes"] <= RAW.MAX_BYTES,
            "Oversized common output",
        )
        names = (
            ("logits", "loss", "gradient")
            if family == "model"
            else ("parameters", "m", "v", "step")
        )
        with io.BytesIO() as stream:
            recipe = RAW.storage(math["root"]).reconstruct(
                job["raw_store"], raw["sha256"], stream, max_bytes=RAW.MAX_BYTES
            )
            C.require(recipe["bytes"] == raw["bytes"], "Raw common output size differs")
            stream.seek(0)
            actual = RAW.decode_arrays(stream, names)
        C.require(all(np.isfinite(v).all() for v in actual.values()), "Nonfinite common output")
        oracle = {}
        if family == "model":
            batch = int(shape.split("-")[0][1:])
            x, y = np.ascontiguousarray(rows[:batch, :-1]), np.ascontiguousarray(rows[:batch, 1:])
            gold = numpy_model(state["parameters"], x, y)
            double = _torch_reference(state["parameters"], x, y, backward=True)
            oracle = {k: independent.accuracy(double[k], gold[k], "oracle") for k in gold}
            C.require(
                actual["logits"].dtype == actual["gradient"].dtype == np.float32
                and actual["loss"].dtype in (np.dtype("f4"), np.dtype("f8")),
                "Wrong model dtype",
            )
            for key, value in actual.items():
                fp = COMMON.fingerprint(value)
                C.require(row["repetitions"][key] == [fp, fp], "Model repetitions differ")
            checks = {
                k: independent.accuracy(
                    actual[k], gold[k], "gradient" if k == "gradient" else "forward"
                )
                for k in gold
            }
        else:
            gold = independent.adam(state, gradient, int(shape.split("-")[1]))
            C.require(
                all(
                    actual[k].dtype == np.float32 and actual[k].shape == (count,)
                    for k in ("parameters", "m", "v")
                )
                and actual["step"].dtype == np.int64
                and actual["step"].shape == ()
                and (actual["v"] >= 0).all()
                and row["repetitions"] is None,
                "Wrong common optimizer state",
            )
            checks = {
                k: independent.accuracy(actual[k], gold[k], "exact" if k == "step" else "optimizer")
                for k in gold
            }
        for saved, fresh in ((row["comparisons"], checks), (row["oracle"], oracle)):
            C.require(set(saved) == set(fresh), "Missing common comparison")
            for name, decision in fresh.items():
                C.require(
                    all(saved[name][key] == decision[key] for key in ("passed", "atol", "rtol")),
                    "Raw common decision differs from reported accuracy",
                )
        passed = all(c["passed"] for c in [*checks.values(), *oracle.values()])
        C.require(row["passed"] is passed, "Common pass flag differs from raw arrays")
        checked.append(dict(name=row["name"], passed=passed, comparisons=checks, oracle=oracle))
    passed = all(r["passed"] for r in checked)
    C.require(result["passed"] is passed, "Common aggregate differs from raw cases")
    if job["engine"] == "metal":
        D.E.validate_capabilities("metal", result["capabilities"])
        C.require(
            result["metal_library_origin"] == C.binding(Path(control["root"]) / A.METAL_LIBRARY),
            "Common Metal library origin differs",
        )
    origins = D.ORIGINS.project_origins(math, dict(sys.modules))
    C.require(
        "torch.nn.precise_fp32" not in sys.modules and "mlx.core" not in sys.modules,
        "Candidate packages must not replace the independent oracle",
    )
    A.verify_role(control, control=True)
    A.verify_role(math)
    C.require(C.binding(args.receipt)["sha256"] == args.receipt_sha256, "Common receipt changed")
    return dict(
        schema="frozen-library-native-metal-common-audit-v1",
        evidence_valid=True,
        numerics_passed=passed,
        cases=checked,
        engine=job["engine"],
        width=job["width"],
        auditor_control_role=control,
        auditor_math_revision=math["revision"],
        auditor_project_origins=origins,
        receipt=C.binding(args.receipt),
        formal_eligible=False,
        formal_fits=0,
        admission_stream_opened=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-revision", required=True)
    parser.add_argument("--math-root", type=Path, required=True)
    for name in ("capture", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    value = execute(args)
    print(json.dumps(value, allow_nan=False))
    raise SystemExit(0 if value["numerics_passed"] else 1)
