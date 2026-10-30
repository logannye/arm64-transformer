"""Archive closed full-shape development evidence by engine and batch shape."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import prepare_evidence as archive

UTILITY_SHA256 = "2a30d78a30cfe75e56bf3a133f7611e0544706c4b1171a40732eb113b0b4ba51"
AUDITOR_SHA256 = "8ff4565f77e7f535ec72a77eb03fe3b210b9ba60d55f28408810319887c344e8"
MODEL = "work/gpu-v4-owned/model-full-shape-v1"
MANIFEST = "work/gpu-v4-owned/model-full-shape-manifest-v1.json"
AUDIT = "work/gpu-v4-owned/model-full-shape-array-audit-v1.json"
MANIFEST_SHA256 = "1617562aa8cd1e04c358aa5e4261fb72a6fad529472115f1762462c3ed470954"
MODEL_PRODUCER_SHA256 = "06a772b3e28819739fc63a3f782631840dcd22cb1eb8d71b9d09b002670b6277"
PROTOCOL = "gpu-v4-owned-full-shape-development-v1"
WIDTHS = (32, 64, 128, 256, 512, 1024)
ENGINES = ("cpu_torch", "mps", "mlx")
SHAPES = ((1, 32), (16, 32))
RAW_NAMES = (
    "operands.npz",
    "initial.npz",
    "oracles.npz",
    "checkpoint-step3.npz",
    "checkpoint-step6.npz",
    "checkpoint-step3-logits.npz",
    "final-logits.npz",
)


def anchors(summary_sha256, audit_sha256):
    def require_hash(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)

    archive.require(
        require_hash(summary_sha256) and require_hash(audit_sha256),
        "External closed summary and audit hashes are required",
    )
    return {MODEL + "/summary.json": summary_sha256, MANIFEST: MANIFEST_SHA256, AUDIT: audit_sha256}


def validate_anchors(value):
    archive.require(
        isinstance(value, dict) and set(value) == {MODEL + "/summary.json", MANIFEST, AUDIT},
        "Wrong full-shape closure identities",
    )
    archive.require(
        value == anchors(value[MODEL + "/summary.json"], value[AUDIT]),
        "Full-shape manifest changed",
    )


def cells():
    return [
        {
            "width": w,
            "batch": b,
            "length": t,
            "engine": e,
            "process_index": p,
            "id": f"w{w}-b{b}-t{t}-{e}-p{p}",
        }
        for w in WIDTHS
        for b, t in SHAPES
        for e in ENGINES
        for p in (0, 1)
    ]


def helper_identity():
    archive.require(
        archive.stable_hash(Path(archive.__file__))["sha256"] == UTILITY_SHA256,
        "Frozen archive utility changed",
    )
    archive.require(
        archive.stable_hash(Path(__file__).with_name("audit_full_shape_probe.py"))["sha256"]
        == AUDITOR_SHA256,
        "Frozen full-shape auditor changed",
    )
    return archive.stable_hash(Path(__file__))["sha256"]


def load_closed(workspace, bound_anchors):
    helper_identity()
    validate_anchors(bound_anchors)
    records = {}
    for name, expected in bound_anchors.items():
        path = archive.source_path(workspace, name)
        archive.require(
            archive.stable_hash(path)["sha256"] == expected, "Closed full-shape anchor changed"
        )
        records[name] = json.loads(path.read_text())
        archive.require(
            archive.stable_hash(path)["sha256"] == expected, "Closure changed during read"
        )
    summary, manifest, audit = (records[MODEL + "/summary.json"], records[MANIFEST], records[AUDIT])
    validate_closed_records(summary, manifest, audit, bound_anchors)
    return summary, manifest, audit


def validate_closed_records(summary, manifest, audit, bound_anchors):
    import audit_full_shape_probe as audited

    audited.validate_manifest(manifest)
    audited.validate_summary(summary, cells())
    audited.validate_pairs(audit["rows"], summary)
    archive.require(
        audit["schema"] == "gpu-owned-full-shape-array-audit-v1"
        and audit["passed"] is True
        and type(audit["workers"]) is int
        and audit["workers"] == 72
        and type(audit["fresh_process_pairs"]) is int
        and audit["fresh_process_pairs"] == 36
        and type(audit["common_fixtures"]) is int
        and audit["common_fixtures"] == 12
        and type(audit["raw_archives_checked"]) is int
        and audit["raw_archives_checked"] == 504
        and audit["auditor_sha256"] == AUDITOR_SHA256
        and audit["declared_producer_sha256"] == MODEL_PRODUCER_SHA256
        and audit["summary_sha256"] == bound_anchors[MODEL + "/summary.json"]
        and audit["manifest_sha256"] == bound_anchors[MANIFEST]
        and audit["formal_eligible"] is False
        and type(audit["formal_fits"]) is int
        and audit["formal_fits"] == 0
        and audit["admission_stream_opened"] is False
        and all(
            type(row["raw_archives_checked"]) is int and row["raw_archives_checked"] == 7
            for row in audit["rows"]
        ),
        "Independent full-shape saved-array audit is missing or mismatched",
    )


def group_specs(workspace, bound_anchors):
    summary, _, _ = load_closed(workspace, bound_anchors)
    specs = []
    for engine in ENGINES:
        for batch, length in SHAPES:
            expected = [
                cell for cell in cells() if cell["engine"] == engine and cell["batch"] == batch
            ]
            selected = [
                row
                for row in summary["workers"]
                if row["cell"]["engine"] == engine and row["cell"]["batch"] == batch
            ]
            archive.require(
                [row["cell"] for row in selected] == expected
                and len(selected) == 12
                and all(
                    row["passed"] is True
                    and type(row["returncode"]) is int
                    and row["returncode"] == 0
                    for row in selected
                ),
                "Engine/shape worker inventory is incomplete or reordered",
            )
            roots, closures = [], []
            for row in selected:
                worker = MODEL + "/" + row["cell"]["id"]
                roots.extend((worker, worker + ".log"))
                closures.append(
                    {"path": worker + "/result.json", "sha256": row["receipt"]["sha256"]}
                )
            specs.append(
                {
                    "name": f"model-full-shape-v1-{engine.replace('_', '-')}-b{batch}-t{length}",
                    "roots": roots,
                    "closures": closures,
                }
            )
    specs.append(
        {
            "name": "model-full-shape-v1-metadata",
            "roots": list(bound_anchors),
            "closures": [{"path": name, "sha256": value} for name, value in bound_anchors.items()],
        }
    )
    archive.validate_specs(specs)
    return specs


def validate_coverage(workspace, groups):
    expected = archive.enumerate_paths(workspace, [MODEL, MANIFEST, AUDIT])
    actual = [member["path"] for group in groups for member in group["members"]]
    archive.require(
        len(actual) == len(set(actual)) and sorted(actual) == expected,
        "Full-shape archives omit, duplicate or add evidence files",
    )
    raw_names = sorted(name for name in actual if name.endswith(".npz"))
    expected_raw = sorted(
        MODEL + "/" + cell["id"] + "/" + name for cell in cells() for name in RAW_NAMES
    )
    archive.require(raw_names == expected_raw, "Wrong named full-shape raw archive inventory")
    return {
        "files": len(actual),
        "raw_archives": sum(name.endswith(".npz") for name in actual),
        "bytes": sum(member["bytes"] for group in groups for member in group["members"]),
    }


def make_plan(workspace, bound_anchors):
    producer = helper_identity()
    specs = group_specs(workspace, bound_anchors)
    archive.verify_closures(workspace, specs)
    groups = [{**spec, "members": archive.inventory(workspace, spec["roots"])} for spec in specs]
    coverage = validate_coverage(workspace, groups)
    archive.require(coverage["raw_archives"] == 504, "Wrong raw archive inventory")
    return {
        "schema": "gpu-owned-full-shape-evidence-plan-v1",
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "producer_sha256": producer,
        "archive_utility_sha256": UTILITY_SHA256,
        "auditor_sha256": AUDITOR_SHA256,
        "maximum_group_bytes_exclusive": archive.MAX_BYTES,
        "anchors": bound_anchors,
        "groups": groups,
        "coverage": coverage,
        "scope": "Closed B1/T32 and B16/T32 development evidence; "
        "prior failures remain preserved separately",
        "runtime_and_source_payloads": "Referenced identities only; "
        "final source archive is separate",
    }


def validate_plan(workspace, plan):
    archive.require(
        plan["schema"] == "gpu-owned-full-shape-evidence-plan-v1"
        and plan["producer_sha256"] == helper_identity()
        and plan["archive_utility_sha256"] == UTILITY_SHA256
        and plan["auditor_sha256"] == AUDITOR_SHA256,
        "Full-shape archive producer or contract changed",
    )
    validate_anchors(plan["anchors"])
    proxy = {
        **plan,
        "schema": "gpu-owned-closed-evidence-plan-v1",
        "producer_sha256": UTILITY_SHA256,
    }
    archive.validate_plan(proxy, group_specs(workspace, plan["anchors"]))
    archive.require(
        plan["coverage"] == validate_coverage(workspace, plan["groups"]),
        "Full-shape evidence coverage changed",
    )
    archive.verify_sources(workspace, plan)


def build(workspace, plan_path, expected, output):
    archive.require(
        archive.stable_hash(plan_path)["sha256"] == expected, "Full-shape archive plan hash changed"
    )
    plan = json.loads(plan_path.read_text())
    validate_plan(workspace, plan)
    archive.reject_nested_output(workspace, output, plan)
    archive.require(
        shutil.disk_usage(output.parent).free > plan["coverage"]["bytes"] + 64 * 1024**2,
        "Insufficient free archive storage",
    )
    output.mkdir(exist_ok=False)
    shutil.copyfile(plan_path, output / "archive-plan.json")
    receipt = {
        "schema": "gpu-owned-full-shape-archive-receipt-v1",
        "passed": False,
        "formal_eligible": False,
        "uploaded": False,
        "plan_sha256": expected,
        "producer_sha256": plan["producer_sha256"],
        "coverage": plan["coverage"],
        "archives": [],
    }
    try:
        for group in plan["groups"]:
            result = archive.write_archive(workspace, group, output / (group["name"] + ".tar.gz"))
            receipt["archives"].append(result)
            print(json.dumps({"verified": result["archive"], "bytes": result["bytes"]}), flush=True)
        validate_plan(workspace, plan)
        archive.require(
            archive.stable_hash(plan_path)["sha256"]
            == expected
            == archive.stable_hash(output / "archive-plan.json")["sha256"],
            "Full-shape archive plan changed during packaging",
        )
        receipt["passed"] = True
    except Exception as error:
        receipt["error_type"] = type(error).__name__
        raise
    finally:
        (output / "publication-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    planning = sub.add_parser("plan")
    packing = sub.add_parser("build")
    for command in (planning, packing):
        command.add_argument("--workspace", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    planning.add_argument("--summary-sha256", required=True)
    planning.add_argument("--audit-sha256", required=True)
    packing.add_argument("--plan", type=Path, required=True)
    packing.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    if args.command == "plan":
        plan = make_plan(workspace, anchors(args.summary_sha256, args.audit_sha256))
        archive.reject_nested_output(workspace, output, plan)
        with output.open("x") as stream:
            stream.write(json.dumps(plan, indent=2) + "\n")
        print(
            json.dumps(
                {"plan_sha256": archive.stable_hash(output)["sha256"], "coverage": plan["coverage"]}
            )
        )
    else:
        print(json.dumps(build(workspace, args.plan, args.plan_sha256, output)))


if __name__ == "__main__":
    main()
