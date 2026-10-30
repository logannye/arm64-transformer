"""Audit exact regression coverage across externally bound replay shards.

This is development evidence. It cannot qualify a runtime or admit formal fits.
The manifest is frozen separately from this audit, including each receipt hash
and the expected implementation identity. A valid collection may fail numerics.
"""

import argparse
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "independent_raw_audit", Path(__file__).with_name("audit_replay.py")
)
RAW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RAW)


def implementation(receipt):
    """Fields that must match across shards, including source and binary bytes."""
    native = receipt.get("native", {})
    metal = receipt.get("metal", {})
    identity = {
        key: receipt.get(key)
        for key in (
            "engine",
            "width",
            "batch_size",
            "gradients_checked",
            "source_sha256",
            "runtime",
            "owned_runtime",
            "torch_diagnostic_policy",
            "python_candidate",
            "metal_candidate_shader_sha256",
        )
    } | {
        "native_library_sha256": native.get("sha256"),
        "native_policy": native.get("precision_policy"),
        "native_selected_kind": native.get("backend", {}).get("selected_kind"),
        "metal_library_sha256": metal.get("binary_sha256"),
        "metal_policy_sha256": metal.get("kernel_policy_sha256"),
    }
    for key in (
        "framework_candidate",
        "framework_candidate_policy",
        "framework_compiled",
        "framework_operation",
    ):
        if key in receipt:
            identity[key] = receipt[key]
    return identity


def audit_group(group, plan, plan_path, plan_sha, evidence_root, workers_root, raw_store=None):
    identity = group["implementation"]
    expected = [c for c in plan["cases"] if c["width"] == identity["width"]]
    RAW.require(expected and group["workers"], "Empty collection")
    RAW.require(identity["source_sha256"] and identity["runtime"], "Missing implementation proof")
    audited, observed, paths, batches = [], [], set(), {}
    for binding in group["workers"]:
        receipt_path = RAW.checked(workers_root, binding["path"], binding["sha256"])
        RAW.require(receipt_path.name == "result.json", "Not a worker receipt")
        RAW.require(receipt_path not in paths, "Duplicate worker")
        paths.add(receipt_path)
        receipt = json.loads(receipt_path.read_text())
        RAW.require(implementation(receipt) == identity, "Mixed implementation identity")
        audit = RAW.audit_worker(
            receipt_path.parent, binding["sha256"], plan_path, plan_sha, evidence_root, raw_store
        )
        audited.append(audit)
        for row, checked in zip(receipt["cases"], audit["cases"], strict=True):
            case = row["case"]
            observed.append(case["id"])
            batch_key = (case["checkpoint_sha256"], case["rows_sha256"], row["first_sequence"])
            # Different sequence anchors may replay the same batch. Require
            # repeated batches to have identical raw results, not just a vote.
            value = (row["raw_sha256"], checked["passed"])
            RAW.require(batches.get(batch_key, value) == value, "Repeated batch bytes differ")
            batches[batch_key] = value
    RAW.require(observed == [c["id"] for c in expected], "Missing, reordered or duplicate cases")
    return {
        "name": group["name"],
        "implementation": identity,
        "evidence_valid": True,
        "numerics_passed": all(a["numerics_passed"] for a in audited),
        "sequence_anchors": len(observed),
        "unique_batches": len(batches),
        "failed_sequence_anchors": sum(not c["passed"] for a in audited for c in a["cases"]),
        "failed_unique_batches": sum(not value[1] for value in batches.values()),
        "receipts": group["workers"],
        "audits": audited,
    }


def audit_collection(
    manifest_path, manifest_sha, plan_path, plan_sha, evidence_root, workers_root, raw_store=None
):
    RAW.require(RAW.sha256(manifest_path) == manifest_sha, "Collection manifest changed")
    RAW.require(RAW.sha256(plan_path) == plan_sha, "Replay inventory changed")
    manifest = json.loads(Path(manifest_path).read_text())
    plan = json.loads(Path(plan_path).read_text())
    RAW.require(
        manifest["schema"] == "trained-precision-collection-v1"
        and manifest["inventory_sha256"] == plan_sha
        and manifest["formal_eligible"] is False,
        "Wrong collection identity",
    )
    groups = manifest["groups"]
    RAW.require(groups and len({g["name"] for g in groups}) == len(groups), "Duplicate group")
    return {
        "schema": "trained-precision-collection-audit-v1",
        "manifest_sha256": manifest_sha,
        "inventory_sha256": plan_sha,
        "formal_eligible": False,
        "admission_stream_opened": False,
        "groups": [
            audit_group(g, plan, plan_path, plan_sha, evidence_root, workers_root, raw_store)
            for g in groups
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "manifest",
        "manifest-sha256",
        "inventory",
        "inventory-sha256",
        "evidence-root",
        "workers-root",
    ):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--raw-store")
    args = parser.parse_args()
    print(json.dumps(audit_collection(*vars(args).values()), indent=2))
