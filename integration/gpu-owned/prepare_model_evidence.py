"""Archive the closed model-v2 evidence by engine using the frozen archive utilities."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import prepare_evidence as archive

UTILITY_SHA256 = "2a30d78a30cfe75e56bf3a133f7611e0544706c4b1171a40732eb113b0b4ba51"
AUDITOR_SHA256 = "095509f8432de50914b4c6bbaed4e31b371c917b587370674693c2bdf9480351"
MODEL = "work/gpu-v4-owned/model-compatibility-v2"
MANIFEST = "work/gpu-v4-owned/model-manifest-v2.json"
AUDIT = "work/gpu-v4-owned/model-array-audit-v1.json"
ANCHORS = {
    MODEL + "/summary.json": "1b72abdfa347c7877ad930c7b4c9721c1f61a2dc1b07f9de3244221959f98f4e",
    MANIFEST: "06c29116993f0adbaa2bfb6f5f04b79de152345c3549cfaf466c17a9f2e32a86",
    AUDIT: "13588eb731711a0ddd299643f0f29dd187238273357fbfbb5e9c57244516e8e0",
}
WIDTHS = (32, 64, 128, 256, 512, 1024)
ENGINES = ("cpu_torch", "mps", "mlx")


def cells():
    return [
        {
            "width": width,
            "engine": engine,
            "process_index": process,
            "id": f"w{width}-{engine}-p{process}",
        }
        for width in WIDTHS
        for engine in ENGINES
        for process in (0, 1)
    ]


def helper_identity():
    archive.require(
        archive.stable_hash(Path(archive.__file__))["sha256"] == UTILITY_SHA256,
        "Frozen archive utility changed",
    )
    return archive.stable_hash(Path(__file__))["sha256"]


def load_closed(workspace):
    records = {}
    for name, expected in ANCHORS.items():
        path = archive.source_path(workspace, name)
        archive.require(
            archive.stable_hash(path)["sha256"] == expected, "Closed model anchor changed"
        )
        records[name] = json.loads(path.read_text())
    summary, manifest, audit = (records[MODEL + "/summary.json"], records[MANIFEST], records[AUDIT])
    archive.require(
        manifest["cells"] == cells()
        and summary["planned_workers"] == cells()
        and [row["cell"] for row in summary["workers"]] == cells(),
        "Model cell inventory changed",
    )
    archive.require(
        summary["complete"] is True
        and summary["passed"] is True
        and summary["broad_framework_suite_remains_blocking"] is True,
        "Model execution is not a closed passing development check",
    )
    for record in (summary, audit):
        archive.require(
            record["formal_eligible"] is False
            and record["formal_fits"] == 0
            and record["admission_stream_opened"] is False,
            "Model scope changed",
        )
    archive.require(
        audit["schema"] == "gpu-owned-model-array-audit-v1"
        and audit["passed"] is True
        and audit["workers"] == 36
        and audit["fresh_process_pairs"] == 18
        and audit["raw_archives_checked"] == 216
        and audit["auditor_sha256"] == AUDITOR_SHA256
        and audit["summary_sha256"] == ANCHORS[MODEL + "/summary.json"]
        and audit["manifest_sha256"] == ANCHORS[MANIFEST]
        and [row["cell"] for row in audit["rows"]] == cells(),
        "Independent saved-array audit is missing or mismatched",
    )
    return summary, manifest, audit


def group_specs(workspace):
    summary, _, _ = load_closed(workspace)
    specs = []
    for engine in ENGINES:
        selected = [row for row in summary["workers"] if row["cell"]["engine"] == engine]
        archive.require(
            len(selected) == 12
            and all(row["passed"] is True and row["returncode"] == 0 for row in selected),
            "Engine workers incomplete",
        )
        roots, closures = [], []
        for row in selected:
            worker = MODEL + "/" + row["cell"]["id"]
            roots.extend((worker, worker + ".log"))
            closures.append({"path": worker + "/result.json", "sha256": row["receipt"]["sha256"]})
        specs.append(
            {
                "name": "model-compatibility-v2-" + engine.replace("_", "-"),
                "roots": roots,
                "closures": closures,
            }
        )
    specs.append(
        {
            "name": "model-compatibility-v2-metadata",
            "roots": list(ANCHORS),
            "closures": [{"path": name, "sha256": value} for name, value in ANCHORS.items()],
        }
    )
    archive.validate_specs(specs)
    return specs


def validate_coverage(workspace, groups):
    expected = archive.enumerate_paths(workspace, [MODEL, MANIFEST, AUDIT])
    actual = [member["path"] for group in groups for member in group["members"]]
    archive.require(
        len(actual) == len(set(actual)) and sorted(actual) == expected,
        "Model archives omit, duplicate or add evidence files",
    )
    return {
        "files": len(actual),
        "raw_archives": sum(name.endswith(".npz") for name in actual),
        "bytes": sum(member["bytes"] for group in groups for member in group["members"]),
    }


def make_plan(workspace):
    producer = helper_identity()
    specs = group_specs(workspace)
    archive.verify_closures(workspace, specs)
    groups = [{**spec, "members": archive.inventory(workspace, spec["roots"])} for spec in specs]
    coverage = validate_coverage(workspace, groups)
    archive.require(coverage["raw_archives"] == 216, "Wrong raw archive inventory")
    return {
        "schema": "gpu-owned-model-evidence-plan-v1",
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "producer_sha256": producer,
        "archive_utility_sha256": UTILITY_SHA256,
        "auditor_sha256": AUDITOR_SHA256,
        "maximum_group_bytes_exclusive": archive.MAX_BYTES,
        "anchors": ANCHORS,
        "groups": groups,
        "coverage": coverage,
        "scope": "Closed B3/T3 development evidence; prior failed model-v1 is preserved separately",
        "runtime_and_source_payloads": "Referenced identities only; "
        "final source archive is separate",
    }


def validate_plan(workspace, plan):
    archive.require(
        plan["schema"] == "gpu-owned-model-evidence-plan-v1"
        and plan["producer_sha256"] == helper_identity()
        and plan["archive_utility_sha256"] == UTILITY_SHA256
        and plan["auditor_sha256"] == AUDITOR_SHA256
        and plan["anchors"] == ANCHORS,
        "Model archive producer or contract changed",
    )
    proxy = {
        **plan,
        "schema": "gpu-owned-closed-evidence-plan-v1",
        "producer_sha256": UTILITY_SHA256,
    }
    archive.validate_plan(proxy, group_specs(workspace))
    archive.require(
        plan["coverage"] == validate_coverage(workspace, plan["groups"]),
        "Model evidence coverage changed",
    )
    archive.verify_sources(workspace, plan)


def build(workspace, plan_path, expected, output):
    archive.require(
        archive.stable_hash(plan_path)["sha256"] == expected, "Model archive plan hash changed"
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
        "schema": "gpu-owned-model-archive-receipt-v1",
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
            "Model archive plan changed during packaging",
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
    packing.add_argument("--plan", type=Path, required=True)
    packing.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    if args.command == "plan":
        plan = make_plan(workspace)
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
