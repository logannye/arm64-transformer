"""Serial all-origin, all-engine best-and-last checkpoint audit after 180 fits."""

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        "library_campaign_audit_" + name, HERE / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R, CAM, Q = (load(n) for n in ("checkpoint_replay", "campaign", "qualify"))
C, F, AD = R.C, R.F, R.AD


def ledger(binding, manifest, manifest_binding, admission):
    value = F.PLAN.load(binding)
    C.require(
        value["schema"] == "frozen-library-formal-campaign-v1"
        and value["manifest"] == manifest_binding
        and value["admission"] == admission
        and value["status"] == "measurement_complete_unaudited"
        and value["scientific_acceptance"] is False
        and type(value["completed_fits"]) is type(value["attempted_fits"]) is int
        and value["completed_fits"] == value["attempted_fits"] == value["planned_fits"] == 180
        and [r["cell"] for r in value["cells"]] == manifest["comparison"]["cells"],
        "Foreign or incomplete formal campaign",
    )
    for row in value["cells"]:
        C.require(row["status"] == "completed" and row["exit_code"] == 0, "Incomplete formal cell")
        receipt, proof = R.cell_receipt(row["receipt"], manifest, manifest_binding, admission)
        C.require(
            receipt["cell"] == row["cell"] and proof["pid"] == row["pid"],
            "Campaign process/cell differs",
        )
        C.bound(row["log"])
    return value


def prepare_replays(campaign, output, manifest_binding, admission, raw_store):
    """Freeze every original job before grouping only its untimed execution."""
    cases, groups = [], {}
    for index, row in enumerate(campaign["cells"]):
        for checkpoint in ("best.f32", "last.f32"):
            label = f"cell-{index:03d}-{checkpoint[:-4]}"
            destination = output / label
            destination.mkdir()
            base = dict(
                manifest=manifest_binding,
                admission=admission,
                cell_receipt=row["receipt"],
                checkpoint=checkpoint,
                raw_store=raw_store,
            )
            jobs = {}
            for engine in (*R.ENGINES, "oracle"):
                path = destination / (engine + "-job.json")
                C.write(
                    path,
                    dict(
                        base,
                        schema="frozen-library-checkpoint-job-v1",
                        engine=engine,
                        output=str(destination / engine),
                        parent_pid=os.getpid(),
                    ),
                )
                jobs[engine] = C.binding(path)
                if engine != "oracle":
                    groups.setdefault((row["cell"]["width"], engine), []).append(jobs[engine])
            cases.append(
                dict(
                    label=label,
                    destination=destination,
                    base=base,
                    row=row,
                    checkpoint=checkpoint,
                    jobs=jobs,
                )
            )
    C.require(
        len(cases) == 360 and len(groups) == 30 and all(len(j) == 60 for j in groups.values()),
        "Full best-and-last, five-engine replay denominator required",
    )
    return cases, groups


def execute(args):
    C.require_execution_support()
    C.require(
        sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode,
        "Use metadata audit coordinator -I -S -B",
    )
    manifest_binding, admission = C.binding(args.manifest), C.binding(args.admission)
    C.require(manifest_binding["sha256"] == args.manifest_sha256, "Manifest trust anchor differs")
    manifest = F.read(args.manifest, args.manifest_sha256)
    AD.verify(manifest, manifest_binding, admission)
    campaign_binding = C.binding(args.campaign)
    campaign = ledger(campaign_binding, manifest, manifest_binding, admission)
    capture = F.PLAN.load(manifest["bindings"]["capture"])
    inventory = F.PLAN.load(capture["runtime_inventory"])
    request = F.PLAN.load(F.PLAN.load(manifest["bindings"]["raw_prerequisites"])["request"])
    roles = manifest["bindings"]["roles"]
    output = Path(args.output).absolute()
    R.W.protected_output(
        dict(output=str(output), raw_store=request["raw_store"]),
        type("MetadataRuntime", (), dict(roles=roles, inventory=inventory))(),
    )
    C.require(
        not output.is_relative_to(Path(args.campaign).parent)
        and not Path(args.campaign).parent.is_relative_to(output),
        "Audit overlaps measured outputs",
    )
    audit_bytes = CAM.storage_estimate()["components_bytes"]["five_engine_and_fp64_best_and_last"]
    C.require(
        shutil.disk_usage(output.parent).free
        > C.MIN_FREE_BYTES + int(audit_bytes * 1.25) + 2 * 1024**3,
        "Insufficient space for complete retained checkpoint audit",
    )
    output.mkdir()
    sys.path.insert(0, roles["math"]["root"])
    from transformer.gpu_study import valid_conditions

    # Audit power checks authenticate the measurement records, never time replay.
    C.require(
        all(
            valid_conditions(r[phase])
            for r in campaign["cells"]
            for phase in ("conditions_before", "conditions_after")
        ),
        "Invalid measured power conditions",
    )
    path = Path(roles["math"]["root"]) / "integration/gpu-owned/owned_mlx_suites.py"
    spec = importlib.util.spec_from_file_location("postrun_environment", path)
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    env = CAM.A.execution_environment(suite.controls(dict(os.environ)))
    result = dict(
        schema="frozen-library-campaign-audit-v1",
        manifest=manifest_binding,
        admission=admission,
        campaign=campaign_binding,
        planned_fits=180,
        completed_fits=180,
        planned_checkpoints=360,
        audited_checkpoints=0,
        reports=[],
        planned_test_token_predictions=1474560,
        scientific_acceptance=False,
    )
    C.write(output / "started.json", result)
    cases, groups = prepare_replays(
        campaign, output, manifest_binding, admission, request["raw_store"]
    )
    batch_root = output / "batches"
    batch_root.mkdir()
    selected_replays, batch_proofs = {}, {}
    for (width, engine), bindings in sorted(groups.items()):
        name = f"w{width}-{engine}"
        path = batch_root / (name + "-job.json")
        C.write(
            path,
            dict(
                schema="frozen-library-checkpoint-batch-job-v1",
                manifest=manifest_binding,
                admission=admission,
                width=width,
                engine=engine,
                jobs=bindings,
                output=str(batch_root / name),
                parent_pid=os.getpid(),
            ),
        )
        reply, _ = Q.run(
            [
                inventory["layout"]["interpreter"],
                "-P",
                "-S",
                "-B",
                str(HERE / "checkpoint_batch.py"),
                "--job",
                str(path),
                "--job-sha256",
                C.binding(path)["sha256"],
            ],
            name + "-execution",
            batch_root,
            dict(env, ARM64_TRANSFORMER_WIDTH=str(width)),
        )
        C.require(reply["complete"] is True, "Incomplete selected replay batch")
        batch = F.PLAN.load(reply["receipt"])
        C.require(
            batch["complete"] is True
            and batch["job"] == C.binding(path)
            and [r["job"] for r in batch["records"]] == bindings,
            "Selected replay batch denominator or authority changed",
        )
        batch_proofs[(width, engine)] = reply["receipt"]
        for record in batch["records"]:
            C.bound(record["receipt"])
            C.require(
                record["job"]["path"] not in selected_replays,
                "Duplicate selected checkpoint replay",
            )
            selected_replays[record["job"]["path"]] = record["receipt"]
    C.require(
        len(selected_replays) == 1800, "All five engines must replay every best and last checkpoint"
    )
    for case in cases:
        row, checkpoint = case["row"], case["checkpoint"]
        label, destination, base = case["label"], case["destination"], case["base"]
        environment = dict(env, ARM64_TRANSFORMER_WIDTH=str(row["cell"]["width"]))
        replays = {e: selected_replays[case["jobs"][e]["path"]] for e in R.ENGINES}
        oracle = case["jobs"]["oracle"]
        reply, _ = Q.run(
            [
                str(args.audit_python),
                "-I",
                "-B",
                str(HERE / "checkpoint_replay.py"),
                "--job",
                oracle["path"],
                "--job-sha256",
                oracle["sha256"],
            ],
            "oracle-execution",
            destination,
            environment,
        )
        C.require(reply["complete"] is True, "Incomplete independent checkpoint oracle")
        replays["oracle"] = reply["receipt"]
        job = dict(
            base,
            schema="frozen-library-checkpoint-audit-job-v2",
            replays=replays,
            replay_root=str(destination),
            batch_proofs={e: batch_proofs[(row["cell"]["width"], e)] for e in R.ENGINES},
        )
        path = destination / "audit-job.json"
        C.write(path, job)
        report, execution = Q.run(
            [
                str(args.audit_python),
                "-I",
                "-B",
                str(HERE / "audit_checkpoint.py"),
                "--job",
                str(path),
                "--job-sha256",
                C.binding(path)["sha256"],
            ],
            "independent-audit",
            destination,
            environment,
        )
        C.require(
            report["evidence_valid"] is True
            and report["job"] == C.binding(path)
            and report["cell"] == row["cell"]
            and report["checkpoint"] == checkpoint,
            "Foreign checkpoint audit",
        )
        result["reports"].append(
            dict(
                cell=row["cell"],
                checkpoint=checkpoint,
                report=execution["stdout"],
                passed=report["passed"],
            )
        )
        result["audited_checkpoints"] += 1
        C.bound(campaign_binding)
        F.verify(manifest)
        C.write(output / (label + "-progress.json"), result)
    cells = {r["cell"]["id"]: r for r in campaign["cells"]}
    repeats = []
    for row in (r for r in campaign["cells"] if r["cell"]["repeat"] == 1):
        cell = row["cell"]
        other = next(
            c
            for c in manifest["comparison"]["cells"]
            if c["repeat"] == 2 and all(c[k] == cell[k] for k in ("engine", "width", "seed"))
        )
        checked = CAM.repeat_pair(
            Path(row["receipt"]["path"]).parent,
            Path(cells[other["id"]]["receipt"]["path"]).parent,
            cell["engine"],
        )
        repeats.append(dict(first=cell["id"], second=other["id"], **checked))
    result.update(
        complete=True,
        repeat_checks=repeats,
        scientific_acceptance=len(repeats) == 90
        and all(r["passed"] for r in repeats)
        and len(result["reports"]) == 360
        and all(r["passed"] for r in result["reports"]),
        audited_test_token_predictions=sum(
            F.PLAN.load(r["report"])["quality"]["tokens"]
            for r in result["reports"]
            if r["checkpoint"] == "best.f32"
        ),
    )
    C.require(
        result["audited_test_token_predictions"] == 1474560,
        "Postrun prediction denominator differs",
    )
    ledger(campaign_binding, manifest, manifest_binding, admission)
    C.write(output / "audit.json", result)
    print(
        json.dumps(
            dict(
                complete=True,
                scientific_acceptance=result["scientific_acceptance"],
                report=C.binding(output / "audit.json"),
            )
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "manifest-sha256", "admission", "campaign", "output", "audit-python"):
        parser.add_argument("--" + name, required=True)
    execute(parser.parse_args())
