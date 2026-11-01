"""Execute the complete source-bound raw prerequisite schedule, serially."""

import argparse
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        "library_qualifier_" + name, HERE / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A, PLAN, FIXTURES, N = (
    load(n) for n in ("authority", "qualification_plan", "fixtures", "native_assets")
)
C = A.C


def require_result(job, value):
    kind = job["kind"]
    if kind in ("primitive", "optimizer", "provenance", "collection"):
        C.require(
            value["schema"] == "frozen-library-raw-audit-dispatch-v1"
            and value["kind"] == kind
            and value["width"] == job["width"],
            "Wrong raw audit dispatch",
        )
        value = value["result"]
    if kind == "collection":
        C.require(len(value["groups"]) == 1, "Wrong collection audit denominator")
        value = value["groups"][0]
        C.require(value["name"] == job["name"].removesuffix("-raw"), "Foreign raw collection")
    C.require(value["evidence_valid"] is True, "Invalid raw prerequisite evidence")
    required = (
        ("exact_passed",)
        if kind == "optimizer"
        else (
            ("exact_passed", "numerics_passed")
            if kind == "repeatability"
            else ("numerics_passed",)
            if job["expected_numerics"] is True
            else ()
        )
    )
    C.require(
        all(value[k] is True for k in required), "Required raw numerical/repetition gate failed"
    )
    return {k: value[k] for k in ("evidence_valid", *required)}


def ci_evidence(records, revision):
    C.require(len(records) == 2, "Both exact source CI runs required")
    events, ids = set(), set()
    for record in records:
        run, jobs = PLAN.load(record["run"]), PLAN.load(record["jobs"])
        C.require(
            run["head_sha"] == revision
            and run["name"] == "Validation"
            and run["status"] == "completed"
            and run["conclusion"] == "success"
            and run["event"] in ("push", "pull_request")
            and run["repository"]["full_name"] == "logannye/arm64-transformer"
            and type(run["run_attempt"]) is int
            and run["run_attempt"] > 0,
            "Hosted source checks are incomplete or belong to another run/source",
        )
        C.require(
            {j["name"] for j in jobs["jobs"]} == {"native", "python"}
            and len(jobs["jobs"]) == 2
            and all(
                j["status"] == "completed"
                and j["conclusion"] == "success"
                and j["run_id"] == run["id"]
                and j["head_sha"] == revision
                and j["run_attempt"] == run["run_attempt"]
                for j in jobs["jobs"]
            ),
            "Hosted job/attempt/source provenance differs",
        )
        C.require(
            run["id"] not in ids and run["event"] not in events, "Duplicate source CI evidence"
        )
        ids.add(run["id"])
        events.add(run["event"])
    C.require(events == {"push", "pull_request"}, "Missing source CI event")


def command(job, roles, capture, audit_python, reuse=None):
    if job.get("previous_context"):
        C.require(reuse is not None, "Missing authenticated pre-FF1 audit context")
    if job["kind"] == "optimizer":
        expected_roles = reuse["roles"] if job.get("previous_context") else roles
        # Adapter receipts bind their actual producer control, not the
        # later coordinating control. Reaudit those unchanged bytes there.
        workers = [
            C.read(job["arguments"][job["arguments"].index("--" + name) + 1])
            for name in ("first", "second", "restart")
        ]
        producer_roles = workers[0]["control_adapter"]["roles"]
        C.require(
            producer_roles["math"] == expected_roles["math"]
            and all(w["control_adapter"]["roles"] == producer_roles for w in workers),
            "Mixed original optimizer producers",
        )
        producer = A.verify_role(producer_roles["control"], control=True)
        for name in ("audit_dispatch.py", "audit_optimizer.py", "audit.py"):
            relative = "integration/library-study-v1/" + name
            C.require(
                producer["files"][relative] == expected_roles["control"]["files"][relative],
                "Original optimizer auditor bytes differ",
            )
        return [
            str(audit_python),
            "-I",
            "-B",
            str(Path(producer["root"]) / "integration/library-study-v1/audit_dispatch.py"),
            "--kind",
            "optimizer",
            "--width",
            str(job["width"]),
            "--control-revision",
            producer["revision"],
            "--math-root",
            producer_roles["math"]["root"],
            "--",
            *job["arguments"],
        ]
    if job.get("previous_context"):
        source = Path(reuse["roles"]["control"]["root"]) / "integration/library-study-v1/qualify.py"
        C.require(
            C.binding(source)["sha256"]
            == reuse["roles"]["control"]["files"]["integration/library-study-v1/qualify.py"],
            "Unbound previous numerical auditor dispatch",
        )
        spec = importlib.util.spec_from_file_location("pre_ff1_qualifier", source)
        previous = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(previous)
        return previous.command(job, reuse["roles"], PLAN.load(reuse["capture"]), audit_python)
    base = [str(audit_python), "-I", "-B"]
    if job["kind"] in ("primitive", "optimizer", "provenance", "collection"):
        return [
            *base,
            str(HERE / "audit_dispatch.py"),
            "--kind",
            job["kind"],
            "--width",
            str(job["width"]),
            "--control-revision",
            roles["control"]["revision"],
            "--math-root",
            roles["math"]["root"],
            "--",
            *job["arguments"],
        ]
    if job["kind"] in ("repeatability", "common_state"):
        return [
            *base,
            str(HERE / "audit.py"),
            "--kind",
            job["kind"],
            "--control-revision",
            roles["control"]["revision"],
            "--math-root",
            roles["math"]["root"],
            *job["arguments"],
        ]
    C.require(job["kind"] in ("native_metal", "native_common_state"), "Unknown prerequisite audit")
    producer = A.source_role(
        job["producer_control_root"], job["producer_control_revision"], control=True
    )
    if job["kind"] == "native_metal":
        # The corrected auditor binds the original producer and accepts only
        # audit-only differences. Do not rerun completed numerical workers.
        return [
            *base,
            str(HERE / "audit_development.py"),
            "--control-revision",
            roles["control"]["revision"],
            "--producer-control-root",
            producer["root"],
            "--producer-control-revision",
            producer["revision"],
            "--math-root",
            roles["math"]["root"],
            *job["arguments"],
        ]
    filename = (
        "audit_development.py" if job["kind"] == "native_metal" else "audit_common_native_metal.py"
    )
    path = Path(producer["root"]) / "integration/library-study-v1" / filename
    C.require(
        C.binding(path)["sha256"]
        == producer["files"][path.relative_to(producer["root"]).as_posix()],
        "Unbound producer-version numerical auditor",
    )
    return [
        *base,
        str(path),
        "--control-revision",
        producer["revision"],
        "--math-root",
        roles["math"]["root"],
        *job["arguments"],
    ]


def run(argv, label, output, environment):
    path = output / (label + ".json")
    errors = output / (label + ".stderr.log")
    with path.open("x") as stdout, errors.open("x") as stderr:
        process = subprocess.Popen(
            argv, stdout=stdout, stderr=stderr, env=environment, start_new_session=True
        )
        started = time.monotonic()
        try:
            while process.poll() is None:
                C.require(time.monotonic() - started < 6 * 3600, "Raw audit time budget reached")
                C.require(
                    shutil.disk_usage(output).free > C.MIN_FREE_BYTES,
                    "Raw audit disk reserve reached",
                )
                C.require(
                    sum(p.stat().st_size for p in output.iterdir() if p.is_file()) < 2 * 1024**3,
                    "Raw audit metadata budget reached",
                )
                time.sleep(2)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    C.require(
        process.returncode == 0, "Raw auditor failed; log, error and original evidence retained"
    )
    value = C.read(path)
    return value, dict(
        pid=process.pid,
        exit_code=process.returncode,
        argv=argv,
        stdout=C.binding(path),
        stderr=C.binding(errors),
    )


def execute(args):
    C.require(
        sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode,
        "Use metadata controller Python -I -S -B",
    )
    request = C.read(args.request, args.request_sha256)
    C.require(
        request["schema"] == "frozen-library-prerequisite-request-v1"
        and set(request) - {"reuse"}
        == {
            "schema",
            "roles",
            "source_ci",
            "capture",
            "fixtures",
            "native_assets",
            "full_suite_audit",
            "collectors",
            "inventories",
            "raw_store",
        },
        "Wrong prerequisite request",
    )
    roles = request["roles"]
    C.require(Path(roles["control"]["root"]) == HERE.parents[1], "Qualifier control origin differs")
    A.verify_role(roles["control"], control=True)
    A.verify_role(roles["math"])
    for role in ("math", "control"):
        ci_evidence(request["source_ci"][role], roles[role]["revision"])
    reuse = PLAN.reuse_context(request.get("reuse"), request["capture"])
    if reuse is not None:
        for role in ("math", "control"):
            ci_evidence(reuse["source_ci"][role], reuse["roles"][role]["revision"])
    capture = PLAN.load(request["capture"])
    inventory = PLAN.load(capture["runtime_inventory"])
    fixtures = PLAN.load(request["fixtures"])
    FIXTURES.verify(fixtures, roles["math"])
    N.verify(request["native_assets"], roles["math"])
    original_suites = PLAN.load(request["full_suite_audit"])
    previous_suites = PLAN.previous_scope(
        original_suites["capture"], request["capture"], reuse, "full_suites"
    )
    suite_roles = reuse["roles"] if previous_suites else roles
    suite_capture = reuse["capture"] if previous_suites else request["capture"]
    C.require(
        original_suites["schema"] == "frozen-library-full-suite-audit-v1"
        and original_suites["full_suites_passed"] is True
        and original_suites["capture"] == suite_capture
        and original_suites["math"] == suite_roles["math"],
        "Wrong raw suite audit scope",
    )
    plan = PLAN.build(
        request["collectors"],
        request["capture"],
        request["inventories"],
        request["raw_store"],
        reuse=request.get("reuse"),
    )
    output = Path(args.output).absolute()
    C.require(
        output.resolve() == output and not output.exists() and not output.is_symlink(),
        "Fresh qualifier output required",
    )
    protected = [Path(r["root"]) for r in roles.values()] + [
        Path(request["raw_store"]),
        Path(fixtures["input_root"]),
    ]
    protected += [Path(p) for p in inventory["layout"]["scan_roots"]]
    C.require(
        all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected),
        "Qualifier output overlaps protected evidence",
    )
    C.require(
        shutil.disk_usage(output.parent).free > C.MIN_FREE_BYTES + 2 * 1024**3,
        "Insufficient qualification metadata budget",
    )
    output.mkdir()
    C.write(output / "plan.json", plan)
    sys.path.insert(0, roles["math"]["root"])
    path = Path(roles["math"]["root"]) / "integration/gpu-owned/owned_mlx_suites.py"
    spec = importlib.util.spec_from_file_location("qualifier_environment", path)
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    environment = suite.controls(dict(os.environ))
    completed = []
    try:
        canonical = inventory["layout"]["interpreter"]
        argv = [
            canonical,
            "-I",
            "-S",
            "-B",
            str(
                Path(suite_roles["control"]["root"])
                / "integration/library-study-v1/qualify_suites.py"
            ),
            "--control-revision",
            suite_roles["control"]["revision"],
            "--math-root",
            suite_roles["math"]["root"],
            "--capture",
            suite_capture["path"],
            "--capture-sha256",
            suite_capture["sha256"],
        ]
        for name, summary in zip(("mps", "mlx"), original_suites["summaries"], strict=True):
            C.bound(summary)
            argv += [
                "--" + name + "-summary",
                summary["path"],
                "--" + name + "-summary-sha256",
                summary["sha256"],
            ]
        argv += ["--output", str(output / "full-suites-result.json")]
        _, execution = run(argv, "full-suites", output, environment)
        C.require(
            C.read(output / "full-suites-result.json")["full_suites_passed"] is True,
            "Raw suite audit failed",
        )
        completed.append(
            dict(
                name="full-suites",
                execution=execution,
                result=C.binding(output / "full-suites-result.json"),
            )
        )
        for index, job in enumerate(plan["jobs"]):
            A.verify_role(roles["control"], control=True)
            A.verify_role(roles["math"])
            env = dict(environment, ARM64_TRANSFORMER_WIDTH=str(job["width"]))
            value, execution = run(
                command(job, roles, capture, args.audit_python, reuse),
                f"audit-{index:04d}",
                output,
                env,
            )
            decision = require_result(job, value)
            completed.append(
                dict(
                    name=job["name"],
                    kind=job["kind"],
                    width=job["width"],
                    decision=decision,
                    execution=execution,
                )
            )
            C.write(output / f"progress-{index:04d}.json", completed[-1])
        C.require(
            len(completed) == 1 + len(plan["jobs"]), "Qualification audit denominator differs"
        )
        C.require(
            C.binding(args.request)["sha256"] == args.request_sha256,
            "Qualification request changed",
        )
        A.verify_role(roles["control"], control=True)
        A.verify_role(roles["math"])
        result = dict(
            schema="frozen-library-raw-prerequisites-v1",
            raw_prerequisites_passed=True,
            request=C.binding(args.request),
            plan=C.binding(output / "plan.json"),
            roles=roles,
            capture=request["capture"],
            completed=completed,
            formal_execution_supported=C.EXECUTION_SUPPORTED,
            formal_eligible=False,
            formal_fits=0,
            admission_stream_opened=False,
        )
        C.write(output / "result.json", result)
        return result
    except BaseException:
        C.write(
            output / "stopped.json",
            dict(
                error=traceback.format_exc(),
                completed=completed,
                formal_eligible=False,
                formal_fits=0,
                admission_stream_opened=False,
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--audit-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(args)
