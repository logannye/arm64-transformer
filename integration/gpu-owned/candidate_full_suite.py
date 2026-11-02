"""Explicit full test_mps replay for an authenticated owned PyTorch candidate.

Preparation authenticates a passing 27-method replay and creates one new stock
conformance-fixture cache; it does not compile or run numerical work. The separate
run command executes every unchanged upstream test, including its bounded C++
fixture compilation. This is development integration, never study admission.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import traceback
import unittest
from pathlib import Path

SCHEMA = "gpu-owned-candidate-full-suite-v1"
PROTOCOL = "gpu-v4-owned-development-v1"
CANDIDATE_SHA = "89fc1708bb0f767ecb628ee09d15333063c2dea98c2e078d440e9794ab4f14d2"
POLICY_SHA = "4582d3a5299ed8f9752297fa52ed47a7d3e5fb030a21c39f4487b86d5fffb583"
NINJA_SHA = "ad44480e2ba27be95a4850720983fd1b123728bb8a45c6c90670f5aa0d3e0069"
LIMITS = {
    "cache_bytes": 1024**3,
    "cache_files": 64,
    "minimum_free_bytes": 5 * 1024**3,
    "jobs": 2,
    "poll_seconds": 1,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def candidate_module():
    path = Path(__file__).resolve().with_name("candidate_validation.py")
    require(
        path.is_file() and not any(p.is_symlink() for p in [path, *path.parents]),
        "Candidate helper path changed",
    )
    with path.open("rb") as stream:
        require(
            hashlib.file_digest(stream, "sha256").hexdigest() == CANDIDATE_SHA,
            "Version-bound candidate helper changed",
        )
    spec = importlib.util.spec_from_file_location("full_suite_candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def authenticate_focused(candidate, plan, plan_sha, path, external_sha):
    identity = candidate.pinned(path, external_sha)
    summary = candidate.json_read(path)
    require(
        summary["schema"] == candidate.SCHEMA
        and summary["protocol"] == PROTOCOL
        and summary["plan"]["sha256"] == plan_sha
        and summary["complete"] is True
        and summary["evidence_valid"] is True
        and summary["passed"] is True
        and summary["formal_eligible"] is False
        and summary["formal_fits"] == 0
        and summary["admission_stream_opened"] is False
        and summary["original_full_suite_failure_unchanged"] is True
        and summary["full_suite_required"] is True
        and summary["full_suite_executed"] is False,
        "A complete passing focused development replay is required",
    )
    candidate.check_binding(summary["plan"])
    candidate.audit_completed_workers(summary["workers"], plan, plan_sha)
    require(
        all(row["status"] == "passed" and row["exit_code"] == 0 for row in summary["workers"]),
        "A retained focused failure cannot be waived",
    )
    return identity


def suite_inventory(candidate, plan):
    original = candidate.json_read(plan["helpers"]["original"]["path"])
    reference = candidate.load(plan["helpers"]["reference"]["path"], "full_retained_replay")
    failed = reference.failure_inventory(original)
    candidate.validate_inventory(plan, original, failed)
    full = plan["full_suite"]
    ids = full["planned_ids"]
    require(len(ids) == len(set(ids)) == 10353, "Full suite must retain exactly 10,353 methods")
    return {
        "test_ids": ids,
        "skip_ids": full["retained_skip_ids"],
        "expected_failure_ids": full["retained_expected_failure_ids"],
    }


def environment(candidate, plan, policy, cache):
    controlled = candidate.controlled_environment(os.environ)
    allowed = {
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "DEVELOPER_DIR",
        "SDKROOT",
        "MACOSX_DEPLOYMENT_TARGET",
        *candidate.PINS,
    }
    # Do not write unrelated parent environment variables or credentials into a plan.
    env = {key: value for key, value in controlled.items() if key in allowed}
    policy.configure_environment(env, cache)
    # Reuse the installed Mach-O Ninja from the retained pinned wheel; install nothing.
    tools = Path(plan["workspace"]) / candidate.OLD / "runtime-env/bin"
    search = [path for path in env.get("PATH", "").split(os.pathsep) if path and path != str(tools)]
    env["PATH"] = os.pathsep.join([str(tools), *search])
    return env


def fixture_prepare(candidate, plan, policy, cache, env):
    """Run only the stock policy's read/hash/cache-marker preparation, no build."""
    old = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(env)
        policy.require_environment(os.environ, cache)
        proof = policy.prepare(cache, Path(plan["source"]), Path(plan["runtime"]))
    finally:
        os.environ.clear()
        os.environ.update(old)
    require(
        proof["runtime_manifest_sha256"] == plan["runtime_manifest"]["sha256"],
        "Stock fixture cache belongs to another runtime",
    )
    require(
        proof["ninja"]
        == {
            "path": str(Path(plan["workspace"]) / candidate.OLD / "runtime-env/bin/ninja"),
            "sha256": NINJA_SHA,
        },
        "Stock fixture Ninja differs from its retained wheel payload",
    )
    require(
        policy.MAX_CACHE_BYTES == LIMITS["cache_bytes"]
        and policy.MAX_CACHE_FILES == LIMITS["cache_files"]
        and policy.MIN_FREE_BYTES == LIMITS["minimum_free_bytes"]
        and policy.CONTROLS["MAX_JOBS"] == str(LIMITS["jobs"])
        and policy.CONTROLS["CMAKE_BUILD_PARALLEL_LEVEL"] == str(LIMITS["jobs"]),
        "Stock fixture bounds changed",
    )
    return proof


def prepare(candidate_plan, candidate_sha, focused, focused_sha, output, timeout):
    candidate = candidate_module()
    plan, _ = candidate.authenticate_plan(candidate_plan, candidate_sha)
    require(
        sys.flags.isolated
        and sys.flags.no_site
        and not sys.flags.optimize
        and Path(sys.executable).resolve(strict=True) == Path(plan["python"]["path"]),
        "Prepare with the bound Python executable and -I -S -B",
    )
    candidate.output_path(Path(plan["workspace"]), output)
    require(type(timeout) is int and 60 <= timeout <= 21600, "Require a bounded full-suite timeout")
    focused_identity = authenticate_focused(candidate, plan, candidate_sha, focused, focused_sha)
    inventory = suite_inventory(candidate, plan)
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "status": "preparing",
    }
    candidate.write(output / "preparation.json", record, new=True)
    try:
        source = Path(__file__).resolve().with_name("candidate_fixture_policy.py.txt")
        require(
            plan["helpers"]["extension_policy"]["sha256"] == POLICY_SHA,
            "Historical stock-fixture policy changed",
        )
        candidate.pinned(source, POLICY_SHA)
        policy_path = output / "test_extension_policy_v2.py"
        with policy_path.open("xb") as stream:
            stream.write(source.read_bytes())
        policy_identity = candidate.pinned(policy_path, POLICY_SHA)
        policy = candidate.load(policy_path, "owned_full_stock_fixture_policy")
        cache = output / "pytorch-test-extensions"
        require(not cache.exists(), "Full suite needs one new runtime-bound fixture cache")
        env = environment(candidate, plan, policy, cache)
        proof = fixture_prepare(candidate, plan, policy, cache, env)
        manifest = {
            **record,
            "status": "frozen_pending_explicit_run",
            "mode": "full",
            "candidate_plan": candidate.binding(candidate_plan),
            "focused_summary": focused_identity,
            "inventory": inventory,
            "helper": candidate.binding(Path(candidate.__file__).resolve()),
            "driver": candidate.binding(Path(__file__).resolve()),
            "policy": policy_identity,
            "cache": str(cache),
            "cache_identity": candidate.binding(cache / "cache-identity.json"),
            "fixture_preparation": proof,
            "limits": dict(LIMITS),
            "timeout_seconds": timeout,
            "execution_environment": env,
            "source_sha256": candidate.TEST_SOURCE_SHA,
            "scope": "Every unchanged test_mps.py method, not broader multi-file --mps",
            "compilation_scope": (
                "Only the unchanged upstream mps_iter_conformance C++ fixture may compile; "
                "MAX_JOBS=2, no debug flags, one bounded cache. No framework source build."
            ),
            "local_framework_compilation": False,
            "standard_study_unchanged": True,
        }
        candidate.write(output / "plan.json", manifest, new=True)
        record.update(
            status="prepared",
            plan=candidate.binding(output / "plan.json"),
            numerical_work_executed=False,
            fixture_compiled=False,
        )
    except BaseException:
        record.update(status="failed", exception=traceback.format_exc())
        raise
    finally:
        candidate.write(output / "preparation.json", record)
    return record


def authenticate(path, sha):
    candidate = candidate_module()
    candidate.pinned(path, sha)
    manifest = candidate.json_read(path)
    require(
        manifest["schema"] == SCHEMA
        and manifest["protocol"] == PROTOCOL
        and manifest["status"] == "frozen_pending_explicit_run"
        and manifest["mode"] == "full"
        and manifest["formal_eligible"] is False
        and manifest["formal_fits"] == 0
        and manifest["admission_stream_opened"] is False
        and manifest["local_framework_compilation"] is False
        and manifest["standard_study_unchanged"] is True
        and manifest["limits"] == LIMITS,
        "Not a frozen full-suite development plan",
    )
    require(
        type(manifest["timeout_seconds"]) is int and 60 <= manifest["timeout_seconds"] <= 21600,
        "Full-suite timeout changed",
    )
    for key in (
        "candidate_plan",
        "focused_summary",
        "helper",
        "driver",
        "policy",
        "cache_identity",
    ):
        candidate.check_binding(manifest[key])
    require(
        manifest["helper"]["sha256"] == CANDIDATE_SHA
        and manifest["helper"]["path"] == str(Path(candidate.__file__).resolve())
        and manifest["driver"]["path"] == str(Path(__file__).resolve())
        and manifest["policy"]["sha256"] == POLICY_SHA
        and manifest["source_sha256"] == candidate.TEST_SOURCE_SHA,
        "Execution source binding changed",
    )
    plan, reference = candidate.authenticate_plan(
        Path(manifest["candidate_plan"]["path"]), manifest["candidate_plan"]["sha256"]
    )
    authenticate_focused(
        candidate,
        plan,
        manifest["candidate_plan"]["sha256"],
        Path(manifest["focused_summary"]["path"]),
        manifest["focused_summary"]["sha256"],
    )
    require(
        manifest["inventory"] == suite_inventory(candidate, plan), "Full test inventory changed"
    )
    cache = Path(manifest["cache"])
    require(
        cache == Path(manifest["cache_identity"]["path"]).parent
        and cache == path.parent / "pytorch-test-extensions",
        "Fixture cache substituted",
    )
    policy = candidate.load(manifest["policy"]["path"], "owned_full_stock_fixture_policy")
    proof = fixture_prepare(candidate, plan, policy, cache, manifest["execution_environment"])
    # The cache grows during the real upstream fixture build. Its identity,
    # source, toolchain, headers, limits, and control inputs remain immutable.
    before = {k: v for k, v in proof.items() if k != "before"}
    expected = {k: v for k, v in manifest["fixture_preparation"].items() if k != "before"}
    require(before == expected, "Candidate stock-fixture inputs changed")
    require(
        manifest["execution_environment"] == environment(candidate, plan, policy, cache),
        "Controlled full-suite environment changed",
    )
    return candidate, manifest, plan, reference, policy


def validate_tests(tests, inventory):
    require(
        type(tests["run"]) is int
        and tests["run"] == len(inventory["test_ids"])
        and tests["executed_ids"] == inventory["test_ids"],
        "Full suite incomplete or reordered",
    )
    require(
        [row["id"] for row in tests["skipped"]] == inventory["skip_ids"],
        "Retained upstream skip eligibility changed",
    )
    require(
        [row["id"] for row in tests["expected_failures"]] == inventory["expected_failure_ids"],
        "Retained expected-failure eligibility changed",
    )
    for kind in ("failures", "errors", "expected_failures"):
        rows = tests[kind]
        require(
            len({row["id"] for row in rows}) == len(rows)
            and all(row["id"] in inventory["test_ids"] and row["traceback"] for row in rows),
            "Missing or substituted full-suite failure record",
        )
    require(all(row["reason"] for row in tests["skipped"]), "Missing upstream skip reason")
    require(
        len(set(tests["unexpected_successes"])) == len(tests["unexpected_successes"])
        and all(identity in inventory["test_ids"] for identity in tests["unexpected_successes"]),
        "Unexpected-success inventory changed",
    )
    successful = not (tests["failures"] or tests["errors"] or tests["unexpected_successes"])
    require(tests["successful"] is successful, "Contradictory full-suite outcome")
    return successful


def cache_content(snapshot):
    """Compare immutable cache contents, not fluctuating disk-free telemetry."""
    return {key: value for key, value in snapshot.items() if key != "free_bytes"}


def worker(path, sha, output):
    candidate = candidate_module()
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "plan_sha256": sha,
        "pid": os.getpid(),
        "complete": False,
        "passed": False,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
    }
    context = None
    try:
        candidate, manifest, plan, reference, policy = authenticate(path, sha)
        context = (manifest, plan, policy)
        require(
            sys.flags.isolated
            and sys.flags.no_site
            and not sys.flags.optimize
            and Path(sys.executable).resolve(strict=True) == Path(plan["python"]["path"]),
            "Full-suite worker requires the isolated bound Python",
        )
        require(dict(os.environ) == manifest["execution_environment"], "Worker environment differs")
        bytecode = output / "unused-bytecode-cache"
        require(
            sys.pycache_prefix == str(bytecode) and not bytecode.exists(),
            "Require a fresh unused bytecode prefix",
        )
        os.environ["PYTHONPYCACHEPREFIX"] = str(bytecode)
        record.update(
            planned_tests=manifest["inventory"]["test_ids"],
            source_manifest_sha256=plan["source_manifest"]["sha256"],
            runtime_manifest_sha256=plan["runtime_manifest"]["sha256"],
            fixture_before=policy.snapshot(Path(manifest["cache"]), hashes=True),
        )
        candidate.write(output / "result.json", record, new=True)
        verifier_path = Path(plan["artifact"]) / "verify_runtime.py"
        verifier = candidate.load(verifier_path, "full_candidate_runtime_verifier")
        sys.argv = [
            str(verifier_path),
            "--runtime",
            plan["runtime"],
            "--dependencies",
            plan["dependencies"],
            "--receipt",
            str(output / "runtime-import.json"),
            "--expected-base",
            candidate.BASE,
        ]
        verifier.main()
        import torch

        require(
            torch.backends.mps.is_available() and not torch.ops.loaded_libraries,
            "Real MPS execution without operator overrides is required",
        )
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        record["state_after_bootstrap"] = reference.state(torch)
        record["native_origins_before"] = candidate.native_origins(Path(plan["runtime"]))
        driver = candidate.load(plan["helpers"]["driver"]["path"], "full_candidate_test_helpers")
        record["conditions_before"] = driver.conditions()
        launcher = output / "archive-python"
        launcher.write_text(
            "#!/bin/sh\nexec " + shlex.quote(plan["python"]["path"]) + ' -S -B "$@"\n'
        )
        launcher.chmod(0o700)
        sys.executable = str(launcher)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [plan["runtime"], plan["dependencies"], str(Path(plan["source"]) / "test")]
        )
        test_source = Path(plan["source"]) / "test/test_mps.py"
        candidate.pinned(test_source, candidate.TEST_SOURCE_SHA)
        sys.argv = [str(test_source)]
        module = driver.load_module(test_source, "integrated_upstream_test_mps")
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        collected = [test.id() for test in driver.flatten(suite)]
        require(
            collected == manifest["inventory"]["test_ids"],
            "Upstream test discovery changed before execution",
        )
        record["state_before_test"] = reference.state(torch)
        candidate.write(output / "result.json", record)
        result = unittest.TextTestRunner(verbosity=2, resultclass=driver.RetainedResult).run(suite)
        record["tests"] = reference.suite_record(result)
        candidate.write(output / "result.json", record)
        record["state_after_test"] = reference.state(torch)
        torch.mps.synchronize()
        record["conditions_after"] = driver.conditions()
        policy.require_environment(os.environ, Path(manifest["cache"]))
        require(
            all(os.environ.get(key) == value for key, value in candidate.PINS.items())
            and all(
                key not in os.environ
                for key in candidate.ABSENT
                if key not in {"PYTHONPATH", "PYTHONPYCACHEPREFIX"}
            )
            and os.environ["PYTHONPYCACHEPREFIX"] == str(bytecode)
            and sys.pycache_prefix == str(bytecode)
            and not bytecode.exists()
            and "_virtualenv" not in sys.modules
            and not torch.ops.loaded_libraries,
            "Execution controls, bytecode policy, or operator libraries changed",
        )
        runtime_manifest = candidate.json_read(plan["runtime_manifest"]["path"])
        origins, synthetic = verifier.verify_module_origins(
            Path(plan["runtime"]), runtime_manifest["files"], sys.modules, torch
        )
        record["postflight_module_origins"] = origins
        record["postflight_synthetic_origins"] = synthetic
        record["native_origins_after"] = candidate.native_origins(Path(plan["runtime"]))
        record["stock_fixture_loaded"] = policy.loaded_extension(module, Path(manifest["cache"]))
        record["normal_import"] = candidate.binding(output / "runtime-import.json")
        record["passed"] = validate_tests(record["tests"], manifest["inventory"])
        record["complete"] = True
    except BaseException:
        record.update(complete=False, passed=False, exception=traceback.format_exc())
    finally:
        if context is not None:
            manifest, plan, policy = context
            try:
                record["fixture_after"] = policy.snapshot(Path(manifest["cache"]), hashes=True)
                # Reauthenticate using the original process environment. Upstream
                # subprocess bootstrap adds only declared PYTHONPATH/cache fields.
                old = dict(os.environ)
                try:
                    os.environ.clear()
                    os.environ.update(manifest["execution_environment"])
                    authenticate(path, sha)
                finally:
                    os.environ.clear()
                    os.environ.update(old)
                record["integrity_after"] = True
            except BaseException:
                record.update(
                    complete=False,
                    passed=False,
                    integrity_after=False,
                    integrity_error=traceback.format_exc(),
                )
        candidate.write(output / "result.json", record)
    return 2 if not record["complete"] else int(not record["passed"])


def validate_worker(candidate, record, manifest, plan, sha, pid, code):
    require(
        record["schema"] == SCHEMA
        and record["protocol"] == PROTOCOL
        and record["plan_sha256"] == sha
        and record["pid"] == pid
        and record["complete"] is True
        and record["integrity_after"] is True
        and record["formal_eligible"] is False
        and record["formal_fits"] == 0
        and record["admission_stream_opened"] is False
        and record["source_manifest_sha256"] == plan["source_manifest"]["sha256"]
        and record["runtime_manifest_sha256"] == plan["runtime_manifest"]["sha256"],
        "Incomplete or mismatched full-suite worker",
    )
    passed = validate_tests(record["tests"], manifest["inventory"])
    require(
        record["passed"] is passed and type(code) is int and code == int(not passed),
        "Full-suite outcome and exit code disagree",
    )
    for key in ("state_after_bootstrap", "state_before_test", "state_after_test"):
        require(
            record[key]["host_threads"] == record[key]["interop_threads"] == 1
            and record[key]["operator_libraries"] == [],
            "Runtime state changed",
        )
    candidate.check_binding(record["normal_import"])
    imported = candidate.json_read(record["normal_import"]["path"])
    require(
        imported["status"] == "passed"
        and imported["runtime"] == plan["runtime"]
        and imported["runtime_manifest_sha256"] == plan["runtime_manifest"]["sha256"]
        and imported["mps_available"] is True
        and imported["loaded_operator_libraries"] == [],
        "Full-suite import proof changed",
    )
    for key in ("native_origins_before", "native_origins_after"):
        require(record[key], "Missing native-origin evidence")
        for row in record[key]:
            require(
                Path(row["path"]).is_relative_to(Path(plan["runtime"])),
                "Foreign native framework library",
            )
            candidate.check_binding(row)
    loaded = record["stock_fixture_loaded"]
    require(
        Path(loaded["path"])
        == Path(manifest["cache"]) / "mps_iter_conformance/mps_iter_conformance.so",
        "Wrong stock extension loaded",
    )
    candidate.pinned(Path(loaded["path"]), loaded["sha256"])
    candidate.pinned(Path(loaded["path"]).with_name("build.ninja"), loaded["build_ninja_sha256"])
    return passed


def run(path, sha, output):
    candidate, manifest, plan, _, policy = authenticate(path, sha)
    candidate.output_path(Path(plan["workspace"]), output)
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "plan": candidate.binding(path),
        "complete": False,
        "evidence_valid": False,
        "passed": False,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "original_full_suite_failure_unchanged": True,
        "scope": manifest["scope"],
    }
    candidate.write(output / "summary.json", summary, new=True)
    destination = output / "full-1"
    log, monitor = output / "full-1.log", output / "cache-monitor.jsonl"
    argv = [
        plan["python"]["path"],
        "-I",
        "-S",
        "-B",
        "-X",
        "pycache_prefix=" + str(destination / "unused-bytecode-cache"),
        str(Path(__file__).resolve()),
        "worker",
        "--plan",
        str(path),
        "--plan-sha256",
        sha,
        "--output",
        str(destination),
    ]
    summary["argv"] = argv
    child = None
    try:
        with log.open("x") as stream, monitor.open("x") as samples:
            child = subprocess.Popen(
                argv,
                env=manifest["execution_environment"],
                cwd=output,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            summary["pid"] = child.pid
            candidate.write(output / "summary.json", summary)
            start = time.monotonic()
            try:
                while child.poll() is None:
                    remaining = manifest["timeout_seconds"] - (time.monotonic() - start)
                    require(remaining > 0, "Full-suite worker timed out")
                    sample = policy.snapshot(Path(manifest["cache"]), enforce=False)
                    sample["elapsed_seconds"] = time.monotonic() - start
                    samples.write(json.dumps(sample, allow_nan=False) + "\n")
                    samples.flush()
                    policy.require_bounds(sample)
                    try:
                        child.wait(timeout=min(LIMITS["poll_seconds"], remaining))
                    except subprocess.TimeoutExpired:
                        pass
            except BaseException:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
        result_path = destination / "result.json"
        summary.update(
            log=candidate.binding(log),
            monitor=candidate.binding(monitor),
            result=candidate.binding(result_path),
            exit_code=child.returncode,
        )
        record = candidate.json_read(result_path)
        passed = validate_worker(
            candidate, record, manifest, plan, sha, child.pid, child.returncode
        )
        authenticate(path, sha)
        require(
            cache_content(policy.snapshot(Path(manifest["cache"]), hashes=True))
            == cache_content(record["fixture_after"]),
            "Fixture cache changed after the worker completed",
        )
        for key in ("log", "monitor", "result"):
            candidate.check_binding(summary[key])
        validate_worker(
            candidate,
            candidate.json_read(result_path),
            manifest,
            plan,
            sha,
            child.pid,
            child.returncode,
        )
        summary.update(
            complete=True,
            evidence_valid=True,
            passed=passed,
            tests_run=record["tests"]["run"],
            failures=len(record["tests"]["failures"]),
            errors=len(record["tests"]["errors"]),
        )
    except BaseException:
        summary.update(exception=traceback.format_exc())
    finally:
        if child is not None:
            summary["exit_code"] = child.returncode
        for key, file in (
            ("log", log),
            ("monitor", monitor),
            ("result", destination / "result.json"),
        ):
            if file.is_file() and not file.is_symlink():
                summary.setdefault(key, candidate.binding(file))
        candidate.write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--candidate-plan", type=Path, required=True)
    preparation.add_argument("--candidate-plan-sha256", required=True)
    preparation.add_argument("--focused-summary", type=Path, required=True)
    preparation.add_argument("--focused-summary-sha256", required=True)
    preparation.add_argument("--output", type=Path, required=True)
    preparation.add_argument("--timeout-seconds", type=int, default=10800)
    for name in ("run", "worker"):
        command = sub.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--plan-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.candidate_plan.absolute(),
            args.candidate_plan_sha256,
            args.focused_summary.absolute(),
            args.focused_summary_sha256,
            args.output.absolute(),
            args.timeout_seconds,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "worker":
        return worker(args.plan.absolute(), args.plan_sha256, args.output.absolute())
    result = run(args.plan.absolute(), args.plan_sha256, args.output.absolute())
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("complete", "evidence_valid", "passed", "formal_eligible")
            },
            indent=2,
        )
    )
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
