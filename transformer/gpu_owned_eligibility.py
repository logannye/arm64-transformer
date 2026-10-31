"""Authenticate owned framework eligibility from full-suite raw evidence.

This gate is separate from numerical admission. It never executes a framework,
opens reserved operands, or promotes old-runtime suite results. XML/log outcomes
and complete case eligibility are independently checked under a frozen inventory.
The original failed broad-suite audit must remain bound alongside repaired runs.
"""

from __future__ import annotations

import math
import re
import runpy
import shlex
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from transformer.gpu_owned_study import HISTORICAL_AUDIT, validate_formal_selection
from transformer.gpu_study import digest, identity, read_json, regular, require, verify_files
from transformer.gpu_study_context import OWNED

INVENTORY_PATH = "integration/gpu-owned/owned-suite-inventory-v1.json"
INVENTORY_SHA256 = "cff557d2b1d68857bc488e77e6672fb0fb386409e88ad6ca4e7fa31bd862d852"
SUITES = ("pytorch_mps", "mlx_cpp", "mlx_python")
PRODUCERS = {
    "pytorch_mps": "integration/gpu-owned/owned_pytorch_suites.py",
    "mlx_cpp": "integration/gpu-owned/owned_mlx_suites.py",
    "mlx_python": "integration/gpu-owned/owned_mlx_suites.py",
}
MLX_SUBPROCESS_AST_SHA256 = "90a6406e042b09e83ed671864ad0cf30cefa2e6c4e148ca60d24ac26cbb6b433"
# The pinned common_mps.py applies bare unittest.skip to these eight already
# eligible FP16 gradient tests. Python records an empty reason for that spelling.
# No new skip ID is allowed; all other reasons must remain nonempty strings.
UPSTREAM_EMPTY_SKIP_REASON_IDS = frozenset(
    "integrated_upstream_test_mps.TestConsistencyMPS.test_output_grad_match_"
    + name
    + "_mps_float16"
    for name in (
        "nn_functional_conv1d",
        "nn_functional_conv2d",
        "nn_functional_conv3d",
        "nn_functional_conv_transpose1d",
        "nn_functional_conv_transpose2d",
        "nn_functional_conv_transpose3d",
        "nn_functional_pairwise_distance",
        "topk",
    )
)
EMPTY_REASON_TEST_SOURCES = {
    "7345b6e229aed240921460a3b598ffd1a47148ab4816aead0d405539a154a396",
    "a7b6420664a9352dcb1fc7eb070744c4f1f869d7498d036e5d73ef83dab27bc7",
}


def suite_contract(path=None):
    path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / INVENTORY_PATH
    require(digest(path) == INVENTORY_SHA256, "Owned suite inventory changed")
    result = read_json(path)
    require(
        result["schema"] == "gpu-owned-suite-inventory-v1" and result["protocol"] == OWNED.protocol,
        "Wrong owned suite inventory",
    )
    return result


def _raw(root, row):
    require(set(row) == {"path", "sha256", "bytes"}, "Malformed raw suite binding")
    path = regular(root, row["path"])
    require(
        path.stat().st_size == row["bytes"] and digest(path) == row["sha256"],
        "Raw suite evidence changed",
    )
    return path


def validate_mps(tests, contract):
    expected = contract["pytorch"]
    require(
        type(tests.get("run")) is int
        and tests["run"] == len(expected["test_ids"])
        and tests.get("executed_ids") == expected["test_ids"],
        "Incomplete, duplicate or reordered full MPS suite",
    )
    for field, key in (("skipped", "skip_ids"), ("expected_failures", "expected_failure_ids")):
        require(
            [row["id"] for row in tests[field]] == expected[key], "Upstream MPS eligibility changed"
        )
    for kind in ("failures", "errors", "expected_failures"):
        rows = tests[kind]
        require(
            len({row["id"] for row in rows}) == len(rows)
            and all(row["id"] in expected["test_ids"] and row["traceback"] for row in rows),
            "Missing or substituted MPS failure evidence",
        )
    require(
        all(
            isinstance(row["reason"], str)
            and (
                row["reason"]
                or (
                    row["id"] in UPSTREAM_EMPTY_SKIP_REASON_IDS
                    and expected["test_source_sha256"] in EMPTY_REASON_TEST_SOURCES
                )
            )
            for row in tests["skipped"]
        ),
        "Missing upstream skip reasons",
    )
    require(
        tests["failures"] == tests["errors"] == tests["unexpected_successes"] == []
        and tests["successful"] is True,
        "Full MPS suite still fails",
    )
    return {
        "cases": len(expected["test_ids"]),
        "skips": len(expected["skip_ids"]),
        "expected_failures": len(expected["expected_failure_ids"]),
    }


def validate_mlx_cpp(log, contract):
    text = Path(log).read_text()
    cases = re.findall(
        r"\[doctest\] test cases:\s*(\d+)\s*\|\s*(\d+) passed"
        r"\s*\|\s*(\d+) failed\s*\|\s*(\d+) skipped",
        text,
    )
    assertions = re.findall(
        r"\[doctest\] assertions:\s*(\d+)\s*\|\s*(\d+) passed\s*\|\s*(\d+) failed", text
    )
    require(len(cases) == len(assertions) == 1, "Missing or duplicate C++ suite totals")
    n, a = contract["mlx"]["cpp_case_count"], contract["mlx"]["cpp_assertions"]
    require(
        tuple(map(int, cases[0])) == (n, n, 0, 0) and tuple(map(int, assertions[0])) == (a, a, 0),
        "Incomplete or failed C++ suite",
    )
    return {"cases": n, "assertions": a}


def validate_mlx_python(xml, log, contract):
    rows = []
    for case in ET.parse(xml).iter("testcase"):
        require(
            case.find("failure") is None and case.find("error") is None,
            "MLX Python suite contains a failure",
        )
        skipped = case.find("skipped")
        row = {
            "class": case.attrib.get("classname", ""),
            "name": case.attrib.get("name", ""),
            "status": "skipped" if skipped is not None else "passed",
        }
        if skipped is not None:
            row["reason"] = skipped.attrib.get("message", "")
        rows.append(row)
    rows.sort(key=lambda item: (item["class"], item["name"]))
    require(
        rows == contract["mlx"]["python_cases"], "MLX Python case or eligibility inventory changed"
    )
    matches = re.findall(
        r"^(\d+) passed, (\d+) skipped, (\d+) subtests passed in [0-9.]+s(?: \([^\n]*\))?$",
        Path(log).read_text(),
        re.M,
    )
    skipped = sum(row["status"] == "skipped" for row in rows)
    require(
        len(matches) == 1
        and tuple(map(int, matches[0]))
        == (len(rows) - skipped, skipped, contract["mlx"]["python_subtests"]),
        "Incomplete MLX Python/subtest log",
    )
    return {"cases": len(rows), "skips": skipped, "subtests": contract["mlx"]["python_subtests"]}


def validate_mlx_descendants(record, root, manifest, runtime_receipt):
    """Re-evaluate both unchanged upstream fresh-process checks, without live paths."""
    from transformer.gpu_owned_backend import validate_compilation_probe

    require(isinstance(runtime_receipt, dict), "Descendant audit needs bound runtime authority")
    binding = manifest["bindings"]
    require(
        runtime_receipt["project"]["files"] == binding["source"],
        "Descendant project authority differs",
    )
    routing = record.get("descendant_routing", {})
    require(
        set(routing)
        == {"contract", "launcher", "producer_sha256", "children", "retained_child_receipts"},
        "Missing complete descendant routing evidence",
    )
    contract = routing["contract"]
    require(
        contract
        == {
            "mode": "explicit_authenticated_interpreter_launcher_conformance_only",
            "calls": contract.get("calls"),
            "calls_sha256": MLX_SUBPROCESS_AST_SHA256,
            "python_invocation": [
                "source/python/tests/test_gather_grad.py",
                "--repeatability-worker",
            ],
            "required_python_children": 2,
            "upstream_child_timeout_seconds": 120,
            "system_query": "sysctl -n machdep.cpu.brand_string",
            "subprocess_function_monkeypatch": False,
            "test_source_modified": False,
            "measured_baseline_launcher": False,
        }
        and identity(contract["calls"]) == MLX_SUBPROCESS_AST_SHA256,
        "Upstream descendant call contract changed",
    )
    producer = PRODUCERS["mlx_python"]
    source = Path(__file__).resolve().parents[1] / producer
    require(
        routing["producer_sha256"] == binding["source"].get(producer) == digest(source),
        "Descendant producer source is missing or changed",
    )
    validator = runpy.run_path(str(source))["validate_descendant_proofs"]
    children = routing["children"]
    require(
        isinstance(children, list)
        and len(children) == 2
        and len({item["path"] for item in children}) == 2
        and children == routing["retained_child_receipts"],
        "Missing, duplicate, or removed upstream descendants",
    )
    process = record["process"]
    command = process["command"]
    values = {}
    for flag in ("--output", "--plan", "--plan-sha256"):
        require(
            command.count(flag) == 1 and command.index(flag) + 1 < len(command),
            "Missing suite launch argument",
        )
        values[flag] = command[command.index(flag) + 1]
    require(
        values["--plan"] == record["plan_path"]
        and values["--plan-sha256"] == record["plan_sha256"],
        "Descendant plan differs from actual parent launch",
    )
    original_output = Path(values["--output"])
    require(original_output.is_absolute(), "Noncanonical original suite output")
    runtime_root = Path(record["runtime_root"])
    layout = binding["runtime_layout"]
    require(
        runtime_root.is_absolute()
        and Path(layout["interpreter"]).is_relative_to(runtime_root)
        and Path(layout["interpreter"]).relative_to(runtime_root).as_posix() in binding["runtime"],
        "Descendant runtime root does not bind actual interpreter",
    )
    launcher = _raw(root, routing["launcher"])
    require(
        routing["launcher"]["path"] == "archive-python", "Unexpected descendant launcher location"
    )
    expected_command = [
        layout["interpreter"],
        "-P",
        "-S",
        "-B",
        str(Path(runtime_receipt["project"]["root"]) / producer),
        "descendant",
        "--plan",
        record["plan_path"],
        "--plan-sha256",
        record["plan_sha256"],
        "--parent-output",
        str(original_output),
        "--parent-pid",
        str(process["pid"]),
        "--",
    ]
    require(
        launcher.read_text()
        == "#!/bin/sh\nexec " + " ".join(map(shlex.quote, expected_command)) + ' "$@"\n',
        "Descendant interpreter launcher changed",
    )
    bundle = runtime_receipt["frameworks"]["mlx"]["payload_root"]
    intervals = []
    for item in children:
        path = _raw(root, item)
        child = read_json(path)
        original_child = original_output / item["path"]
        require(
            child.get("schema") == "gpu-owned-mlx-suite-descendant-v1"
            and child.get("protocol") == OWNED.protocol
            and child.get("producer") == producer
            and child.get("complete") is True
            and child.get("integrity_after") is True
            and type(child.get("returncode")) is int
            and child["returncode"] == 0
            and child.get("formal_admission_opened") is False
            and child.get("formal_fits") == 0
            and "error" not in child
            and "integrity_error" not in child,
            "Failed or incomplete upstream descendant",
        )
        require(
            type(child.get("pid")) is int
            and child["pid"] > 0
            and child["pid"] != process["pid"]
            and child.get("parent_pid") == process["pid"]
            and path.parent.name.startswith("child-" + str(child["pid"]) + "-")
            and child.get("parent_output") == str(original_output)
            and child.get("plan_path") == record["plan_path"]
            and child.get("plan_sha256") == record["plan_sha256"]
            and child.get("bytecode_prefix")
            == str(original_child.parent / "unused-bytecode-cache"),
            "Descendant launch identity changed",
        )
        require(
            child.get("runtime_identity_sha256_before")
            == child.get("runtime_identity_sha256_after")
            == identity(binding["runtime"])
            and child.get("source_sha256_before")
            == child.get("source_sha256_after")
            == binding["source"]
            and child.get("runtime_receipt_sha256") == record["runtime_receipt_sha256"]
            and child.get("framework_manifest_sha256") == record["framework_manifest_sha256"],
            "Descendant runtime or source identity changed",
        )
        require(
            child.get("launcher")
            == {**routing["launcher"], "path": str(original_output / "archive-python")}
            and child.get("invocation")
            == [str(Path(bundle) / contract["python_invocation"][0]), "--repeatability-worker"],
            "Descendant invocation changed",
        )
        begin, end = child.get("started_unix"), child.get("ended_unix")
        require(
            type(begin) in (int, float)
            and type(end) in (int, float)
            and math.isfinite(begin)
            and math.isfinite(end)
            and process["started_unix"] <= begin <= end <= process["ended_unix"],
            "Descendant interval is outside its parent",
        )
        intervals.append((begin, end))
        for phase in ("before", "after"):
            validate_compilation_probe(child["compilation_" + phase])
        validator(
            child,
            {"files": binding["runtime"], "layout": layout},
            str(runtime_root),
            runtime_receipt["project"],
            bundle,
        )
    intervals.sort()
    require(intervals[0][1] <= intervals[1][0], "Upstream descendant GPU processes overlapped")
    for item in children:
        _raw(root, item)


def validate_mps_archive(record, root, manifest, runtime_receipt):
    """Authenticate the portable auditor before reading any archived MPS inputs."""
    require(isinstance(runtime_receipt, dict), "MPS archive needs bound runtime authority")
    source = Path(__file__).resolve().parents[1]
    names = {
        "producer": PRODUCERS["pytorch_mps"],
        "policy": "integration/gpu-owned/owned_pytorch_fixture.py",
        "archive": "integration/gpu-owned/owned_pytorch_archive.py",
    }
    for name in names.values():
        require(
            manifest["bindings"]["source"].get(name) == digest(source / name),
            "MPS archive validator is outside the selected project authority",
        )
    modules = {key: runpy.run_path(str(source / name)) for key, name in names.items()}
    return modules["archive"]["audit"](
        root,
        record,
        manifest,
        runtime_receipt,
        SimpleNamespace(**modules["producer"]),
        SimpleNamespace(**modules["policy"]),
    )


def validate_suite(record, root, manifest, contract, name, runtime_receipt=None):
    require(
        name in SUITES
        and record.get("schema") == "gpu-owned-suite-execution-v1"
        and record.get("protocol") == OWNED.protocol
        and record.get("name") == name,
        "Wrong owned suite receipt",
    )
    require(
        record.get("complete") is True
        and record.get("integrity_after") is True
        and "error" not in record
        and "integrity_error" not in record,
        "Suite did not finish all runtime and evidence checks",
    )
    if name == "pytorch_mps" and not record.get("portable_evidence"):
        raise ValueError(
            "Owned MPS archive qualification remains blocked: complete portable "
            "descendant, stock-fixture, helper and full-source evidence is required"
        )
    binding = manifest["bindings"]
    selected = manifest["owned_frameworks"]
    runtime_hash = identity(binding["runtime"])
    require(
        record.get("formal_admission_opened") is False and record.get("formal_fits") == 0,
        "Framework development is not formal admission",
    )
    require(
        record.get("runtime_identity_sha256_before")
        == record.get("runtime_identity_sha256_after")
        == runtime_hash
        and record.get("runtime_receipt_sha256")
        == binding["baseline"][selected["runtime_receipt"]],
        "Suite runtime/dependency identity is stale",
    )
    framework = "pytorch" if name == "pytorch_mps" else "mlx"
    require(
        record.get("framework_manifest_sha256")
        == selected["frameworks"][framework]["runtime_manifest_sha256"],
        "Suite ran an unselected native build",
    )
    require(
        record.get("source_sha256_before") == record.get("source_sha256_after") == binding["source"]
        and record.get("producer") == PRODUCERS[name]
        and PRODUCERS[name] in binding["source"],
        "Suite producer sources changed or are missing",
    )
    sources = (
        {"test/test_mps.py": contract["pytorch"]["test_source_sha256"]}
        if name == "pytorch_mps"
        else contract["mlx"]["test_source_sha256"]
    )
    require(record.get("test_source_sha256") == sources, "Full upstream test sources changed")
    require(
        record.get("execution_scope") == "full_gpu_default_suite"
        and record.get("no_filters") is True
        and record.get("no_test_source_modifications") is True,
        "Suite scope was narrowed",
    )
    process = record["process"]
    require(
        type(process.get("pid")) is int
        and process["pid"] > 0
        and type(process.get("returncode")) is int
        and process["returncode"] == 0
        and process.get("timed_out") is False,
        "Suite execution failed or is incomplete",
    )
    require(
        all(
            type(process.get(key)) in (int, float) and math.isfinite(process[key])
            for key in ("started_unix", "ended_unix")
        )
        and process["ended_unix"] >= process["started_unix"],
        "Invalid suite execution interval",
    )
    require(
        isinstance(process.get("command"), list)
        and process["command"]
        and all(isinstance(arg, str) and arg for arg in process["command"]),
        "Missing exact suite argv",
    )
    native = record.get("native_artifacts")
    require(
        isinstance(native, dict)
        and native
        and all(binding["runtime"].get(path) == sha for path, sha in native.items()),
        "Suite native provenance is outside current runtime",
    )
    require(
        any(
            path.endswith("libtorch_cpu.dylib" if framework == "pytorch" else "libmlx.dylib")
            for path in native
        ),
        "Required framework native image absent",
    )
    if name == "mlx_python":
        require(
            any("/mlx/core." in path and path.endswith(".so") for path in native),
            "Actual MLX core image is absent",
        )
    executable = record["executable"]
    require(
        binding["runtime"].get(executable["runtime_relative"]) == executable["sha256"],
        "Suite executable differs from authenticated runtime",
    )
    proof = record["native_origin_verification"]
    require(
        proof.get("method")
        == ("direct_pinned_executable_and_linkage" if name == "mlx_cpp" else "loaded_images")
        and isinstance(proof.get("before"), dict)
        and proof["before"]
        and isinstance(proof.get("after"), dict)
        and proof["after"],
        "Missing actual native-origin proof",
    )
    require(
        proof["before"].get("files") == proof["after"].get("files") == native,
        "Native origin evidence does not bind the retained artifacts",
    )
    if name == "pytorch_mps":
        validate_mps_archive(record, root, manifest, runtime_receipt)
    if name.startswith("mlx_"):
        from transformer.gpu_owned_backend import validate_compilation_probe

        require(
            record.get("natural_gpu_default") is True
            and record.get("numpy", {}).get("version") == "1.26.4",
            "MLX suite did not use the selected natural GPU/joint dependency environment",
        )
        require(
            record["numpy"]["path"] in binding["runtime_layout"]["packages"]["numpy"],
            "MLX suite imported a different NumPy distribution",
        )
        for phase in ("before", "after"):
            validate_compilation_probe(record["compilation_" + phase])
        retained = record.get("retained_files")
        require(isinstance(retained, dict) and retained, "Missing retained MLX suite outputs")
        for item in retained.values():
            _raw(root, item)
        if name == "mlx_python":
            validate_mlx_descendants(record, root, manifest, runtime_receipt)
    files = record["raw"]
    require(
        set(files)
        == (
            {"log", "tests"}
            if name == "pytorch_mps"
            else {"log", "xml"}
            if name == "mlx_python"
            else {"log"}
        ),
        "Raw full-suite evidence inventory changed",
    )
    paths = {key: _raw(root, row) for key, row in files.items()}
    if name == "pytorch_mps":
        result = validate_mps(read_json(paths["tests"]), contract)
    elif name == "mlx_cpp":
        result = validate_mlx_cpp(paths["log"], contract)
    else:
        result = validate_mlx_python(paths["xml"], paths["log"], contract)
    for row in files.values():
        _raw(root, row)
    return result


def verify_eligibility(manifest, roots):
    selected = validate_formal_selection(manifest)
    root = Path(roots["baseline"])
    bindings = manifest["bindings"]["baseline"]
    require(
        selected["eligibility_receipt"] in bindings and selected["runtime_receipt"] in bindings,
        "Owned framework eligibility is missing; admission stays unopened",
    )
    verify_files(root, bindings)
    receipt = read_json(regular(root, selected["eligibility_receipt"]))
    require(
        receipt.get("schema") == "gpu-owned-framework-eligibility-v1"
        and receipt.get("protocol") == OWNED.protocol
        and receipt.get("formal_admission_opened") is False,
        "Not owned framework eligibility evidence",
    )
    require(
        receipt.get("runtime_identity_sha256") == identity(manifest["bindings"]["runtime"])
        and receipt.get("runtime_receipt_sha256") == bindings[selected["runtime_receipt"]]
        and receipt.get("suite_inventory_sha256") == INVENTORY_SHA256,
        "Eligibility is stale or belongs to another contract",
    )
    runtime = read_json(regular(root, selected["runtime_receipt"]))
    require(
        runtime.get("schema") == "gpu-owned-runtime-identity-v1"
        and runtime.get("protocol") == OWNED.protocol
        and runtime.get("runtime_identity_sha256") == receipt["runtime_identity_sha256"]
        and runtime.get("project", {}).get("files") == manifest["bindings"]["source"]
        and {key: row["selection"] for key, row in runtime["frameworks"].items()}
        == selected["frameworks"],
        "Owned runtime selection or project authority differs",
    )
    history = receipt["historical_failure"]
    require(
        history.get("sha256") == HISTORICAL_AUDIT["sha256"]
        and history.get("bytes") == HISTORICAL_AUDIT["bytes"],
        "Historical failure evidence was replaced",
    )
    historical = read_json(_raw(root, history))
    require(
        historical.get("development_qualification_passed") is False
        and historical.get("formal_admission_eligible") is False,
        "Historical failure was relabeled",
    )
    suites = receipt.get("suites", [])
    require(
        [row.get("name") for row in suites] == list(SUITES),
        "Missing, duplicate or reordered full framework suites",
    )
    contract = suite_contract()
    require(
        manifest["bindings"]["source"].get(INVENTORY_PATH) == INVENTORY_SHA256,
        "Suite inventory is outside the frozen source contract",
    )
    for row in suites:
        path = regular(root, row["receipt"])
        require(row["receipt"] in bindings, "Unbound full-suite receipt")
        validate_suite(read_json(path), path.parent, manifest, contract, row["name"], runtime)
    verify_files(root, bindings)
    return True
