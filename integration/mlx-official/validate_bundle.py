"""Replay one prospective official matrix cell using its natural device default.

No installation or compilation occurs here. Use the retained dependency lock in
an isolated environment first. CPU-only must have no Metal backend; both Metal
settings require actual GPU execution. Earlier CPU-on-Metal failures are untouched.
"""

import argparse
import ast
import importlib.metadata
import json
import os
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from integration_common import digest, inventory, run, write


def required_gpu_tests(source):
    names = []
    for filename in ("test_gather_grad.py", "test_fast_sdpa_training.py"):
        tree = ast.parse((source / "python/tests" / filename).read_text())
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                names += [
                    (node.name, item.name)
                    for item in node.body
                    if isinstance(item, ast.FunctionDef) and item.name.startswith("test_")
                ]
    return names


def test_inventory(xml, required=()):
    tree = ET.parse(xml)
    cases = list(tree.iter("testcase"))
    result = {"testcases": len(cases), "skipped": [], "failures": []}
    for case in cases:
        entry = {"class": case.attrib.get("classname", ""), "name": case.attrib.get("name", "")}
        if case.find("skipped") is not None:
            result["skipped"].append(
                {**entry, "reason": case.find("skipped").attrib.get("message", "")}
            )
        if case.find("failure") is not None or case.find("error") is not None:
            result["failures"].append(entry)
    for classname, name in required:
        matches = [
            c
            for c in cases
            if c.attrib.get("classname", "").split(".")[-1] == classname
            and c.attrib.get("name") == name
        ]
        if len(matches) != 1 or any(
            matches[0].find(tag) is not None for tag in ("skipped", "failure", "error")
        ):
            raise ValueError(
                f"Required GPU regression missing, skipped, or failed: {classname}.{name}"
            )
    if not cases or result["failures"]:
        raise ValueError("Empty or failed Python test inventory")
    return result


def validate(bundle, output, python=sys.executable):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(bundle):
        raise ValueError("Use a new results directory outside the immutable bundle")
    if (
        platform.system() != "Darwin"
        or platform.machine() != "arm64"
        or sys.version_info[:2] != (3, 12)
    ):
        raise ValueError("Replay requires native macOS arm64 CPython 3.12")
    expected = json.loads((bundle / "MANIFEST.json").read_text())
    if expected.get("scope") != "mlx_official_three_mode_integration_v1":
        raise ValueError("Wrong integration contract")
    if expected.get("mode") not in ("cpu-only", "jit-on", "jit-off"):
        raise ValueError("Unknown matrix mode")
    device = "cpu" if expected["mode"] == "cpu-only" else "gpu"
    observed = inventory(bundle)
    observed.pop("MANIFEST.json")
    if observed != expected["files"]:
        raise ValueError("Bundle member inventory or bytes changed")
    for name, version in expected["runtime_dependencies"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(
                f"Replay dependency differs: {name}; "
                "use runtime-lock.txt in an isolated environment"
            )
    output.mkdir()
    env = dict(
        os.environ,
        PYTHONPATH=str(bundle / "runtime/python"),
        PYTHONDONTWRITEBYTECODE="1",
        VECLIB_MAXIMUM_THREADS="1",
        OMP_NUM_THREADS="1",
        METAL_DEBUG_ERROR_MODE="0",
        METAL_DEVICE_WRAPPER_TYPE="1",
        TF_CPP_MIN_LOG_LEVEL="2",
    )
    for name in (
        "MLX_METALLIB_PATH",
        "DYLD_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DEVICE",
    ):
        env.pop(name, None)
    result = {
        "scope": "mlx_official_three_mode_integration_v1",
        "mode": expected["mode"],
        "manifest_sha256": digest(bundle / "MANIFEST.json"),
        "formal_fits": 0,
        "admission_stream_opened": False,
        "required_default_device": device,
        "device_override": None,
        "cpp": {"status": "not_executed"},
        "python": {"status": "not_executed"},
        "qualified": False,
    }
    try:
        probe = output / f"{device}-probe.json"
        command = [
            python,
            str(bundle / "tools/runtime_probe.py"),
            "--bundle",
            str(bundle),
            "--device",
            device,
            "--output",
            str(probe),
        ]
        execution = run(
            command,
            cwd=bundle,
            output=output,
            name="natural-default-probe-command",
            env=env,
            timeout=300,
            check=False,
        )
        result["probe"] = (
            json.loads(probe.read_text()) if probe.is_file() else {"status": "missing"}
        )
        if execution["returncode"] != 0 or result["probe"].get("status") != "passed":
            return result
        cpp = run(
            [str(bundle / "runtime/bin/tests")],
            cwd=bundle,
            output=output,
            name="full-cpp",
            env=env,
            timeout=1800,
            check=False,
        )
        result["cpp"] = {
            **cpp,
            "status": "passed" if cpp["returncode"] == 0 and not cpp.get("timed_out") else "failed",
        }
        xml = output / "full-python.xml"
        tests = run(
            [
                python,
                "-m",
                "pytest",
                str(bundle / "source/python/tests"),
                "-q",
                "-ra",
                "--durations=20",
                f"--junitxml={xml}",
                "-o",
                f"cache_dir={output / 'pytest-cache'}",
            ],
            cwd=bundle / "source",
            output=output,
            name="full-python",
            env=env,
            timeout=3600,
            check=False,
        )
        result["python"] = {**tests, "status": "failed"}
        try:
            result["python"]["inventory"] = test_inventory(
                xml, required_gpu_tests(bundle / "source") if device == "gpu" else ()
            )
            if tests["returncode"] == 0 and not tests.get("timed_out"):
                result["python"]["status"] = "passed"
        except (ValueError, OSError, ET.ParseError) as error:
            result["python"]["error"] = str(error)
        result["qualified"] = result["cpp"]["status"] == result["python"]["status"] == "passed"
        return result
    finally:
        after = inventory(bundle)
        after.pop("MANIFEST.json")
        result["bundle_unchanged"] = after == expected["files"]
        result["qualified"] = result["qualified"] and result["bundle_unchanged"]
        write(output / "summary.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return 0 if validate(args.bundle, args.output)["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
