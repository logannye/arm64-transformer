"""Derive a test-only MLX capsule while retaining its authenticated native build.

Only the prerelease-version parser in one Python test module changes. No native
source, runtime, test method, assertion, tolerance or eligibility rule changes.
"""

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("mlx_compat_authority", HERE / "owned_mlx_suites.py")
SUITE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUITE)
require = SUITE.require
CONTRACT_PATH = HERE / "mlx-test-compat-v1/contract.json"
CONTRACT_SHA = "9c9df29cf7fc02b76330966a06a37681376b1c68bc47b7e4ac95b78a146a46da"
PATCH_PATH = HERE / "mlx-test-compat-v1/changes.patch"
PATCH_MEMBER = "inputs/torch-version-compat.patch"
RECEIPT = "TEST-COMPATIBILITY.json"
OLD_LINE = '    torch_version = [int(v) for v in torch.__version__.split("+")[0].split(".")]'
NEW_LINE = (
    "    from packaging.version import Version\n\n"
    "    torch_version = list(Version(torch.__version__).release)"
)


def contract():
    SUITE.pinned(CONTRACT_PATH, CONTRACT_SHA)
    value = SUITE.read_json(CONTRACT_PATH)
    SUITE.pinned(PATCH_PATH, value["patch_sha256"])
    return value


def row(path):
    value = SUITE.binding(path)
    return {
        "sha256": value["sha256"],
        "bytes": value["bytes"],
        "executable": bool(Path(path).stat().st_mode & 0o111),
    }


def files(root):
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Symlink in compatibility capsule")
        if path.is_dir():
            continue
        require(path.is_file(), "Special file in compatibility capsule")
        name = path.relative_to(root).as_posix()
        if name != "MANIFEST.json":
            result[name] = row(path)
    return result


def original(bundle, selected):
    require(bundle.resolve(strict=True) == bundle, "Original capsule is not canonical")
    SUITE.pinned(bundle / "MANIFEST.json", selected["original_manifest_sha256"])
    manifest = SUITE.read_json(bundle / "MANIFEST.json")
    require(files(bundle) == manifest["files"], "Original capsule payload changed")
    require(
        manifest["files"][selected["target"]]["sha256"] == selected["original_sha256"],
        "Original test parser differs",
    )
    return manifest


def rewritten(data, selected):
    require(hashlib.sha256(data).hexdigest() == selected["original_sha256"], "Test source changed")
    text = data.decode()
    require(text.count(OLD_LINE) == 1, "Expected one original version parser")
    changed = text.replace(OLD_LINE, NEW_LINE)
    require(
        hashlib.sha256(changed.encode()).hexdigest() == selected["patched_sha256"],
        "Unexpected parser patch result",
    )

    # Independently preserve the complete function/class ASTs, including every
    # test body, decorator and skip condition. Only module-level parsing changes.
    def methods(source):
        return [
            ast.dump(node, include_attributes=False)
            for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]

    require(methods(text) == methods(changed), "Test methods or eligibility changed")
    return changed.encode()


def authenticate(bundle, receipt_sha):
    selected = contract()
    SUITE.pinned(bundle / RECEIPT, receipt_sha)
    proof = SUITE.read_json(bundle / RECEIPT)
    require(
        proof["schema"] == selected["schema"]
        and proof["native_compilation"] is False
        and proof["native_source_changes"] is False
        and proof["test_assertion_changes"] is False
        and proof["test_eligibility_changes"] is False
        and proof["formal_eligible"] is False
        and proof["native_runtime_bytes_reused"] is True
        and proof["source_methods_unchanged"] is True
        and proof["native_build_run"] == selected["native_build_run"],
        "Not a test-only compatibility derivation",
    )
    for key, expected in (("producer", Path(__file__).resolve()), ("contract", CONTRACT_PATH)):
        SUITE.binding(proof[key]["path"], proof[key])
        require(proof[key]["sha256"] == SUITE.sha256(expected), "Compatibility producer differs")
    old = Path(proof["original_bundle"])
    manifest = original(old, selected)
    require(
        proof["original_manifest"] == SUITE.binding(old / "MANIFEST.json"),
        "Original authority differs",
    )
    new = SUITE.read_json(bundle / "MANIFEST.json")
    expected = copy.deepcopy(manifest)
    expected["files"][selected["target"]] = row(bundle / selected["target"])
    expected["files"][PATCH_MEMBER] = row(bundle / PATCH_MEMBER)
    expected["files"][RECEIPT] = row(bundle / RECEIPT)
    expected["patches"]["torch-version-compat.patch"] = selected["patch_sha256"]
    expected["test_source_compatibility"] = SUITE.binding(bundle / RECEIPT)
    require(new == expected, "Derived manifest has undeclared changes")
    require(files(bundle) == new["files"], "Derived capsule payload changed")
    require(
        (bundle / selected["target"]).read_bytes()
        == rewritten((old / selected["target"]).read_bytes(), selected),
        "Derived test source differs",
    )
    SUITE.pinned(bundle / PATCH_MEMBER, selected["patch_sha256"])
    require(
        set(new["files"]) == set(manifest["files"]) | {PATCH_MEMBER, RECEIPT}
        and all(
            new["files"][name] == value
            for name, value in manifest["files"].items()
            if name != selected["target"]
        ),
        "Native or unrelated source payload changed",
    )
    return new, selected


def derive(bundle, output):
    selected = contract()
    manifest = original(bundle, selected)
    require(not output.exists() and output.resolve() == output, "Fresh canonical output required")
    require(
        not output.is_relative_to(bundle) and not bundle.is_relative_to(output),
        "Compatibility output overlaps original capsule",
    )
    logical = sum(value["bytes"] for value in manifest["files"].values())
    require(shutil.disk_usage(output.parent).free > 6 * 1024**3 + logical, "Disk reserve too small")
    # APFS clone preserves old bytes without creating another native build tree.
    subprocess.run(["/bin/cp", "-cR", str(bundle), str(output)], check=True)
    require(files(output) == manifest["files"], "Cloned payload differs")
    (output / selected["target"]).write_bytes(
        rewritten((bundle / selected["target"]).read_bytes(), selected)
    )
    (output / PATCH_MEMBER).write_bytes(PATCH_PATH.read_bytes())
    proof = {
        key: selected[key]
        for key in (
            "schema",
            "native_compilation",
            "native_source_changes",
            "test_assertion_changes",
            "test_eligibility_changes",
            "formal_eligible",
        )
    }
    proof.update(
        original_bundle=str(bundle),
        original_manifest=SUITE.binding(bundle / "MANIFEST.json"),
        producer=SUITE.binding(Path(__file__).resolve()),
        contract=SUITE.binding(CONTRACT_PATH),
        native_build_run=selected["native_build_run"],
        cloned_logical_bytes=logical,
        source_methods_unchanged=True,
        native_runtime_bytes_reused=True,
    )
    SUITE.write_json(output / RECEIPT, proof)
    changed = copy.deepcopy(manifest)
    changed["patches"]["torch-version-compat.patch"] = selected["patch_sha256"]
    changed["test_source_compatibility"] = SUITE.binding(output / RECEIPT)
    changed["files"] = files(output)
    # This is a new task-owned copy; the original capsule is never modified.
    (output / "MANIFEST.json").write_text(json.dumps(changed, sort_keys=True, indent=2) + "\n")
    receipt = SUITE.binding(output / RECEIPT)
    authenticate(output, receipt["sha256"])
    return {
        "manifest": SUITE.binding(output / "MANIFEST.json"),
        "receipt": receipt,
        "native_compilation": False,
        "formal_eligible": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.original.resolve(strict=True), args.output.absolute()), indent=2))
