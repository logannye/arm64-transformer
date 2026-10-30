"""Authenticate and replay a complete owned PyTorch candidate, without substitutions.

``prepare`` is stdlib-only: it verifies externally pinned hosted evidence, expands
the authenticated source/runtime into one new directory, verifies the existing
45-wheel dependency installation, and freezes the retrospective 27-method plan.
``run`` executes that exact plan in fresh serial processes. Neither operation is
admission or a formal experiment. Complete test_mps coverage remains required;
its stock conformance extension must be prepared for the candidate separately.
No command in this module builds, installs, downloads, or changes a framework.
"""

import argparse
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import traceback
import unittest
import zipfile
from pathlib import Path, PurePosixPath

SCHEMA = "gpu-owned-pytorch-candidate-validation-v1"
PROTOCOL = "gpu-v4-owned-development-v1"
HOSTED_RUN = "41092391888"
BASE = "71d2a08507ab47dd6b2fa9b66319d17ce2377de4"
CONTRACT_SHA = "68822f52a0c5932141cf60a0ac8ab858ce90dd5dbdc3052f364234303c6964ab"
BUILD_SHA = "b48ca14933ac32f204f931b96e86b52c6379be3337154c00b417b2ec46ee5a87"
VERIFIER_SHA = "b30e26ba6436832862ce1e8d7b9ca1a306c281395917395a57ec6762a38055f4"
TEST_SOURCE_SHA = "7345b6e229aed240921460a3b598ffd1a47148ab4816aead0d405539a154a396"
ORIGINAL_SHA = "eab61f146499b51c30e9cf4ec4b422914f2c89accebd3ef8fbde4dedd68919e2"
OLD = Path("work/upstream-integration-v2/pytorch")
ORIGINAL = OLD / "device-format-v2-v5/full-suite-v1/full-1/result.json"
DEPENDENCIES = OLD / "runtime-env/lib/python3.12/site-packages"
PYTHON = OLD / "runtime-env/bin/python"
HELPERS = {
    "reference": (
        OLD / "failure-replay-v2/replay.py",
        "6b0a5253526d9ac7123247b4695932c394eb8e95a1229e2657e42d5c37880e56",
    ),
    "driver": (
        OLD / "formatted-v2-runtime-v5/run_integrated_v2.py",
        "440cbf36b9bbcf818555c4be488e62207012aab22702419bcd3c1f93708ca982",
    ),
    "extension_policy": (
        OLD / "formatted-v2-runtime-v5/test_extension_policy_v2.py",
        "4582d3a5299ed8f9752297fa52ed47a7d3e5fb030a21c39f4487b86d5fffb583",
    ),
}
PINS = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    "PYTORCH_MPS_FAST_MATH": "0",
    "PYTORCH_TEST_WITH_SLOW": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "SKBUILD_EDITABLE_REBUILD": "false",
}
ABSENT = (
    "PYTHONPATH",
    "PYTHONHOME",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH",
    "PYTORCH_MPS_PREFER_METAL",
    "MLX_DISABLE_COMPILE",
    "PYTORCH_OPINFO_SAMPLE_INPUT_INDEX",
    "PYTHONPYCACHEPREFIX",
)
GIB = 1024**3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=unique,
        parse_constant=lambda _: require(False, "Nonfinite JSON value"),
    )


def stable_stat(info):
    return tuple(
        getattr(info, key)
        for key in (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_mode",
            "st_nlink",
        )
    )


def no_symlink(path):
    path = Path(path)
    require(not any(p.is_symlink() for p in [path, *path.parents]), "Symlink input path")
    return path


def binding(path):
    path = no_symlink(path)
    require(path.is_absolute() and path.is_file(), "Require an absolute regular input")
    require(path.resolve(strict=True) == path, "Noncanonical input path")
    before = path.stat()
    result = {"path": str(path), "sha256": digest(path), "bytes": before.st_size}
    require(stable_stat(before) == stable_stat(path.stat()), "Input changed while hashing")
    return result


def check_binding(expected):
    require(set(expected) == {"path", "sha256", "bytes"}, "Malformed file binding")
    require(binding(expected["path"]) == expected, "Bound file changed: " + expected["path"])


def pinned(path, expected):
    result = binding(path)
    require(result["sha256"] == expected, "Pinned source/evidence changed: " + str(path))
    return result


def write(path, value, *, new=False):
    path = Path(path)
    require(not path.is_symlink(), "Symlink output")
    if new:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        temporary = path.with_suffix(path.suffix + ".partial")
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safe_relative(name):
    require(isinstance(name, str) and name and "\\" not in name, "Invalid member path")
    path = PurePosixPath(name)
    require(
        not path.is_absolute()
        and ".." not in path.parts
        and "." not in path.parts
        and str(path) == name,
        "Unsafe or noncanonical member path",
    )
    return path


def manifest_files(manifest, *, source):
    files = manifest["files"]
    require(isinstance(files, dict) and files, "Empty manifest inventory")
    total = 0
    for name, row in files.items():
        path = safe_relative(name)
        require(
            all(str(parent) not in files for parent in path.parents if str(parent) != "."),
            "Manifest file/directory collision",
        )
        kind = row.get("kind", "file")
        require(kind == "file" or (source and kind == "symlink"), "Unsupported member kind")
        if kind == "file":
            require(
                type(row["bytes"]) is int
                and row["bytes"] >= 0
                and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]),
                "Invalid member binding",
            )
            total += row["bytes"]
        else:
            target = row["target"]
            require(
                isinstance(target, str) and not PurePosixPath(target).is_absolute(),
                "Absolute source symlink",
            )
            # Resolve lexically before any symlink exists on disk.
            parts = list(path.parent.parts)
            for part in PurePosixPath(target).parts:
                if part == "..":
                    require(parts, "Source symlink escapes archive")
                    parts.pop()
                elif part != ".":
                    parts.append(part)
    require(total <= 8 * GIB, "Payload exceeds declared 8 GiB bound")
    return files, total


def archive_inventory(archive_path, manifest, *, source):
    """Check all tar members, including content; no extraction or imports."""
    files, total = manifest_files(manifest, source=source)
    prefix = "source/" if source else "runtime/"
    expected = {prefix + name for name in files}
    expected |= (
        {"source-manifest.json", "changes.patch"} if source else {"runtime/runtime-manifest.json"}
    )
    seen = set()
    metadata = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            name = member.name
            safe_relative(name)
            require(name not in seen, "Duplicate archive member")
            seen.add(name)
            if member.isdir():
                require(
                    not source
                    and name == "runtime"
                    or (not source and any(item.startswith(name + "/") for item in expected)),
                    "Unexpected archive directory",
                )
                continue
            require(
                name in expected and not member.islnk(), "Unexpected/hard-linked archive member"
            )
            relative = name.removeprefix(prefix)
            if name.startswith(prefix) and relative in files:
                row = files[relative]
                if row.get("kind", "file") == "symlink":
                    require(
                        member.issym()
                        and member.linkname == row["target"]
                        and member.mode == row["mode"],
                        "Source symlink differs",
                    )
                    continue
                require(member.isfile() and member.size == row["bytes"], "Archive member type/size")
                if source:
                    require(member.mode == row["mode"], "Source mode differs")
                stream = archive.extractfile(member)
                require(
                    stream is not None
                    and hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"],
                    "Archive member bytes differ",
                )
            else:
                require(member.isfile() and member.size <= 64 * 1024**2, "Invalid archive metadata")
                metadata[name] = archive.extractfile(member).read()
    require({name for name in seen if name in expected} == expected, "Missing archive members")
    return metadata, total


def expand_verified(archive_path, destination, *, source):
    """Called only after full archive verification; keep partial evidence on failure."""
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if member.isdir() or member.issym():
                continue
            path = destination / member.name
            no_symlink(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output, archive.extractfile(member) as stream:
                shutil.copyfileobj(stream, output)
            if source and member.name.startswith("source/"):
                path.chmod(member.mode)
        if source:
            for member in archive.getmembers():
                if member.issym():
                    path = destination / member.name
                    no_symlink(path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.symlink_to(member.linkname)


def verify_tree(root, manifest, *, source):
    files, _ = manifest_files(manifest, source=source)
    no_symlink(root)
    actual = {
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()
    }
    if not source:
        actual.discard("runtime-manifest.json")
    require(actual == set(files), "Expanded inventory differs")
    for name, row in files.items():
        path = root / name
        if row.get("kind", "file") == "symlink":
            require(
                path.is_symlink()
                and os.readlink(path) == row["target"]
                and path.resolve().is_relative_to(root),
                "Expanded source symlink differs",
            )
        else:
            value = binding(path)
            require(
                value["sha256"] == row["sha256"] and value["bytes"] == row["bytes"],
                "Expanded bytes differ: " + name,
            )
            if source:
                require(
                    stat.S_IMODE(path.stat().st_mode) == row["mode"], "Expanded source mode differs"
                )
    return {
        "files": len(files),
        "manifest_json_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def allowed_dependency_extra(relative):
    parts = PurePosixPath(relative).parts
    return (
        relative in {"_virtualenv.py", "_virtualenv.pth"}
        or ("__pycache__" in parts and parts[-1].endswith(".pyc"))
        or (
            len(parts) == 2
            and parts[0].endswith(".dist-info")
            and parts[1] in {"RECORD", "INSTALLER", "REQUESTED", "uv_cache.json"}
        )
    )


def verify_dependencies(dependencies, wheels, contract):
    """Verify the retained wheel payloads and exact installed distributions."""
    no_symlink(dependencies)
    expected_versions = {}
    checked = 0
    wheel_paths = set()
    for name, identity in contract["wheel_inventory"].items():
        safe_relative(name)
        require("/" not in name, "Nested wheel name")
        path = wheels / name
        require(binding(path) == {"path": str(path), **identity}, "Wheel transport differs")
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            require(
                len({row.filename for row in members}) == len(members), "Duplicate wheel member"
            )
            meta = [
                row
                for row in members
                if row.filename.endswith(".dist-info/METADATA")
                and len(PurePosixPath(row.filename).parts) == 2
            ]
            require(len(meta) == 1, "Wheel metadata is ambiguous")
            from email.parser import BytesParser

            metadata = BytesParser().parsebytes(archive.read(meta[0]))
            package = normalize(metadata["Name"])
            require(package not in expected_versions, "Duplicate wheel distribution")
            expected_versions[package] = metadata["Version"]
            for row in members:
                if row.is_dir() or (
                    row.filename.endswith(".dist-info/RECORD")
                    and len(PurePosixPath(row.filename).parts) == 2
                ):
                    continue
                relative = safe_relative(row.filename)
                if any(part.endswith(".data") for part in relative.parts):
                    if len(relative.parts) < 3 or relative.parts[1] not in {"purelib", "platlib"}:
                        continue  # Installer scripts are not imported payloads.
                    relative = PurePosixPath(*relative.parts[2:])
                wheel_paths.add(str(relative))
                installed = dependencies / str(relative)
                require(
                    binding(installed)["sha256"] == hashlib.sha256(archive.read(row)).hexdigest(),
                    "Installed dependency payload changed: " + str(relative),
                )
                checked += 1
    distributions = list(importlib.metadata.distributions(path=[str(dependencies)]))
    actual = {normalize(d.metadata["Name"]): d.version for d in distributions}
    require(
        len(distributions) == len(expected_versions) == 45 and actual == expected_versions,
        "Exact 45-distribution dependency inventory differs",
    )
    actual_files = {
        path.relative_to(dependencies).as_posix()
        for path in dependencies.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    extras = actual_files - wheel_paths
    require(
        all(allowed_dependency_extra(name) for name in extras),
        "Unexpected non-wheel dependency code or payload: "
        + repr(sorted(name for name in extras if not allowed_dependency_extra(name))[:8]),
    )
    extra_bindings = [binding(dependencies / name) for name in sorted(extras)]
    return {
        "versions": expected_versions,
        "payload_files": checked,
        "wheel_count": 45,
        "environment_modified": False,
        "non_wheel_files": extra_bindings,
        "bytecode_policy": "Fresh unused pycache prefix and -B; installed caches never executed",
        "site_hooks_executed": False,
    }


def helpers(workspace):
    rows = {name: pinned(workspace / path, sha) for name, (path, sha) in HELPERS.items()}
    rows["original"] = pinned(workspace / ORIGINAL, ORIGINAL_SHA)
    return rows


def original_inventory(workspace):
    rows = helpers(workspace)
    reference = load(rows["reference"]["path"], "owned_candidate_retained_replay")
    original = json_read(rows["original"]["path"])
    inventory = reference.failure_inventory(original)
    return rows, original, inventory


def validate_hosted(receipt, contract, head):
    require(
        re.fullmatch(r"[0-9a-f]{40}", head), "An externally verified hosted source SHA is required"
    )
    require(
        receipt["schema"] == "pytorch-owned-layout-hosted-v1"
        and receipt["contract_sha256"] == CONTRACT_SHA
        and receipt["source_revision"] == BASE
        and receipt["patch_sha256"] == contract["combined_patch_sha256"]
        and receipt["component_patches"] == contract["component_patches"],
        "Wrong candidate source contract",
    )
    require(
        receipt["github"]["GITHUB_REPOSITORY"] == "logannye/arm64-transformer"
        and receipt["github"]["GITHUB_RUN_ID"] == HOSTED_RUN
        and receipt["github"]["GITHUB_SHA"] == head,
        "Hosted origin does not match external selection",
    )
    for label in (
        "full-framework-build",
        "normal-editable-import",
        "transported-runtime-import",
        "archive-roundtrip-import",
    ):
        rows = [row for row in receipt["commands"] if row["label"] == label]
        require(
            len(rows) == 1 and rows[0]["status"] == "passed" and rows[0]["exit_code"] == 0,
            "Required hosted build/import did not pass: " + label,
        )
    for name in ("Embedding.mm", "Matmul.mm", "Convolution.mm"):
        rows = receipt["changed_translation_units"][name]
        expected = contract["source_sha256"]["aten/src/ATen/native/mps/operations/" + name]
        require(
            rows
            and all(
                row["source_sha256"] == expected
                and re.fullmatch(r"[0-9a-f]{64}", row["object_sha256"])
                for row in rows
            ),
            "Candidate compilation provenance differs",
        )
    require(
        receipt["embedding_metal_object"]["source_sha256"]
        == contract["source_sha256"]["aten/src/ATen/native/mps/kernels/Embedding.metal"],
        "Embedding Metal source differs",
    )
    require(
        receipt["runtime_archive"]["qualification"]
        == "import-and-portability-validated; device-numerics-unvalidated",
        "Archive import proof missing",
    )


def artifact_inputs(artifact, receipt_sha, head):
    receipt_row = pinned(artifact / "receipt.json", receipt_sha)
    contract_row = pinned(artifact / "contract.json", CONTRACT_SHA)
    pinned(artifact / "hosted_build.py", BUILD_SHA)
    pinned(artifact / "verify_runtime.py", VERIFIER_SHA)
    receipt, contract = json_read(receipt_row["path"]), json_read(contract_row["path"])
    validate_hosted(receipt, contract, head)
    rows = [
        receipt_row,
        contract_row,
        binding(artifact / "hosted_build.py"),
        binding(artifact / "verify_runtime.py"),
    ]
    rows.append(pinned(artifact / "changes.patch", contract["combined_patch_sha256"]))
    for command in receipt["commands"]:
        safe_relative(command["log"])
        require("/" not in command["log"], "Nested command log")
        if "log_sha256" in command:
            rows.append(pinned(artifact / command["log"], command["log_sha256"]))
    for kind in ("source_archive", "runtime_archive"):
        record = receipt[kind]
        safe_relative(record["file"])
        require("/" not in record["file"], "Nested archive name")
        row = binding(artifact / record["file"])
        require(
            row["sha256"] == record["sha256"] and row["bytes"] == record["bytes"],
            "Archive transport differs",
        )
        rows.append(row)
    for name in ("runtime-import.json", "archive-roundtrip-import.json"):
        value = json_read(artifact / name)
        require(
            value["status"] == "passed"
            and value["source_revision"] == BASE
            and value["runtime_manifest_sha256"] == receipt["runtime_archive"]["manifest_sha256"]
            and value["loaded_operator_libraries"] == []
            and value["mps_built"] is True,
            "Hosted import receipt differs",
        )
        rows.append(binding(artifact / name))
    return receipt, contract, rows


def output_path(workspace, output):
    workspace = workspace.resolve(strict=True)
    no_symlink(output)
    require(
        output.is_absolute()
        and output.resolve(strict=False) == output
        and output.is_relative_to(workspace / "work/gpu-v4-owned")
        and output != workspace / "work/gpu-v4-owned"
        and not output.exists(),
        "Require a fresh owned-development output path",
    )
    return output


def prepare(workspace, artifact, receipt_sha, head, output):
    output_path(workspace, output)
    require(re.fullmatch(r"[0-9a-f]{64}", receipt_sha), "Require external hosted receipt SHA256")
    receipt, contract, inputs = artifact_inputs(artifact, receipt_sha, head)
    bound_helpers, original, failed = original_inventory(workspace)
    source_manifest = json_read(artifact / "source-manifest.json")
    inputs.append(
        pinned(artifact / "source-manifest.json", receipt["source_archive"]["manifest_sha256"])
    )
    require(
        source_manifest["source_revision"] == BASE
        and source_manifest["patch_sha256"] == contract["combined_patch_sha256"],
        "Source manifest identity",
    )
    source_meta, source_bytes = archive_inventory(
        artifact / receipt["source_archive"]["file"], source_manifest, source=True
    )
    require(
        hashlib.sha256(source_meta["source-manifest.json"]).hexdigest()
        == receipt["source_archive"]["manifest_sha256"]
        and hashlib.sha256(source_meta["changes.patch"]).hexdigest()
        == contract["combined_patch_sha256"],
        "Source archive metadata differs",
    )
    runtime_archive = artifact / receipt["runtime_archive"]["file"]
    with tarfile.open(runtime_archive, "r:gz") as archive:
        member = archive.getmember("runtime/runtime-manifest.json")
        require(member.isfile() and member.size <= 64 * 1024**2, "Invalid runtime manifest")
        raw = archive.extractfile(member).read()
    require(
        hashlib.sha256(raw).hexdigest() == receipt["runtime_archive"]["manifest_sha256"],
        "Runtime manifest binding differs",
    )
    runtime_manifest = json.loads(raw)
    require(
        runtime_manifest["source_revision"] == BASE
        and runtime_manifest["patch_sha256"] == contract["combined_patch_sha256"],
        "Runtime manifest identity",
    )
    archive_inventory(runtime_archive, runtime_manifest, source=False)
    _, runtime_bytes = manifest_files(runtime_manifest, source=False)
    for relative, sha in contract["source_sha256"].items():
        require(source_manifest["files"][relative]["sha256"] == sha, "Required source differs")
    require(
        shutil.disk_usage(workspace).free >= source_bytes + runtime_bytes + 16 * GIB,
        "Insufficient disk reserve for authenticated expansion",
    )
    dependencies = workspace / DEPENDENCIES
    dependency_proof = verify_dependencies(dependencies, artifact / "wheels", contract)
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "status": "preparing",
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
    }
    write(output / "preparation.json", record, new=True)
    try:
        expand_verified(artifact / receipt["source_archive"]["file"], output, source=True)
        expand_verified(runtime_archive, output, source=False)
        verify_tree(output / "source", source_manifest, source=True)
        verify_tree(output / "runtime", runtime_manifest, source=False)
        plan = {
            **record,
            "status": "frozen_development_only",
            "workspace": str(workspace),
            "artifact": str(artifact),
            "hosted_receipt_sha256": receipt_sha,
            "hosted_head": head,
            "source": str(output / "source"),
            "runtime": str(output / "runtime"),
            "dependencies": str(dependencies),
            "dependency_proof": dependency_proof,
            "python": binding((workspace / PYTHON).resolve(strict=True)),
            "inputs": inputs,
            "helpers": bound_helpers,
            "driver": binding(Path(__file__).resolve()),
            "source_manifest": binding(output / "source-manifest.json"),
            "runtime_manifest": binding(output / "runtime/runtime-manifest.json"),
            "environment": dict(PINS),
            "absent_environment": list(ABSENT),
            "focused_inventory": failed,
            "workers": [
                {"index": i, "id": row["id"], "status": "not_started"}
                for i, row in enumerate(failed)
            ],
            "full_suite": {
                "required": True,
                "planned_ids": original["planned_tests"],
                "retained_skip_ids": [row["id"] for row in original["tests"]["skipped"]],
                "retained_expected_failure_ids": [
                    row["id"] for row in original["tests"]["expected_failures"]
                ],
                "status": "pending_candidate_stock_conformance_fixture",
                "local_fixture_compilation_enabled": False,
                "fixture_policy": (
                    "Retain existing bounded upstream fixture policy; explicitly prepare a new "
                    "runtime-specific cache before full-suite execution."
                ),
                "scope": "Every test in unchanged test_mps.py; not broader multi-file --mps suite",
            },
            "limitations": [
                "Focused replay never clears the failed full-suite gate or grants admission.",
                "Python executable bytes are bound; standard library and OS libraries are not.",
                "Receipt SHA anchors transport; it is not an independent build attestation.",
                "Retrospective failing tests are development data, not numerical admission.",
                "No model, embedding-matrix, MLX, CPU-anchor, or formal qualification is inferred.",
            ],
        }
        write(output / "plan.json", plan, new=True)
        record.update(status="prepared", plan=binding(output / "plan.json"))
    except BaseException:
        record.update(status="failed", exception=traceback.format_exc())
        raise
    finally:
        write(output / "preparation.json", record)
    return record


def validate_inventory(plan, original, failed):
    require(
        plan["focused_inventory"] == failed
        and plan["workers"]
        == [{"index": i, "id": row["id"], "status": "not_started"} for i, row in enumerate(failed)],
        "Missing, duplicate, reordered, or altered focused tests",
    )
    full = plan["full_suite"]
    require(
        full["required"] is True
        and full["planned_ids"] == original["planned_tests"]
        and full["retained_skip_ids"] == [row["id"] for row in original["tests"]["skipped"]]
        and full["retained_expected_failure_ids"]
        == [row["id"] for row in original["tests"]["expected_failures"]]
        and full["status"] == "pending_candidate_stock_conformance_fixture"
        and full["local_fixture_compilation_enabled"] is False,
        "Full-suite requirement or retained eligibility changed",
    )


def authenticate_plan(path, external_sha):
    pinned(path, external_sha)
    plan = json_read(path)
    require(
        plan["schema"] == SCHEMA
        and plan["protocol"] == PROTOCOL
        and plan["status"] == "frozen_development_only"
        and plan["formal_eligible"] is False
        and plan["formal_fits"] == 0
        and plan["admission_stream_opened"] is False,
        "Not a frozen development plan",
    )
    require(
        plan["environment"] == PINS and plan["absent_environment"] == list(ABSENT),
        "Execution controls changed",
    )
    check_binding(plan["driver"])
    require(plan["driver"]["path"] == str(Path(__file__).resolve()), "Wrong validation driver")
    check_binding(plan["python"])
    for row in [
        *plan["inputs"],
        *plan["helpers"].values(),
        plan["source_manifest"],
        plan["runtime_manifest"],
    ]:
        check_binding(row)
    workspace = Path(plan["workspace"])
    require(plan["helpers"] == helpers(workspace), "Pinned helper identity changed")
    reference = load(plan["helpers"]["reference"]["path"], "owned_candidate_retained_replay")
    original = json_read(plan["helpers"]["original"]["path"])
    failed = reference.failure_inventory(original)
    validate_inventory(plan, original, failed)
    receipt, contract, inputs = artifact_inputs(
        Path(plan["artifact"]), plan["hosted_receipt_sha256"], plan["hosted_head"]
    )
    require(all(row in plan["inputs"] for row in inputs), "Artifact evidence absent from plan")
    proof = verify_dependencies(
        Path(plan["dependencies"]), Path(plan["artifact"]) / "wheels", contract
    )
    require(
        proof == plan["dependency_proof"]
        and Path(plan["dependencies"]) == workspace / DEPENDENCIES,
        "Dependency installation changed",
    )
    require(
        plan["source_manifest"]["sha256"] == receipt["source_archive"]["manifest_sha256"]
        and plan["runtime_manifest"]["sha256"] == receipt["runtime_archive"]["manifest_sha256"],
        "Expanded manifest binding differs",
    )
    for kind in ("source", "runtime"):
        verify_tree(
            Path(plan[kind]), json_read(plan[kind + "_manifest"]["path"]), source=kind == "source"
        )
    return plan, reference


def controlled_environment(environment):
    require(
        "PYTORCH_OPINFO_SAMPLE_INPUT_INDEX" not in environment,
        "Partial OpInfo sample selection forbidden",
    )
    result = dict(environment)
    for key in ABSENT:
        result.pop(key, None)
    # Unknown framework controls could alter kernels without touching a source file.
    require(
        not [
            key
            for key in result
            if key.startswith(("PYTORCH_", "MLX_", "DYLD_")) and key not in PINS
        ],
        "Unrecorded framework environment control",
    )
    result.update(PINS)
    return result


def native_origins(runtime):
    """Check already loaded framework dylibs, without loading any candidate code."""
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.argtypes = []
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    origins = []
    for index in range(dyld._dyld_image_count()):
        raw = dyld._dyld_get_image_name(index)
        if not raw:
            continue
        path = Path(os.fsdecode(raw))
        if path.name.startswith(("libtorch", "libc10", "libomp", "libshm")):
            path = path.resolve(strict=True)
            require(
                path.is_relative_to(runtime), "Native framework library loaded outside candidate"
            )
            origins.append(binding(path))
    require(
        any(Path(row["path"]).name == "libtorch_cpu.dylib" for row in origins),
        "Candidate libtorch_cpu is not loaded",
    )
    return sorted(origins, key=lambda row: row["path"])


def validate_result(record, row, pid, exit_code, plan_sha):
    require(
        record.get("schema") == SCHEMA
        and record.get("protocol") == PROTOCOL
        and record.get("plan_sha256") == plan_sha
        and record.get("pid") == pid
        and record.get("index") == row["index"]
        and record.get("id") == row["id"]
        and record.get("complete") is True
        and record.get("formal_eligible") is False
        and record.get("integrity_after") is True,
        "Incomplete, stale, or mismatched worker",
    )
    tests = record["tests"]
    require(tests["run"] == 1 and tests["executed_ids"] == [row["id"]], "Wrong test execution")
    require(
        not any(tests[key] for key in ("skipped", "expected_failures", "unexpected_successes")),
        "Focused eligibility changed",
    )
    for key in ("failures", "errors"):
        require(
            all(item["id"] == row["id"] and item["traceback"] for item in tests[key]),
            "Failure record missing or substituted",
        )
    failed = bool(tests["failures"] or tests["errors"])
    require(
        tests["successful"] is (not failed)
        and type(exit_code) is int
        and exit_code == int(failed)
        and record["outcome"] == ("failed" if failed else "passed"),
        "Outcome/exit mismatch",
    )
    for key in ("state_after_bootstrap", "state_before_test", "state_after_test"):
        state = record[key]
        require(
            state["operator_libraries"] == []
            and state["host_threads"] == 1
            and state["interop_threads"] == 1,
            "Override library or host threading changed",
        )
    require(record["controlled_environment"] == PINS, "Worker environment differs")


def worker(plan_path, plan_sha, index, output):
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "plan_sha256": plan_sha,
        "index": index,
        "pid": os.getpid(),
        "complete": False,
        "outcome": "runtime_error",
        "formal_eligible": False,
    }
    torch = None
    plan = None
    try:
        require(
            sys.flags.isolated and sys.flags.no_site and not sys.flags.optimize,
            "Worker requires -I -S without -O",
        )
        require(
            sys.platform == "darwin"
            and platform.machine() == "arm64"
            and sys.version_info[:2] == (3, 12),
            "Candidate requires macOS ARM64 CPython 3.12",
        )
        require(
            all(os.environ.get(k) == v for k, v in PINS.items())
            and all(k not in os.environ for k in ABSENT),
            "Worker environment differs",
        )
        bytecode_prefix = output / "unused-bytecode-cache"
        require(
            sys.pycache_prefix == str(bytecode_prefix) and not bytecode_prefix.exists(),
            "Worker bytecode cache must use a fresh unused prefix",
        )
        plan, reference = authenticate_plan(plan_path, plan_sha)
        require(
            Path(sys.executable).resolve(strict=True) == Path(plan["python"]["path"]),
            "Worker uses another Python executable",
        )
        require(type(index) is int and 0 <= index < len(plan["workers"]), "Unknown worker")
        row = plan["workers"][index]
        record.update(
            id=row["id"],
            controlled_environment=dict(PINS),
            source_manifest_sha256=plan["source_manifest"]["sha256"],
            runtime_manifest_sha256=plan["runtime_manifest"]["sha256"],
            bytecode_prefix=str(bytecode_prefix),
        )
        os.environ["PYTHONPYCACHEPREFIX"] = str(bytecode_prefix)
        verifier_path = Path(plan["artifact"]) / "verify_runtime.py"
        verifier = load(verifier_path, "owned_candidate_runtime_verifier")
        sys.argv = [
            str(verifier_path),
            "--runtime",
            plan["runtime"],
            "--dependencies",
            plan["dependencies"],
            "--receipt",
            str(output / "runtime-import.json"),
            "--expected-base",
            BASE,
        ]
        verifier.main()
        import torch

        require(torch.backends.mps.is_available(), "Real MPS device required; skip is not a pass")
        require(not torch.ops.loaded_libraries, "Operator override library loaded")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        record["state_after_bootstrap"] = reference.state(torch)
        record["native_origins_before"] = native_origins(Path(plan["runtime"]))
        driver = load(plan["helpers"]["driver"]["path"], "owned_candidate_test_helpers")
        record["conditions_before"] = driver.conditions()
        # Only bootstrap/source/test-selection helpers are reused. Old admission,
        # source provenance, model diagnostics, and parent entry points are never called.
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
        pinned(test_source, TEST_SOURCE_SHA)
        sys.argv = [str(test_source)]
        module = driver.load_module(test_source, "integrated_upstream_test_mps")
        record["state_before_test"] = reference.state(torch)
        test = reference.select_test(module, row["id"])
        write(output / "result.json", record)
        result = unittest.TextTestRunner(verbosity=2, resultclass=driver.RetainedResult).run(
            unittest.TestSuite([test])
        )
        record["tests"] = reference.suite_record(result)
        record["state_after_test"] = reference.state(torch)
        torch.mps.synchronize()
        record["conditions_after"] = driver.conditions()
        require(
            all(os.environ.get(k) == v for k, v in PINS.items())
            and all(
                k not in os.environ
                for k in ABSENT
                if k not in {"PYTHONPATH", "PYTHONPYCACHEPREFIX"}
            ),
            "Execution controls mutated",
        )
        require(
            sys.pycache_prefix == str(bytecode_prefix)
            and os.environ.get("PYTHONPYCACHEPREFIX") == str(bytecode_prefix)
            and not bytecode_prefix.exists()
            and "_virtualenv" not in sys.modules,
            "Unverified dependency bytecode or site hook became active",
        )
        manifest = json_read(plan["runtime_manifest"]["path"])
        origins, synthetic = verifier.verify_module_origins(
            Path(plan["runtime"]), manifest["files"], sys.modules, torch
        )
        record["postflight_module_origins"] = origins
        record["postflight_synthetic_origins"] = synthetic
        record["native_origins_after"] = native_origins(Path(plan["runtime"]))
        record["normal_import"] = binding(output / "runtime-import.json")
        record["outcome"] = "passed" if result.wasSuccessful() else "failed"
        record["complete"] = True
    except BaseException:
        record.update(complete=False, outcome="runtime_error", exception=traceback.format_exc())
    finally:
        if plan is not None:
            try:
                authenticate_plan(plan_path, plan_sha)
                record["integrity_after"] = True
            except BaseException:
                record.update(
                    complete=False,
                    outcome="runtime_error",
                    integrity_after=False,
                    integrity_error=traceback.format_exc(),
                )
        if record["complete"]:
            try:
                validate_result(
                    record,
                    plan["workers"][index],
                    os.getpid(),
                    int(record["outcome"] != "passed"),
                    plan_sha,
                )
            except BaseException:
                record.update(
                    complete=False, outcome="runtime_error", validation_error=traceback.format_exc()
                )
        write(output / "result.json", record)
    return 2 if not record["complete"] else int(record["outcome"] != "passed")


def audit_completed_workers(rows, plan, plan_sha):
    require(len(rows) == len(plan["workers"]), "Missing completed workers")
    for row, expected in zip(rows, plan["workers"]):
        require(
            row["index"] == expected["index"] and row["id"] == expected["id"],
            "Completed worker order changed",
        )
        check_binding(row["log"])
        check_binding(row["result"])
        result = json_read(row["result"]["path"])
        validate_result(result, expected, row["pid"], row["exit_code"], plan_sha)
        require(row["status"] == result["outcome"], "Retained worker outcome changed")
        imported = result["normal_import"]
        require(
            Path(imported["path"]) == Path(row["result"]["path"]).parent / "runtime-import.json",
            "Worker import receipt substituted",
        )
        check_binding(imported)
        proof = json_read(imported["path"])
        require(
            proof["schema"] == "pytorch-runtime-import-v1"
            and proof["status"] == "passed"
            and proof["runtime"] == plan["runtime"]
            and proof["source_revision"] == BASE
            and proof["runtime_manifest_sha256"] == plan["runtime_manifest"]["sha256"]
            and proof["loaded_operator_libraries"] == []
            and proof["mps_built"] is True
            and proof["mps_available"] is True,
            "Retained candidate import identity changed",
        )
        require(
            result["source_manifest_sha256"] == plan["source_manifest"]["sha256"]
            and result["runtime_manifest_sha256"] == plan["runtime_manifest"]["sha256"],
            "Retained source/runtime identity changed",
        )
        for key in ("native_origins_before", "native_origins_after"):
            require(result[key], "Missing native-library origin proof")
            for native in result[key]:
                require(
                    Path(native["path"]).is_relative_to(Path(plan["runtime"])),
                    "Native library originated outside candidate",
                )
                check_binding(native)


def run(plan_path, plan_sha, output, timeout):
    plan, _ = authenticate_plan(plan_path, plan_sha)
    output_path(Path(plan["workspace"]), output)
    require(1 <= timeout <= 7200, "Bounded worker timeout required")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "plan": binding(plan_path),
        "workers": [dict(row) for row in plan["workers"]],
        "complete": False,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "original_full_suite_failure_unchanged": True,
        "full_suite_required": True,
        "full_suite_executed": False,
    }
    write(output / "summary.json", record, new=True)
    try:
        env = controlled_environment(os.environ)
        for row in record["workers"]:
            authenticate_plan(plan_path, plan_sha)
            destination = output / f"case-{row['index']:02d}"
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
                str(plan_path),
                "--plan-sha256",
                plan_sha,
                "--index",
                str(row["index"]),
                "--output",
                str(destination),
            ]
            row.update(status="running", argv=argv)
            write(output / "summary.json", record)
            log = output / f"case-{row['index']:02d}.log"
            with log.open("x") as stream:
                child = subprocess.Popen(
                    argv,
                    cwd=output,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    code = child.wait(timeout=timeout)
                except BaseException:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    row.update(
                        status="interrupted_or_timeout", exit_code=child.returncode, pid=child.pid
                    )
                    raise
                finally:
                    row.update(exit_code=child.returncode, pid=child.pid)
            row["log"] = binding(log)
            result_path = destination / "result.json"
            row["result"] = binding(result_path)
            result = json_read(result_path)
            validate_result(result, row, child.pid, code, plan_sha)
            check_binding(result["normal_import"])
            row["status"] = result["outcome"]
            write(output / "summary.json", record)
        authenticate_plan(plan_path, plan_sha)
        audit_completed_workers(record["workers"], plan, plan_sha)
        record.update(
            complete=True,
            evidence_valid=True,
            passed=all(row["status"] == "passed" for row in record["workers"]),
        )
    except BaseException:
        record.update(evidence_valid=False, passed=False, exception=traceback.format_exc())
    finally:
        write(output / "summary.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    for name in ("workspace", "artifact", "output"):
        prepare_parser.add_argument("--" + name, type=Path, required=True)
    prepare_parser.add_argument("--receipt-sha256", required=True)
    prepare_parser.add_argument("--hosted-head", required=True)
    for name in ("run", "worker"):
        command = sub.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--plan-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "worker":
            command.add_argument("--index", type=int, required=True)
        else:
            command.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.workspace.resolve(strict=True),
            args.artifact.resolve(strict=True),
            args.receipt_sha256,
            args.hosted_head,
            args.output.absolute(),
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "worker":
        return worker(args.plan.absolute(), args.plan_sha256, args.index, args.output.absolute())
    result = run(
        args.plan.absolute(), args.plan_sha256, args.output.absolute(), args.timeout_seconds
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in ("complete", "passed", "evidence_valid", "formal_eligible")
            },
            indent=2,
        )
    )
    return int(not result.get("passed", False))


if __name__ == "__main__":
    raise SystemExit(main())
