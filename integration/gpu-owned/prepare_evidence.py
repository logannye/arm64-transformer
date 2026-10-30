"""Freeze, package and verify closed owned-study evidence; no uploads or framework imports."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
from pathlib import Path, PurePosixPath

PREFIX = "work/gpu-v4-owned/"
MAX_BYTES = (
    2_000_000_000  # Both uncompressed input and compressed archive must be strictly smaller.
)
CHUNK = 1024 * 1024
SUFFIXES = {".json", ".log", ".npz", ".npy", ".txt", ".py"}
NATIVE_MAGIC = {
    b"\x7fELF",
    b"\xcf\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xce",
    b"\xca\xfe\xba\xbe",
}


def group(name, roots, closures):
    return {
        "name": name,
        "roots": [PREFIX + value for value in roots],
        "closures": [{"path": PREFIX + path, "sha256": sha} for path, sha in closures],
    }


GROUPS = [
    group(
        "joint-runtime-v1-through-v4",
        [
            "joint-runtime-v1",
            "joint-runtime-v2",
            "joint-runtime-v3",
            "joint-runtime-v4",
            "failed-runtime-probe-v1",
            "failed-runtime-probe-v2",
            "joint-runtime-v3-retry.json",
        ],
        [
            (
                "joint-runtime-v1/summary.json",
                "a0df7a678c6749d7deb5d7e636c541a3537bdadd3812c5ba8b29567fde213dbb",
            ),
            (
                "joint-runtime-v2/summary.json",
                "28006a5e3b52b6aaee075f180956a83e74514f757e687dcae4792b3eb26537dd",
            ),
            (
                "joint-runtime-v3/summary.json",
                "3484b72f1951247c68d78b5ad167ea0d0d4355a902b20d6e60d81ad41a71e3b0",
            ),
            (
                "joint-runtime-v4/summary.json",
                "f747bad17d83bf0af85fa6a8bdf73643e9c0abb989e2ff56cb7cd4525bdf9f4b",
            ),
            (
                "failed-runtime-probe-v1/snapshot.json",
                "34fba1abd506535b035d6e1af624b8c4e149ba3173345aca87bf2c45693c48e8",
            ),
            (
                "failed-runtime-probe-v2/snapshot.json",
                "7ff2aa0704086e6dbe2f83e07795895783ab7478b7a6c23d4ed6305b746e6326",
            ),
        ],
    ),
    group(
        "layout-diagnostic-v1",
        ["layout-diagnostic-v1"],
        [
            (
                "layout-diagnostic-v1/summary.json",
                "08f98e4ea6adf7b1257d6dcf598205fd0414333d5f286c3177f64b6f81ad6f48",
            ),
        ],
    ),
    group(
        "half-index-v1",
        ["half-index-v1", "half-index-independent-audit-v1.json", "half-index-audit-v1.py"],
        [
            (
                "half-index-v1/summary.json",
                "77a5c232387b9a609115a2f76f9e79a165bfdbd943b60d6dbbd712ef03996db1",
            ),
            (
                "half-index-independent-audit-v1.json",
                "74f316b2e50d737494fd1add73b87f38c0d69b490aee490b1d169c5ec5c28b30",
            ),
        ],
    ),
    group(
        "failed-model-v1",
        ["model-compatibility-v1", "failed-model-probe-v1", "model-manifest-v1.json"],
        [
            (
                "model-compatibility-v1/summary.json",
                "0a3a5c50acb3f8265f9b14944c4248319aaf1dbe2572ddf540d8fdad8caa272b",
            ),
            (
                "failed-model-probe-v1/snapshot.json",
                "0f15f37e96b17b138667d9d076f5962fc1cf9cafa7e65653ab8aaaa75448053b",
            ),
        ],
    ),
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_name(name):
    require(isinstance(name, str) and name.startswith(PREFIX), "Path outside owned evidence")
    parts = PurePosixPath(name).parts
    require(
        not name.startswith("/")
        and "\\" not in name
        and "\x00" not in name
        and all(part not in ("", ".", "..") for part in name.split("/"))
        and str(PurePosixPath(name)) == name
        and len(parts) > 2,
        "Unsafe member path",
    )
    return name


def source_path(workspace, name):
    safe_name(name)
    path = workspace
    for part in PurePosixPath(name).parts:
        path /= part
        require(not path.is_symlink(), f"Symlink source: {name}")
    return path


def fingerprint(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def stable_hash(path):
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), f"Nonregular source: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        require(
            fingerprint(before) == fingerprint(os.fstat(stream.fileno())),
            "Source replaced before read",
        )
        prefix = stream.read(4)
        require(prefix not in NATIVE_MAGIC, "Native binary payload is outside evidence scope")
        digest = hashlib.sha256(prefix)
        while chunk := stream.read(CHUNK):
            digest.update(chunk)
        require(
            fingerprint(before) == fingerprint(os.fstat(stream.fileno())),
            "Source changed during read",
        )
    require(fingerprint(before) == fingerprint(path.lstat()), "Source replaced after read")
    return {
        "sha256": digest.hexdigest(),
        "bytes": before.st_size,
        "mode": stat.S_IMODE(before.st_mode),
    }


def enumerate_paths(workspace, roots):
    require(len(roots) == len(set(roots)), "Duplicate input root")
    names = []
    for root in roots:
        path = source_path(workspace, root)
        require(path.exists(), f"Missing source root: {root}")
        for other in roots:
            require(root == other or not root.startswith(other + "/"), "Overlapping input roots")
        pending = [path]
        while pending:
            entry = pending.pop()
            require(not entry.is_symlink(), f"Symlink source: {entry}")
            info = entry.lstat()
            if stat.S_ISDIR(info.st_mode):
                pending.extend(sorted(entry.iterdir(), reverse=True))
            else:
                require(stat.S_ISREG(info.st_mode), f"Nonregular source: {entry}")
                require(entry.suffix in SUFFIXES, f"Excluded payload type: {entry}")
                name = entry.relative_to(workspace).as_posix()
                safe_name(name)
                names.append(name)
    require(names and len(names) == len(set(names)), "Empty or duplicate file inventory")
    return sorted(names)


def inventory(workspace, roots):
    names = enumerate_paths(workspace, roots)
    entries = [{"path": name, **stable_hash(source_path(workspace, name))} for name in names]
    require(names == enumerate_paths(workspace, roots), "Source inventory changed during hashing")
    require(sum(item["bytes"] for item in entries) < MAX_BYTES, "Input group exceeds storage bound")
    return entries


def validate_specs(specs):
    names, roots = [], []
    for spec in specs:
        require(set(spec) == {"name", "roots", "closures"}, "Unexpected group contract fields")
        require(re.fullmatch(r"[a-z0-9][a-z0-9-]*", spec["name"]), "Unsafe group name")
        names.append(spec["name"])
        roots.extend(spec["roots"])
        require(spec["roots"] and spec["closures"], "Group lacks roots or closure anchors")
        for name in spec["roots"]:
            safe_name(name)
        for item in spec["closures"]:
            safe_name(item["path"])
            require(re.fullmatch(r"[a-f0-9]{64}", item["sha256"]), "Invalid closure hash")
            require(
                any(
                    item["path"] == root or item["path"].startswith(root + "/")
                    for root in spec["roots"]
                ),
                "Closure outside group roots",
            )
    require(names and len(names) == len(set(names)), "Missing or duplicate group")
    require(len(roots) == len(set(roots)), "Duplicate root across groups")
    for first in roots:
        require(
            not any(second != first and first.startswith(second + "/") for second in roots),
            "Overlapping roots across groups",
        )


def verify_closures(workspace, specs):
    for spec in specs:
        for closure in spec["closures"]:
            path = source_path(workspace, closure["path"])
            require(stable_hash(path)["sha256"] == closure["sha256"], "Changed closure receipt")
            record = json.loads(path.read_text())
            if path.name == "summary.json":
                require(
                    record.get("formal_eligible") is False
                    and record.get("formal_fits") == 0
                    and record.get("admission_stream_opened") is False,
                    "Wrong evidence scope",
                )
                require(
                    record.get("complete") is True or bool(record.get("error")),
                    "Attempt not closed",
                )


def make_plan(workspace, specs=None):
    specs = GROUPS if specs is None else specs
    validate_specs(specs)
    verify_closures(workspace, specs)
    groups = [{**spec, "members": inventory(workspace, spec["roots"])} for spec in specs]
    return {
        "schema": "gpu-owned-closed-evidence-plan-v1",
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "maximum_group_bytes_exclusive": MAX_BYTES,
        "producer_sha256": stable_hash(Path(__file__))["sha256"],
        "groups": groups,
        "excluded": [
            "model-compatibility-v2 is still active when this plan is declared",
            "NumPy overlay, full native/runtime trees and wheels are referenced, not duplicated",
            "The layout candidate native build remains pending and is a separate identity",
        ],
        "qualification": "Closed evidence preservation only; failures are preserved, not waived",
    }


def validate_plan(plan, specs=None):
    specs = GROUPS if specs is None else specs
    validate_specs(specs)
    require(
        plan.get("schema") == "gpu-owned-closed-evidence-plan-v1"
        and plan.get("formal_eligible") is False
        and plan.get("formal_fits") == 0
        and plan.get("admission_stream_opened") is False,
        "Wrong archive plan scope",
    )
    require(
        plan.get("producer_sha256") == stable_hash(Path(__file__))["sha256"],
        "Archive helper changed",
    )
    require(plan.get("maximum_group_bytes_exclusive") == MAX_BYTES, "Archive budget changed")
    require(
        [{key: item[key] for key in ("name", "roots", "closures")} for item in plan["groups"]]
        == specs,
        "Missing, reordered or altered group contract",
    )
    for item in plan["groups"]:
        names = [member["path"] for member in item["members"]]
        require(names and names == sorted(set(names)), "Missing, reordered or duplicate members")
        for member in item["members"]:
            safe_name(member["path"])
            require(
                set(member) == {"path", "sha256", "bytes", "mode"}, "Unexpected member metadata"
            )
            require(re.fullmatch(r"[a-f0-9]{64}", member["sha256"]), "Invalid member hash")
            require(
                type(member["bytes"]) is int and 0 <= member["bytes"] < MAX_BYTES,
                "Invalid member size",
            )
            require(
                type(member["mode"]) is int and 0 <= member["mode"] <= 0o777, "Unsafe member mode"
            )
            require(
                any(
                    member["path"] == root or member["path"].startswith(root + "/")
                    for root in item["roots"]
                ),
                "Member outside declared roots",
            )
        require(
            sum(member["bytes"] for member in item["members"]) < MAX_BYTES,
            "Input group exceeds bound",
        )


def verify_sources(workspace, plan):
    verify_closures(workspace, plan["groups"])
    for item in plan["groups"]:
        require(
            inventory(workspace, item["roots"]) == item["members"],
            "Source bytes or inventory changed",
        )


def verify_archive(path, members):
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size < MAX_BYTES,
        "Missing, symlinked or oversized archive",
    )
    before = fingerprint(path.lstat())
    names = [item["path"] for item in members]
    require(names and names == sorted(set(names)), "Invalid expected archive inventory")
    seen = []
    with gzip.open(path, "rb") as decompressed:
        with tarfile.open(fileobj=decompressed, mode="r:") as archive:
            last_data_end = 0
            for member in archive:
                safe_name(member.name)
                require(
                    member.isfile() and not member.issparse() and not member.pax_headers,
                    "Archive member is not a plain regular file",
                )
                require(
                    member.uid == member.gid == member.mtime == 0
                    and not member.uname
                    and not member.gname,
                    "Unexpected tar ownership metadata",
                )
                require(
                    member.offset == last_data_end and member.offset_data == last_data_end + 512,
                    "Unexpected extended header or archive gap",
                )
                require(len(seen) < len(members), "Extra archive member")
                expected = members[len(seen)]
                require(
                    member.name == expected["path"],
                    "Missing, reordered, duplicate or unexpected member",
                )
                require(
                    member.size == expected["bytes"] and member.mode == expected["mode"],
                    "Archive member metadata differs",
                )
                stream = archive.extractfile(member)
                require(stream is not None, "Unreadable archive member")
                with stream:
                    require(
                        hashlib.file_digest(stream, "sha256").hexdigest() == expected["sha256"],
                        "Archive member bytes differ",
                    )
                seen.append(member.name)
                last_data_end = member.offset_data + ((member.size + 511) // 512) * 512
            # Reject concatenated tar payloads and anything hidden behind tar's first EOF.
            decompressed.seek(last_data_end)
            expected_padding = ((last_data_end + 1024 + 10239) // 10240) * 10240 - last_data_end
            padding = decompressed.read(expected_padding + 1)
            require(
                len(padding) == expected_padding and not any(padding), "Unexpected tar tail bytes"
            )
    require(seen == names, "Incomplete archive inventory")
    require(fingerprint(path.lstat()) == before, "Archive changed during verification")
    return {
        "archive": path.name,
        **stable_hash(path),
        "members": len(seen),
        "uncompressed_bytes": sum(item["bytes"] for item in members),
        "all_member_hashes_sizes_and_inventory_passed": True,
    }


class DigestReader:
    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.sha256()

    def read(self, size):
        data = self.stream.read(min(size, CHUNK))
        self.digest.update(data)
        return data


def write_archive(workspace, group_plan, destination):
    require(not destination.exists(), "Archive destination already exists")
    with destination.open("xb") as output:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=output, mtime=0, compresslevel=6
        ) as packed:
            with tarfile.open(fileobj=packed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                for member in group_plan["members"]:
                    path = source_path(workspace, member["path"])
                    before = path.lstat()
                    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                    with os.fdopen(descriptor, "rb") as stream:
                        require(
                            fingerprint(before) == fingerprint(os.fstat(stream.fileno()))
                            and stat.S_ISREG(before.st_mode),
                            "Source replaced before archiving",
                        )
                        require(
                            before.st_size == member["bytes"]
                            and stat.S_IMODE(before.st_mode) == member["mode"],
                            "Source metadata changed",
                        )
                        reader = DigestReader(stream)
                        info = tarfile.TarInfo(member["path"])
                        info.size, info.mode = member["bytes"], member["mode"]
                        info.uid = info.gid = info.mtime = 0
                        archive.addfile(info, reader)
                        require(
                            reader.digest.hexdigest() == member["sha256"],
                            "Source changed while archiving",
                        )
                        require(
                            fingerprint(before)
                            == fingerprint(os.fstat(stream.fileno()))
                            == fingerprint(path.lstat()),
                            "Source changed during archiving",
                        )
                    require(output.tell() < MAX_BYTES, "Compressed archive exceeds bound")
    return verify_archive(destination, group_plan["members"])


def reject_nested_output(workspace, output, plan):
    for item in plan["groups"]:
        for name in item["roots"]:
            source = workspace / name
            require(
                output != source and source not in output.parents,
                "Output lies inside evidence roots",
            )


def build(workspace, plan_path, expected_sha256, output, specs=None):
    require(stable_hash(plan_path)["sha256"] == expected_sha256, "Plan hash mismatch")
    plan = json.loads(plan_path.read_text())
    validate_plan(plan, specs)
    reject_nested_output(workspace, output, plan)
    verify_sources(workspace, plan)
    require(
        shutil.disk_usage(output.parent).free
        > 64 * 1024 * 1024
        + sum(member["bytes"] for item in plan["groups"] for member in item["members"]),
        "Insufficient disk space for bounded archives",
    )
    output.mkdir(exist_ok=False)
    shutil.copyfile(plan_path, output / "archive-plan.json")
    receipt = {
        "schema": "gpu-owned-closed-evidence-publication-v1",
        "passed": False,
        "formal_eligible": False,
        "uploaded": False,
        "plan_sha256": expected_sha256,
        "producer_sha256": plan["producer_sha256"],
        "archives": [],
    }
    try:
        for item in plan["groups"]:
            result = write_archive(workspace, item, output / (item["name"] + ".tar.gz"))
            receipt["archives"].append(result)
        verify_sources(workspace, plan)
        require(
            stable_hash(plan_path)["sha256"]
            == expected_sha256
            == stable_hash(output / "archive-plan.json")["sha256"],
            "Plan changed during build",
        )
        receipt["passed"] = True
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "publication-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    planning = subparsers.add_parser("plan")
    planning.add_argument("--workspace", type=Path, required=True)
    planning.add_argument("--output", type=Path, required=True)
    packing = subparsers.add_parser("build")
    packing.add_argument("--workspace", type=Path, required=True)
    packing.add_argument("--plan", type=Path, required=True)
    packing.add_argument("--plan-sha256", required=True)
    packing.add_argument("--output", type=Path, required=True)
    checking = subparsers.add_parser("verify")
    checking.add_argument("--plan", type=Path, required=True)
    checking.add_argument("--plan-sha256", required=True)
    checking.add_argument("--archive-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        workspace, output = args.workspace.resolve(), args.output.resolve()
        plan = make_plan(workspace)
        reject_nested_output(workspace, output, plan)
        with output.open("x") as stream:
            stream.write(json.dumps(plan, indent=2) + "\n")
        print(json.dumps({"plan": str(output), **stable_hash(output)}))
    elif args.command == "build":
        print(
            json.dumps(
                build(
                    args.workspace.resolve(),
                    args.plan.resolve(),
                    args.plan_sha256,
                    args.output.resolve(),
                )
            )
        )
    else:
        require(stable_hash(args.plan)["sha256"] == args.plan_sha256, "Plan hash mismatch")
        plan = json.loads(args.plan.read_text())
        validate_plan(plan)
        results = [
            verify_archive(args.archive_directory / (item["name"] + ".tar.gz"), item["members"])
            for item in plan["groups"]
        ]
        print(json.dumps({"passed": True, "archives": results, "uploaded": False}))


if __name__ == "__main__":
    main()
