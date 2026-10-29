"""Archive GPU development evidence without claiming formal experiment admission.

Run only after source/evidence writers have stopped. Reads source and evidence;
never installs dependencies, runs GPU code, deletes files, or mutates Git. Output
must be outside both input trees. A verification sidecar marks successful archive
verification; an interrupted or failed archive remains preserved without it.
"""

import argparse
import datetime
import gzip
import hashlib
import io
import json
import os
import stat
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_EVIDENCE = {"uv-cache", "python-site"}
CHUNK = 1024 * 1024
MAX_METADATA = 32 * 1024 * 1024
CPU_V6 = {
    "repository": "https://github.com/logannye/arm64-transformer",
    "measured_revision": "2c5191047068c16b12e3312652b118d48a5f3744",
    "publication_revision": "7504ae04684eb7e5ec89904471360c11930fabb7",
    "merged_main_revision": "04a69ba7b13e4c0800a58be400c62d8095d5acae",
    "release_tag": "transformer-attention-precision-v6",
    "release_url": "https://github.com/logannye/arm64-transformer/releases/tag/transformer-attention-precision-v6",
    "ci_run": "40815358560",
    "status": "native_precision_limit",
    "scope": "Historical CPU references only; this packager does not rerun or re-audit them",
}
FORMAL = {
    "planned_fits": 108,
    "completed_fits": 0,
    "status": "not_admitted",
    "admitted": False,
    "frozen": False,
    "numerical_gate_waived": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_name(name):
    path = PurePosixPath(name)
    require(
        bool(name)
        and bool(path.parts)
        and not path.is_absolute()
        and str(path) == name
        and ".." not in path.parts
        and not any(char in name for char in "\\\r\n\x00"),
        f"Unsafe archive member name: {name!r}",
    )
    return name


def no_symlinks(path):
    path = Path(path).absolute()
    for candidate in (path, *path.parents):
        require(not candidate.is_symlink(), f"Symlink is not allowed: {candidate}")
    return path


def regular_file(path):
    no_symlinks(path)
    info = path.stat()
    require(stat.S_ISREG(info.st_mode), f"Expected a regular file: {path}")
    return info


def signature(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_mode,
    )


def stream_digest(handle):
    digest, size = hashlib.sha256(), 0
    while data := handle.read(CHUNK):
        digest.update(data)
        size += len(data)
    return digest.hexdigest(), size


def file_identity(path):
    before = regular_file(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        require(signature(os.fstat(handle.fileno())) == signature(before), f"File changed: {path}")
        digest, size = stream_digest(handle)
        require(signature(os.fstat(handle.fileno())) == signature(before), f"File changed: {path}")
    require(signature(regular_file(path)) == signature(before), f"File changed: {path}")
    require(size == before.st_size, f"File size changed: {path}")
    return {
        "sha256": digest,
        "bytes": size,
        "mode": 0o755 if before.st_mode & 0o111 else 0o644,
    }, signature(before)


def git_bytes(repo, arguments):
    # Even status should not refresh/write Git's optional index caches.
    return subprocess.check_output(["git", "--no-optional-locks", *arguments], cwd=repo)


def git_state(repo):
    revision = git_bytes(repo, ["rev-parse", "HEAD"]).decode().strip()
    status = git_bytes(repo, ["status", "--porcelain=v1", "-z", "--untracked-files=all"])
    return {
        "revision": revision,
        "dirty": bool(status),
        "status_porcelain_v1_nul": status.decode(),
        "status_sha256": hashlib.sha256(status).hexdigest(),
    }


def inventory(repo, evidence, binary):
    payload = {}

    def add(name, path, category):
        safe_name(name)
        require(name not in payload, f"Duplicate archive member: {name}")
        regular_file(path)
        payload[name] = {"path": path, "category": category}

    raw = git_bytes(repo, ["ls-files", "-z", "--cached", "--others", "--exclude-standard"])
    names = [name.decode() for name in raw.split(b"\0") if name]
    require(len(names) == len(set(names)), "Duplicate Git source paths")
    for name in sorted(names):
        safe_name(name)
        path = repo / name
        # Dependency installations stay excluded even if accidentally Git-listed.
        if path.is_relative_to(repo / ".venv") or any(
            path.is_relative_to(evidence / excluded) for excluded in EXCLUDED_EVIDENCE
        ):
            continue
        add("source/" + name, path, "working_tree_source")
    require(binary.is_relative_to(repo), "Metal binary must be inside the project")
    add("runtime/libtransformer_metal.dylib", binary, "native_metal_binary")
    require(evidence.is_dir(), "Missing GPU development evidence directory")
    pending = [evidence]
    while pending:
        directory = pending.pop()
        no_symlinks(directory)
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(evidence)
            if relative.parts[0] in EXCLUDED_EVIDENCE:
                continue
            no_symlinks(path)
            if path.is_dir():
                pending.append(path)
            else:
                add("evidence/" + relative.as_posix(), path, "development_evidence")
    require(
        any(name.startswith("evidence/fixtures/") for name in payload),
        "Missing development fixtures",
    )
    for name in (
        "source/requirements-gpu-development.txt",
        "source/scripts/capture_gpu_environment.py",
        "source/experiments/GPU-PLAN-v1.md",
        "source/experiments/PROTOCOL-gpu-v1.md",
        "evidence/discovery/runtime-identity.json",
    ):
        require(name in payload, f"Required development evidence is absent: {name}")
    return dict(sorted(payload.items()))


def read_json(path):
    require(regular_file(path).st_size <= MAX_METADATA, f"Metadata exceeds bounded size: {path}")
    result = json.loads(path.read_text())
    require(isinstance(result, dict), f"Expected JSON object: {path}")
    return result


def development_metadata(payload, files):
    attempts = []
    for name, item in payload.items():
        if item["category"] == "development_evidence" and item["path"].name == "runtime.json":
            try:
                receipt = read_json(item["path"])
            except (ValueError, UnicodeError) as exc:
                # A crashed writer may leave a partial receipt. Preserve its
                # bytes and the parse failure instead of discarding the attempt.
                attempts.append(
                    {
                        "receipt": name,
                        "status": "unreadable_receipt",
                        "parse_error": type(exc).__name__,
                        "scope_verified": False,
                    }
                )
                continue
            require(
                receipt.get("scope") in (None, "development_only"), f"Non-development run: {name}"
            )
            model = receipt.get("model")
            attempts.append(
                {
                    "receipt": name,
                    "status": receipt.get("status"),
                    "engine": receipt.get("engine"),
                    "width": model.get("width") if isinstance(model, dict) else None,
                    "scope_verified": receipt.get("scope") == "development_only",
                }
            )
    runtime = read_json(payload["evidence/discovery/runtime-identity.json"]["path"])
    for key, name in (
        ("requirements_identity", "source/requirements-gpu-development.txt"),
        ("capture_script_identity", "source/scripts/capture_gpu_environment.py"),
    ):
        require(
            runtime.get(key, {}).get("sha256") == files[name]["sha256"],
            f"Runtime identity does not match retained source: {name}",
        )
    return {
        "development_attempt_receipts": attempts,
        "dependency_identity": {
            "receipt": "evidence/discovery/runtime-identity.json",
            "package_count": runtime.get("package_count"),
            "installed_file_count": runtime.get("installed_file_count"),
            "record_integrity_passed": runtime.get("record_integrity_passed"),
            "installed_runtime_bytes_included": False,
            "wheel_archive_provenance_verified": False,
        },
    }


class HashingReader:
    def __init__(self, handle):
        self.handle, self.digest, self.size = handle, hashlib.sha256(), 0

    def read(self, count=-1):
        data = self.handle.read(count)
        self.digest.update(data)
        self.size += len(data)
        return data


def tar_info(name, size, mode=0o644):
    info = tarfile.TarInfo(safe_name(name))
    info.size, info.mode, info.mtime = size, mode, 0
    return info


def write_archive(output, payload, files, signatures, manifest_bytes):
    expected = {name: item["sha256"] for name, item in files.items()}
    expected["MANIFEST.json"] = hashlib.sha256(manifest_bytes).hexdigest()
    checksums = "".join(f"{digest}  {name}\n" for name, digest in sorted(expected.items())).encode()
    # Exclusive creation preserves any partial archive on failure; never overwrite.
    with (
        output.open("xb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped,
    ):
        with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for name, data in (("MANIFEST.json", manifest_bytes), ("SHA256SUMS", checksums)):
                archive.addfile(tar_info(name, len(data)), io.BytesIO(data))
            for name, item in payload.items():
                path, expected_file = item["path"], files[name]
                require(signature(regular_file(path)) == signatures[name], f"Input changed: {path}")
                with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
                    require(
                        signature(os.fstat(handle.fileno())) == signatures[name],
                        f"Input changed: {path}",
                    )
                    reader = HashingReader(handle)
                    archive.addfile(
                        tar_info(name, expected_file["bytes"], expected_file["mode"]), reader
                    )
                    require(
                        reader.size == expected_file["bytes"]
                        and reader.digest.hexdigest() == expected_file["sha256"],
                        f"Input bytes changed: {path}",
                    )
                    require(not handle.read(1), f"Input grew: {path}")
                    require(
                        signature(os.fstat(handle.fileno())) == signatures[name],
                        f"Input changed: {path}",
                    )
    return expected


def verify_archive(path):
    """Read every archived byte against both embedded manifests, without extraction."""
    seen, observed, manifest, checksums = set(), {}, None, None
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            safe_name(member.name)
            require(
                member.isfile() and not member.issparse(), "Only regular archive files are allowed"
            )
            require(member.name not in seen, f"Duplicate archive member: {member.name}")
            seen.add(member.name)
            handle = archive.extractfile(member)
            require(handle is not None, "Missing archive file payload")
            if member.name in ("MANIFEST.json", "SHA256SUMS"):
                require(member.size <= MAX_METADATA, "Oversized embedded metadata")
                data = handle.read()
                digest, size = hashlib.sha256(data).hexdigest(), len(data)
                if member.name == "MANIFEST.json":
                    manifest = json.loads(data)
                else:
                    checksums = {}
                    for line in data.decode().splitlines():
                        value, name = line.split("  ", 1)
                        safe_name(name)
                        require(name not in checksums, "Duplicate checksum manifest member")
                        checksums[name] = value
            else:
                digest, size = stream_digest(handle)
            require(size == member.size, f"Truncated member: {member.name}")
            observed[member.name] = {"sha256": digest, "bytes": size, "mode": member.mode}
    require(
        isinstance(manifest, dict) and isinstance(checksums, dict), "Missing embedded manifests"
    )
    require(
        manifest.get("scope") == "development_only" and manifest.get("formal_campaign") == FORMAL,
        "Archive does not retain the unresolved formal-admission boundary",
    )
    files = manifest.get("files")
    require(isinstance(files, dict) and bool(files), "Empty archive payload inventory")
    require(seen == set(files) | {"MANIFEST.json", "SHA256SUMS"}, "Archive member set mismatch")
    require(set(checksums) == set(files) | {"MANIFEST.json"}, "Checksum member set mismatch")
    for name, expected in checksums.items():
        require(observed[name]["sha256"] == expected, f"Archive checksum mismatch: {name}")
    for name, expected in files.items():
        require(
            all(observed[name][key] == expected[key] for key in ("sha256", "bytes", "mode")),
            f"Archive manifest mismatch: {name}",
        )
    return {
        "payload_files": len(files),
        "payload_bytes": sum(item["bytes"] for item in files.values()),
        "manifest_sha256": observed["MANIFEST.json"]["sha256"],
        "archive_integrity_verified": True,
    }


def output_paths(output, repo, evidence):
    output = no_symlinks(output).resolve()
    safe_name(output.name)
    require(output.name.endswith(".tar.gz"), "Archive output must end in .tar.gz")
    base = output.name.removesuffix(".tar.gz")
    paths = {
        "archive": output,
        "manifest": output.with_name(base + "-manifest.json"),
        "verification": output.with_name(base + "-verification.json"),
        "checksums": output.with_name(base + "-SHA256SUMS.txt"),
    }
    for path in paths.values():
        no_symlinks(path)
        require(
            not path.is_relative_to(repo) and not path.is_relative_to(evidence),
            "Outputs must be outside source and evidence trees",
        )
        if path.exists():
            raise FileExistsError(path)
    require(output.parent.is_dir(), "Output parent must already exist")
    return paths


def package(repo, evidence, binary, output):
    repo, evidence, binary = (no_symlinks(path).resolve() for path in (repo, evidence, binary))
    require(repo.is_dir() and evidence.is_relative_to(repo), "Evidence must be inside the project")
    paths = output_paths(output, repo, evidence)
    source_state = git_state(repo)
    payload = inventory(repo, evidence, binary)
    files, signatures = {}, {}
    for name, item in payload.items():
        identity, signatures[name] = file_identity(item["path"])
        files[name] = identity | {
            "category": item["category"],
            "project_path": item["path"].relative_to(repo).as_posix(),
        }
    manifest = {
        "schema": 1,
        "scope": "development_only",
        "description": (
            "Source and complete retained GPU development evidence, "
            "including failures and superseded attempts"
        ),
        "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "formal_campaign": FORMAL,
        "formal_count_basis": "Declared development phase; formal execution is not admitted",
        "source": source_state,
        "cpu_v6_references": CPU_V6,
        "exclusions": [
            {
                "path": evidence.relative_to(repo).as_posix() + "/" + name,
                "reason": (
                    "Dependency installation/cache; installed identity receipt "
                    "and version pins are retained"
                ),
            }
            for name in sorted(EXCLUDED_EVIDENCE)
        ]
        + [
            {
                "path": ".venv",
                "reason": "Installed dependency tree; identity is retained separately",
            }
        ],
        "files": files,
        **development_metadata(payload, files),
        "limits": [
            "Archive integrity does not establish numerical qualification or formal admission.",
            "Current source cannot authenticate early diagnostics that lack source receipts.",
            "Failed and superseded outputs retain their original claims and failure records.",
            "Original CPU release archives are not copied or modified; tracked summaries remain.",
        ],
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    write_archive(paths["archive"], payload, files, signatures, manifest_bytes)
    verified = verify_archive(paths["archive"])
    require(
        verified["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest(),
        "Embedded manifest changed",
    )
    require(git_state(repo) == source_state, "Git/source status changed while packaging")
    after = inventory(repo, evidence, binary)
    require(after.keys() == payload.keys(), "Input inventory changed while packaging")
    for name, item in after.items():
        require(signature(regular_file(item["path"])) == signatures[name], f"Input changed: {name}")
    archive_identity, _ = file_identity(paths["archive"])
    verified.update(
        scope="development_only",
        formal_campaign=FORMAL,
        archive=paths["archive"].name,
        archive_sha256=archive_identity["sha256"],
        archive_bytes=archive_identity["bytes"],
        sources_and_evidence_unchanged=True,
        numerical_qualification_claimed=False,
    )
    with paths["manifest"].open("xb") as handle:
        handle.write(manifest_bytes)
    with paths["verification"].open("x") as handle:
        handle.write(json.dumps(verified, indent=2, sort_keys=True) + "\n")
    lines = []
    for key in ("archive", "manifest", "verification"):
        identity, _ = file_identity(paths[key])
        lines.append(f"{identity['sha256']}  {paths[key].name}\n")
    with paths["checksums"].open("x") as handle:
        handle.writelines(lines)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--metal-binary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence or args.repo / "work/gpu-v1"
    binary = args.metal_binary or args.repo / "native/metal/build/libtransformer_metal.dylib"
    print(json.dumps(package(args.repo, evidence, binary, args.output), indent=2))


if __name__ == "__main__":
    main()
