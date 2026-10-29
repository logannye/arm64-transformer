"""Stream and verify a split archive of retained, blocked GPU-v2 engineering work.

Run after all writers stop. No model execution, dependency installation, original
held-out-array access, deletion, or Git mutation. Every retained evidence file is
included except explicitly inventoried dependency/cache directories. One gzip/tar
stream is split below the release asset limit, including files larger than a part.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import package_gpu_development as old  # noqa: E402

CHUNK = old.CHUNK
MAX_PART_BYTES = 1_900_000_000
EXCLUDED = {"__pycache__", ".pytest_cache", ".ruff_cache", "python-site", "uv-cache", ".venv"}
SCOPE = "gpu_v2_blocked_engineering_release"
FORMAL = {
    "planned_fits": 180,
    "completed_fits": 0,
    "status": "not_admitted",
    "admitted": False,
    "numerical_gate_waived": False,
    "heldout_arrays_read_by_packager": False,
}


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_new(path, value):
    with Path(path).open("xb") as handle:
        handle.write(json_bytes(value))


def tree_files(root):
    root = old.no_symlinks(root).resolve()
    old.require(root.is_dir(), f"Missing tree: {root}")
    files, excluded, pending = [], [], [root]
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir()):
            old.no_symlinks(path)
            if path.is_dir() and path.name in EXCLUDED:
                excluded.append(path.relative_to(root).as_posix())
            elif path.is_dir():
                pending.append(path)
            else:
                old.regular_file(path)
                files.append(path)
    return sorted(files), sorted(excluded)


def inventory(repo, evidence, inputs, binary, report=None, snapshots=None):
    payload, exclusions = {}, []

    def add(name, path, category):
        old.safe_name(name)
        old.require(name not in payload, f"Duplicate member: {name}")
        old.regular_file(path)
        payload[name] = {"path": Path(path), "category": category}

    names = old.git_bytes(repo, ["ls-files", "-z", "--cached", "--others", "--exclude-standard"])
    for raw in sorted(set(names.split(b"\0")) - {b""}):
        relative = raw.decode()
        old.safe_name(relative)
        path = repo / relative
        if any(part in EXCLUDED for part in Path(relative).parts):
            exclusions.append("source/" + relative)
            continue
        old.require(
            not path.is_relative_to(inputs), "CPU inputs must use the explicit input inventory"
        )
        add("source/" + relative, path, "current_working_tree_source")
    for prefix, root, category in (
        ("evidence", evidence, "retained_development_evidence"),
        *([("report", report, "post_development_report")] if report is not None else []),
    ):
        files, omitted = tree_files(root)
        exclusions.extend(f"{prefix}/{name}" for name in omitted)
        for path in files:
            add(f"{prefix}/{path.relative_to(root).as_posix()}", path, category)
    add("runtime/libtransformer_metal.dylib", binary, "executed_metal_binary")
    original_inputs = []
    for width in (32, 64, 128, 256, 512, 1024):
        for seed in (42, 43, 44):
            directory = inputs / f"width-{width}/seed-{seed}"
            manifest = old.read_json(directory / "manifest.json")
            old.require(
                manifest["config"]["width"] == width and manifest["config"]["seed"] == seed,
                "CPU input manifest identity differs",
            )
            prefix = f"cpu-inputs/width-{width}/seed-{seed}"
            add(
                prefix + "/manifest.json",
                directory / "manifest.json",
                "original_cpu_input_manifest",
            )
            original_inputs.append(
                {
                    "width": width,
                    "seed": seed,
                    "manifest": prefix + "/manifest.json",
                    "recorded_original_sha256": manifest["sha256"],
                    "heldout_hashes_reused_without_reading_arrays": True,
                }
            )
            if seed == 42:
                for name in ("initial.f32", "train.i32"):
                    add(
                        prefix + "/" + name, directory / name, "original_training_development_input"
                    )
    for path in snapshots or []:
        add("source-snapshots/" + path.name, path, "immutable_git_source_snapshot")
    for name in (
        "evidence/runtime-identity.json",
        "evidence/cpu-reference-receipt.json",
        "source/experiments/PROTOCOL-gpu-v2.md",
        "source/scripts/package_gpu_v2_engineering.py",
        "source/scripts/package_gpu_development.py",
    ):
        old.require(name in payload, f"Missing required release evidence: {name}")
    old.require(
        any(name.startswith("evidence/cpu-reference/native/build/") for name in payload),
        "Frozen CPU binaries missing",
    )
    return dict(sorted(payload.items())), sorted(exclusions), original_inputs


def validate_report_inventory(summary):
    """A blocked completed engineering report is required, never an admission pass."""
    old.require(summary.get("scope") == SCOPE, "Expected a blocked v2 engineering report")
    old.require(
        summary.get("formal_admission_opened") is False
        and summary.get("formal_fits_completed") == 0
        and summary.get("framework_or_cpu_speedup_claimed") is False,
        "Blocked release boundary differs",
    )
    widths = {32, 64, 128, 256, 512, 1024}
    baselines, metal = summary.get("baselines", []), summary.get("metal", [])
    old.require(
        len(baselines) == 12
        and {(row["engine"], row["width"]) for row in baselines}
        == {(engine, width) for engine in ("mps", "mlx") for width in widths},
        "Report omits a standard baseline width",
    )
    old.require(
        any(row.get("standard_baseline_qualified") is False for row in baselines),
        "No failed standard baseline gate is retained",
    )
    old.require(
        len(metal) == 6
        and {row["width"] for row in metal} == widths
        and all(row.get("search_status") in ("completed", "failed") for row in metal),
        "Report requires all six finalized searches, including failures or no improvement",
    )
    old.require(
        set(summary.get("input_roots", {})) == {"baseline", "profiles", "searches"},
        "Report does not identify its source roots",
    )
    old.require(bool(summary.get("source_artifacts")), "Report has no authenticated source files")
    old.require(
        isinstance(summary.get("eligible_search_attempt"), str)
        and bool(summary["eligible_search_attempt"]),
        "Report lacks an eligible attempt",
    )


def bind_report_files(evidence, summary, files):
    evidence = Path(evidence).resolve()
    roots = {role: old.no_symlinks(path).resolve() for role, path in summary["input_roots"].items()}
    old.require(
        all(root.is_relative_to(evidence) for root in roots.values()),
        "Report source root lies outside retained evidence",
    )
    bound = {}
    for key, expected in summary["source_artifacts"].items():
        role, relative = key.split("/", 1)
        old.require(role in roots, "Unknown report source role")
        old.safe_name(relative)
        path = roots[role] / relative
        member = "evidence/" + path.relative_to(evidence).as_posix()
        old.require(
            files.get(member, {}).get("sha256") == expected,
            f"Reported source bytes are absent or changed: {key}",
        )
        bound[key] = member
    for engine in ("mps", "mlx"):
        for width in (32, 64, 128, 256, 512, 1024):
            for process in range(2):
                key = f"baseline/{engine}-w{width}-process{process}/worker.json"
                old.require(key in bound, "Report omits a fresh-process baseline worker")
    for role in ("profiles", "searches"):
        for width in (32, 64, 128, 256, 512, 1024):
            for suffix in (".json", ".json.ledger.json"):
                old.require(
                    f"{role}/width-{width}{suffix}" in bound,
                    "Report omits a profile/search or frozen ledger",
                )
    old.require("searches/attempt.json" in bound, "Eligible-attempt marker is not authenticated")
    old.require(
        not (roots["searches"] / "invalidated.json").exists(),
        "Reported search attempt has been invalidated",
    )
    attempt = old.read_json(roots["searches"] / "attempt.json")
    old.require(
        attempt.get("attempt_id") == summary["eligible_search_attempt"]
        and attempt.get("eligible_for_reporting") is True,
        "Search attempt eligibility changed",
    )
    return roots, bound


def validate_retained_baseline_members(evidence, baseline, files):
    """Verify retention of complete inventory, failures, first arrays and divergences."""

    def retained(path):
        member = "evidence/" + Path(path).relative_to(evidence).as_posix()
        old.require(member in files, f"Referenced baseline artifact missing: {member}")

    for engine in ("mps", "mlx"):
        for width in (32, 64, 128, 256, 512, 1024):
            retained(baseline / f"{engine}-w{width}-cross-process.json")
            for process in range(2):
                root = baseline / f"{engine}-w{width}-process{process}"
                worker = old.read_json(root / "worker.json")
                old.require(
                    worker.get("engine") == engine and worker.get("width") == width,
                    "Wrong baseline worker identity",
                )
                cases, expected = worker.get("cases", []), worker.get("expected_cases", [])
                names = [case["name"] for case in cases]
                old.require(
                    len(names) == len(set(names)) == len(expected) == 81
                    and set(names) == set(expected),
                    "Baseline case inventory is incomplete",
                )
                for case in cases:
                    executions = case.get("executions", [])
                    observed = [item["execution"] for item in executions]
                    skipped = case.get("unexecuted_repetitions", [])
                    old.require(
                        observed + skipped == list(range(10)),
                        "Missing baseline execution or explicit runtime-error skip",
                    )
                    if skipped:
                        old.require(
                            bool(executions) and "error" in executions[-1],
                            "Skipped repetitions lack a retained runtime error",
                        )
                    if case.get("first_arrays"):
                        retained(root / old.safe_name(case["first_arrays"]))
                    else:
                        old.require(
                            executions and "error" in executions[0],
                            "Successful baseline case has no first arrays",
                        )
                    saved = set()
                    for execution in executions:
                        divergent = {
                            key
                            for key, value in execution.get("exact", {}).items()
                            if value.get("passed") is False
                        } - saved
                        if divergent:
                            retained(
                                root
                                / old.safe_name(case["name"])
                                / f"first-divergence-{execution['execution']:02}.npz"
                            )
                            saved.update(divergent)
    for relative in (
        "repeatability/full-v1/aborted-source-change.json",
        "repeatability/full-v2/aborted-harness.json",
    ):
        old.require(
            "evidence/" + relative in files,
            "A superseded failure attempt is missing its closing receipt",
        )


class SplitWriter:
    """Bounded streaming sink: no complete archive or complete part in memory."""

    def __init__(self, output, basename, limit):
        old.require(0 < limit <= MAX_PART_BYTES, "Part limit exceeds release-safe maximum")
        self.output, self.basename, self.limit = Path(output), basename, limit
        self.records, self.handle, self.hash, self.part_bytes, self.total = [], None, None, 0, 0

    def _finish(self):
        if self.handle is not None:
            self.handle.close()
            self.records.append(
                {
                    "name": Path(self.handle.name).name,
                    "bytes": self.part_bytes,
                    "sha256": self.hash.hexdigest(),
                }
            )
            self.handle = None

    def write(self, value):
        view = memoryview(value)
        size = len(view)
        while view:
            if self.handle is None:
                name = f"{self.basename}.tar.gz.part-{len(self.records) + 1:05}"
                self.handle = (self.output / name).open("xb")
                self.hash, self.part_bytes = hashlib.sha256(), 0
            piece = view[: self.limit - self.part_bytes]
            self.handle.write(piece)
            self.hash.update(piece)
            self.part_bytes += len(piece)
            self.total += len(piece)
            view = view[len(piece) :]
            if self.part_bytes == self.limit:
                self._finish()
        return size

    def tell(self):
        return self.total

    def flush(self):
        if self.handle is not None:
            self.handle.flush()

    def close(self):
        self._finish()


class JoinedReader:
    def __init__(self, paths):
        self.paths, self.index, self.handle, self.position = list(paths), 0, None, 0

    def read(self, size=-1):
        old.require(0 <= size <= 16 * CHUNK, "Unbounded archive read requested")
        result = bytearray()
        while len(result) < size:
            if self.handle is None:
                if self.index == len(self.paths):
                    break
                path = self.paths[self.index]
                self.handle = os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb")
                self.index += 1
            value = self.handle.read(size - len(result))
            if value:
                result.extend(value)
            else:
                self.handle.close()
                self.handle = None
        self.position += len(result)
        return bytes(result)

    def tell(self):
        return self.position

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def write_split_archive(output, basename, limit, payload, files, signatures, manifest):
    data = json_bytes(manifest)
    checksums = {name: record["sha256"] for name, record in files.items()}
    checksums["MANIFEST.json"] = hashlib.sha256(data).hexdigest()
    sums = "".join(f"{value}  {name}\n" for name, value in sorted(checksums.items())).encode()
    sink = SplitWriter(output, basename, limit)
    try:
        with gzip.GzipFile(fileobj=sink, mode="wb", filename="", mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for name, value in (("MANIFEST.json", data), ("SHA256SUMS", sums)):
                    archive.addfile(old.tar_info(name, len(value)), io.BytesIO(value))
                for name, item in payload.items():
                    path, expected = item["path"], files[name]
                    old.require(
                        old.signature(old.regular_file(path)) == signatures[name],
                        f"Input changed: {path}",
                    )
                    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
                        reader = old.HashingReader(handle)
                        archive.addfile(
                            old.tar_info(name, expected["bytes"], expected["mode"]), reader
                        )
                        old.require(
                            reader.size == expected["bytes"]
                            and reader.digest.hexdigest() == expected["sha256"],
                            f"Input bytes changed: {path}",
                        )
                        old.require(not handle.read(1), f"Input grew: {path}")
                        old.require(
                            old.signature(os.fstat(handle.fileno())) == signatures[name],
                            f"Input changed: {path}",
                        )
    finally:
        sink.close()
    return {
        "schema": 1,
        "scope": SCOPE,
        "format": "ordered_byte_parts_of_one_tar_gzip_stream",
        "parts": sink.records,
        "manifest_sha256": checksums["MANIFEST.json"],
        "max_part_bytes": limit,
    }


def verify_archive(index_path):
    """Authenticate every compressed part and every member, without extraction."""
    index_path = old.no_symlinks(index_path).resolve()
    index = old.read_json(index_path)
    old.require(index.get("scope") == SCOPE, "Not a v2 engineering archive")
    parts, signatures = [], {}
    seen_parts = set()
    for record in index.get("parts", []):
        name = old.safe_name(record["name"])
        old.require(Path(name).name == name and name not in seen_parts, "Invalid archive part")
        seen_parts.add(name)
        path = index_path.parent / name
        actual, signatures[name] = old.file_identity(path)
        old.require(
            actual["sha256"] == record["sha256"] and actual["bytes"] == record["bytes"],
            "Archive part changed",
        )
        old.require(
            0 < actual["bytes"] <= min(index["max_part_bytes"], MAX_PART_BYTES),
            "Oversized archive part",
        )
        parts.append(path)
    old.require(bool(parts), "Missing archive parts")
    reader = JoinedReader(parts)
    observed, manifest, checksums = {}, None, None
    try:
        with gzip.GzipFile(fileobj=reader, mode="rb") as uncompressed:
            with tarfile.open(fileobj=uncompressed, mode="r|") as archive:
                for member in archive:
                    name = old.safe_name(member.name)
                    old.require(
                        member.isfile() and not member.issparse() and name not in observed,
                        "Unsafe archive member",
                    )
                    handle = archive.extractfile(member)
                    old.require(handle is not None, "Missing member content")
                    if name in ("MANIFEST.json", "SHA256SUMS"):
                        old.require(member.size <= old.MAX_METADATA, "Oversized manifest")
                        value = handle.read()
                        sha, size = hashlib.sha256(value).hexdigest(), len(value)
                        if name == "MANIFEST.json":
                            manifest = json.loads(value)
                        else:
                            checksums = {}
                            for line in value.decode().splitlines():
                                expected, entry = line.split("  ", 1)
                                old.safe_name(entry)
                                old.require(entry not in checksums, "Duplicate checksum")
                                checksums[entry] = expected
                    else:
                        sha, size = old.stream_digest(handle)
                    old.require(size == member.size, "Truncated member")
                    observed[name] = {"sha256": sha, "bytes": size, "mode": member.mode}
            # Consume gzip trailer and any tar padding, rejecting nonzero extras.
            while extra := uncompressed.read(CHUNK):
                old.require(not extra.strip(b"\0"), "Unexpected trailing archive content")
    finally:
        reader.close()
    old.require(isinstance(manifest, dict) and isinstance(checksums, dict), "Missing manifests")
    old.require(
        manifest.get("scope") == SCOPE and manifest.get("formal_campaign") == FORMAL,
        "Formal boundary changed",
    )
    old.require(
        observed["MANIFEST.json"]["sha256"] == index["manifest_sha256"], "Manifest identity changed"
    )
    files = manifest["files"]
    old.require(
        set(observed) == set(files) | {"MANIFEST.json", "SHA256SUMS"}, "Member inventory changed"
    )
    old.require(set(checksums) == set(files) | {"MANIFEST.json"}, "Checksum inventory changed")
    for name, expected in checksums.items():
        old.require(observed[name]["sha256"] == expected, f"Checksum mismatch: {name}")
    for name, expected in files.items():
        old.require(
            all(observed[name][key] == expected[key] for key in ("sha256", "bytes", "mode")),
            f"Manifest mismatch: {name}",
        )
    for path in parts:
        old.require(
            old.signature(old.regular_file(path)) == signatures[path.name],
            "Part changed during verification",
        )
    return {
        "scope": SCOPE,
        "archive_integrity_verified": True,
        "payload_files": len(files),
        "payload_bytes": sum(item["bytes"] for item in files.values()),
        "parts": len(parts),
        "manifest_sha256": index["manifest_sha256"],
        "numerical_qualification_claimed": False,
    }


def package(
    repo,
    evidence,
    inputs,
    binary,
    output,
    *,
    report=None,
    revisions=(),
    max_part_bytes=MAX_PART_BYTES,
    basename="gpu-v2-engineering",
):
    repo, evidence, inputs, binary = [
        old.no_symlinks(path).resolve() for path in (repo, evidence, inputs, binary)
    ]
    report = old.no_symlinks(report).resolve() if report is not None else None
    old.require(report is not None, "A finalized blocked engineering report is required")
    output = old.no_symlinks(output).resolve()
    old.safe_name(basename)
    old.require(Path(basename).name == basename, "Basename must be a single filename")
    old.require(
        not any(
            output.is_relative_to(root)
            for root in (repo, evidence, inputs, *((report,) if report else ()))
        ),
        "Archive output must be outside all input trees",
    )
    old.require(output.parent.is_dir(), "Archive parent must exist")
    old.require(not output.exists(), "Archive output already exists")
    source_state = old.git_state(repo)
    output.mkdir()
    snapshots = []
    for revision in dict.fromkeys((source_state["revision"], *revisions)):
        commit = (
            old.git_bytes(repo, ["rev-parse", "--verify", revision + "^{commit}"]).decode().strip()
        )
        destination = output / f"source-{commit}.tar"
        if destination in snapshots:
            continue
        with destination.open("xb") as handle:
            subprocess.run(
                ["git", "--no-optional-locks", "archive", "--format=tar", commit],
                cwd=repo,
                stdout=handle,
                check=True,
            )
        snapshots.append(destination)
    payload, exclusions, original_inputs = inventory(
        repo, evidence, inputs, binary, report, snapshots
    )
    files, signatures = {}, {}
    for name, item in payload.items():
        value, signatures[name] = old.file_identity(item["path"])
        files[name] = {**value, "category": item["category"]}
    report_summary = old.read_json(report / "summary.json")
    validate_report_inventory(report_summary)
    old.require(
        report_summary.get("reporter_sha256")
        == files["source/scripts/report_gpu_v2_engineering.py"]["sha256"],
        "Executed reporter source is absent or changed",
    )
    report_roots, report_bindings = bind_report_files(evidence, report_summary, files)
    validate_retained_baseline_members(evidence, report_roots["baseline"], files)
    for record in original_inputs:
        if record["seed"] == 42:
            for name in ("initial.f32", "train.i32"):
                member = str(Path(record["manifest"]).parent / name)
                old.require(
                    files[member]["sha256"] == record["recorded_original_sha256"][name],
                    "Original training operand changed",
                )
    cpu = old.read_json(evidence / "cpu-reference-receipt.json")
    for relative, expected in cpu["binary_sha256"].items():
        member = "evidence/cpu-reference/" + old.safe_name(relative)
        old.require(
            files.get(member, {}).get("sha256") == expected, "Frozen CPU binary/layout mismatch"
        )
    raw_bytes = sum(value["bytes"] for value in files.values())
    old.require(
        shutil.disk_usage(output).free > raw_bytes + 64 * CHUNK,
        "Insufficient free disk for bounded archive creation",
    )
    manifest = {
        "schema": 1,
        "scope": SCOPE,
        "formal_campaign": FORMAL,
        "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "description": (
            "Complete retained v2 evidence, including failed and superseded attempts; "
            "current and explicit historical sources"
        ),
        "source": source_state,
        "files": files,
        "original_cpu_inputs": original_inputs,
        "report_source_bindings": report_bindings,
        "report_summary_sha256": files["report/summary.json"]["sha256"],
        "excluded_paths": exclusions,
        "dependency_installed_bytes_included": False,
        "limits": [
            "Integrity verifies retained bytes, not numerical qualification or formal admission.",
            "Runtime identities are retained; installed packages and caches are not copied.",
            "Original validation/test arrays remain unread; "
            "their original manifest hashes are preserved.",
            "Current source does not replace missing historical execution identities; "
            "explicit Git source snapshots are separately labeled.",
        ],
    }
    old.require(
        len(json_bytes(manifest)) <= old.MAX_METADATA, "Archive manifest exceeds verifier bound"
    )
    index = write_split_archive(
        output, basename, max_part_bytes, payload, files, signatures, manifest
    )
    index_path = output / f"{basename}-archive.json"
    write_new(index_path, index)
    verified = verify_archive(index_path)
    after, after_exclusions, after_inputs = inventory(
        repo, evidence, inputs, binary, report, snapshots
    )
    old.require(
        after == payload and after_exclusions == exclusions and after_inputs == original_inputs,
        "Input inventory changed while packaging",
    )
    old.require(old.git_state(repo) == source_state, "Git/source state changed while packaging")
    for name, item in payload.items():
        old.require(
            old.signature(old.regular_file(item["path"])) == signatures[name],
            f"Input changed while packaging: {name}",
        )
    write_new(output / f"{basename}-manifest.json", manifest)
    verified.update(sources_and_evidence_unchanged=True, formal_campaign=FORMAL)
    write_new(output / f"{basename}-verification.json", verified)
    publication = [
        index_path,
        output / f"{basename}-manifest.json",
        output / f"{basename}-verification.json",
        *(output / part["name"] for part in index["parts"]),
    ]
    with (output / f"{basename}-SHA256SUMS.txt").open("x") as handle:
        for path in publication:
            identity, _ = old.file_identity(path)
            handle.write(f"{identity['sha256']}  {path.name}\n")
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    creation = commands.add_parser("create")
    creation.add_argument("--repo", type=Path, default=ROOT)
    creation.add_argument("--evidence", type=Path)
    creation.add_argument("--inputs", type=Path)
    creation.add_argument("--metal-binary", type=Path)
    creation.add_argument("--report", type=Path, required=True)
    creation.add_argument("--revision", action="append", default=[])
    creation.add_argument("--output", type=Path, required=True)
    creation.add_argument("--max-part-bytes", type=int, default=MAX_PART_BYTES)
    creation.add_argument("--basename", default="gpu-v2-engineering")
    check = commands.add_parser("verify")
    check.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_archive(args.index)
    else:
        result = package(
            args.repo,
            args.evidence or args.repo / "work/gpu-v2",
            args.inputs or args.repo / "results/inputs-v6",
            args.metal_binary or args.repo / "native/metal/build/libtransformer_metal.dylib",
            args.output,
            report=args.report,
            revisions=args.revision,
            max_part_bytes=args.max_part_bytes,
            basename=args.basename,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
