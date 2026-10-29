"""Package the GPU-v3 QKV development release after every evidence writer stops.

Only new-phase evidence, source, selected training inputs, and the runtime receipt
are copied. Historical v2 evidence is linked. No framework imports, model work,
admission access, held-out input reads, dependency installation, or Git mutation
occur. Output is a verified, bounded sequence of parts of one gzip/tar stream.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import package_gpu_development as old  # noqa: E402
from scripts import package_gpu_v2_engineering as v2  # noqa: E402

SCOPE = "gpu_v3_qkv_blocked_engineering_release"
SEARCH_SCOPE = "qkv_training_partition_development_only"
WIDTHS = (32, 64, 128, 256, 512, 1024)
MAX_PART_BYTES = v2.MAX_PART_BYTES
FORMAL = {
    "planned_fits": 180,
    "completed_fits": 0,
    "status": "not_admitted",
    "admitted": False,
    "numerical_gate_waived": False,
    "reserved_admission_stream_opened": False,
    "heldout_arrays_read_by_packager": False,
}
HISTORICAL_V2 = {
    "release_tag": "transformer-gpu-engineering-v2",
    "release_url": (
        "https://github.com/logannye/arm64-transformer/releases/tag/transformer-gpu-engineering-v2"
    ),
    "scope": "Historical evidence is linked, not recopied, rerun, or requalified.",
}
SOURCE_ROOTS = {"transformer", "native", "scripts", "tests", ".github"}
SOURCE_FILES = {
    "experiments/PROTOCOL-gpu-v3-qkv.md",
    "experiments/gpu-v2/policies/selected.json",
    "experiments/gpu-v2/blocked-study.json",
}
BINARY_PATH = "native/metal/build/libtransformer_metal.dylib"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
json_bytes, write_new, tree_files = v2.json_bytes, v2.write_new, v2.tree_files


def source_member(relative):
    """Select source and phase metadata, never a historical evidence/input tree."""
    old.safe_name(relative)
    parts = Path(relative).parts
    if any(part in v2.EXCLUDED | {".git", "build"} for part in parts):
        return False
    if len(parts) == 1:
        return Path(relative).suffix in {".md", ".toml", ".txt", ".yml", ".yaml"} or relative in {
            "Makefile",
            "LICENSE",
            ".gitignore",
            ".gitattributes",
        }
    return (
        parts[0] in SOURCE_ROOTS
        or relative in SOURCE_FILES
        or parts[:2] == ("experiments", "gpu-v3-qkv")
    )


def input_inventory(inputs):
    """Enumerate two known training files explicitly; never inspect held-out files."""
    result = []
    for width in WIDTHS:
        root = Path(inputs) / f"width-{width}/seed-42"
        manifest_path = root / "manifest.json"
        manifest = old.read_json(manifest_path)
        old.require(
            manifest.get("config", {}).get("width") == width
            and manifest["config"].get("seed") == 42,
            "Development input manifest identity differs",
        )
        hashes = {name: manifest["sha256"][name] for name in ("initial.f32", "train.i32")}
        old.require(all(HEX64.fullmatch(h) for h in hashes.values()), "Invalid training hashes")
        result.append(
            {
                "width": width,
                "seed": 42,
                "directory": root,
                "manifest": manifest_path,
                "sha256": hashes,
            }
        )
    return result


def inventory(repo, evidence, inputs, binary, runtime, report, snapshots=()):
    payload, exclusions = {}, []

    def add(name, path, category):
        old.safe_name(name)
        old.require(name not in payload, f"Duplicate member: {name}")
        old.regular_file(path)
        payload[name] = {"path": Path(path), "category": category}

    names = old.git_bytes(repo, ["ls-files", "-z", "--cached", "--others", "--exclude-standard"])
    for raw in sorted(set(names.split(b"\0")) - {b""}):
        relative = raw.decode()
        if source_member(relative):
            add("source/" + relative, repo / relative, "current_working_tree_source")
    for prefix, directory, category in (
        ("evidence", evidence, "all_retained_new_phase_evidence_including_failures"),
        ("report", report, "completed_development_report"),
    ):
        paths, omitted = tree_files(directory)
        exclusions.extend(f"{prefix}/{p}" for p in omitted)
        for path in paths:
            add(f"{prefix}/{path.relative_to(directory).as_posix()}", path, category)
    add("runtime/libtransformer_metal.dylib", binary, "current_metal_binary")
    add("runtime/runtime-identity.json", runtime, "measurement_runtime_identity")
    inputs_inventory = input_inventory(inputs)
    for item in inputs_inventory:
        prefix = f"training-inputs/width-{item['width']}/seed-42"
        add(prefix + "/manifest.json", item["manifest"], "original_input_manifest")
        for name in ("initial.f32", "train.i32"):
            add(prefix + "/" + name, item["directory"] / name, "training_development_input")
    for path in snapshots:
        add("source-snapshots/" + path.name, path, "filtered_immutable_git_source_snapshot")
    for relative in (*SOURCE_FILES, "scripts/package_gpu_qkv.py", "scripts/report_gpu_qkv.py"):
        old.require("source/" + relative in payload, f"Missing release source: {relative}")
    return dict(sorted(payload.items())), sorted(exclusions), inputs_inventory


def validate_report(summary):
    old.require(
        summary.get("scope") == SEARCH_SCOPE
        and summary.get("status") == "completed_development_report",
        "A completed QKV development report is required",
    )
    old.require(
        summary.get("formal_campaign_allowed") is False
        and summary.get("formal_fits") == 0
        and summary.get("framework_or_cpu_speed_comparisons_claimed") is False,
        "Formal-study boundary differs",
    )
    old.require(
        summary.get("numerically_qualified_development_result") is True
        and isinstance(summary.get("repeatability"), dict)
        and summary.get("baseline_feasibility", {}).get("reserved_admission_opened") is False,
        "Final development repeatability or closed admission evidence missing",
    )
    widths = summary.get("widths", [])
    old.require(
        len(widths) == len(WIDTHS) and {row.get("width") for row in widths} == set(WIDTHS),
        "Report requires exactly six widths",
    )
    old.require(bool(summary.get("evidence_sha256")), "Report lacks evidence bindings")
    return widths


def bind_report(summary, payload, files, roots):
    """Every report hash must identify a retained member, including raw replay arrays."""
    by_path = {item["path"].resolve(): name for name, item in payload.items()}
    bindings = {}
    for key, expected in summary["evidence_sha256"].items():
        role, relative = old.safe_name(key).split("/", 1)
        old.require(role in roots, f"Unknown report evidence role: {role}")
        old.safe_name(relative)
        path = roots[role] / relative
        old.no_symlinks(path)
        member = by_path.get(path.resolve())
        old.require(
            member is not None and files[member]["sha256"] == expected,
            f"Report evidence is absent or changed: {key}",
        )
        bindings[key] = member
    return bindings


def measured_ledgers(evidence):
    """Only frozen candidate ledgers identify measured commits; smoke remains evidence."""
    records = []
    paths, _ = tree_files(evidence)
    for path in paths:
        if not path.name.endswith(".ledger.json"):
            continue
        value = old.read_json(path)
        old.require(value.get("scope") == SEARCH_SCOPE, "Unexpected candidate-ledger scope")
        old.require(
            value.get("frozen_before_performance_scoring") is True
            and HEX40.fullmatch(value.get("source_revision", ""))
            and isinstance(value.get("source_sha256"), dict)
            and bool(value["source_sha256"]),
            "Unbound measured candidate ledger",
        )
        for relative, expected in value["source_sha256"].items():
            old.safe_name(relative)
            old.require(HEX64.fullmatch(expected), "Invalid measured source hash")
            old.require(
                relative == BINARY_PATH or source_member(relative),
                f"Measured source is outside the explicit source inventory: {relative}",
            )
        records.append(
            {
                "member": "evidence/" + path.relative_to(evidence).as_posix(),
                "revision": value["source_revision"],
                "source_sha256": value["source_sha256"],
            }
        )
    old.require(bool(records), "No frozen measured candidate ledgers")
    return records


def snapshot(repo, revision, output):
    """Archive source-only paths at an exact commit, with no results/input checkout."""
    old.require(HEX40.fullmatch(revision), "Expected a full source commit")
    resolved = (
        old.git_bytes(repo, ["rev-parse", "--verify", revision + "^{commit}"]).decode().strip()
    )
    old.require(resolved == revision, "Source revision did not resolve exactly")
    selected = []
    for record in old.git_bytes(repo, ["ls-tree", "-r", "-z", revision]).split(b"\0"):
        if not record:
            continue
        metadata, raw_name = record.split(b"\t", 1)
        name = raw_name.decode()
        if source_member(name):
            mode, kind, _ = metadata.decode().split()
            old.require(kind == "blob" and mode in ("100644", "100755"), "Unsafe source snapshot")
            selected.append(name)
    old.require(bool(selected), "Empty source snapshot")
    path = output / f"source-{revision}.tar"
    with path.open("xb") as handle:
        subprocess.run(
            [
                "git",
                "--no-optional-locks",
                "--literal-pathspecs",
                "archive",
                "--format=tar",
                revision,
                "--",
                *selected,
            ],
            cwd=repo,
            stdout=handle,
            check=True,
        )
    return path


def bind_sources(repo, ledgers, payload, files):
    """Bind immutable source bytes and retained binaries without executing either."""
    binaries = {
        record["sha256"]: name
        for name, record in files.items()
        if payload[name]["path"].suffix == ".dylib"
    }
    verified, snapshot_hashes, bindings = {}, {}, []
    for ledger in ledgers:
        commit, bound = ledger["revision"], {}
        old.require(
            "source-snapshots/source-" + commit + ".tar" in files,
            "Measured source snapshot missing",
        )
        snapshot_name = f"source-snapshots/source-{commit}.tar"
        if commit not in snapshot_hashes:
            hashes = {}
            with tarfile.open(payload[snapshot_name]["path"], mode="r|") as archive:
                for member in archive:
                    name = old.safe_name(member.name.rstrip("/"))
                    if member.isdir():
                        continue
                    old.require(
                        member.isfile()
                        and not member.issparse()
                        and name not in hashes
                        and source_member(name),
                        "Unsafe or out-of-scope snapshot member",
                    )
                    handle = archive.extractfile(member)
                    old.require(handle is not None, "Missing snapshot content")
                    digest, size = old.stream_digest(handle)
                    old.require(size == member.size, "Truncated source snapshot")
                    hashes[name] = digest
            snapshot_hashes[commit] = hashes
        for relative, expected in ledger["source_sha256"].items():
            if relative == BINARY_PATH:
                old.require(expected in binaries, "Measured Metal binary missing")
                bound[relative] = binaries[expected]
            else:
                key = (commit, relative)
                if key not in verified:
                    verified[key] = hashlib.sha256(
                        old.git_bytes(repo, ["show", f"{commit}:{relative}"])
                    ).hexdigest()
                old.require(
                    verified[key] == expected, f"Measured commit/source mismatch: {relative}"
                )
                old.require(
                    snapshot_hashes[commit].get(relative) == expected,
                    f"Measured snapshot/source mismatch: {relative}",
                )
                bound[relative] = {
                    "snapshot": f"source-snapshots/source-{commit}.tar",
                    "path": relative,
                }
        bindings.append({"ledger": ledger["member"], "revision": commit, "sources": bound})
    return bindings


def bind_selected_searches(summary, searches, files, report_bindings):
    """Prevent a genuine archive from silently retaining substituted summary rows."""
    for row in summary["widths"]:
        relative = f"width-{row['width']}.json"
        result = old.read_json(searches / relative)
        old.require(
            result.get("scope") == SEARCH_SCOPE
            and result.get("status") == "completed"
            and result.get("width") == row["width"]
            and result.get("formal_fits") == 0
            and result.get("admission_stream_opened") is False,
            "Selected search is incomplete or outside development scope",
        )
        for key in (
            "source_revision",
            "source_sha256",
            "runtime_identity",
            "input_sha256",
            "selected_policy",
            "incumbent_policy",
        ):
            old.require(result.get(key) == row.get(key), f"Report/search identity differs: {key}")
        ledger = old.read_json(searches / (relative + ".ledger.json"))
        for key in ("source_revision", "source_sha256", "runtime_identity", "input_sha256"):
            old.require(
                result.get(key) == ledger.get(key), f"Result/ledger identity differs: {key}"
            )
        for suffix in ("", ".ledger.json"):
            member = report_bindings.get("search/" + relative + suffix)
            old.require(member in files, "Selected search or ledger is not report-bound")
            if suffix:
                old.require(
                    files[member]["sha256"] == result.get("ledger_sha256"),
                    "Selected ledger hash differs",
                )


def bind_runtime_inputs(summary, files):
    runtime_hash = files["runtime/runtime-identity.json"]["sha256"]
    for row in summary["widths"]:
        old.require(
            row.get("runtime_identity", {}).get("inventory_sha256") == runtime_hash,
            "Measured runtime inventory differs",
        )
        old.require(
            set(row.get("input_sha256", {})) == {"initial.f32", "train.i32"},
            "Measured input inventory differs",
        )
        for name, expected in row["input_sha256"].items():
            member = f"training-inputs/width-{row['width']}/seed-42/{name}"
            old.require(files[member]["sha256"] == expected, "Measured training input changed")


def write_split_archive(output, basename, limit, payload, files, signatures, manifest):
    # This existing streaming writer embeds the caller's manifest verbatim. Only
    # its index scope is v2-specific; assign the new scope before publishing it.
    index = v2.write_split_archive(output, basename, limit, payload, files, signatures, manifest)
    index["scope"] = SCOPE
    return index


def verify_archive(index_path):
    """Check every part and member without extracting or loading arrays."""
    index_path = old.no_symlinks(index_path).resolve()
    index = old.read_json(index_path)
    old.require(index.get("scope") == SCOPE, "Not a QKV engineering archive")
    old.require(
        index.get("format") == "ordered_byte_parts_of_one_tar_gzip_stream", "Unknown archive format"
    )
    parts, signatures, names = [], {}, set()
    for position, record in enumerate(index.get("parts", []), 1):
        name = old.safe_name(record["name"])
        old.require(
            Path(name).name == name
            and name not in names
            and name.endswith(f".tar.gz.part-{position:05}"),
            "Invalid or reordered archive part",
        )
        names.add(name)
        path = index_path.parent / name
        actual, signatures[name] = old.file_identity(path)
        old.require(
            actual["sha256"] == record["sha256"] and actual["bytes"] == record["bytes"],
            "Archive part changed",
        )
        old.require(
            0 < actual["bytes"] <= min(index["max_part_bytes"], MAX_PART_BYTES), "Oversized part"
        )
        parts.append(path)
    old.require(bool(parts), "Missing archive parts")
    reader = v2.JoinedReader(parts)
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
            while extra := uncompressed.read(old.CHUNK):
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
        "payload_bytes": sum(v["bytes"] for v in files.values()),
        "parts": len(parts),
        "manifest_sha256": index["manifest_sha256"],
        "numerical_qualification_claimed": False,
    }


def package(
    repo,
    evidence,
    inputs,
    binary,
    runtime,
    report,
    searches,
    repeatability,
    output,
    *,
    max_part_bytes=MAX_PART_BYTES,
    basename="gpu-v3-qkv",
):
    repo, evidence, inputs, binary, runtime, report, searches, repeatability, output = [
        old.no_symlinks(p).resolve()
        for p in (repo, evidence, inputs, binary, runtime, report, searches, repeatability, output)
    ]
    old.require(
        evidence == repo / "work/gpu-v3-qkv", "Only the new-phase evidence namespace is allowed"
    )
    old.require(
        searches.is_relative_to(evidence) and repeatability.is_relative_to(evidence),
        "Evidence roots escaped new phase",
    )
    old.require(
        not any(output.is_relative_to(r) for r in (repo, evidence, inputs, report)),
        "Archive output must be outside all input trees",
    )
    old.require(
        output.parent.is_dir() and not output.exists(),
        "Archive output must be new with an existing parent",
    )
    old.safe_name(basename)
    old.require(Path(basename).name == basename, "Basename must be a single filename")
    source_state = old.git_state(repo)
    summary = old.read_json(report / "results.json")
    validate_report(summary)
    ledgers = measured_ledgers(evidence)
    output.mkdir()
    revisions = sorted({source_state["revision"], *(r["revision"] for r in ledgers)})
    snapshots = [snapshot(repo, revision, output) for revision in revisions]
    payload, exclusions, original_inputs = inventory(
        repo, evidence, inputs, binary, runtime, report, snapshots
    )
    files, signatures = {}, {}
    for name, item in payload.items():
        identity, signatures[name] = old.file_identity(item["path"])
        files[name] = {**identity, "category": item["category"]}
    roots = {"search": searches, "repeatability": repeatability, "source": repo, "historical": repo}
    report_bindings = bind_report(summary, payload, files, roots)
    bind_selected_searches(summary, searches, files, report_bindings)
    source_bindings = bind_sources(repo, ledgers, payload, files)
    bind_runtime_inputs(summary, files)
    retained_inputs = []
    for item in original_inputs:
        prefix = f"training-inputs/width-{item['width']}/seed-42"
        for name, expected in item["sha256"].items():
            old.require(
                files[prefix + "/" + name]["sha256"] == expected, "Original training input changed"
            )
        retained_inputs.append(
            {
                "width": item["width"],
                "seed": 42,
                "manifest": prefix + "/manifest.json",
                "sha256": item["sha256"],
            }
        )
    raw_bytes = sum(f["bytes"] for f in files.values())
    old.require(
        shutil.disk_usage(output).free > raw_bytes + 64 * old.CHUNK,
        "Insufficient free disk for bounded archive creation",
    )
    manifest = {
        "schema": 1,
        "scope": SCOPE,
        "formal_campaign": FORMAL,
        "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "historical_gpu_v2": HISTORICAL_V2,
        "source": source_state,
        "files": files,
        "measured_source_bindings": source_bindings,
        "report_source_bindings": report_bindings,
        "report_results_sha256": files["report/results.json"]["sha256"],
        "runtime_inventory": {
            "member": "runtime/runtime-identity.json",
            "measurement_path": "work/gpu-v2/runtime-identity.json",
            "installed_runtime_bytes_revalidated_by_packager": False,
        },
        "training_inputs": retained_inputs,
        "excluded_paths": exclusions,
        "dependency_installed_bytes_included": False,
        "limits": [
            "Integrity verification does not re-run or establish numerical qualification.",
            "All new-phase failure evidence is preserved; "
            "source snapshots bind frozen measured ledgers only.",
            "Source snapshots are filtered source trees, not full repository archives.",
            "Runtime package identities are retained; "
            "installed package bytes and caches are excluded.",
            "Original validation/test arrays and reserved admission operands are never opened.",
            "The linked historical CPU/GPU releases and their limitations remain unchanged.",
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
    after = inventory(repo, evidence, inputs, binary, runtime, report, snapshots)
    old.require(
        after == (payload, exclusions, original_inputs), "Input inventory changed while packaging"
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
        *(output / p["name"] for p in index["parts"]),
    ]
    with (output / f"{basename}-SHA256SUMS.txt").open("x") as handle:
        for path in publication:
            identity, _ = old.file_identity(path)
            handle.write(f"{identity['sha256']}  {path.name}\n")
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser(
        "create", help="Run only after all evidence writers and scoring stop"
    )
    create.add_argument("--repo", type=Path, default=ROOT)
    create.add_argument("--evidence", type=Path, default=ROOT / "work/gpu-v3-qkv")
    create.add_argument("--inputs", type=Path, default=ROOT / "results/inputs-v6")
    create.add_argument("--metal-binary", type=Path, default=ROOT / BINARY_PATH)
    create.add_argument("--runtime", type=Path, default=ROOT / "work/gpu-v2/runtime-identity.json")
    create.add_argument("--report", type=Path, default=ROOT / "results/gpu-v3-qkv")
    create.add_argument("--searches", type=Path, required=True)
    create.add_argument("--repeatability", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--max-part-bytes", type=int, default=MAX_PART_BYTES)
    create.add_argument("--basename", default="gpu-v3-qkv")
    verify = commands.add_parser("verify")
    verify.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_archive(args.index)
    else:
        result = package(
            args.repo,
            args.evidence,
            args.inputs,
            args.metal_binary,
            args.runtime,
            args.report,
            args.searches,
            args.repeatability,
            args.output,
            max_part_bytes=args.max_part_bytes,
            basename=args.basename,
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
