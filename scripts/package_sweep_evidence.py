"""Stream a completed v3 width sweep into a deterministic, verified release archive.

This utility reads saved evidence only. It performs no training or numerical audit.
Temporary disk use is one Git source tar plus the eventual compressed archive;
raw checkpoints and predictions are never copied into a staging tree or read whole.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

WIDTHS = (32, 64, 128, 256)
SEEDS = (42, 43, 44)
CHUNK = 1024 * 1024
FIXTURE_FILES = {
    "config.bin",
    "initial.f32",
    "train.i32",
    "validation.i32",
    "test.i32",
    "schedule.i32",
}
RUN_FILES = {
    "best.f32",
    "last.f32",
    "adam_m.f32",
    "adam_v.f32",
    "history.f64",
    "performance.f64",
    "timings.f64",
    "test_logits.f32",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stream_digest(handle):
    value = hashlib.sha256()
    while block := handle.read(CHUNK):
        value.update(block)
    return value.hexdigest()


def digest(path):
    with Path(path).open("rb") as handle:
        return stream_digest(handle)


def read_json(path):
    return json.loads(Path(path).read_text())


def safe_name(name):
    path = PurePosixPath(name)
    require(
        bool(name)
        and not path.is_absolute()
        and path.as_posix() == name
        and ".." not in path.parts
        and "." not in path.parts
        and not any(character in name for character in ("\n", "\r", "\0", "\\")),
        f"Unsafe archive or evidence path: {name!r}",
    )
    return name


def file_in(directory, name):
    path = Path(directory) / safe_name(name)
    require(path.is_file() and not path.is_symlink(), f"Missing or linked evidence file: {path}")
    for parent in path.parents:
        require(not parent.is_symlink(), f"Symlink in evidence path: {path}")
        if parent == Path(directory):
            break
    return path


def check_hash(path, expected):
    require(digest(path) == expected, f"Evidence hash mismatch: {path}")


def check_audit(audit, label):
    records = audit.get("checks", [])
    require(
        audit.get("passed") is True
        and bool(records)
        and audit.get("check_count") == len(records)
        and all(row.get("passed") is True for row in records)
        and audit.get("failed_checks") == []
        and audit.get("error") is None,
        f"Incomplete or failed audit: {label}",
    )


def expected_order():
    result = []
    for repeat in (1, 2):
        for width in WIDTHS if repeat == 1 else WIDTHS[::-1]:
            for index, seed in enumerate(SEEDS):
                pair = ("python", "arm64")
                if (WIDTHS.index(width) + index) % 2:
                    pair = pair[::-1]
                if repeat == 2:
                    pair = pair[::-1]
                result.extend((width, seed, repeat, impl) for impl in pair)
    return result


def identity(row, width=None):
    return (
        row["width"] if width is None else width,
        row["seed"],
        row["repeat"],
        row["implementation"],
    )


def validate_evidence(inputs, results, report, preflight_dir, repo):
    """Authenticate all reported raw bytes and the complete cross-file identity chain."""
    manifest_path = file_in(results, "manifest.json")
    manifest = read_json(manifest_path)
    summary = read_json(file_in(report, "summary.json"))
    audit = read_json(file_in(report, "audit.json"))
    check_audit(audit, "global report")
    for value in (manifest, summary):
        require(
            value.get("status") == "completed" and value.get("protocol_version") == "v3",
            "Only completed v3 studies and reports can be packaged",
        )
        require(value.get("widths") == list(WIDTHS), "Wrong width ladder")
    require(bool(manifest.get("source_sha256")), "Missing measured source hashes")
    for field in ("revision", "source_sha256", "models"):
        require(summary[field] == manifest[field], f"Global report {field} mismatch")
    require(summary["raw_manifest_sha256"] == digest(manifest_path), "Global raw manifest changed")
    require(summary["audit_check_count"] == audit["check_count"], "Global audit count mismatch")
    order = expected_order()
    require(
        [identity(row) for row in manifest["runs"]] == order, "Incomplete or reordered 48-run sweep"
    )
    require([tuple(row) for row in manifest["order"]] == order, "Declared sweep order mismatch")
    global_runs = {identity(row): row for row in summary["runs"]}
    require(
        len(summary["runs"]) == 48 and set(global_runs) == set(order),
        "Global report run identities mismatch",
    )
    require(
        summary["retained_test_prediction_rows"] == 48 * 256 * 32,
        "Global prediction count mismatch",
    )
    natives, preflights = {}, {}
    total_preflight = total_child_audit = 0
    for width in WIDTHS:
        key, tag = str(width), f"width-{width}"
        model = manifest["models"][key]
        expected_model = {
            "width": width,
            "heads": 4,
            "head_width": width // 4,
            "hidden": 2 * width,
            "context": 32,
            "vocab": 128,
            "parameters": 8 * width**2 + 301 * width + 128,
        }
        require(
            all(model.get(name) == value for name, value in expected_model.items()),
            f"Wrong model: {tag}",
        )
        selected_path = file_in(results, tag + "/manifest.json")
        selected = read_json(selected_path)
        width_report = Path(report) / tag
        child_summary = read_json(file_in(width_report, "summary.json"))
        child_audit = read_json(file_in(width_report, "audit.json"))
        check_audit(child_audit, tag)
        for value in (selected, child_summary):
            require(
                value.get("status") == "completed" and value.get("protocol_version") == "v3",
                f"Incomplete {tag}",
            )
            for field in ("revision", "source_sha256"):
                require(value[field] == manifest[field], f"{tag} {field} mismatch")
            require(value["model"] == model, f"{tag} model identity mismatch")
        require(
            selected["width"] == width
            and selected["model_width"] == width
            and selected["parameter_count"] == model["parameters"],
            f"{tag} compiled model mismatch",
        )
        selected_entries = [row for row in manifest["runs"] if row["width"] == width]
        require(selected["runs"] == selected_entries, f"{tag} runner entries mismatch")
        require(
            selected["order"] == [list(row[1:]) for row in order if row[0] == width],
            f"{tag} order mismatch",
        )
        require(
            selected["fixtures"] == manifest["fixtures"][key], f"{tag} fixture manifest mismatch"
        )
        require(selected["preflight"] == manifest["preflights"][key], f"{tag} preflight mismatch")
        require(
            child_summary["raw_manifest_sha256"] == digest(selected_path),
            f"{tag} raw manifest changed",
        )
        require(
            child_summary["audit_check_count"] == child_audit["check_count"],
            f"{tag} audit count mismatch",
        )
        require(
            child_summary["retained_test_prediction_rows"] == 12 * 256 * 32,
            f"{tag} prediction count mismatch",
        )
        aggregate = summary["by_width"][key]
        for field in (
            "model",
            "aggregate",
            "audit_check_count",
            "preflight_check_count",
            "retained_test_prediction_rows",
            "raw_manifest_sha256",
        ):
            require(
                aggregate[field] == child_summary[field], f"{tag} aggregate field mismatch: {field}"
            )
        check_hash(width_report / "summary.json", aggregate["report_summary_sha256"])
        check_hash(width_report / "audit.json", aggregate["report_audit_sha256"])
        preflight_path = file_in(preflight_dir, tag + ".json")
        preflight = read_json(preflight_path)
        require(
            preflight
            == selected["preflight"]
            == read_json(file_in(width_report, "preflight.json")),
            f"{tag} retained preflight mismatch",
        )
        check_audit(preflight, tag + " preflight")
        require(
            preflight.get("runtime_included") is True
            and preflight.get("model_width") == width
            and preflight.get("parameter_count") == model["parameters"]
            and preflight.get("model_identity")
            == {"width": width, "parameter_count": model["parameters"], "verified": True},
            f"{tag} preflight identity mismatch",
        )
        require(
            child_summary["preflight_check_count"] == preflight["check_count"],
            f"{tag} preflight count mismatch",
        )
        require(
            bool(preflight.get("source_sha256"))
            and all(
                manifest["source_sha256"].get(name) == value
                for name, value in preflight["source_sha256"].items()
            ),
            f"{tag} preflight/source mismatch",
        )
        for filename, stem in (
            ("transformer-arm64", "executable"),
            ("libtransformer.dylib", "library"),
            ("layout.inc", "generated_layout"),
        ):
            relative = f"native/build/{tag}/{filename}"
            native = file_in(repo, relative)
            recorded = Path(preflight[stem + "_path"])
            recorded = recorded if recorded.is_absolute() else Path(repo) / recorded
            require(recorded.resolve() == native.resolve(), f"{tag} native path mismatch")
            check_hash(native, preflight[stem + "_sha256"])
            natives[f"bin/{tag}/{filename}"] = native
        preflights[f"preflights/{tag}.json"] = preflight_path
        total_preflight += preflight["check_count"]
        total_child_audit += child_audit["check_count"]
        require(
            len(child_summary["runs"]) == 12
            and {identity(row, width) for row in child_summary["runs"]}
            == {identity(row) for row in selected_entries},
            f"{tag} report identities mismatch",
        )
        for seed in SEEDS:
            directory = Path(inputs) / tag / f"seed-{seed}"
            fixture = read_json(file_in(directory, "manifest.json"))
            require(
                fixture == selected["fixtures"][str(seed)],
                f"{tag}/seed-{seed} fixture identity mismatch",
            )
            require(set(fixture["sha256"]) == FIXTURE_FILES, "Fixture payload set incomplete")
            for name, expected in fixture["sha256"].items():
                check_hash(file_in(directory, name), expected)
        for run in child_summary["runs"]:
            run_key = identity(run, width)
            name = f"seed-{run['seed']}-repeat-{run['repeat']}-{run['implementation']}"
            require(
                run["name"] == run["run"] == name and run["fixture"] == f"seed-{run['seed']}",
                "Unsafe or mismatched run identity",
            )
            require(
                global_runs[run_key] == run | {"width": width, "relative_run": tag + "/" + name},
                f"Global/child report run mismatch: {tag}/{name}",
            )
            require(
                run["fixture_sha256"] == selected["fixtures"][str(run["seed"])]["sha256"],
                "Run fixture identity mismatch",
            )
            require(
                run["checkpoint_predictions_recomputed"] is True
                and run["quality"]["tokens"] == 256 * 32,
                "Missing independent checkpoint prediction audit",
            )
            directory = Path(results) / tag / name
            hashes = run["artifact_sha256"]
            actual = {path.name for path in directory.iterdir() if path.is_file()}
            require(
                RUN_FILES <= set(hashes) and actual == set(hashes),
                f"Run artifact set changed: {tag}/{name}",
            )
            for filename, expected in hashes.items():
                check_hash(file_in(directory, filename), expected)
        for row in selected_entries:
            require(
                row.get("status") == "completed"
                and row.get("returncode") == 0
                and all(
                    row.get(flag) is True
                    for flag in (
                        "conditions_valid",
                        "source_unchanged",
                        "binary_unchanged",
                        "fixtures_unchanged",
                    )
                ),
                f"Invalid measured child: {row['name']}",
            )
    require(summary["preflight_check_count"] == total_preflight, "Total preflight count mismatch")
    require(
        summary["audit_check_count"] == summary["sweep_audit_check_count"] + total_child_audit,
        "Total report audit count mismatch",
    )
    return manifest, summary, natives | preflights


def archive_source(repo, revision, destination, expected):
    require(
        len(revision) == 40 and all(char in "0123456789abcdef" for char in revision),
        "Measured revision must be a full Git commit hash",
    )
    with Path(destination).open("xb") as handle:
        subprocess.run(
            ["git", "archive", "--format=tar", revision], cwd=repo, stdout=handle, check=True
        )
    seen = set()
    with tarfile.open(destination, "r|") as archive:
        for member in archive:
            if member.isdir():
                continue
            require(
                member.isfile() and member.name not in seen,
                "Unsupported or duplicate measured-source member",
            )
            safe_name(member.name)
            require(member.name in expected, f"Unrecorded source member: {member.name}")
            handle = archive.extractfile(member)
            require(
                stream_digest(handle) == expected[member.name],
                f"Measured source hash mismatch: {member.name}",
            )
            seen.add(member.name)
    require(seen == set(expected), "Measured source archive is incomplete")


def add_tree(payload, directory, prefix):
    directory = Path(directory)
    require(
        directory.is_dir() and not directory.is_symlink(),
        f"Invalid evidence directory: {directory}",
    )
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), f"Symlink in release inputs: {path}")
        if path.is_file():
            name = safe_name(prefix + "/" + path.relative_to(directory).as_posix())
            require(name not in payload, f"Duplicate archive name: {name}")
            payload[name] = path


def write_archive(output, payload):
    hashes = {safe_name(name): digest(path) for name, path in sorted(payload.items())}
    checksums = "".join(f"{value}  {name}\n" for name, value in hashes.items()).encode()
    with Path(output).open("xb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=6, fileobj=raw, mtime=0
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.GNU_FORMAT) as archive:
                for name in sorted([*payload, "SHA256SUMS"]):
                    info = tarfile.TarInfo(name)
                    info.mode = (
                        0o755
                        if name.startswith("bin/") and name.endswith("/transformer-arm64")
                        else 0o644
                    )
                    info.mtime = info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    if name == "SHA256SUMS":
                        info.size = len(checksums)
                        archive.addfile(info, io.BytesIO(checksums))
                    else:
                        info.size = payload[name].stat().st_size
                        with payload[name].open("rb") as handle:
                            archive.addfile(info, handle)
    return hashes


def verify_archive(path, expected_payload=None):
    expected, seen = None, set()
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            safe_name(member.name)
            require(
                member.isfile() and member.name not in seen,
                "Unsupported or duplicate archive member",
            )
            seen.add(member.name)
            handle = archive.extractfile(member)
            if member.name == "SHA256SUMS":
                require(expected is None and len(seen) == 1, "Checksum manifest must be first")
                require(member.size <= 16 * CHUNK, "Unreasonably large checksum manifest")
                expected = {}
                for line in handle.read().decode().splitlines():
                    value, name = line.split("  ", 1)
                    safe_name(name)
                    require(
                        name not in expected and len(value) == 64, "Malformed checksum manifest"
                    )
                    expected[name] = value
                if expected_payload is not None:
                    require(
                        expected == expected_payload,
                        "Archive checksum manifest differs from the input inventory",
                    )
            else:
                require(
                    expected is not None and member.name in expected, "Unmanifested archive payload"
                )
                require(
                    stream_digest(handle) == expected[member.name],
                    f"Archive checksum failed: {member.name}",
                )
    require(
        expected is not None and seen == set(expected) | {"SHA256SUMS"},
        "Archive payload set is incomplete",
    )
    return expected


def package(args):
    output = Path(args.output).resolve()
    require(output.name.endswith(".tar.gz"), "Output must end in .tar.gz")
    base = output.name.removesuffix(".tar.gz")
    checksum = output.with_name(base + "-SHA256SUMS.txt")
    verification = output.with_name(base + "-verification.json")
    partial = output.with_name(output.name + ".partial")
    for path in (output, checksum, verification, partial):
        if path.exists():
            raise FileExistsError(path)
    repo = Path(args.repo).resolve()
    inputs, results, report, preflight_dir = (
        Path(value).resolve()
        for value in (args.inputs, args.results, args.report, args.preflight_dir)
    )
    for directory in (inputs, results, report, preflight_dir):
        require(
            not output.is_relative_to(directory),
            "Archive output must be outside all input evidence directories",
        )
    manifest, summary, supplemental = validate_evidence(
        inputs, results, report, preflight_dir, repo
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sweep-source-", dir=output.parent) as temporary:
        source = Path(temporary) / "measured-source.tar"
        archive_source(repo, manifest["revision"], source, manifest["source_sha256"])
        payload = {"measured-source.tar": source, **supplemental}
        add_tree(payload, inputs, "inputs")
        add_tree(payload, results, "results")
        add_tree(payload, report, "report")
        tool = Path(__file__).resolve()
        payload["packaging/package_sweep_evidence.py"] = tool
        raw_bytes = sum(path.stat().st_size for path in payload.values())
        require(
            shutil.disk_usage(output.parent).free > raw_bytes + 32 * CHUNK,
            "Insufficient free space for bounded archive creation",
        )
        # Capture all inputs before writing, then verify streamed bytes and reread inputs.
        expected = write_archive(partial, payload)
        verified = verify_archive(partial, expected)
        for name, path in payload.items():
            check_hash(path, verified[name])
        rechecked_manifest, rechecked_summary, _ = validate_evidence(
            inputs, results, report, preflight_dir, repo
        )
        require(
            rechecked_manifest == manifest and rechecked_summary == summary,
            "Evidence identities changed during packaging",
        )
        archive_hash = digest(partial)
        evidence = {
            "verified": True,
            "schema_version": 1,
            "protocol_version": "v3",
            "archive": output.name,
            "sha256": archive_hash,
            "archive_bytes": partial.stat().st_size,
            "payload_bytes": raw_bytes,
            "verified_payload_files": len(verified),
            "measured_revision": manifest["revision"],
            "widths": list(WIDTHS),
            "measured_fits": 48,
            "models": manifest["models"],
            "preflight_checks": summary["preflight_check_count"],
            "postrun_checks": summary["audit_check_count"],
            "retained_test_prediction_rows": summary["retained_test_prediction_rows"],
            "raw_manifest_sha256": digest(results / "manifest.json"),
            "global_report_sha256": digest(report / "summary.json"),
            "packaging_tool_sha256": digest(tool),
            "packaging_tool_is_measured_source": False,
            "verification_scope": (
                "All payload bytes rehashed from the archive and inputs; measured source, "
                "binary/layout, fixture, raw-run and report identity chains checked. "
                "Saved numerical audits are authenticated, not rerun."
            ),
            "determinism": (
                "Sorted members, fixed modes/owners/mtime, empty gzip filename and mtime zero; "
                "identical inputs and Python/zlib produce identical archive bytes."
            ),
        }
        # Linking publishes the verified archive without ever replacing an existing path.
        os.link(partial, output)
        partial.unlink()
        with checksum.open("x") as handle:
            handle.write(f"{archive_hash}  {output.name}\n")
        with verification.open("x") as handle:
            handle.write(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args), indent=2))


if __name__ == "__main__":
    main()
