"""Archive an authenticated completed sweep whose post-run numerical audit failed.

This separate, post-study publication tool never calls or weakens the successful
study packager's validation routine. It authenticates saved evidence through the
failed-outcome reporter, then uses the existing bounded streaming archive helpers.
Archive integrity is distinct from numerical validation: the latter remains false.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTER_PATH = Path(__file__).with_name("report_failed_sweep.py")
HELPER_PATH = Path(__file__).with_name("package_sweep_evidence.py")
ALL_WIDTHS = (32, 64, 128, 256, 512, 1024)


def load_sibling(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


archive = load_sibling(HELPER_PATH, "failed_sweep_archive_helpers")


def authenticate(args):
    reporter = load_sibling(REPORTER_PATH, "failed_sweep_reporter")
    authentication = reporter.authenticate_completed_failed_campaign(
        Path(args.results).resolve(),
        Path(args.inputs).resolve(),
        Path(args.parent_failure).resolve(),
        {512: Path(args.width_512_report).resolve(), 1024: Path(args.width_1024_report).resolve()},
        root=Path(args.repo).resolve(),
    )
    summary = reporter.validate_failed_report(Path(args.report).resolve(), authentication)
    return authentication, summary


def validate_outcome(authentication, summary):
    """Reject a success-shaped report or incomplete failure evidence before writing."""
    manifest = authentication["manifest"]
    failed = authentication["failed_checks"]
    integrity = authentication["integrity_checks"]
    archive.require(
        manifest.get("status") == "completed"
        and manifest.get("protocol_version") == "v5"
        and manifest.get("widths") == [512, 1024]
        and len(manifest.get("runs", [])) == 24,
        "Expected the completed 24-fit v5 campaign",
    )
    archive.require(
        bool(failed) and all(row.get("passed") is False for row in failed),
        "Failed-outcome archive requires retained failed numerical check records",
    )
    archive.require(
        bool(integrity) and all(row.get("passed") is True for row in integrity),
        "Evidence integrity authentication failed or is incomplete",
    )
    archive.require(
        summary.get("status") == "audit_failed"
        and summary.get("evidence_integrity_passed") is True
        and summary.get("revision") == manifest.get("revision")
        and summary.get("post_run_failed_check_count") == len(failed),
        "Failed-outcome report identity/status mismatch",
    )
    archive.require(
        authentication.get("measured_source_sha256") == manifest.get("source_sha256")
        and bool(manifest.get("source_sha256")),
        "Measured-source identity mismatch",
    )
    archive.require(
        bool(authentication.get("verified_files")), "No authenticated evidence inventory"
    )
    return manifest


def extra_directories(items):
    extras = {}
    for item in items:
        name, separator, value = item.partition("=")
        archive.require(
            separator and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name),
            "Extra evidence must be NAME=DIRECTORY with a simple unique name",
        )
        archive.require(name not in extras, "Duplicate extra-evidence name")
        directory = Path(value).resolve()
        archive.require(directory.is_dir(), "Extra-evidence directory is missing")
        extras[name] = directory
    return extras


def admitted_payload(manifest, repo, verified):
    """Keep all six models and all 18 policy gates, authenticated at admission."""
    result = {}
    regression = manifest.get("regression_preflights", {})
    archive.require(
        set(regression) == {str(width) for width in ALL_WIDTHS},
        "Missing admitted regression width",
    )
    directory = Path(manifest["preflight_directory"])
    if not directory.is_absolute():
        directory = repo / directory
    for width in ALL_WIDTHS:
        modes = regression[str(width)]
        archive.require(set(modes) == {"auto", "neon", "sme"}, "Expected all 18 policy gates")
        for mode, gate in modes.items():
            suffix = "" if mode == "auto" else "-" + mode
            name = f"width-{width}{suffix}.json"
            path = archive.file_in(directory, name).resolve()
            archive.require(str(path) in verified, f"Gate is not authenticated: {path}")
            archive.require(archive.read_json(path) == gate, f"Admission gate changed: {path}")
            result["preflights/" + name] = path
        gate = modes["auto"]
        for filename, stem in (
            ("transformer-arm64", "executable"),
            ("libtransformer.dylib", "library"),
            ("layout.inc", "generated_layout"),
        ):
            path = archive.file_in(repo, f"native/build/width-{width}/{filename}").resolve()
            recorded = Path(gate[stem + "_path"])
            recorded = recorded if recorded.is_absolute() else repo / recorded
            archive.require(recorded.resolve() == path, "Admitted binary/layout path mismatch")
            archive.require(
                verified.get(str(path)) == gate[stem + "_sha256"],
                f"Binary/layout is not authenticated: {path}",
            )
            result[f"bin/width-{width}/{filename}"] = path
    return result


def checked_inventory(authentication):
    verified = {}
    for raw_path, expected in authentication["verified_files"].items():
        path = Path(raw_path)
        archive.require(path.is_absolute(), "Authenticated evidence paths must be absolute")
        archive.require(
            isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            "Malformed authenticated file hash",
        )
        archive.require(path.is_file() and not path.is_symlink(), "Missing/linked evidence file")
        resolved = str(path.resolve())
        archive.require(resolved not in verified, "Duplicate authenticated file identity")
        verified[resolved] = expected
    return verified


def package(args):
    output = Path(args.output).resolve()
    archive.require(output.name.endswith(".tar.gz"), "Output must end in .tar.gz")
    base = output.name.removesuffix(".tar.gz")
    checksum = output.with_name(base + "-SHA256SUMS.txt")
    verification = output.with_name(base + "-verification.json")
    partial = output.with_name(output.name + ".partial")
    for path in (output, checksum, verification, partial):
        if path.exists():
            raise FileExistsError(path)
    repo = Path(args.repo).resolve()
    inputs, results, report, parent = (
        Path(value).resolve()
        for value in (args.inputs, args.results, args.report, args.parent_failure)
    )
    widths = {
        512: Path(args.width_512_report).resolve(),
        1024: Path(args.width_1024_report).resolve(),
    }
    extras = extra_directories(getattr(args, "extra_evidence", []))
    trees = {
        "inputs": inputs,
        "results": results,
        "outcome-report": report,
        "original-failed-reports/parent": parent,
        **{f"original-failed-reports/width-{width}": path for width, path in widths.items()},
        **{"development/" + name: path for name, path in extras.items()},
    }
    for directory in trees.values():
        archive.require(directory.is_dir(), f"Evidence directory missing: {directory}")
        archive.require(
            not output.is_relative_to(directory),
            "Archive output must be outside all input evidence directories",
        )
    authentication, summary = authenticate(args)
    manifest = validate_outcome(authentication, summary)
    verified = checked_inventory(authentication)
    supplemental = admitted_payload(manifest, repo, verified)
    tool = Path(__file__).resolve()
    for path in (tool, REPORTER_PATH):
        archive.require(
            not archive.tool_is_measured_source(path, repo, manifest["source_sha256"]),
            "Failed-outcome tools must be identified as post-study tools",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="failed-sweep-source-", dir=output.parent) as temp:
        source = Path(temp) / "measured-source.tar"
        archive.archive_source(repo, manifest["revision"], source, manifest["source_sha256"])
        payload = {"measured-source.tar": source, **supplemental}
        for prefix, directory in trees.items():
            archive.add_tree(payload, directory, prefix)
        payload.update(
            {
                "packaging/package_failed_sweep.py": tool,
                "packaging/report_failed_sweep.py": REPORTER_PATH,
                "packaging/package_sweep_evidence.py": HELPER_PATH,
            }
        )
        archived_paths = {str(path.resolve()) for path in payload.values()}
        archive.require(
            set(verified) <= archived_paths,
            "An authenticated raw evidence file would be omitted from the archive",
        )
        metadata = {
            "schema_version": 1,
            "status": "audit_failed",
            "outcome": "audit_failed",
            "numerical_validation_passed": False,
            "integrity_verified": True,
            "integrity_scope": (
                "Saved evidence identities and archive payload bytes; no numerical pass claimed."
            ),
            "protocol_version": "v5",
            "measured_revision": manifest["revision"],
            "widths": manifest["widths"],
            "regression_widths": list(ALL_WIDTHS),
            "measured_fits": 24,
            "post_run_failed_check_count": len(authentication["failed_checks"]),
            "legacy_diagnostic_count": authentication.get("legacy_diagnostic_count"),
            "legacy_failed_diagnostic_count": authentication.get("legacy_failed_diagnostic_count"),
            "gate_counts": authentication.get("gate_counts"),
            "raw_manifest_sha256": archive.digest(results / "manifest.json"),
            "outcome_summary_sha256": archive.digest(report / "summary.json"),
            "packager_is_measured_source": False,
            "reporter_is_measured_source": False,
            "post_study_tool_sha256": {
                "packaging/package_failed_sweep.py": archive.digest(tool),
                "packaging/report_failed_sweep.py": archive.digest(REPORTER_PATH),
            },
            "archive_helper_sha256": archive.digest(HELPER_PATH),
            "archive_helper_is_measured_source": archive.tool_is_measured_source(
                HELPER_PATH, repo, manifest["source_sha256"]
            ),
            "development_extras": sorted(extras),
            "scope": (
                "The campaign completed, but its post-run numerical checkpoint audit failed. "
                "Original false check records are preserved. Descriptive measurements do not "
                "establish an admitted implementation speedup or numerical equivalence. "
                "No training, model inference, or numerical audit was rerun by this packager."
            ),
        }
        metadata_path = Path(temp) / "outcome.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        payload["outcome.json"] = metadata_path
        raw_bytes = sum(path.stat().st_size for path in payload.values())
        archive.require(
            shutil.disk_usage(output.parent).free > raw_bytes + 32 * archive.CHUNK,
            "Insufficient free space for bounded archive creation",
        )
        expected = archive.write_archive(partial, payload)
        by_path = {str(path.resolve()): expected[name] for name, path in payload.items()}
        for path, recorded_hash in verified.items():
            archive.require(
                by_path[path] == recorded_hash, f"Evidence changed after authentication: {path}"
            )
        archive.verify_archive(partial, expected)
        for name, path in payload.items():
            archive.check_hash(path, expected[name])
        for prefix, directory in trees.items():
            current = {}
            archive.add_tree(current, directory, prefix)
            original = {
                name: path for name, path in payload.items() if name.startswith(prefix + "/")
            }
            archive.require(current == original, "Evidence file set changed during packaging")
        evidence = metadata | {
            "archive": output.name,
            "sha256": archive.digest(partial),
            "archive_bytes": partial.stat().st_size,
            "payload_bytes": raw_bytes,
            "verified_payload_files": len(expected),
            "authenticated_original_files": len(verified),
            "determinism": (
                "Sorted members and fixed gzip/tar metadata; "
                "same inputs and Python/zlib give same bytes."
            ),
        }
        os.link(partial, output)
        partial.unlink()
        with checksum.open("x") as handle:
            handle.write(f"{evidence['sha256']}  {output.name}\n")
        with verification.open("x") as handle:
            handle.write(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    for name in (
        "inputs",
        "results",
        "parent-failure",
        "width-512-report",
        "width-1024-report",
        "report",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--extra-evidence", action="append", default=[], metavar="NAME=DIRECTORY")
    print(json.dumps(package(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
