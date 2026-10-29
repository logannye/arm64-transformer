"""Create a compact repository view of an unchanged, complete v6 report.

This is a publication projection, not a numerical re-audit. The full report is
preserved for the release archive. No qualification rules or outcomes change.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse

HELPER_PATH = Path(__file__).with_name("package_sweep_evidence.py")
HELPER_SPEC = importlib.util.spec_from_file_location("publication_evidence_checks", HELPER_PATH)
evidence = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(evidence)

RECORD_ARRAYS = {"checks", "numerical_checks", "legacy_diagnostics", "diagnostics"}
SCOPE = (
    "Compact publication view only; complete check records are omitted here and retained "
    "unchanged in the linked release archive. Counts, failures, qualification flags and "
    "metrics are copied from the full report. This projection performs no model execution."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compact(value):
    """Remove only full check-record arrays, retaining their summaries and failures."""
    if isinstance(value, dict):
        output = {}
        omitted = {}
        for key, item in value.items():
            if key in RECORD_ARRAYS and isinstance(item, list):
                omitted[key] = len(item)
            else:
                output[key] = compact(item)
        if omitted:
            output["omitted_full_record_counts"] = omitted
        return output
    if isinstance(value, list):
        return [compact(item) for item in value]
    return value


def source_pointer(relative, sha256, release_url):
    return {
        "archive_path": "report/" + relative,
        "sha256": sha256,
        "release_url": release_url,
    }


def project_json(value, relative, sha256, release_url):
    projected = compact(value)
    if isinstance(projected, dict):
        projected["publication"] = {
            "artifact_type": "audit_index"
            if relative.endswith("audit.json")
            else "compact_report_view",
            "complete_check_records_present": False,
            "scope": SCOPE,
            "full_source": source_pointer(relative, sha256, release_url),
        }
    else:
        # Process manifests are lists, so keep their original data structure and
        # rely on the publication receipt for their authenticated source mapping.
        require(isinstance(projected, list), "Unexpected report JSON value")
    return projected


def publication_readme(text, release_url):
    """Keep conclusions and tables; clarify links whose targets are now indices."""
    text = text.replace(
        "their full numerical evidence are retained in [summary.json](summary.json) and\n"
        "[audit.json](audit.json).",
        "their per-run metrics and numerical counts appear in [summary.json](summary.json).\n"
        "[audit.json](audit.json) indexes the complete numerical evidence in the release archive.",
    )
    text = text.replace("for every post-run check", "for the index of complete post-run evidence")
    text = text.replace(
        "each audit's `numerical_checks`", "each archived audit's `numerical_checks`"
    )
    text = text.replace(
        "checks and their diagnostics are retained under ",
        "checks and their diagnostics are indexed under ",
    )
    text = text.replace(": full report]", ": detailed report]")
    notice = (
        "**Publication view:** this directory contains compact metrics and audit indices. "
        "Full correctness checks, numerical failure records, admission gates, and raw evidence "
        f"are retained in the [release archive]({release_url}). Each JSON index records the "
        "SHA-256 and archive path of its complete source. Numerical outcomes and qualification "
        "statements below are unchanged.\n\n"
    )
    heading, separator, rest = text.partition("\n")
    return heading + "\n\n" + notice + rest.lstrip("\n") if separator else text + "\n\n" + notice


def check_report_pair(summary, audit, *, width=None):
    evidence.check_audit(audit, "publication source", protocol="v6")
    require(
        summary.get("protocol_version") == "v6", "Only v6 reports have this publication contract"
    )
    require(
        all(summary.get(key) == audit.get(key) for key in evidence.V6_FLAGS),
        "Source summary and audit disagree on qualification",
    )
    require(
        summary.get("audit_check_count") == audit["check_count"]
        and summary.get("numerical_check_count") == audit["numerical_check_count"],
        "Source summary and audit counts differ",
    )
    expected_runs = 72 if width is None else 12
    require(
        len(summary.get("runs", [])) == expected_runs
        and audit["numerical_check_count"] == expected_runs * 2050,
        "Source publication report is incomplete",
    )
    if width is None:
        require(
            summary.get("widths") == list(evidence.REGRESSION_WIDTHS)
            and set(summary.get("by_width", {})) == {str(w) for w in evidence.REGRESSION_WIDTHS},
            "Source publication report has the wrong width ladder",
        )
    else:
        require(summary.get("model", {}).get("width") == width, "Source width report mismatch")


def publish(report, output, release_url):
    report, output = Path(report).resolve(), Path(output).resolve()
    require(report.is_dir(), "Full report directory is missing")
    require(not output.exists(), "Publication destination already exists")
    require(not output.is_relative_to(report), "Publication output cannot be inside its source")
    parsed = urlparse(release_url)
    require(
        parsed.scheme == "https"
        and parsed.netloc == "github.com"
        and "/releases/tag/" in parsed.path
        and not parsed.query
        and not parsed.fragment
        and not any(character.isspace() for character in release_url),
        "Use a direct HTTPS GitHub release-tag URL",
    )
    sources = {}
    for path in sorted(report.rglob("*")):
        require(not path.is_symlink(), "Symlinks are not admitted in full report evidence")
        if path.is_file():
            relative = path.relative_to(report).as_posix()
            sources[relative] = {"sha256": evidence.digest(path), "bytes": path.stat().st_size}
    require("summary.json" in sources and "audit.json" in sources, "Incomplete full report")
    summary, audit = (evidence.read_json(report / name) for name in ("summary.json", "audit.json"))
    check_report_pair(summary, audit)
    for width in evidence.REGRESSION_WIDTHS:
        prefix = f"width-{width}"
        child_summary = evidence.read_json(evidence.file_in(report, prefix + "/summary.json"))
        child_audit = evidence.read_json(evidence.file_in(report, prefix + "/audit.json"))
        check_report_pair(child_summary, child_audit, width=width)
        recorded = summary["by_width"][str(width)]
        require(
            recorded.get("report_summary_sha256") == sources[prefix + "/summary.json"]["sha256"]
            and recorded.get("report_audit_sha256") == sources[prefix + "/audit.json"]["sha256"],
            "Full child report differs from its recorded identity",
        )
    del audit, child_audit, child_summary
    output.mkdir(parents=True, exist_ok=False)
    for relative, identity in sources.items():
        source = report / relative
        destination = output / relative
        if source.suffix.lower() == ".json":
            destination.parent.mkdir(parents=True, exist_ok=True)
            value = project_json(
                evidence.read_json(source), relative, identity["sha256"], release_url
            )
            destination.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        elif source.name == "README.md":
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(publication_readme(source.read_text(), release_url))
        elif source.suffix.lower() == ".png":
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    for relative, identity in sources.items():
        require(
            evidence.digest(report / relative) == identity["sha256"],
            "Full report changed during publication projection",
        )
    outputs = {
        path.relative_to(output).as_posix(): {
            "sha256": evidence.digest(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    receipt = {
        "schema_version": 1,
        "artifact_type": "compact_publication_receipt",
        "scope": SCOPE,
        "protocol_version": "v6",
        "measured_revision": summary["revision"],
        "release_url": release_url,
        "qualification": {key: summary[key] for key in evidence.V6_FLAGS},
        "full_report_files": sources,
        "full_summary_sha256": sources["summary.json"]["sha256"],
        "full_audit_sha256": sources["audit.json"]["sha256"],
        "compact_output_files": outputs,
        "output_hashes_exclude": ["publication.json"],
        "publisher_sha256": evidence.digest(Path(__file__)),
        "evidence_checker_sha256": evidence.digest(HELPER_PATH),
    }
    (output / "publication.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-url", required=True)
    args = parser.parse_args()
    result = publish(args.report, args.output, args.release_url)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "files": len(result["compact_output_files"]),
                "qualification": result["qualification"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
