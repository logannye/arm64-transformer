"""Package and independently checksum a completed matched study for release."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def package(args):
    output = args.output.resolve()
    checksum = output.with_name(output.name.removesuffix(".tar.gz") + "-SHA256SUMS.txt")
    verification = output.with_name(output.name.removesuffix(".tar.gz") + "-verification.json")
    for path in (output, checksum, verification):
        if path.exists():
            raise FileExistsError(path)
    manifest = json.loads((args.results / "manifest.json").read_text())
    summary = json.loads((args.report / "summary.json").read_text())
    preflight = json.loads(args.preflight.read_text())
    if manifest["status"] != "completed" or summary["status"] != "completed":
        raise ValueError("Only completed, independently audited studies can be packaged")
    if not preflight["passed"] or not preflight["runtime_included"]:
        raise ValueError("Passing numerical/runtime preflight required")
    if preflight != manifest["preflight"]:
        raise ValueError("Preflight differs from the one used by the measured runner")
    if summary["raw_manifest_sha256"] != digest((args.results / "manifest.json").read_bytes()):
        raise ValueError("Raw results changed after report generation")
    if summary["revision"] != manifest["revision"]:
        raise ValueError("Mismatched measured source revisions")
    for run in summary["runs"]:
        for name, expected in run["artifact_sha256"].items():
            if digest((args.results / run["run"] / name).read_bytes()) != expected:
                raise ValueError(f"Audited artifact changed: {run['run']}/{name}")
        for name, expected in run["fixture_sha256"].items():
            if digest((args.inputs / run["fixture"] / name).read_bytes()) != expected:
                raise ValueError(f"Audited input changed: {run['fixture']}/{name}")
    root = Path(__file__).resolve().parents[1]
    model_width = manifest.get("model", {}).get("width", 32)
    source = subprocess.check_output(
        ["git", "archive", "--format=tar", manifest["revision"]], cwd=root
    )
    with tarfile.open(fileobj=io.BytesIO(source)) as archive:
        for name, expected in manifest["source_sha256"].items():
            if digest(archive.extractfile(name).read()) != expected:
                raise ValueError(f"Source snapshot does not authenticate {name}")
    payload = {"measured-source.tar": source, "preflight.json": args.preflight.read_bytes()}
    for name, key in (
        ("transformer-arm64", "executable_sha256"),
        ("libtransformer.dylib", "library_sha256"),
    ):
        path = root / f"native/build/width-{model_width}" / name
        if not path.exists():
            path = root / "native/build" / name
        data = path.read_bytes()
        if digest(data) != preflight[key]:
            raise ValueError(f"Native binary changed: {name}")
        payload["bin/" + name] = data

    def add_tree(directory, prefix):
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Symlink is not allowed in release inputs: {path}")
            if path.is_file():
                name = prefix + "/" + path.relative_to(directory).as_posix()
                if name in payload:
                    raise ValueError(f"Duplicate archive name: {name}")
                payload[name] = path.read_bytes()

    add_tree(args.inputs, "inputs")
    add_tree(args.results, "results")
    add_tree(args.report, "report")
    for path in args.development:
        add_tree(path, "development/" + path.name)
    payload["SHA256SUMS"] = "".join(
        f"{digest(data)}  {name}\n" for name, data in sorted(payload.items())
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for name, data in sorted(payload.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755 if name == "bin/transformer-arm64" else 0o644
            archive.addfile(info, io.BytesIO(data))
    # Verify every archived byte against an independently reread embedded manifest.
    with tarfile.open(output, "r:gz") as archive:
        expected = {}
        for line in archive.extractfile("SHA256SUMS").read().decode().splitlines():
            value, name = line.split("  ", 1)
            expected[name] = value
        names = [item.name for item in archive.getmembers()]
        if len(names) != len(set(names)) or set(names) != set(expected) | {"SHA256SUMS"}:
            raise ValueError("Archive member set differs from the checksum manifest")
        for name, value in expected.items():
            if digest(archive.extractfile(name).read()) != value:
                raise ValueError(f"Archive checksum failed: {name}")
    archive_hash = digest(output.read_bytes())
    checksum.write_text(f"{archive_hash}  {output.name}\n")
    evidence = {
        "verified": True,
        "archive": output.name,
        "sha256": archive_hash,
        "archive_bytes": output.stat().st_size,
        "verified_payload_files": len(expected),
        "measured_revision": manifest["revision"],
        "preflight_checks": preflight["check_count"],
        "postrun_checks": summary["audit_check_count"],
        "retained_test_prediction_rows": summary["retained_test_prediction_rows"],
    }
    verification.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", type=Path, action="append", default=[])
    package(parser.parse_args())
