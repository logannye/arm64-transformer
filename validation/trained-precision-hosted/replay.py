"""Hosted NEON historical-gradient replay with independent, retained raw audit."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, choices=(512, 1024), required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        os.environ.get("GITHUB_REPOSITORY") != "logannye/arm64-transformer"
        or os.environ.get("GITHUB_ACTIONS") != "true"
    ):
        raise ValueError("Only the authorized own-project hosted workflow may execute")
    if shutil.disk_usage(ROOT).free < 24 * 1024**3:
        raise ValueError("Preserve initial hosted replay allocation")
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = ROOT / "experiments/trained-precision-v1/replay-inventory.json"
    inventory_sha = "021f8e232863b437198c65e35c06dc6f95162cc07294a3f30714b60e6aeef18e"
    if sha(inventory) != inventory_sha:
        raise ValueError("Frozen historical inventory changed")
    library = ROOT / f"native/build/width-{args.width}/libtransformer.dylib"
    binary = args.output / "libtransformer.dylib"
    shutil.copy2(library, binary)
    library_sha = sha(binary)
    env = dict(
        os.environ, ARM64_TRANSFORMER_WIDTH=str(args.width), ARM64_TRANSFORMER_BACKEND="neon"
    )
    total = sum(c["width"] == args.width for c in json.loads(inventory.read_text())["cases"])
    jobs = [dict(offset=offset, limit=min(16, total - offset)) for offset in range(0, total, 16)]
    write(
        args.output / "plan.json",
        dict(
            schema="hosted-native-trained-precision-replay-v1",
            jobs=jobs,
            width=args.width,
            inventory_sha256=inventory_sha,
            native_library_sha256=library_sha,
            native_policy=15,
            selected_backend="neon",
            formal_eligible=False,
            admission_stream_opened=False,
            github_revision=os.environ["GITHUB_SHA"],
            producer_sha256=sha(Path(__file__)),
            platform=subprocess.check_output(["sw_vers"], text=True),
            cpu=subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip(),
        ),
    )
    workers = []
    for job in jobs:
        output = args.output / f"worker-{job['offset']:03d}"
        command = [
            sys.executable,
            "-B",
            "-m",
            "transformer.trained_precision",
            "worker",
            "--inventory",
            str(inventory),
            "--inventory-sha256",
            inventory_sha,
            "--evidence-root",
            str(args.inputs),
            "--engine",
            "native",
            "--width",
            str(args.width),
            "--batch-size",
            "16",
            "--gradients",
            "--library",
            str(binary),
            "--library-sha256",
            library_sha,
            "--case-offset",
            str(job["offset"]),
            "--case-limit",
            str(job["limit"]),
            "--output",
            str(output),
            "--raw-store",
            str(args.output / "raw-store"),
        ]
        with (args.output / f"worker-{job['offset']:03d}.log").open("x") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            try:
                while process.poll() is None:
                    if shutil.disk_usage(ROOT).free < 8 * 1024**3:
                        raise RuntimeError("Preserve 8 GiB hosted disk reserve")
                    time.sleep(5)
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        receipt = json.loads((output / "result.json").read_text())
        if not receipt["complete"]:
            raise RuntimeError("Incomplete replay; original failure retained")
        if (
            receipt["native"]["precision_policy"] != 15
            or receipt["native"]["backend"]["selected_kind"] != 0
        ):
            raise ValueError("Wrong candidate/backend executed")
        workers.append(
            dict(
                path=str((output / "result.json").relative_to(args.output)),
                sha256=sha(output / "result.json"),
            )
        )
    spec = importlib.util.spec_from_file_location(
        "hosted_collection_audit", ROOT / "integration/trained-precision/audit_collection.py"
    )
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    manifest = dict(
        schema="trained-precision-collection-v1",
        inventory_sha256=inventory_sha,
        formal_eligible=False,
        groups=[
            dict(
                name=f"native-neon-policy15-width-{args.width}",
                implementation=audit.implementation(receipt),
                workers=workers,
            )
        ],
    )
    write(args.output / "manifest.json", manifest)
    result = audit.audit_collection(
        args.output / "manifest.json",
        sha(args.output / "manifest.json"),
        inventory,
        inventory_sha,
        args.inputs,
        args.output,
        args.output / "raw-store",
    )
    write(args.output / "audit.json", result)
    group = result["groups"][0]
    summary = {k: v for k, v in group.items() if k not in ("audits", "implementation")}
    summary.update(
        audit_sha256=sha(args.output / "audit.json"),
        formal_eligible=False,
        admission_stream_opened=False,
        scope="Hosted M1/NEON replay; not M4/SME or timing qualification",
    )
    write(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if group["numerics_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
