"""Apply the prospective patch to a fresh eight-file copy; never build or import it."""

import argparse
import hashlib
import json
import subprocess
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(source, output):
    source = source.resolve(strict=True)
    require(
        not output.exists() and not output.is_relative_to(source), "Fresh external output required"
    )
    require(not any(p.is_symlink() for p in [output, *output.parents]), "Symlink output")
    contract = json.loads((HERE / "contract.json").read_text())
    patch = HERE / "changes.patch"
    require(digest(patch) == contract["patch_sha256"], "Candidate patch changed")
    files = contract["source_files"]
    require(
        len(files) == 8
        and all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in files),
        "Unexpected source inventory",
    )
    for name, identity in files.items():
        path = source / name
        require(path.resolve(strict=True) == path and path.is_file(), "Noncanonical source file")
        require(digest(path) == identity["before_sha256"], "Original source differs: " + name)
    output.mkdir(parents=True, exist_ok=False)
    tree = output / "source"
    receipt = {
        "schema": "gpu-owned-half-index-static-validation-v1",
        "passed": False,
        "native_compiled": False,
        "device_numerics_validated": False,
        "formal_eligible": False,
        "patch_sha256": digest(patch),
        "contract_sha256": digest(HERE / "contract.json"),
        "sources": files,
        "commands": [],
    }
    try:
        for name in files:
            path = tree / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write((source / name).read_bytes())
        for arguments in (["--check", "--whitespace=error-all"], ["--whitespace=error-all"]):
            command = ["git", "apply", *arguments, str(patch)]
            result = subprocess.run(command, cwd=tree, capture_output=True, text=True)
            receipt["commands"].append(
                {
                    "command": command,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
            require(result.returncode == 0, "Patch application failed")
        for name, identity in files.items():
            require(digest(tree / name) == identity["after_sha256"], "Patched bytes differ")
            require(digest(source / name) == identity["before_sha256"], "Original source changed")
        receipt.update(passed=True, original_source_unchanged=True)
    except BaseException:
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        with (output / "receipt.json").open("x") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.output.resolve())
    print(
        json.dumps(
            {key: result[key] for key in ("passed", "native_compiled", "device_numerics_validated")}
        )
    )
