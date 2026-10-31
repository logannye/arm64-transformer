"""Add explicit generic Python APIs to authenticated full maintained packages.

No package installation, native compilation or old-file modification occurs.
The original build remains an input, never a qualification for the new package.
"""

import argparse
import copy
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parent / "precision-library-v1"
CONTRACT_SHA = "7b20e72aa969732a1348b4d01281f604dee8ba043433191ce4d71a1ce4604718"
RECEIPT = "PRECISION-API.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUITE = load("precision_derivation_checks", HERE / "owned_mlx_suites.py")
CANDIDATE = load("precision_derivation_candidate", HERE / "candidate_half_v3_validation.py")
require = SUITE.require


def contract():
    SUITE.pinned(API / "contract.json", CONTRACT_SHA)
    value = SUITE.read_json(API / "contract.json")
    for name, digest in value["source_sha256"].items():
        SUITE.pinned(API / name, digest)
    return value


def additions(framework, selected):
    require(framework in {"pytorch", "mlx"}, "Unknown maintained package")
    result = {}
    for tree in ("source", "runtime"):
        if framework == "pytorch":
            prefix = "torch/nn/"
        else:
            prefix = "python/mlx/nn/"
        for source in (
            "torch_api.py" if framework == "pytorch" else "mlx_api.py",
            "_fp32_precise_dense.py",
            "_fp32_precise_math.py",
        ):
            target = "precise_fp32.py" if source.endswith("_api.py") else source
            require(
                SUITE.sha256(API / source) == selected["source_sha256"][source],
                "API source changed",
            )
            result[tree + "/" + prefix + target] = API / source
    return result


def authority(framework, path, digest):
    """Authenticate complete native/source lineage through the original producer."""
    SUITE.pinned(path, digest)
    if framework == "pytorch":
        plan, _ = CANDIDATE.authenticate_plan(path, digest)
        manifests = {
            kind: SUITE.read_json(plan[kind + "_manifest"]["path"])
            for kind in ("source", "runtime")
        }
        return dict(
            roots={kind: plan[kind] for kind in manifests},
            manifests=manifests,
            native_build_run=int(
                CANDIDATE.candidate_profile(plan.get("candidate", "half-v3"))["run"]
            ),
        )
    require(framework == "mlx", "Unknown maintained package")
    require(path.name == "TEST-COMPATIBILITY.json", "Require the original MLX derivation receipt")
    compat = load("precision_derivation_mlx_compat", HERE / "derive_mlx_test_compat.py")
    manifest, selected = compat.authenticate(path.parent, digest)
    return dict(
        roots={"bundle": str(path.parent)},
        manifests={"bundle": manifest},
        native_build_run=selected["native_build_run"],
    )


def row(path, kind):
    bound = SUITE.binding(path)
    value = {"bytes": bound["bytes"], "sha256": bound["sha256"]}
    if kind == "source":
        value.update(kind="file", mode=0o644)
    if kind == "bundle":
        value["executable"] = False
    return value


def manifest_path(output, kind):
    return (
        output
        / {
            "source": "source-manifest.json",
            "runtime": "runtime/runtime-manifest.json",
            "bundle": "MANIFEST.json",
        }[kind]
    )


def expected_manifests(original, output, framework, selected):
    """The complete allowed delta: six new Python files and explicit lineage."""
    extra = additions(framework, selected)
    result = copy.deepcopy(original["manifests"])
    validate_additive(original, extra)
    for kind, manifest in result.items():
        added = {
            name: source
            for name, source in extra.items()
            if kind == "bundle" or name.startswith(kind + "/")
        }
        for name, source in added.items():
            relative = name if kind == "bundle" else name.split("/", 1)[1]
            require(relative not in manifest["files"], "API would replace existing library source")
            target = output / name
            require(
                target.read_bytes() == source.read_bytes(),
                "Installed API differs from frozen source",
            )
            manifest["files"][relative] = row(target, kind)
        if kind == "bundle":
            require(RECEIPT not in manifest["files"], "Original capsule already has an API receipt")
            manifest["files"][RECEIPT] = row(output / RECEIPT, kind)
        if kind == "source":
            manifest["regular_file_bytes"] += sum(
                source.stat().st_size for source in added.values()
            )
        manifest["precision_api_derivation"] = SUITE.binding(output / RECEIPT)
    return result


def validate_additive(original, extra):
    for kind, manifest in original["manifests"].items():
        for name in extra:
            if kind == "bundle" or name.startswith(kind + "/"):
                relative = name if kind == "bundle" else name.split("/", 1)[1]
                require(
                    relative not in manifest["files"], "API would replace existing library source"
                )


def verify_payload(output, manifests):
    for kind, manifest in manifests.items():
        root = output if kind == "bundle" else output / kind
        if kind == "bundle":
            compat = load("precision_derivation_mlx_files", HERE / "derive_mlx_test_compat.py")
            require(compat.files(root) == manifest["files"], "Derived MLX payload changed")
        else:
            CANDIDATE.verify_tree(root, manifest, source=kind == "source")


def authenticate(output, digest):
    selected = contract()
    SUITE.pinned(output / RECEIPT, digest)
    proof = SUITE.read_json(output / RECEIPT)
    require(
        set(proof)
        == {
            "schema",
            "framework",
            "contract",
            "producer",
            "original_authority",
            "original",
            "added_sources",
            "native_compilation",
            "formal_eligible",
            "qualification_transferred",
            "logical_clone_bytes",
        },
        "Undeclared precision derivation receipt fields",
    )
    require(
        proof["schema"] == selected["schema"]
        and proof["native_compilation"] is False
        and proof["formal_eligible"] is False
        and proof["qualification_transferred"] is False,
        "Precision derivation cannot confer qualification",
    )
    for key, path in (("contract", API / "contract.json"), ("producer", Path(__file__).resolve())):
        SUITE.binding(proof[key]["path"], proof[key])
        require(proof[key]["sha256"] == SUITE.sha256(path), "Precision derivation producer changed")
    bound = proof["original_authority"]
    original = authority(proof["framework"], Path(bound["path"]), bound["sha256"])
    SUITE.binding(bound["path"], bound)
    require(original == proof["original"], "Original maintained-build authority changed")
    extra = additions(proof["framework"], selected)
    require(
        proof["added_sources"] == {name: SUITE.binding(source) for name, source in extra.items()},
        "API source bindings differ",
    )
    logical = sum(
        item["bytes"]
        for manifest in original["manifests"].values()
        for item in manifest["files"].values()
        if item.get("kind") != "symlink"
    )
    require(proof["logical_clone_bytes"] == logical, "Clone storage accounting differs")
    expected = expected_manifests(original, output, proof["framework"], selected)
    for kind, manifest in expected.items():
        require(
            SUITE.read_json(manifest_path(output, kind)) == manifest,
            "Undeclared maintained-library manifest change",
        )
    verify_payload(output, expected)
    return proof, expected


def derive(framework, original_path, original_sha, output):
    selected = contract()
    original = authority(framework, original_path, original_sha)
    require(
        output.is_absolute()
        and output.resolve(strict=False) == output
        and not output.exists()
        and output.parent.is_dir(),
        "Fresh canonical output required",
    )
    require(
        all(
            not output.is_relative_to(Path(root)) and not Path(root).is_relative_to(output)
            for root in original["roots"].values()
        ),
        "Output overlaps original package",
    )
    logical = sum(
        item["bytes"]
        for manifest in original["manifests"].values()
        for item in manifest["files"].values()
        if item.get("kind") != "symlink"
    )
    extra = additions(framework, selected)
    validate_additive(original, extra)
    added_bytes = sum(source.stat().st_size for source in extra.values())
    require(added_bytes <= selected["maximum_added_bytes"], "API addition budget exceeded")
    require(
        shutil.disk_usage(output.parent).free
        > selected["minimum_free_bytes"] + logical + added_bytes,
        "Insufficient disk reserve for bounded clone",
    )
    if framework == "mlx":
        subprocess.run(["/bin/cp", "-cR", original["roots"]["bundle"], str(output)], check=True)
    else:
        output.mkdir()
        for kind, root in original["roots"].items():
            subprocess.run(["/bin/cp", "-cR", root, str(output / kind)], check=True)
        SUITE.write_json(output / "source-manifest.json", original["manifests"]["source"])
    verify_payload(output, original["manifests"])
    for name, source in extra.items():
        target = output / name
        require(
            target.parent.is_dir() and not target.exists(), "API must be an additive package module"
        )
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        target.chmod(0o644)
    proof = dict(
        schema=selected["schema"],
        framework=framework,
        contract=SUITE.binding(API / "contract.json"),
        producer=SUITE.binding(Path(__file__).resolve()),
        original_authority=SUITE.binding(original_path),
        original=original,
        added_sources={name: SUITE.binding(source) for name, source in extra.items()},
        native_compilation=False,
        formal_eligible=False,
        qualification_transferred=False,
        logical_clone_bytes=logical,
    )
    SUITE.write_json(output / RECEIPT, proof)
    manifests = expected_manifests(original, output, framework, selected)
    for kind, manifest in manifests.items():
        SUITE.write_json(manifest_path(output, kind), manifest)
    digest = SUITE.sha256(output / RECEIPT)
    authenticate(output, digest)
    return SUITE.binding(output / RECEIPT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("derive")
    prepare.add_argument("--framework", choices=("pytorch", "mlx"), required=True)
    prepare.add_argument("--original-authority", type=Path, required=True)
    prepare.add_argument("--original-sha256", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    audit = sub.add_parser("authenticate")
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args()
    if args.command == "derive":
        result = derive(
            args.framework,
            args.original_authority.resolve(strict=True),
            args.original_sha256,
            args.output.absolute(),
        )
    else:
        authenticate(args.output.resolve(strict=True), args.receipt_sha256)
        result = SUITE.binding(args.output / RECEIPT)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
