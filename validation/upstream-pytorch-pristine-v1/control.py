"""Metadata controls for a prospective exact-base, full-framework build."""

import ast
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHODS = {
    "TestConsistency": ["test_output_grad_match", "_run_op", "_compute_tolerances"],
    "TestConv3dChannelsLast3dMPS": [
        "test_conv3d_channels_last_3d_fwd_bwd",
        "_run_conv3d_cl3d",
        "test_conv_transpose3d_channels_last_3d",
    ],
    "TestMPS": [
        "test_bmm_strided_output",
        "test_mm_strided_output",
        "test_gemv_strided_output_falls_back",
    ],
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def stable_stat(value):
    return tuple(
        getattr(value, key)
        for key in (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_mode",
            "st_nlink",
        )
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_control_contract():
    result = json.loads((HERE / "contract.json").read_text())
    require(result["schema"] == "pytorch-pristine-control-contract-v1", "Wrong contract")
    require(len(result["wheel_inventory"]) == 45, "Expected exact45-wheel comparison")
    require(
        (HERE / "requirements-full.txt").read_text() == result["dependency_freeze"],
        "Dependency constraint bytes differ",
    )
    return result


def check_host(contract, record):
    expected = contract["host"]
    require(sys.version == expected["python"], "Python toolchain differs")
    require(platform.platform() == expected["platform"], "Host OS/architecture differs")
    require(
        {k: os.environ.get(k) for k in expected["image"]} == expected["image"],
        "Hosted runner image differs",
    )
    require(record["build_environment"] == contract["build_environment"], "Build controls differ")


def normalize_tool(text):
    # The Metal toolchain's signed mount suffix may change between identical hosted images.
    # Preserve full raw logs, but compare compiler identity/target/thread model independently
    # of that mount location. Clang/Xcode/SDK identities remain exact.
    return (
        "\n".join(line for line in text.splitlines() if not line.startswith("InstalledDir:")) + "\n"
    )


def libomp_identity(value):
    formulae = value["formulae"]
    require(len(formulae) == 1, "Ambiguous libomp formula")
    formula = formulae[0]
    keys = ["name", "versions", "revision", "version_scheme", "bottle", "ruby_source_checksum"]
    result = {k: formula[k] for k in keys}
    result["installed"] = [
        {k: row[k] for k in ["version", "used_options", "built_as_bottle", "poured_from_bottle"]}
        for row in formula["installed"]
    ]
    return result


def check_tool(label, text, contract):
    if label in contract["tool_checks"]:
        require(
            normalize_tool(text) == contract["tool_checks"][label]["normalized_stdout"],
            "Compiler/tool differs: " + label,
        )
    elif label == "libomp-package":
        require(libomp_identity(json.loads(text)) == contract["libomp"], "OpenMP bottle differs")
    elif label != "metal-path":
        raise RuntimeError("Undeclared tool identity: " + label)


def submodule_identities(text):
    result = []
    for line in text.splitlines():
        match = re.fullmatch(r" ([0-9a-f]{40}) ([^ ]+)(?: \(.*\))?", line)
        require(match is not None, "Missing or changed recursive submodule")
        revision, name = match.groups()
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe submodule path")
        result.append((name, revision))
    require(
        len(result) == 61 and len({x[0] for x in result}) == 61,
        "Recursive submodule inventory differs",
    )
    return sorted(result)


def canonical_ast(node):
    # ast.dump's default empty-field presentation changed after Python3.12.
    # Serialize every declared field explicitly; locations are not AST fields.
    if isinstance(node, ast.AST):
        return [
            type(node).__name__,
            {name: canonical_ast(value) for name, value in ast.iter_fields(node)},
        ]
    if isinstance(node, list):
        return [canonical_ast(value) for value in node]
    if node is Ellipsis:
        return {"constant_type": "ellipsis"}
    if isinstance(node, bytes):
        return {"constant_type": "bytes", "hex": node.hex()}
    if isinstance(node, complex):
        return {"constant_type": "complex", "real": node.real.hex(), "imag": node.imag.hex()}
    if isinstance(node, float):
        return {"constant_type": "float", "hex": node.hex()}
    return node


def method_hashes(code):
    result = {}
    for cls in ast.parse(code).body:
        if isinstance(cls, ast.ClassDef) and cls.name in METHODS:
            for method in cls.body:
                if isinstance(method, ast.FunctionDef) and method.name in METHODS[cls.name]:
                    key = cls.name + "." + method.name
                    require(key not in result, "Duplicate test/helper method")
                    result[key] = hashlib.sha256(
                        json.dumps(
                            canonical_ast(method), sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest()
    require(len(result) == 9, "Missing unchanged test/helper methods")
    return result


def retain_test_source(run, contract):
    """Transport only the complete test module and its source/Metal fixture closure."""
    fixed = [
        "LICENSE",
        "NOTICE",
        "test/test_mps.py",
        "aten/src/ATen/native/mps/MetalShaderLibrary.h",
        "aten/src/ATen/native/mps/kernels/Convolution.metal",
    ]
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--", *fixed, "test/metal", "c10/metal"], cwd=run.source
    )
    names = output.decode().rstrip("\0").split("\0")
    require(all(x in names for x in fixed), "Missing complete test module/fixture closure")
    require(len(names) == len(set(names)) and len(names) < 256, "Test source inventory bound")
    files = {}
    total = 0
    for name in sorted(names):
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe source member")
        path = run.source / relative
        require(path.is_file() and not path.is_symlink(), "Nonregular source member")
        before = path.stat()
        raw = path.read_bytes()
        total += len(raw)
        require(total <= 16 * 1024**2, "Compact source closure exceeded16MiB")
        blob = (
            subprocess.check_output(["git", "rev-parse", "HEAD:" + name], cwd=run.source)
            .decode()
            .strip()
        )
        require(
            hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == blob,
            "Test source differs from pristine Git blob: " + name,
        )
        require(stable_stat(before) == stable_stat(path.stat()), "Source changed during transport")
        files[name] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob": blob,
        }
    expected = contract["unchanged_test_method_ast_sha256"]
    require(
        method_hashes((run.source / "test/test_mps.py").read_bytes()) == expected,
        "Failed-test methods/helper equations differ from formatted source",
    )
    manifest = {
        "schema": "pytorch-pristine-test-source-v1",
        "source_revision": contract["source_revision"],
        "scope": "Complete untouched test_mps.py and its source/Metal fixture closure",
        "files": files,
        "unchanged_test_method_ast_sha256": expected,
        "formatted_test_sha256": contract["formatted_test_sha256"],
    }
    manifest_path = run.output / "test-source-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    archive = run.output / "pytorch-pristine-test-source.tar.gz"
    require(not archive.exists(), "Source archive exists")
    with tarfile.open(archive, "w:gz") as tar:
        for name in files:
            tar.add(run.source / name, arcname="test-source/" + name, recursive=False)
        tar.add(manifest_path, arcname="test-source/manifest.json", recursive=False)
    require(
        all(sha(run.source / name) == row["sha256"] for name, row in files.items()),
        "Source changed after archive creation",
    )
    run.record["test_source_archive"] = {
        "file": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": sha(archive),
        "manifest_sha256": sha(manifest_path),
        "files": len(files),
        "uncompressed_bytes": total,
    }
    run.record["unchanged_failed_test_methods"] = expected
    run.save()
