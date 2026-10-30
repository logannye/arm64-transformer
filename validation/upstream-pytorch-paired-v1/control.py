"""Metadata controls for sequential pristine/formatted builds on one hosted machine."""

import ast
import hashlib
import json
import re
import stat
import subprocess
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
    require(result["schema"] == "pytorch-paired-control-contract-v1", "Wrong contract")
    require(len(result["wheel_inventory"]) == 45, "Expected exact45-wheel comparison")
    require(
        (HERE / "requirements-full.txt").read_text() == result["dependency_freeze"],
        "Dependency constraint bytes differ",
    )
    variants = result["variants"]
    require(set(variants) == {"pristine", "formatted"}, "Wrong paired variant inventory")
    require(
        variants["pristine"]["patch_sha256"] is None
        and variants["pristine"]["sources"] == result["pristine_source_sha256"]
        and variants["formatted"]["patch_sha256"]
        == "3834dca844ea3de6b40467ce214b99bce83fa06dae2bc454623516dafc0fdd71"
        and set(variants["formatted"]["sources"]) == set(result["pristine_source_sha256"])
        and variants["formatted"]["sources"]["test/test_mps.py"] == result["formatted_test_sha256"],
        "Variant source or patch identity differs",
    )
    return result


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
    """Retain each variant's test closure with exact base and working blob identities."""
    require(run.variant in ("pristine", "formatted"), "Unknown source variant")
    expected_variant = contract["variants"][run.variant]
    require(
        run.patch_sha256 == expected_variant["patch_sha256"]
        and run.expected_sources == expected_variant["sources"],
        "Run variant/source/patch identity differs",
    )
    revision = contract["source_revision"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=run.source).decode().strip()
    require(head == revision, "Compact source HEAD differs from pinned base")
    fixed = [
        "LICENSE",
        "NOTICE",
        "test/test_mps.py",
        "aten/src/ATen/native/mps/MetalShaderLibrary.h",
        "aten/src/ATen/native/mps/kernels/Convolution.metal",
    ]
    output = subprocess.check_output(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            "-z",
            revision,
            "--",
            *fixed,
            "test/metal",
            "c10/metal",
        ],
        cwd=run.source,
    )
    names = output.decode().rstrip("\0").split("\0")
    require(all(x in names for x in fixed), "Missing complete test module/fixture closure")
    require(len(names) == len(set(names)) and len(names) < 256, "Test source inventory bound")
    require(run.source == run.source.resolve(), "Source root must be canonical")
    files, stats = {}, {}
    total = 0
    for name in sorted(names):
        relative = Path(name)
        require(
            name
            and not relative.is_absolute()
            and all(x not in ("", ".", "..") for x in name.split("/")),
            "Unsafe source member",
        )
        require(
            name in fixed or relative.parts[:2] in (("test", "metal"), ("c10", "metal")),
            "Source outside declared fixture closure",
        )
        path = run.source / relative
        require(
            path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            "Nonregular or symlink source member",
        )
        before = path.stat()
        require(stat.S_ISREG(before.st_mode), "Nonregular source mode")
        require(total + before.st_size <= 16 * 1024**2, "Compact source closure exceeded16MiB")
        raw = path.read_bytes()
        total += len(raw)
        blob = (
            subprocess.check_output(["git", "rev-parse", revision + ":" + name], cwd=run.source)
            .decode()
            .strip()
        )
        require(re.fullmatch("[0-9a-f]{40}", blob) is not None, "Malformed base Git blob")
        working_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        digest = hashlib.sha256(raw).hexdigest()
        if run.variant == "formatted" and name == "test/test_mps.py":
            require(digest == run.expected_sources[name], "Formatted test source differs")
        else:
            require(working_blob == blob, "Test fixture differs from exact base: " + name)
        if name in run.expected_sources:
            require(digest == run.expected_sources[name], "Declared source digest differs")
        require(stable_stat(before) == stable_stat(path.stat()), "Source changed during transport")
        stats[name] = stable_stat(before)
        files[name] = {
            "bytes": len(raw),
            "sha256": digest,
            "base_git_blob": blob,
            "working_git_blob": working_blob,
        }
    expected = contract["unchanged_test_method_ast_sha256"]
    require(
        method_hashes((run.source / "test/test_mps.py").read_bytes()) == expected,
        "Failed-test methods/helper equations differ between variants",
    )
    manifest = {
        "schema": "pytorch-paired-test-source-v1",
        "source_variant": run.variant,
        "source_revision": revision,
        "patch_sha256": run.patch_sha256,
        "expected_sources": run.expected_sources,
        "scope": "Complete variant test_mps.py and unchanged base source/Metal fixture closure",
        "files": files,
        "unchanged_test_method_ast_sha256": expected,
        "formatted_test_sha256": contract["formatted_test_sha256"],
    }
    manifest_path = run.output / "test-source-manifest.json"
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    archive = run.output / f"pytorch-paired-{run.variant}-test-source.tar.gz"
    require(not archive.exists(), "Source archive exists")
    with tarfile.open(archive, "x:gz") as tar:
        for name in files:
            tar.add(run.source / name, arcname="test-source/" + name, recursive=False)
        tar.add(manifest_path, arcname="test-source/manifest.json", recursive=False)
    require(
        all(
            sha(run.source / name) == row["sha256"]
            and stable_stat((run.source / name).stat()) == stats[name]
            for name, row in files.items()
        ),
        "Source changed after archive creation",
    )
    run.record["test_source_archive"] = {
        "file": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": sha(archive),
        "manifest_sha256": sha(manifest_path),
        "files": len(files),
        "uncompressed_bytes": total,
        "source_variant": run.variant,
    }
    run.record["unchanged_failed_test_methods"] = expected
    run.save()
