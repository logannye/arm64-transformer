"""Owned v1 policy for the unchanged stock conformance extension.

Source bytes come from the selected externally authenticated full-source manifest.
The extension code AST is separately pinned to the original stock implementation.
The frozen historical fixture policy is intentionally not modified.

This module uses only the standard library. The upstream test owns compilation.
The 1 GiB cache ceiling and 5 GiB free-space floor are checked before execution,
once per second by the parent while full mode runs, and after execution. These
are monitored limits, not filesystem quotas; one polling interval may overshoot.
"""

import ast
import hashlib
import json
import os
import shutil
import stat
import sysconfig
from pathlib import Path

GIB = 1024**3
MAX_CACHE_BYTES = GIB
MIN_FREE_BYTES = 5 * GIB
MAX_CACHE_FILES = 64
CACHE_NAME = "pytorch-test-extensions"
SCHEMA = "owned-stock-mps-conformance-extension-cache-v1"
CONFORMANCE_AST_SHA = "f942d37394ecc6edafd19a295e69a0634311bf6d73b98ed66c89723935c6da59"
SOURCE_INPUTS = (
    "test/test_mps.py",
    "test/metal/test_kernels.metallib",
    "test/metal/test_kernels.metal",
    "aten/src/ATen/native/mps/MetalShaderLibrary.h",
    "c10/metal/float8.h",
    "c10/metal/common.h",
    "c10/metal/utils.h",
    "c10/metal/indexing.h",
    "aten/src/ATen/native/mps/kernels/Convolution.metal",
)


def extension_ast(path):
    tree = ast.parse(Path(path).read_text())
    selected = []
    for node in tree.body:
        names = [item.id for item in getattr(node, "targets", []) if isinstance(item, ast.Name)]
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("_conformance")
        ) or any(name.startswith(("_CONFORMANCE", "_conformance")) for name in names):
            selected.append(node)
    value = hashlib.sha256(
        "\n".join(ast.dump(node, include_attributes=False) for node in selected).encode()
    ).hexdigest()
    if value != CONFORMANCE_AST_SHA or len(selected) != 10:
        raise RuntimeError("The stock conformance extension code changed")
    return value


def verify_inputs(source, source_files, test_source_sha):
    expected = {}
    for relative in SOURCE_INPUTS:
        path = source / relative
        if path.resolve(strict=True) != path or path.is_symlink():
            raise RuntimeError("Conformance fixture source must be canonical")
        wanted = source_files.get(relative)
        if not wanted or sha(path) != wanted:
            raise RuntimeError(f"Stock test source/fixture changed: {relative}")
        expected[relative] = wanted
    if expected["test/test_mps.py"] != test_source_sha:
        raise RuntimeError("Stock fixture uses another selected test inventory")
    extension_ast(source / "test/test_mps.py")
    return expected


CONTROLS = {
    "MAX_JOBS": "2",
    "CMAKE_BUILD_PARALLEL_LEVEL": "2",
    "DEBUG": "0",
    "REL_WITH_DEB_INFO": "0",
    "CXX": "/usr/bin/clang++",
    "TORCH_NO_COMPILER_WRAPPER": "1",
}
ABSENT_FLAGS = ("CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "TORCH_DONT_CHECK_COMPILER_ABI")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_environment(env, cache):
    env.update(CONTROLS)
    env["TORCH_EXTENSIONS_DIR"] = str(Path(cache).resolve())
    for key in ABSENT_FLAGS:
        env.pop(key, None)


def require_environment(env, cache):
    expected = {**CONTROLS, "TORCH_EXTENSIONS_DIR": str(Path(cache).resolve())}
    if any(env.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Stock test-extension execution controls differ")
    if any(key in env for key in ABSENT_FLAGS):
        raise RuntimeError("Unrecorded external compiler flags are forbidden")
    return expected


def require_child_receipt(record, mode, process, pid, exit_code):
    permitted = {"passed"}
    if mode == "full":
        permitted.add("completed_with_skips_or_expected_failures")
    if (
        not isinstance(record, dict)
        or exit_code != 0
        or record.get("status") not in permitted
        or record.get("mode") != mode
        or type(record.get("process")) is not int
        or record["process"] != process
        or type(record.get("pid")) is not int
        or record["pid"] != pid
    ):
        raise RuntimeError("Integrated worker receipt is not a complete matching success")


def snapshot(cache, *, hashes=False, enforce=True):
    cache = Path(cache)
    if cache.is_symlink() or not cache.is_dir():
        raise RuntimeError("Extension cache must be an existing real directory")
    files = {}
    for entry in sorted(cache.rglob("*")):
        try:
            info = entry.lstat()
        except FileNotFoundError:
            # Ninja/compiler temporary files may disappear during monitoring.
            # This is a size sample, not an immutable before/after hash pass.
            if hashes:
                raise
            continue
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError("Symlinks are forbidden inside the extension cache")
        if stat.S_ISREG(info.st_mode):
            files[str(entry.relative_to(cache))] = {"bytes": info.st_size}
        elif not stat.S_ISDIR(info.st_mode):
            raise RuntimeError("Special files are forbidden inside the extension cache")
    value = {
        "path": str(cache.resolve()),
        "bytes": sum(row["bytes"] for row in files.values()),
        "file_count": len(files),
        "free_bytes": shutil.disk_usage(cache).free,
        "max_cache_bytes": MAX_CACHE_BYTES,
        "min_free_bytes": MIN_FREE_BYTES,
        "max_cache_files": MAX_CACHE_FILES,
        "files": files,
    }
    if enforce:
        require_bounds(value)
    if hashes:
        require_bounds(value)
        for relative, row in files.items():
            row["sha256"] = sha(cache / relative)
    return value


def require_bounds(value):
    if value["bytes"] > MAX_CACHE_BYTES:
        raise RuntimeError("Stock test-extension cache exceeds its 1 GiB ceiling")
    if value["file_count"] > MAX_CACHE_FILES:
        raise RuntimeError("Stock test-extension cache exceeds its 64-file ceiling")
    if value["free_bytes"] < MIN_FREE_BYTES:
        raise RuntimeError("Stock test-extension execution requires at least 5 GiB free")


def prepare(cache, source, runtime, source_files, test_source_sha, runtime_manifest_sha):
    cache, source, runtime = Path(cache), Path(source), Path(runtime)
    if cache.name != CACHE_NAME or cache.is_symlink():
        raise RuntimeError("Use the dedicated task-owned pytorch-test-extensions directory")
    expected = verify_inputs(source, source_files, test_source_sha)
    if sha(runtime / "runtime-manifest.json") != runtime_manifest_sha:
        raise RuntimeError("Stock fixture runtime manifest differs from selected build")
    if not cache.parent.is_dir() or cache.parent.resolve(strict=True) != cache.parent:
        raise RuntimeError("Stock fixture parent must exist and be canonical")
    if shutil.disk_usage(cache.parent).free < MIN_FREE_BYTES:
        raise RuntimeError("Insufficient free space before fixture preparation")
    cache.mkdir(exist_ok=True)
    # load_inline would delete these when use_pch=False. Refuse beforehand so
    # the authenticated runtime cannot be mutated as a side effect of the test.
    for name in ("extension.h.gch", "extension.h.sign"):
        if (runtime / "torch/include/torch" / name).exists():
            raise RuntimeError("Unexpected precompiled header in immutable runtime")
    includes = [
        runtime / "torch/include/torch/extension.h",
        runtime / "torch/include/torch/csrc/api/include/torch/torch.h",
        source / "aten/src/ATen/native/mps/MetalShaderLibrary.h",
        Path(sysconfig.get_path("include", scheme="posix_prefix")) / "Python.h",
    ]
    if any(not path.is_file() for path in includes):
        raise RuntimeError(
            "The stock test extension requires complete runtime/source/Python headers"
        )
    ninja = shutil.which("ninja")
    if not ninja or not Path(CONTROLS["CXX"]).is_file():
        raise RuntimeError("The stock test extension requires Ninja and /usr/bin/clang++")
    # Bind reuse to the runtime AND recorded toolchain/header/control inputs.
    # Ninja does not necessarily rebuild when the compiler executable changes.
    identity = {
        "schema": SCHEMA,
        "runtime_manifest_sha256": runtime_manifest_sha,
        "source_inputs": expected,
        "conformance_ast_sha256": CONFORMANCE_AST_SHA,
        "controls": dict(CONTROLS),
        "ninja": {"path": ninja, "sha256": sha(ninja)},
        "compiler": {"path": CONTROLS["CXX"], "sha256": sha(CONTROLS["CXX"])},
        "include_inputs": {str(path): sha(path) for path in includes},
        "toolchain_environment": {
            name: os.environ.get(name)
            for name in ("DEVELOPER_DIR", "SDKROOT", "MACOSX_DEPLOYMENT_TARGET")
        },
    }
    marker = cache / "cache-identity.json"
    if marker.exists():
        if marker.is_symlink() or json.loads(marker.read_text()) != identity:
            raise RuntimeError("Stock test-extension cache identity changed")
    else:
        if any(cache.iterdir()):
            raise RuntimeError("Refuse to claim an existing unowned extension cache")
        marker.write_text(json.dumps(identity, indent=2) + "\n")
    return {
        **identity,
        "scope": (
            "One unchanged upstream C++ conformance test extension; "
            "complete framework compilation remains hosted"
        ),
        "tracked_metallib": "test/metal/test_kernels.metallib",
        "offline_metal_compiler_required": False,
        "debug_policy": (
            "Stock load_inline cflags have no -g; DEBUG=0 is recorded, "
            "but this loader does not read DEBUG/CXXFLAGS"
        ),
        "limits_are_monitored_not_quotas": True,
        "poll_seconds": 1,
        "before": snapshot(cache, hashes=True),
    }


def loaded_extension(module, cache):
    extension = getattr(module, "_conformance_ext", None)
    if extension is None:
        raise RuntimeError("The full upstream suite did not load its conformance test extension")
    path = Path(extension.__file__).resolve(strict=True)
    expected = Path(cache).resolve() / "mps_iter_conformance/mps_iter_conformance.so"
    if path != expected:
        raise RuntimeError("The conformance extension loaded outside its declared cache")
    ninja = path.parent / "build.ninja"
    content = ninja.read_text()
    for line in content.splitlines():
        if line.startswith(("cflags =", "post_cflags =")):
            if any(flag.startswith("-g") and flag != "-g0" for flag in line.split()):
                raise RuntimeError(
                    "The stock test extension unexpectedly enabled debug information"
                )
    return {
        "path": str(path),
        "sha256": sha(path),
        "build_ninja_sha256": sha(ninja),
        "build_ninja": content,
    }
