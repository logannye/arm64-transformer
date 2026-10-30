"""Prospective matched Release main/patch runtime preparation, no benchmark claims.

Four native builds run serially on hosted macOS in one reusable build tree.
This helper does not run the transformer study or perform performance scoring.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from integration_common import digest, inventory, run, write

BASE = "a21f892a6e8e83afb117cef1d1ceae3e82f48c32"
CONTRACT = "mlx_matched_release_operator_build_v1"
MODES = (("jit-on", "ON"), ("jit-off", "OFF"))
VARIANTS = ("main", "patched")
DEPENDENCY_LOCK_SHA256 = "2e0126378496b93bd034b5cdc7cda4eaf95ffa262f8c84225f870c9789a0d333"
PATCHES = {
    "embedding-fix.patch": "9aecf0c9b0131fe0673493035fd3c47fba3e4ffb1cfc04d00a506702e1d1067c",
    "attention-fix.patch": "b9787a423920641034f42e611f0d8529f6dd49022d094e280fa5ce52a38b7cc3",
}
UNTOUCHED_BUILD_PATHS = (
    "mlx/backend/metal/kernels/CMakeLists.txt",
    "mlx/backend/metal/make_compiled_preamble.sh",
)
HERE = Path(__file__).resolve().parent


def git(source, *args):
    return subprocess.check_output(["git", "--no-optional-locks", *args], cwd=source)


def source_files(source):
    names = git(source, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    return sorted({p.decode() for p in names.split(b"\0") if p})


def source_identity(source):
    result = {}
    for name in source_files(source):
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Unexpected source file type: {name}")
        result[name] = digest(path)
    return result


def storage(root):
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())
    free = shutil.disk_usage(root).free
    if free < 2 * 1024**3 or size > 10 * 1024**3:
        raise RuntimeError(f"Storage limit: {size} bytes retained; {free} bytes free")
    return {"retained_bytes": size, "free_bytes": free}


def verify_patch_inputs(inputs):
    for name, expected in PATCHES.items():
        if digest(inputs / name) != expected:
            raise ValueError(f"Substituted patch: {name}")
    embedding = json.loads((inputs / "embedding-patch-check.json").read_text())
    attention = json.loads((inputs / "attention-patch-check.json").read_text())
    if embedding["base_commit"] != BASE or attention["base_commit"] != BASE:
        raise ValueError("Patch receipt base differs")
    if (
        embedding["patch_sha256"] != PATCHES["embedding-fix.patch"]
        or attention["patch_sha256"] != PATCHES["attention-fix.patch"]
    ):
        raise ValueError("Patch receipt hash differs")
    expected = {p: v["result_sha256"] for p, v in embedding["source_files"].items()}
    expected.update({p: v["patched_sha256"] for p, v in attention["files"].items()})
    if len(expected) != 11:
        raise ValueError("Wrong patch path inventory")
    return expected


def check_source(source, expected):
    if git(source, "rev-parse", "HEAD").decode().strip() != BASE:
        raise ValueError("Checkout is not the exact base")
    changed = {n.decode() for n in git(source, "diff", "--name-only", "-z").split(b"\0") if n}
    changed |= {
        n.decode()
        for n in git(source, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        if n
    }
    if changed != set(expected):
        raise ValueError(f"Unexpected changed-source inventory: {changed ^ set(expected)}")
    for relative, wanted in expected.items():
        if digest(source / relative) != wanted:
            raise ValueError(f"Patched source bytes differ: {relative}")
    for relative in UNTOUCHED_BUILD_PATHS:
        if (source / relative).read_bytes() != git(source, "show", f"{BASE}:{relative}"):
            raise ValueError(f"Validation-only build substitution detected: {relative}")


def copy_source(source, output):
    for relative in source_files(source):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)


def install_public_headers(source, python_package):
    """Match the pinned CMake headers component, without a build or substitution."""
    source, python_package = Path(source), Path(python_package)
    header_root = source / "mlx"
    destination = python_package / "include"
    if destination.exists() or destination.is_symlink():
        raise ValueError("Public-header destination already exists")
    if header_root.is_symlink() or not header_root.is_dir():
        raise ValueError("Public-header source must be a real directory")
    if any(path.is_symlink() for path in header_root.rglob("*")):
        raise ValueError("Symlink in public-header source")
    headers = sorted(
        path
        for path in header_root.rglob("*.h")
        if path.is_file() and path.relative_to(header_root).as_posix() != "backend/metal/kernels.h"
    )
    if not headers:
        raise ValueError("Empty public-header inventory")
    if not (header_root / "backend/cpu/compiled_preamble.h").is_file():
        raise ValueError("Missing CPU JIT public preamble")
    identities = {
        path.relative_to(source).as_posix(): {
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
        for path in headers
    }
    for relative, expected in identities.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
        if digest(target) != expected["sha256"] or target.stat().st_size != expected["bytes"]:
            raise ValueError(f"Public header changed while copying: {relative}")
        if digest(source / relative) != expected["sha256"]:
            raise ValueError(f"Public header source changed while copying: {relative}")
    copied = {
        name: {key: item[key] for key in ("sha256", "bytes")}
        for name, item in inventory(destination).items()
    }
    if copied != identities:
        raise ValueError("Installed public-header inventory differs")
    if any(digest(source / name) != item["sha256"] for name, item in identities.items()):
        raise ValueError("Public-header source changed during transport")
    return {
        "schema": 1,
        "scope": "upstream_cmake_headers_component_transport",
        "base": BASE,
        "install_rule": "mlx/**/*.h except mlx/backend/metal/kernels.h -> include/mlx/",
        "cmake_source_sha256": digest(source / "CMakeLists.txt"),
        "file_count": len(headers),
        "bytes": sum(row["bytes"] for row in identities.values()),
        "headers": identities,
    }


def package_mode(source, build, output, mode, provenance, env, logs):
    variant = provenance["variant"]
    output.mkdir(parents=True)
    python_package = output / "runtime/python/mlx"
    shutil.copytree(
        source / "python/mlx",
        python_package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so", "*.dylib", "*.metallib"),
    )
    write(output / "PUBLIC-HEADERS.json", install_public_headers(source, python_package))
    core = list((build / "python/mlx").glob("core*.so"))
    if len(core) != 1:
        raise ValueError("Missing or ambiguous newly built core extension")
    shutil.copy2(core[0], python_package / core[0].name)
    lib = python_package / "lib"
    lib.mkdir(exist_ok=True)
    for relative in ("libmlx.dylib", "jaccl/libjaccl.dylib"):
        path = build / relative
        if path.is_file():
            shutil.copy2(path, lib / path.name)
    if not (lib / "libmlx.dylib").is_file():
        raise ValueError("Expected a normal shared MLX library")
    if mode != "cpu-only":
        shutil.copy2(build / "mlx/backend/metal/kernels/mlx.metallib", lib / "mlx.metallib")
    (output / "runtime/bin").mkdir()
    shutil.copy2(build / "tests/tests", output / "runtime/bin/tests")
    copy_source(source, output / "source")
    tools = output / "tools"
    tools.mkdir()
    for name in ("integration_common.py", "transport_probe.py"):
        shutil.copy2(HERE / name, tools / name)
    shutil.copytree(HERE / "inputs", output / "inputs")
    shutil.copy2(HERE / "CONTRACT.md", output / "CONTRACT.md")
    shutil.copy2(HERE / "dependencies-exact.txt", output / "dependencies-exact.txt")
    shutil.copy2(logs / "runtime-lock.txt", output / "runtime-lock.txt")
    shutil.copy2(logs / "dependency-install-report.json", output / "dependency-install-report.json")
    shutil.copy2(build / "CMakeCache.txt", output / "CMakeCache.txt")
    write(output / "PROVENANCE.json", provenance)
    dependency_names = ("numpy", "torch", "tensorflow", "ml_dtypes", "pytest", "pytest-xdist")
    write(
        output / "MANIFEST.json",
        {
            "schema": 1,
            "scope": CONTRACT,
            "mode": mode,
            "base": BASE,
            "patches": PATCHES if variant == "patched" else {},
            "variant": variant,
            "runtime_dependencies": {n: importlib.metadata.version(n) for n in dependency_names},
            "files": inventory(output),
        },
    )


def controlled_environment(root):
    env = dict(os.environ)
    for name in (
        "MLX_DISABLE_COMPILE",
        "PYTORCH_MPS_PREFER_METAL",
        "PYTHONPATH",
        "PYTHONHOME",
        "MLX_METALLIB_PATH",
        "DYLD_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DEVICE",
        "CFLAGS",
        "CXXFLAGS",
        "CPPFLAGS",
        "LDFLAGS",
        "CC",
        "CXX",
        "CMAKE_TOOLCHAIN_FILE",
        "CMAKE_C_COMPILER_LAUNCHER",
        "CMAKE_CXX_COMPILER_LAUNCHER",
    ):
        env.pop(name, None)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        PIP_NO_CACHE_DIR="1",
        PRE_COMMIT_HOME=str(root / "pre-commit-cache"),
        CMAKE_BUILD_PARALLEL_LEVEL="2",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MLX_ENABLE_TF32="0",
        MLX_SDPA_VJP_FALLBACK="0",
        PYTORCH_ENABLE_MPS_FALLBACK="0",
        PYTORCH_MPS_FAST_MATH="0",
    )
    return env


def hosted_identity(environment, root):
    if (
        environment.get("GITHUB_ACTIONS") != "true"
        or environment.get("GITHUB_REPOSITORY") != "logannye/arm64-transformer"
    ):
        raise ValueError(
            "Native preparation is restricted to the authorized project GitHub Actions"
        )
    for name in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
        if not re.fullmatch(r"[1-9][0-9]*", environment.get(name, "")):
            raise ValueError(f"Missing or malformed hosted identity: {name}")
    if not re.fullmatch(r"[0-9a-f]{40}", environment.get("GITHUB_SHA", "")):
        raise ValueError("Missing or malformed workflow source revision")
    if environment.get("RUNNER_OS") != "macOS" or environment.get("RUNNER_ARCH") != "ARM64":
        raise ValueError("Requires an Apple-silicon macOS hosted runner")
    temporary = Path(environment.get("RUNNER_TEMP", ""))
    if not temporary.is_absolute() or not root.is_relative_to(temporary.resolve(strict=True)):
        raise ValueError("Build root must be inside the authorized runner temporary directory")
    keys = (
        "GITHUB_ACTIONS",
        "GITHUB_REPOSITORY",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_SHA",
        "GITHUB_REF",
        "GITHUB_WORKFLOW_REF",
        "GITHUB_WORKFLOW_SHA",
        "RUNNER_NAME",
        "RUNNER_OS",
        "RUNNER_ARCH",
        "RUNNER_ENVIRONMENT",
        "RUNNER_TEMP",
        "ImageOS",
        "ImageVersion",
        "ImageVersionFull",
        "ImageRelease",
        "ImageUrl",
    )
    return {name: environment.get(name) for name in keys}


def dependency_map(text):
    result = {}
    for line in text.splitlines():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+==[^\s]+", line):
            raise ValueError(f"Unsupported dependency entry: {line!r}")
        name, version = line.split("==", 1)
        name = re.sub(r"[-_.]+", "-", name).lower()
        if name in result or name in {"mlx", "mlx-metal", "mlx-cuda"}:
            raise ValueError("Duplicate or alternate MLX dependency")
        result[name] = version
    if not result:
        raise ValueError("Empty dependency lock")
    return result


def configure_command(source, build, jit, python=sys.executable):
    if jit not in {"ON", "OFF"}:
        raise ValueError("Unknown shader-JIT setting")
    return [
        str(Path(python).parent / "cmake"),
        "-S",
        str(source),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -g0",
        "-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG -g0",
        "-DCMAKE_COMPILE_WARNING_AS_ERROR=ON",
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0",
        "-DCMAKE_OSX_ARCHITECTURES=arm64",
        "-DBUILD_SHARED_LIBS=ON",
        "-DMLX_EXPORT_ALL_SYMBOLS=ON",
        "-DMLX_BUILD_CPU=ON",
        "-DMLX_BUILD_METAL=ON",
        "-DMLX_BUILD_CUDA=OFF",
        f"-DMLX_METAL_JIT={jit}",
        "-DMLX_BUILD_TESTS=ON",
        "-DMLX_BUILD_PYTHON_BINDINGS=ON",
        "-DMLX_BUILD_PYTHON_STUBS=OFF",
        "-DMLX_BUILD_EXAMPLES=OFF",
        "-DMLX_BUILD_BENCHMARKS=OFF",
        "-DMLX_USE_CCACHE=OFF",
        "-DMLX_METAL_DEBUG=OFF",
        f"-DPython_EXECUTABLE={python}",
        f"-DMLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY={build / 'python/mlx'}",
        "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
        "-DCMAKE_INSTALL_RPATH=@loader_path;@loader_path/..;@loader_path/jaccl;@loader_path/../python/mlx/lib",
    ]


def require_cache(cache_path, source, build, jit, python=sys.executable):
    expected = {}
    for item in configure_command(source, build, jit, python):
        if item.startswith("-D"):
            name, value = item[2:].split("=", 1)
            expected[name] = value
    expected.update(CMAKE_C_FLAGS="", CMAKE_CXX_FLAGS="")
    observed = {}
    for line in cache_path.read_text().splitlines():
        if not line or line.startswith(("#", "//")):
            continue
        match = re.fullmatch(r"([^:=]+):[^=]+=(.*)", line)
        if not match:
            raise ValueError("Malformed CMake cache line")
        name, value = match.groups()
        if name in observed:
            raise ValueError("Duplicate CMake cache key")
        observed[name] = value
    for name, value in expected.items():
        if observed.get(name) != value:
            raise ValueError(f"Effective CMake setting differs: {name}")
    for name in (
        "CMAKE_C_COMPILER_LAUNCHER",
        "CMAKE_CXX_COMPILER_LAUNCHER",
        "CMAKE_TOOLCHAIN_FILE",
    ):
        if observed.get(name, ""):
            raise ValueError(f"Undeclared compiler cache or toolchain: {name}")
    return {
        "sha256": digest(cache_path),
        "required_settings": expected,
        "undeclared_launchers_or_toolchains": False,
    }


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def retain_archive(artifact, publish, label):
    archive = publish / f"mlx-release-{label}-macos14-arm64-cp312.tar.gz"
    partial = archive.with_suffix(archive.suffix + ".partial")
    if archive.exists() or partial.exists():
        raise ValueError("Refusing to overwrite a runtime archive")
    with (
        partial.open("xb") as stream,
        tarfile.open(fileobj=stream, mode="w:gz", compresslevel=1) as tar,
    ):
        tar.add(artifact, arcname=label, recursive=True)
    partial.replace(archive)
    return {
        "name": archive.name,
        "sha256": digest(archive),
        "bytes": archive.stat().st_size,
        "manifest_sha256": digest(artifact / "MANIFEST.json"),
        "retained_before_execution_probe": True,
        "qualification": "unvalidated",
    }


def verify_archive(archive, artifact, label):
    expected = inventory(artifact)
    observed = set()
    import hashlib

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            parts = member.name.split("/")
            if (
                not parts
                or parts[0] != label
                or "\\" in member.name
                or any(p in {"", ".", ".."} for p in parts)
            ):
                raise ValueError("Unexpected archive path")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Unexpected nonregular archive member")
            relative = "/".join(parts[1:])
            if relative in observed or relative not in expected:
                raise ValueError("Duplicate or extra archive member")
            observed.add(relative)
            value = hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(block)
            row = expected[relative]
            if member.size != row["bytes"] or value.hexdigest() != row["sha256"]:
                raise ValueError("Archive payload identity differs")
            if bool(member.mode & 0o111) != row["executable"]:
                raise ValueError("Archive executable permission differs")
    if observed != set(expected):
        raise ValueError("Archive inventory incomplete")
    return {"verified": True, "files": len(expected), "sha256": digest(archive)}


def check_relocation(logs, label, build, artifact):
    package = artifact / "runtime/python/mlx"
    cores = list(package.glob("core*.so"))
    if len(cores) != 1:
        raise ValueError("Ambiguous packaged core")
    for index, binary in enumerate(
        (cores[0], package / "lib/libmlx.dylib", artifact / "runtime/bin/tests")
    ):
        audit = subprocess.check_output(["otool", "-L", str(binary)], text=True)
        (logs / f"{label}-otool-{index}.txt").write_text(audit)
        commands = subprocess.check_output(["otool", "-l", str(binary)], text=True)
        (logs / f"{label}-load-commands-{index}.txt").write_text(commands)
    for index in range(3):
        dependencies = (logs / f"{label}-otool-{index}.txt").read_text()
        commands = (logs / f"{label}-load-commands-{index}.txt").read_text()
        if str(build) in dependencies or str(build) in commands:
            raise ValueError("Nonrelocatable build-library path")
        if index == 0 and "path @loader_path/lib (offset " not in commands:
            raise ValueError("Missing upstream extension RPATH")


def require_transport(execution, record, row):
    if (
        type(execution.get("returncode")) is not int
        or execution["returncode"] != 0
        or execution.get("timed_out")
        or record.get("status") != "passed"
        or "error" in record
        or record.get("manifest_sha256") != row["artifact"]["manifest_sha256"]
        or record.get("variant") != row["variant"]
        or record.get("mode") != row["mode"]
        or record.get("device_numerics_validated") is not False
        or record.get("performance_measured") is not False
        or not record.get("native_libraries")
    ):
        raise RuntimeError(
            "Transported runtime import failed or receipt mismatched; archive retained"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--parallel", type=int, choices=(2,), default=2)
    args = parser.parse_args()
    root = args.work_root.resolve()
    host = hosted_identity(os.environ, root)
    if root.exists() or root.is_symlink():
        raise ValueError("Use a fresh isolated work root")
    if (
        platform.system() != "Darwin"
        or platform.machine() != "arm64"
        or sys.version_info[:2] != (3, 12)
    ):
        raise ValueError("Requires native macOS arm64 CPython3.12")
    if sys.prefix == sys.base_prefix:
        raise ValueError("Requires a dedicated interpreter environment")
    root.mkdir(parents=True)
    logs, source, build, publish = (root / n for n in ("logs", "source", "build", "publish"))
    logs.mkdir()
    publish.mkdir()
    env = controlled_environment(root)
    rows = [
        {"id": f"{v}-{m}", "variant": v, "mode": m, "jit": j, "status": "planned"}
        for v in VARIANTS
        for m, j in MODES
    ]
    summary = {
        "schema": CONTRACT,
        "base": BASE,
        "patches": PATCHES,
        "cells": rows,
        "status": "running",
        "build_and_transport_complete": False,
        "device_numerics_validated": False,
        "performance_measured": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "source_sha256": inventory(HERE),
        "hosted_identity": host,
        "scope": (
            "Four matched Release runtime builds; "
            "operator performance and device tests remain separate"
        ),
    }

    def save():
        atomic_json(publish / "summary.json", summary)

    save()
    try:
        lock = HERE / "dependencies-exact.txt"
        if digest(lock) != DEPENDENCY_LOCK_SHA256:
            raise ValueError("Frozen dependency lock changed")
        dependencies = dependency_map(lock.read_text())
        expected_patch = verify_patch_inputs(HERE / "inputs")
        write(logs / "storage-before.json", storage(root))
        for name, command in (
            ("developer-directory", ["xcode-select", "-p"]),
            ("metal-compiler", ["xcrun", "-sdk", "macosx", "metal", "--version"]),
            ("macos-sdk", ["xcrun", "-sdk", "macosx", "--show-sdk-version"]),
            ("xcode-version", ["xcodebuild", "-version"]),
            ("clang-version", ["xcrun", "clang", "--version"]),
        ):
            run(command, cwd=root, output=logs, name=name, env=env, timeout=120)
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                "--no-cache-dir",
                "--report",
                str(logs / "dependency-install-report.json"),
                "-r",
                str(lock),
            ],
            cwd=root,
            output=logs,
            name="dependencies",
            env=env,
            timeout=1800,
        )
        frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], env=env)
        (logs / "runtime-lock.txt").write_bytes(frozen)
        if dependency_map(frozen.decode()) != dependencies:
            raise ValueError("Installed dependencies differ from exact frozen lock")
        run(["git", "init", str(source)], cwd=root, output=logs, name="git-init", env=env)
        run(
            ["git", "remote", "add", "origin", "https://github.com/ml-explore/mlx.git"],
            cwd=source,
            output=logs,
            name="git-origin",
            env=env,
        )
        run(
            ["git", "fetch", "--depth=1", "origin", BASE],
            cwd=source,
            output=logs,
            name="git-fetch",
            env=env,
        )
        run(
            ["git", "checkout", "--detach", BASE],
            cwd=source,
            output=logs,
            name="git-checkout",
            env=env,
        )
        pristine = source_identity(source)
        summary["pristine_source_sha256"] = pristine
        summary["toolchain_logs"] = {p.name: digest(p) for p in logs.glob("*-version.log")}
        for variant in VARIANTS:
            expected = {} if variant == "main" else expected_patch
            if variant == "patched":
                check_source(source, {})
                for name in PATCHES:
                    run(
                        ["git", "apply", "--check", str(HERE / "inputs" / name)],
                        cwd=source,
                        output=logs,
                        name=name + "-check",
                        env=env,
                    )
                    run(
                        ["git", "apply", str(HERE / "inputs" / name)],
                        cwd=source,
                        output=logs,
                        name=name + "-apply",
                        env=env,
                    )
            check_source(source, expected)
            before = source_identity(source)
            changed = {n for n in set(before) | set(pristine) if before.get(n) != pristine.get(n)}
            if changed != set(expected):
                raise ValueError("Source tree changed outside declared patch paths")
            for row in (r for r in rows if r["variant"] == variant):
                label, mode, jit = row["id"], row["mode"], row["jit"]
                row["status"] = "running"
                save()
                try:
                    write(logs / f"{label}-storage-before.json", storage(root))
                    check_source(source, expected)
                    if source_identity(source) != before:
                        raise ValueError("Source changed before build")
                    command = configure_command(source, build, jit)
                    run(
                        command,
                        cwd=root,
                        output=logs,
                        name=f"{label}-configure",
                        env=env,
                        timeout=1800,
                    )
                    row["cache_after_configure"] = require_cache(
                        build / "CMakeCache.txt", source, build, jit
                    )
                    run(
                        [
                            str(Path(sys.executable).parent / "cmake"),
                            "--build",
                            str(build),
                            "--parallel",
                            "2",
                            "--target",
                            "core",
                            "tests",
                        ],
                        cwd=root,
                        output=logs,
                        name=f"{label}-build",
                        env=env,
                        timeout=5400,
                    )
                    row["cache_after_build"] = require_cache(
                        build / "CMakeCache.txt", source, build, jit
                    )
                    write(logs / f"{label}-storage-after-build.json", storage(root))
                    if source_identity(source) != before:
                        raise ValueError("Build changed source")
                    artifact = root / "artifacts" / label
                    provenance = {
                        "base": BASE,
                        "variant": variant,
                        "patches": PATCHES if expected else {},
                        "source_sha256": before,
                        "pristine_source_sha256": pristine,
                        "build_mode": mode,
                        "configure_command": command,
                        "normal_offline_compiler": True,
                        "validation_build_substitutions": False,
                        "python": sys.version,
                        "platform": platform.platform(),
                        "toolchain_logs": summary["toolchain_logs"],
                        "dependency_lock_sha256": digest(logs / "runtime-lock.txt"),
                        "hosted_identity": host,
                        "cmake_cache_confirmation": row["cache_after_build"],
                    }
                    package_mode(source, build, artifact, mode, provenance, env, logs)
                    packaged_identity = inventory(artifact)
                    row["artifact"] = retain_archive(artifact, publish, label)
                    save()
                    row["archive_verification"] = verify_archive(
                        publish / row["artifact"]["name"], artifact, label
                    )
                    check_relocation(logs, label, build, artifact)
                    hidden = root / "build-relocation-hold"
                    build.rename(hidden)
                    try:
                        probe = logs / f"{label}-transport.json"
                        execution = run(
                            [
                                sys.executable,
                                "-I",
                                "-S",
                                str(HERE / "transport_probe.py"),
                                "--bundle",
                                str(artifact),
                                "--dependencies",
                                str(Path(sys.prefix) / "lib/python3.12/site-packages"),
                                "--output",
                                str(probe),
                            ],
                            cwd=root,
                            output=logs,
                            name=f"{label}-transport-command",
                            env=env,
                            timeout=300,
                            check=False,
                        )
                        row["transport_execution"] = execution
                        row["transport"] = (
                            json.loads(probe.read_text())
                            if probe.is_file()
                            else {"status": "missing"}
                        )
                        require_transport(execution, row["transport"], row)
                    finally:
                        hidden.rename(build)
                    if inventory(artifact) != packaged_identity:
                        raise RuntimeError("Packaged bytes changed during import")
                    row["artifact"]["qualification"] = (
                        "archive-and-import-validated; device-numerics-unvalidated"
                    )
                    row["status"] = "passed"
                except BaseException as error:
                    row.update(status="failed", error=f"{type(error).__name__}: {error}")
                save()
        if inventory(HERE) != summary["source_sha256"]:
            raise RuntimeError("Build helper/input source changed during execution")
        summary["build_and_transport_complete"] = all(r["status"] == "passed" for r in rows)
        summary["status"] = "passed" if summary["build_and_transport_complete"] else "failed"
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        if summary["status"] != "passed":
            for row in rows:
                if row["status"] in {"planned", "running"}:
                    row["status"] = "not_completed_after_abort"
        save()
        shutil.copytree(logs, publish / "logs")
        shutil.copytree(HERE, publish / "technical-inputs")
        write(publish / "publication-files.json", inventory(publish))
    return 0 if summary["build_and_transport_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
