"""Prospective official MLX three-mode integration matrix, version 1.

Builds CPU-only, Metal JIT ON, then OFF in one reusable build directory. Both Metal modes use
the normal upstream offline compiler and preamble generator. No historical
metallib, preamble, installed MLX binary, or validation-only patch is accepted.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from integration_common import digest, inventory, run, write
from validate_bundle import validate

BASE = "a21f892a6e8e83afb117cef1d1ceae3e82f48c32"
CONTRACT = "mlx_official_three_mode_integration_v1"
MODES = (("cpu-only", "OFF", "OFF"), ("jit-on", "ON", "ON"), ("jit-off", "ON", "OFF"))
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
    if len(headers) != 250 or sum(path.stat().st_size for path in headers) != 1837067:
        raise ValueError("Pinned public-header inventory must be 250 files / 1837067 bytes")
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
        "file_count": 250,
        "bytes": 1837067,
        "headers": identities,
    }


def package_mode(source, build, output, mode, provenance, env, logs):
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
    for name in ("integration_common.py", "runtime_probe.py", "validate_bundle.py"):
        shutil.copy2(HERE / name, tools / name)
    shutil.copytree(HERE / "inputs", output / "inputs")
    shutil.copy2(HERE / "CONTRACT.md", output / "CONTRACT.md")
    shutil.copy2(HERE / "hosted-requirements.txt", output / "hosted-requirements.txt")
    shutil.copy2(logs / "runtime-lock.txt", output / "runtime-lock.txt")
    shutil.copy2(logs / "dependency-install-report.json", output / "dependency-install-report.json")
    shutil.copy2(build / "CMakeCache.txt", output / "CMakeCache.txt")
    write(output / "PROVENANCE.json", provenance)
    for index, binary in enumerate(
        (python_package / core[0].name, lib / "libmlx.dylib", output / "runtime/bin/tests")
    ):
        audit = subprocess.check_output(["otool", "-L", str(binary)], text=True)
        (logs / f"{mode}-otool-{index}.txt").write_text(audit)
        if str(build) in audit:
            raise ValueError(f"Nonrelocatable build-library reference: {binary}")
        load_commands = subprocess.check_output(["otool", "-l", str(binary)], text=True)
        (logs / f"{mode}-load-commands-{index}.txt").write_text(load_commands)
        if str(build) in load_commands:
            raise ValueError(f"Nonrelocatable build RPATH: {binary}")
        if index == 0 and "path @loader_path/lib (offset " not in load_commands:
            raise ValueError("Core extension is missing its upstream @loader_path/lib RPATH")
    dependency_names = ("numpy", "torch", "tensorflow", "ml_dtypes", "pytest", "pytest-xdist")
    write(
        output / "MANIFEST.json",
        {
            "schema": 1,
            "scope": CONTRACT,
            "mode": mode,
            "base": BASE,
            "patches": PATCHES,
            "runtime_dependencies": {n: importlib.metadata.version(n) for n in dependency_names},
            "files": inventory(output),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()
    root = args.work_root.resolve()
    if root.exists() or args.parallel not in range(1, 9):
        raise ValueError("Use a new isolated work root and 1–8 compilation workers")
    if (
        platform.system() != "Darwin"
        or platform.machine() != "arm64"
        or sys.version_info[:2] != (3, 12)
    ):
        raise ValueError("Build requires native macOS arm64 CPython 3.12")
    if sys.prefix == sys.base_prefix:
        raise ValueError("Run inside a new dedicated virtual environment")
    root.mkdir(parents=True)
    logs, source, build, publish = (root / name for name in ("logs", "source", "build", "publish"))
    logs.mkdir()
    publish.mkdir()
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PRE_COMMIT_HOME=str(root / "pre-commit-cache"),
        PIP_NO_CACHE_DIR="1",
        DEVICE="cpu",
        VECLIB_MAXIMUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    summary = {
        "scope": CONTRACT,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "qualified": False,
        "modes": {},
        "required_modes": [mode for mode, _, _ in MODES],
        "prior_cpu_on_metal_diagnostics": {
            "status": "retained_failures_not_reclassified",
            "evidence_directory": "work/upstream-integration-v2/mlx",
            "scope": (
                "additional CPU-default execution on a Metal-enabled build; "
                "outside the upstream CI matrix"
            ),
        },
        "outside_scope": [
            "distributed MPI/ring tests",
            "example-extension builds",
            "array-API external suite",
            "Linux/CUDA/Windows",
            "other macOS deployment targets",
        ],
    }
    expected = verify_patch_inputs(HERE / "inputs")
    try:
        write(logs / "storage-before.json", storage(root))
        for name, command in (
            ("developer-directory", ["xcode-select", "-p"]),
            ("metal-compiler", ["xcrun", "-sdk", "macosx", "metal", "--version"]),
            ("macos-sdk", ["xcrun", "-sdk", "macosx", "--show-sdk-version"]),
            ("xcode-version", ["xcodebuild", "-version"]),
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
                str(HERE / "hosted-requirements.txt"),
            ],
            cwd=root,
            output=logs,
            name="dependencies",
            env=env,
            timeout=1800,
        )
        (logs / "runtime-lock.txt").write_bytes(
            subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], env=env)
        )
        write(logs / "storage-after-dependencies.json", storage(root))
        installed = {
            d.metadata["Name"].lower().replace("_", "-") for d in importlib.metadata.distributions()
        }
        if installed & {"mlx", "mlx-metal", "mlx-cuda"}:
            raise RuntimeError("Integration dependencies must not install an alternate MLX")
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
        # Include new regression files in pre-commit's all-files inventory, without a commit.
        run(
            ["git", "add", "--intent-to-add", "--", *sorted(expected)],
            cwd=source,
            output=logs,
            name="lint-inventory",
            env=env,
        )
        before = source_identity(source)
        lint = run(
            [sys.executable, "-m", "pre_commit", "run", "--all-files", "--show-diff-on-failure"],
            cwd=source,
            output=logs,
            name="pre-commit-all",
            env=env,
            timeout=1800,
            check=False,
        )
        write(
            logs / "lint-source-identity.json", {"before": before, "after": source_identity(source)}
        )
        if lint["returncode"] or source_identity(source) != before:
            raise RuntimeError("Pinned upstream hooks failed or changed source; no build allowed")
        check_source(source, expected)
        for mode, metal, jit in MODES:
            row = {"status": "failed", "qualified": False}
            summary["modes"][mode] = row
            try:
                write(logs / f"{mode}-storage-before.json", storage(root))
                check_source(source, expected)
                command = [
                    str(Path(sys.executable).parent / "cmake"),
                    "-S",
                    str(source),
                    "-B",
                    str(build),
                    "-DCMAKE_BUILD_TYPE=Debug",
                    "-DCMAKE_CXX_FLAGS_DEBUG=-O0 -g0",
                    "-DCMAKE_C_FLAGS_DEBUG=-O0 -g0",
                    "-DCMAKE_COMPILE_WARNING_AS_ERROR=ON",
                    "-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0",
                    "-DBUILD_SHARED_LIBS=ON",
                    "-DMLX_EXPORT_ALL_SYMBOLS=ON",
                    "-DMLX_BUILD_CPU=ON",
                    f"-DMLX_BUILD_METAL={metal}",
                    "-DMLX_BUILD_CUDA=OFF",
                    f"-DMLX_METAL_JIT={jit}",
                    "-DMLX_BUILD_TESTS=ON",
                    "-DMLX_BUILD_PYTHON_BINDINGS=ON",
                    "-DMLX_BUILD_PYTHON_STUBS=OFF",
                    "-DMLX_BUILD_EXAMPLES=OFF",
                    "-DMLX_BUILD_BENCHMARKS=OFF",
                    "-DMLX_USE_CCACHE=OFF",
                    f"-DPython_EXECUTABLE={sys.executable}",
                    f"-DMLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY={build / 'python/mlx'}",
                    "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
                    # CMake executes tests for discovery before packaging: allow
                    # build/tests -> build/libmlx and build/libmlx -> build/jaccl.
                    "-DCMAKE_INSTALL_RPATH=@loader_path;@loader_path/..;@loader_path/jaccl;@loader_path/../python/mlx/lib",
                ]
                run(command, cwd=root, output=logs, name=f"{mode}-configure", env=env, timeout=1800)
                run(
                    [
                        str(Path(sys.executable).parent / "cmake"),
                        "--build",
                        str(build),
                        "--parallel",
                        str(args.parallel),
                        "--target",
                        "core",
                        "tests",
                    ],
                    cwd=root,
                    output=logs,
                    name=f"{mode}-build",
                    env=env,
                    timeout=5400,
                )
                if source_identity(source) != before:
                    raise RuntimeError("Build changed source files")
                write(logs / f"{mode}-storage-after-build.json", storage(root))
                artifact = root / "artifacts" / mode
                provenance = {
                    "base": BASE,
                    "patches": PATCHES,
                    "source_sha256": before,
                    "build_mode": mode,
                    "configure_command": command,
                    "normal_offline_compiler": metal == "ON",
                    "metal_backend": metal,
                    "validation_build_substitutions": False,
                    "python": sys.version,
                    "platform": platform.platform(),
                    "hook_configuration_sha256": digest(source / ".pre-commit-config.yaml"),
                }
                package_mode(source, build, artifact, mode, provenance, env, logs)
                # Make compiled-in build paths unavailable during packaged execution.
                hidden = root / f"build-relocation-hold-{mode}"
                build.rename(hidden)
                try:
                    try:
                        result = validate(artifact, root / "checks" / mode)
                        row.update(status="checked", qualified=result["qualified"], checks=result)
                    except BaseException as error:
                        row["validation_error"] = f"{type(error).__name__}: {error}"
                finally:
                    hidden.rename(build)
                archive = publish / f"mlx-{mode}-macos14-arm64-cp312.tar.gz"
                with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
                    tar.add(artifact, arcname=mode, recursive=True)
                # The directory's manifest verifies the payload after download/extraction.
                row["artifact"] = {
                    "name": archive.name,
                    "sha256": digest(archive),
                    "bytes": archive.stat().st_size,
                }
                write(logs / f"{mode}-storage-after.json", storage(root))
            except BaseException as error:
                row["error"] = f"{type(error).__name__}: {error}"
            write(logs / f"{mode}-result.json", row)
        summary["qualified"] = set(summary["modes"]) == {m for m, _, _ in MODES} and all(
            r["qualified"] for r in summary["modes"].values()
        )
    except BaseException as error:
        summary["error"] = f"{type(error).__name__}: {error}"
    finally:
        write(root / "summary.json", summary)
        write(publish / "summary.json", summary)
        shutil.copytree(logs, publish / "logs")
        if (root / "checks").is_dir():
            shutil.copytree(root / "checks", publish / "checks")
        shutil.copytree(HERE / "inputs", publish / "inputs")
        write(publish / "publication-files.json", inventory(publish))
    return 0 if summary["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
