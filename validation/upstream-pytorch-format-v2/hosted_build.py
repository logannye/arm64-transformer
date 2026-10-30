"""Prospective own-project CI helper. Do not run before environment approval.

Builds the complete framework only through the mandated editable pip command.
It does not submit upstream contributions or run device numerical tests.
"""

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

BASE = "71d2a08507ab47dd6b2fa9b66319d17ce2377de4"
PATCH_SHA256 = "3834dca844ea3de6b40467ce214b99bce83fa06dae2bc454623516dafc0fdd71"
SOURCES = {
    "aten/src/ATen/native/mps/kernels/Embedding.metal": (
        "e49bd60d9a396272af948ec4a482987916a9952118447b5c5bc01e41350a5e85"
    ),
    "aten/src/ATen/native/mps/operations/Embedding.mm": (
        "e7e6fca63cc984cfd61b7c2333b35971446144b9488c9375beda59b905118310"
    ),
    "test/test_mps.py": "7345b6e229aed240921460a3b598ffd1a47148ab4816aead0d405539a154a396",
    "torch/__init__.py": "18053b2d5cf73c20fee2f29173ba4219b586c92e15f00a5b14c7dacd90bd35c9",
}
BUILD_ENV = {
    "MAX_JOBS": "2",
    "CMAKE_BUILD_PARALLEL_LEVEL": "2",
    "DEBUG": "0",
    "REL_WITH_DEB_INFO": "0",
    "CMAKE_BUILD_TYPE": "Release",
    "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
    "USE_DISTRIBUTED": "0",
    "USE_CUDA": "0",
    "USE_MPS": "1",
    "USE_OPENMP": "1",
    "USE_PYTORCH_METAL": "1",
    "BUILD_TEST": "OFF",
    "WERROR": "1",
    "TORCH_NATIVE_AOT": "0",
    "SKBUILD_EDITABLE_REBUILD": "false",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "UV_NO_BUILD": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
GIB = 1024**3


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def inventory(folder):
    return {
        str(path.relative_to(folder)): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def read_cache(path):
    values = {}
    for line in path.read_text().splitlines():
        if line.startswith(("//", "#")) or ":" not in line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.split(":", 1)[0]] = value
    return values


class Run:
    def __init__(self, source, output):
        self.source = source
        self.output = output
        self.env = os.environ.copy()
        for name in ("PYTHONPATH", "PYTHONHOME", "DYLD_INSERT_LIBRARIES", "CMAKE_ARGS"):
            self.env.pop(name, None)
        self.env.update(BUILD_ENV)
        self.env.update(
            PIP_CACHE_DIR=str(output / "cache/pip"),
            UV_CACHE_DIR=str(output / "cache/uv"),
            UV_PYTHON_INSTALL_DIR=str(output / "cache/uv-python"),
        )
        self.record = {
            "schema": "pytorch-hosted-integration-v1",
            "patch_revision": "format-only-v2",
            "prior_patch_sha256": (
                "3dd6bf6567284d798074e7d6102ff7f8b2a051a3a4ff2824f86184c120eb0ce9"
            ),
            "status": "running",
            "scope": (
                "Own-project full-framework build/lint/import validation; no GPU numerical tests"
            ),
            "source_revision": BASE,
            "patch_sha256": PATCH_SHA256,
            "source": str(source),
            "output": str(output),
            "build_environment": BUILD_ENV,
            "commands": [],
            "github": {
                key: os.environ.get(key)
                for key in (
                    "GITHUB_REPOSITORY",
                    "GITHUB_SHA",
                    "GITHUB_RUN_ID",
                    "GITHUB_RUN_ATTEMPT",
                    "GITHUB_WORKFLOW_REF",
                    "RUNNER_OS",
                    "RUNNER_ARCH",
                    "ImageOS",
                    "ImageVersion",
                )
            },
        }

    def save(self):
        write_json(self.output / "receipt.json", self.record)

    def command(self, label, argv, *, cwd=None, timeout=1200, check=True):
        log = self.output / f"{len(self.record['commands']):02d}-{label}.log"
        item = {
            "label": label,
            "argv": list(map(str, argv)),
            "cwd": str(cwd or self.source),
            "log": log.name,
            "status": "running",
        }
        self.record["commands"].append(item)
        self.save()
        start = time.monotonic()
        with log.open("w") as stream:
            process = subprocess.Popen(
                argv,
                cwd=cwd or self.source,
                env=self.env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            last_storage_check = 0
            try:
                while process.poll() is None:
                    if time.monotonic() - start > timeout:
                        raise TimeoutError(f"Command exceeded {timeout}s: {label}")
                    if shutil.disk_usage(self.output).free < 8 * GIB:
                        raise RuntimeError("Stopped to preserve 8 GiB free disk space")
                    if time.monotonic() - last_storage_check > 60:
                        sizes = subprocess.check_output(
                            ["du", "-sk", str(self.source), str(self.output)], text=True
                        )
                        allocated = sum(int(line.split()[0]) * 1024 for line in sizes.splitlines())
                        self.record["last_allocated_bytes"] = allocated
                        if allocated > 60 * GIB:
                            raise RuntimeError(
                                "Source/build/cache/output exceeded the 60 GiB allocation budget"
                            )
                        last_storage_check = time.monotonic()
                    time.sleep(5)
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                item["status"] = "interrupted"
                raise
            finally:
                item.update(exit_code=process.returncode, elapsed_seconds=time.monotonic() - start)
                self.save()
        item["status"] = "passed" if process.returncode == 0 else "failed"
        item["log_sha256"] = sha256(log)
        self.save()
        if check and process.returncode:
            raise RuntimeError(f"Command failed: {label}; see {log}")
        return log.read_text(errors="replace")

    def source_guard(self):
        head = self.command("source-head", ["git", "rev-parse", "HEAD"]).strip()
        if head != BASE:
            raise RuntimeError(f"Expected source base {BASE}, got {head}")
        changed = set(
            self.command("source-diff-names", ["git", "diff", "--name-only"]).splitlines()
        )
        if changed != set(SOURCES):
            raise RuntimeError(f"Unexpected modified source inventory: {sorted(changed)}")
        for name, expected in SOURCES.items():
            if sha256(self.source / name) != expected:
                raise RuntimeError(f"Source hash mismatch: {name}")

    def retain_runtime(self, runtime):
        """Retain unqualified build bytes before any import or lint can fail."""
        archive = self.output / "pytorch-integrated-runtime.tar.gz"
        partial = archive.with_suffix(archive.suffix + ".partial")
        if archive.exists() or partial.exists():
            raise RuntimeError("Refusing to replace an existing runtime archive")
        size = sum(path.stat().st_size for path in runtime.rglob("*") if path.is_file())
        # Allow an uncompressed-sized archive plus tar headers and the same
        # reserve used during compilation. Compression is not assumed.
        if shutil.disk_usage(self.output).free < size + 9 * GIB:
            raise RuntimeError("Insufficient disk reserve to retain the runtime archive")
        with tarfile.open(partial, "w:gz") as target:
            target.add(runtime, arcname="runtime")
        partial.rename(archive)
        self.record["runtime_archive"] = {
            "file": archive.name,
            "sha256": sha256(archive),
            "bytes": archive.stat().st_size,
            "manifest_sha256": sha256(runtime / "runtime-manifest.json"),
            "qualification": "unvalidated",
            "retained_before_validation": True,
        }
        self.save()
        return archive

    def run(self, patch):
        if (
            os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_REPOSITORY") != "logannye/arm64-transformer"
        ):
            raise RuntimeError("This helper is restricted to explicitly dispatched own-project CI")
        if (
            sys.platform != "darwin"
            or platform.machine() != "arm64"
            or sys.version_info[:2] != (3, 12)
        ):
            raise RuntimeError("Select a macOS arm64 CPython 3.12 builder")
        if shutil.disk_usage(self.output).free < 40 * GIB:
            raise RuntimeError("At least 40 GiB free disk space is required before preparation")
        self.record["python"] = sys.version
        self.record["platform"] = platform.platform()
        self.record["free_before_bytes"] = shutil.disk_usage(self.output).free
        self.record["helpers"] = inventory(Path(__file__).parent)
        if sha256(patch) != PATCH_SHA256:
            raise RuntimeError("Patch bytes differ from the reviewed technical artifact")
        if (self.source / "build").exists() or (self.source / ".venv").exists():
            raise RuntimeError("Use a fresh hosted checkout, one build tree, and one environment")
        if self.command(
            "clean-source", ["git", "status", "--porcelain", "--untracked-files=no"]
        ).strip():
            raise RuntimeError("Hosted source is not clean before applying the patch")
        if self.command("base", ["git", "rev-parse", "HEAD"]).strip() != BASE:
            raise RuntimeError("Hosted source base differs from the frozen revision")
        self.command("patch-check", ["git", "apply", "--check", str(patch)])
        self.command("patch-apply", ["git", "apply", str(patch)])
        self.source_guard()
        for label, command in (
            ("xcode-selection", ["xcode-select", "-p"]),
            ("xcode-version", ["xcodebuild", "-version"]),
            ("metal-path", ["xcrun", "--find", "metal"]),
            ("metal-version", ["xcrun", "metal", "--version"]),
            ("clang-version", ["xcrun", "clang", "--version"]),
            ("sdk-version", ["xcrun", "--sdk", "macosx", "--show-sdk-version"]),
            ("libomp-package", ["brew", "info", "--json=v2", "libomp"]),
        ):
            self.command(label, command)
        omp = self.command("libomp-prefix", ["brew", "--prefix", "libomp"]).strip()
        if not (Path(omp) / "lib/libomp.dylib").is_file():
            raise RuntimeError("libomp must be provisioned before this helper runs")
        self.env["CMAKE_PREFIX_PATH"] = omp
        self.record["libomp_prefix"] = omp
        self.command(
            "submodules",
            ["git", "submodule", "update", "--init", "--recursive", "--depth", "1", "--jobs", "2"],
            timeout=3600,
        )
        submodules = self.command(
            "submodule-identities", ["git", "submodule", "status", "--recursive"]
        )
        if any(line.startswith(("-", "+", "U")) for line in submodules.splitlines()):
            raise RuntimeError("A submodule is missing or differs from its pinned revision")
        environment = self.source / ".venv"
        self.command("create-environment", [sys.executable, "-m", "venv", str(environment)])
        python = environment / "bin/python"
        pip = environment / "bin/pip"
        self.env["PATH"] = str(environment / "bin") + os.pathsep + self.env["PATH"]
        self.env["VIRTUAL_ENV"] = str(environment)
        wheels = self.output / "wheels"
        wheels.mkdir()
        constraints = Path(__file__).with_name("hosted-constraints.txt")
        self.command(
            "download-dependencies",
            [
                str(pip),
                "download",
                "--only-binary=:all:",
                "--dest",
                str(wheels),
                "--constraint",
                str(constraints),
                "--requirement",
                "requirements.txt",
                "pytest",
                "pytest-xdist",
                "pytest-subtests",
                "pytest-rerunfailures",
                "uv",
            ],
            timeout=1800,
        )
        self.record["wheel_inventory"] = inventory(wheels)
        write_json(self.output / "wheel-inventory.json", self.record["wheel_inventory"])
        self.command(
            "install-dependencies",
            [
                str(pip),
                "install",
                "--no-index",
                "--find-links",
                str(wheels),
                "--only-binary=:all:",
                "--constraint",
                str(constraints),
                "--requirement",
                "requirements.txt",
                "pytest",
                "pytest-xdist",
                "pytest-subtests",
                "pytest-rerunfailures",
                "uv",
                "--report",
                str(self.output / "dependency-install.json"),
            ],
        )
        self.command("dependency-freeze", [str(pip), "freeze", "--all"])
        self.command("dependency-check", [str(pip), "check"])
        self.command("spin-help", [str(environment / "bin/spin"), "help"])
        self.source_guard()
        self.env["PIP_NO_INDEX"] = "1"
        self.env["PIP_FIND_LINKS"] = str(wheels)
        # This is the only command that builds PyTorch. No wheel/build/cmake
        # entry point, operator override, or automatic import rebuild is used.
        self.command(
            "full-framework-build",
            [str(pip), "install", "-e", ".", "-v", "--no-build-isolation"],
            timeout=18000,
        )
        self.source_guard()
        build = self.source / "build"
        cache = read_cache(build / "CMakeCache.txt")
        if cache.get("CAN_COMPILE_METAL") not in {"YES", "ON", "TRUE", "1"}:
            raise RuntimeError("The requested offline Metal build path was not enabled")
        if cache.get("USE_MPS") not in {"YES", "ON", "TRUE", "1"}:
            raise RuntimeError("MPS was disabled in the full build")
        self.record["cmake_cache"] = cache
        shutil.copy2(build / "CMakeCache.txt", self.output / "CMakeCache.txt")
        shutil.copy2(build / "compile_commands.json", self.output / "compile_commands.json")
        commands = json.loads((build / "compile_commands.json").read_text())
        source = (self.source / "aten/src/ATen/native/mps/operations/Embedding.mm").resolve()
        matches = []
        for item in commands:
            directory = Path(item["directory"])
            filename = Path(item["file"])
            if not filename.is_absolute():
                filename = directory / filename
            if filename.resolve() != source:
                continue
            arguments = item.get("arguments") or shlex.split(item["command"])
            object_path = Path(arguments[arguments.index("-o") + 1])
            if not object_path.is_absolute():
                object_path = directory / object_path
            matches.append(
                {
                    "compile_command": item,
                    "source_sha256": sha256(source),
                    "object_file": str(object_path),
                    "object_sha256": sha256(object_path),
                }
            )
        if not matches:
            raise RuntimeError("Embedding.mm was absent from the full-framework compile database")
        self.record["embedding_translation_units"] = matches
        air = list(build.rglob("Embedding_32.air"))
        if len(air) != 1:
            raise RuntimeError(f"Expected one offline Embedding_32.air object, found {air}")
        self.record["embedding_metal_object"] = {
            "path": str(air[0]),
            "sha256": sha256(air[0]),
            "source_sha256": SOURCES["aten/src/ATen/native/mps/kernels/Embedding.metal"],
        }
        installed = list((build / "install").glob("**/torch/_C.*.so"))
        if len(installed) != 1:
            raise RuntimeError(f"Expected one editable install tree, found {installed}")
        install_root = installed[0].parent.parent
        runtime = self.output / "runtime"
        runtime.mkdir()
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".DS_Store")
        for package in ("torch", "torchgen", "functorch"):
            shutil.copytree(self.source / package, runtime / package, ignore=ignore)
            if (install_root / package).exists():
                shutil.copytree(
                    install_root / package, runtime / package, dirs_exist_ok=True, ignore=ignore
                )
        sites = list((environment / "lib").glob("python3.12/site-packages"))
        if len(sites) != 1:
            raise RuntimeError("Expected one isolated dependency site-packages directory")
        site = sites[0]
        metadata = list(site.glob("torch-*.dist-info"))
        if len(metadata) != 1:
            raise RuntimeError("Expected one full-framework metadata directory")
        shutil.copytree(metadata[0], runtime / metadata[0].name)
        files = inventory(runtime)
        if sum(item["bytes"] for item in files.values()) > 8 * GIB:
            raise RuntimeError("Runtime archive staging exceeded its 8 GiB budget")
        write_json(
            runtime / "runtime-manifest.json",
            {"source_revision": BASE, "patch_sha256": PATCH_SHA256, "files": files},
        )
        self.record["editable_install_root"] = str(install_root)
        self.record["runtime_bytes"] = sum(item["bytes"] for item in files.values())
        archive = self.retain_runtime(runtime)
        self.command(
            "normal-editable-import",
            [
                str(python),
                "-I",
                "-c",
                "import json,torch; print(json.dumps({'file':torch.__file__,"
                "'extension':torch._C.__file__,'git_version':torch.version.git_version,"
                "'mps_built':torch.backends.mps.is_built(),"
                "'loaded_libraries':sorted(torch.ops.loaded_libraries)}))",
            ],
            cwd=self.output,
        )
        verifier = Path(__file__).with_name("verify_runtime.py")
        self.command(
            "transported-runtime-import",
            [
                str(python),
                "-I",
                "-S",
                str(verifier),
                "--runtime",
                str(runtime),
                "--dependencies",
                str(site),
                "--expected-base",
                BASE,
                "--receipt",
                str(self.output / "runtime-import.json"),
            ],
            cwd=self.output,
        )
        macho = []
        for path in runtime.rglob("*"):
            if not path.is_file():
                continue
            with path.open("rb") as stream:
                magic = stream.read(4)
            if magic in {
                b"\xcf\xfa\xed\xfe",
                b"\xfe\xed\xfa\xcf",
                b"\xca\xfe\xba\xbe",
                b"\xbe\xba\xfe\xca",
            }:
                deps = self.command("dylib-dependencies", ["otool", "-l", str(path)])
                load_command = ""
                for line in deps.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("cmd "):
                        load_command = stripped.split()[1]
                    elif load_command in {
                        "LC_LOAD_DYLIB",
                        "LC_LOAD_WEAK_DYLIB",
                        "LC_REEXPORT_DYLIB",
                        "LC_LAZY_LOAD_DYLIB",
                        "LC_RPATH",
                    } and stripped.startswith(("name ", "path ")):
                        dependency = stripped.split(" ", 1)[1].split(" (offset ", 1)[0]
                        if dependency.startswith("/") and not dependency.startswith(
                            ("/System/Library/", "/usr/lib/")
                        ):
                            raise RuntimeError(
                                f"Nonportable absolute dylib dependency or rpath: "
                                f"{path}: {dependency}"
                            )
                macho.append(
                    {
                        "path": str(path.relative_to(runtime)),
                        "sha256": sha256(path),
                        "dependencies": deps,
                    }
                )
        self.record["native_libraries"] = macho
        # Verify the bytes actually transported, not only the staging directory.
        extracted = self.output / "archive-check"
        extracted.mkdir()
        with tarfile.open(archive, "r:gz") as source_archive:
            source_archive.extractall(extracted, filter="data")
        self.command(
            "archive-roundtrip-import",
            [
                str(python),
                "-I",
                "-S",
                str(verifier),
                "--runtime",
                str(extracted / "runtime"),
                "--dependencies",
                str(site),
                "--expected-base",
                BASE,
                "--receipt",
                str(self.output / "archive-roundtrip-import.json"),
            ],
            cwd=self.output,
        )
        shutil.rmtree(extracted)
        self.record["runtime_archive"]["qualification"] = (
            "import-and-portability-validated; device-numerics-unvalidated"
        )
        self.save()
        # Keep all lint evidence, including base findings. No rule exclusions.
        self.record["lint_scope"] = (
            "Unmodified upstream spin lint: all fast checks on all files; "
            "slow checks on changed files; no rule exclusions"
        )
        try:
            self.command(
                "upstream-spin-lint", [str(environment / "bin/spin"), "lint"], timeout=3600
            )
        finally:
            lintbin = self.source / ".lintbin"
            self.record["lint_binary_inventory"] = inventory(lintbin) if lintbin.exists() else {}
            self.record["lint_configuration"] = {
                name: sha256(self.source / name)
                for name in (
                    ".lintrunner.toml",
                    ".spin/cmds.py",
                    "tools/linter/adapters/s3_init_config.json",
                    "tools/linter/adapters/ruff_linter.py",
                )
            }
        self.source_guard()
        self.record.update(
            status="passed",
            device_numerical_tests="not_executed",
            upstream_ci="not_executed",
            free_after_bytes=shutil.disk_usage(self.output).free,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = Run(args.source.resolve(strict=True), output)
    try:
        run.run(args.patch.resolve(strict=True))
    except BaseException:
        run.record["status"] = "failed"
        run.record["exception"] = traceback.format_exc()
        raise
    finally:
        run.save()


if __name__ == "__main__":
    main()
