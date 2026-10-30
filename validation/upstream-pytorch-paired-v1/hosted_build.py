"""Prospective own-project CI helper. Do not run before environment approval.

Builds the complete framework only through the mandated editable pip command.
It does not submit upstream contributions or run device numerical tests.
"""

import argparse
import copy
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

from control import (
    libomp_identity,
    load_control_contract,
    normalize_tool,
    require,
    retain_test_source,
    submodule_identities,
)

BASE = "71d2a08507ab47dd6b2fa9b66319d17ce2377de4"
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
    def __init__(self, source, output, variant="pristine"):
        self.variant = variant
        variant_contract = load_control_contract()["variants"][variant]
        self.patch_sha256 = variant_contract["patch_sha256"]
        self.expected_sources = variant_contract["sources"]
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
            "schema": "pytorch-hosted-paired-stage-v1",
            "source_variant": variant,
            "formal_eligible": False,
            "comparison_contract_sha256": sha256(Path(__file__).with_name("contract.json")),
            "status": "running",
            "scope": (
                "Own-project full-framework build/lint/import validation; no GPU numerical tests"
            ),
            "source_revision": BASE,
            "patch_sha256": self.patch_sha256,
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
        if self.command("source-index-diff", ["git", "diff", "--cached", "--name-only"]).strip():
            raise RuntimeError("Pristine source index differs")
        if changed != (set(self.expected_sources) if self.variant == "formatted" else set()):
            raise RuntimeError(f"Unexpected modified source inventory: {sorted(changed)}")
        for name, expected in self.expected_sources.items():
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

    def setup(self):
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
        contract = load_control_contract()
        require(sys.dont_write_bytecode, "Invoke this producer with Python -B")
        require(
            self.record["build_environment"] == contract["build_environment"],
            "Declared build controls differ",
        )
        self.record["reference_control"] = contract["reference_hosted_run"]
        if (self.source / "build").exists() or (self.source / ".venv").exists():
            raise RuntimeError("Use a fresh hosted checkout, one build tree, and one environment")
        if self.command(
            "clean-source", ["git", "status", "--porcelain", "--untracked-files=no"]
        ).strip():
            raise RuntimeError("Hosted source is not pristine")
        if self.command("base", ["git", "rev-parse", "HEAD"]).strip() != BASE:
            raise RuntimeError("Hosted source base differs from the frozen revision")
        # This checkout remains pristine; no patch or operator override is applied.
        self.record["patch_applied"] = False
        self.source_guard()
        self.record["toolchain_initial"] = self.observe_tools()
        omp = self.record["toolchain_initial"]["tools"]["libomp-prefix"].strip()
        require((Path(omp) / "lib/libomp.dylib").is_file(), "Missing binary OpenMP runtime")
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
        require(
            submodule_identities(submodules)
            == submodule_identities(contract["submodule_identities"]),
            "Recursive base submodules differ",
        )
        environment = self.source / ".venv"
        self.command("create-environment", [sys.executable, "-m", "venv", str(environment)])
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
                "--constraint",
                str(Path(__file__).with_name("requirements-full.txt")),
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
        require(
            self.record["wheel_inventory"] == contract["wheel_inventory"],
            "Downloaded wheels differ from formatted control",
        )
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
                "--constraint",
                str(Path(__file__).with_name("requirements-full.txt")),
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
        freeze = self.command("dependency-freeze", [str(pip), "freeze", "--all"])
        require(freeze == contract["dependency_freeze"], "Installed dependency freeze differs")
        self.command("dependency-check", [str(pip), "check"])
        self.command("spin-help", [str(environment / "bin/spin"), "help"])
        self.source_guard()
        self.env["PIP_NO_INDEX"] = "1"
        self.env["PIP_FIND_LINKS"] = str(wheels)
        self.record["setup_complete"] = True
        self.save()
        return environment

    def observe_tools(self):
        tools = {}
        for label, argv in (
            ("xcode-selection", ["xcode-select", "-p"]),
            ("xcode-version", ["xcodebuild", "-version"]),
            ("metal-path", ["xcrun", "--find", "metal"]),
            ("metal-version", ["xcrun", "metal", "--version"]),
            ("clang-version", ["xcrun", "clang", "--version"]),
            ("sdk-version", ["xcrun", "--sdk", "macosx", "--show-sdk-version"]),
            ("libomp-package", ["brew", "info", "--json=v2", "libomp"]),
            ("libomp-prefix", ["brew", "--prefix", "libomp"]),
        ):
            stdout = self.command("pair-" + label, argv)
            if label == "libomp-package":
                tools[label] = libomp_identity(json.loads(stdout))
            elif label != "metal-path":
                tools[label] = normalize_tool(stdout)
        return {
            "python": sys.version,
            "platform": platform.platform(),
            "image": {
                key: os.environ.get(key)
                for key in ["RUNNER_OS", "RUNNER_ARCH", "ImageOS", "ImageVersion", "RUNNER_NAME"]
            },
            "tools": tools,
            "build_environment": dict(BUILD_ENV),
        }

    def package_runtime(self, environment):
        python = environment / "bin/python"
        build = self.source / "build"
        cache = read_cache(build / "CMakeCache.txt")
        if cache.get("CAN_COMPILE_METAL") not in {"YES", "ON", "TRUE", "1"}:
            raise RuntimeError("The requested offline Metal build path was not enabled")
        if cache.get("USE_MPS") not in {"YES", "ON", "TRUE", "1"}:
            raise RuntimeError("MPS was disabled in the full build")
        require(cache.get("CMAKE_BUILD_TYPE") == "Release", "Expected actual Release build")
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
            "source_sha256": self.expected_sources[
                "aten/src/ATen/native/mps/kernels/Embedding.metal"
            ],
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
            {"source_revision": BASE, "patch_sha256": self.patch_sha256, "files": files},
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

    def build_variant(self, environment, contract, toolchain, source_wheels):
        pip = environment / "bin/pip"
        self.record["build_mode"] = (
            "pristine_full" if self.variant == "pristine" else "formatted_incremental"
        )
        self.record["toolchain_before"] = self.observe_tools()
        require(self.record["toolchain_before"] == toolchain, "Within-pair toolchain changed")
        self.source_guard()
        freeze = self.command(
            "dependencies-before-build", [str(pip), "freeze", "--all", "--exclude", "torch"]
        )
        require(
            freeze == contract["dependency_freeze"], "Dependency environment changed before build"
        )
        self.record["wheel_inventory"] = inventory(source_wheels)
        require(
            self.record["wheel_inventory"] == contract["wheel_inventory"],
            "Shared wheel payload changed",
        )
        retain_test_source(self, contract)
        self.command(
            "full-framework-build",
            [str(pip), "install", "-e", ".", "-v", "--no-build-isolation"],
            timeout=18000,
        )
        self.source_guard()
        self.package_runtime(environment)
        after = self.command(
            "dependencies-after-build", [str(pip), "freeze", "--all", "--exclude", "torch"]
        )
        require(
            after == contract["dependency_freeze"], "Dependency environment changed after build"
        )
        self.record["toolchain_after"] = self.observe_tools()
        require(
            self.record["toolchain_after"] == toolchain, "Within-pair toolchain changed after build"
        )
        self.record["native_import_portability_passed"] = True
        self.record["lint_scope"] = (
            "Unchanged upstream bare spin lint; no exclusions; failure retained"
        )
        self.command(
            "upstream-spin-lint", [str(environment / "bin/spin"), "lint"], timeout=3600, check=False
        )
        self.record["lint_binary_inventory"] = inventory(self.source / ".lintbin")
        self.source_guard()
        # Resolve the unique lint command by identity, not its index among source-guard commands.
        lint = [row for row in self.record["commands"] if row["label"] == "upstream-spin-lint"]
        require(len(lint) == 1, "Missing or duplicate unchanged lint invocation")
        self.record["status"] = (
            "passed" if lint[0]["status"] == "passed" else "completed_with_lint_failures"
        )
        self.record["device_numerical_tests"] = "not_executed"
        self.save()


def historical_drift(current, contract):
    previous = contract["host"]
    report = {}
    for key in ["python", "platform"]:
        if current[key] != previous[key]:
            report[key] = {"historical": previous[key], "pair": current[key]}
    for key, value in previous["image"].items():
        if current["image"].get(key) != value:
            report["image/" + key] = {"historical": value, "pair": current["image"].get(key)}
    for key, value in contract["tool_checks"].items():
        if current["tools"].get(key) != value["normalized_stdout"]:
            report["tools/" + key] = {
                "historical": value["normalized_stdout"],
                "pair": current["tools"].get(key),
            }
    if current["tools"]["libomp-package"] != contract["libomp"]:
        report["libomp"] = {
            "historical": contract["libomp"],
            "pair": current["tools"]["libomp-package"],
        }
    return report


def validate_pair_records(records, contract):
    require(
        [row["source_variant"] for row in records] == ["pristine", "formatted"], "Wrong pair order"
    )
    require(
        all(row.get("native_import_portability_passed") is True for row in records),
        "Incomplete native pair",
    )
    require(
        records[0]["toolchain_before"]
        == records[0]["toolchain_after"]
        == records[1]["toolchain_before"]
        == records[1]["toolchain_after"],
        "Unmatched pair toolchain",
    )
    keys = list(contract["cmake_cache"])
    require(
        all(
            key in row["cmake_cache"] and isinstance(row["cmake_cache"][key], str)
            for row in records
            for key in keys
        ),
        "Missing or malformed paired CMake setting",
    )
    require(
        {k: records[0]["cmake_cache"].get(k) for k in keys}
        == {k: records[1]["cmake_cache"].get(k) for k in keys},
        "CMake compiler/configuration changed within pair",
    )
    require(
        records[0]["embedding_metal_object"]["sha256"]
        != records[1]["embedding_metal_object"]["sha256"],
        "Patched Metal object did not change",
    )
    first = records[0]["embedding_translation_units"]
    second = records[1]["embedding_translation_units"]
    require(
        len(first) == len(second)
        and first
        and all(
            a["compile_command"] == b["compile_command"]
            and a["object_file"] == b["object_file"]
            and a["object_sha256"] != b["object_sha256"]
            for a, b in zip(first, second)
        ),
        "Embedding translation unit/configuration was not rebuilt in the shared tree",
    )
    require(
        all(row["wheel_inventory"] == contract["wheel_inventory"] for row in records),
        "Pair wheels differ",
    )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        os.environ.get("GITHUB_ACTIONS") == "true"
        and os.environ.get("GITHUB_REPOSITORY") == "logannye/arm64-transformer",
        "Only authorized own-repository hosted execution is supported",
    )
    require(sys.dont_write_bytecode, "Invoke this producer with Python -B")
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    setup = Run(source, output)
    contract = load_control_contract()
    pair = {
        "schema": "pytorch-same-host-pair-v1",
        "status": "running",
        "complete": False,
        "evidence_valid": False,
        "diagnostic_control_ready": False,
        "formal_eligible": False,
        "source_revision": BASE,
        "variant_order": ["pristine", "formatted"],
        "variants": [],
        "scope": "Retrospective27-failed-test attribution; one hosted environment/build tree",
        "contract_sha256": sha256(Path(__file__).with_name("contract.json")),
    }
    try:
        environment = setup.setup()
        setup_record = copy.deepcopy(setup.record)
        write_json(output / "setup.json", setup_record)
        toolchain = setup_record["toolchain_initial"]
        pair["historical_drift"] = historical_drift(toolchain, contract)
        for variant in pair["variant_order"]:
            destination = output / variant
            destination.mkdir()
            stage = Run(source, destination, variant)
            stage.env = setup.env.copy()
            stage.record["patch_applied"] = variant == "formatted"
            stage.record["shared_environment"] = str(environment)
            stage.record["shared_build_tree"] = str(source / "build")
            row = {"variant": variant, "status": "running", "receipt": f"{variant}/receipt.json"}
            pair["variants"].append(row)
            write_json(output / "pair.json", pair)
            try:
                if variant == "formatted":
                    patch = Path(__file__).with_name("changes.patch")
                    require(sha256(patch) == stage.patch_sha256, "Formatted patch changed")
                    stage.command("patch-check", ["git", "apply", "--check", str(patch)])
                    stage.command("patch-apply", ["git", "apply", str(patch)])
                stage.build_variant(environment, contract, toolchain, output / "wheels")
                row["status"] = stage.record["status"]
            except BaseException:
                stage.record.update(status="failed", exception=traceback.format_exc())
                row["status"] = "failed"
                raise
            finally:
                stage.save()
                row["receipt_sha256"] = sha256(destination / "receipt.json")
                write_json(output / "pair.json", pair)
        records = [json.loads((output / row["receipt"]).read_text()) for row in pair["variants"]]
        validate_pair_records(records, contract)
        for row, record in zip(pair["variants"], records):
            destination = output / row["variant"]
            require(
                sha256(destination / "receipt.json") == row["receipt_sha256"],
                "Completed stage receipt changed",
            )
            for key in ["runtime_archive", "test_source_archive"]:
                payload = destination / record[key]["file"]
                require(
                    payload.stat().st_size == record[key]["bytes"]
                    and sha256(payload) == record[key]["sha256"],
                    "Retained variant archive changed",
                )
        require(
            inventory(output / "wheels") == contract["wheel_inventory"],
            "Shared wheelhouse changed during pair",
        )
        pair.update(
            complete=True,
            evidence_valid=True,
            diagnostic_control_ready=True,
            status="passed"
            if all(row["status"] == "passed" for row in records)
            else "completed_with_lint_failures",
        )
    except BaseException:
        pair.update(status="failed", exception=traceback.format_exc())
        raise
    finally:
        setup.save()
        write_json(output / "pair.json", pair)
    if pair["status"] != "passed":
        raise RuntimeError(
            "Both native variants retained; upstream lint failures remain unqualified"
        )


if __name__ == "__main__":
    main()
