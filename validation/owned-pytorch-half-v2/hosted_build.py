"""Owned candidate full-framework build, restricted to the project hosted workflow.

Builds the complete framework only through the mandated editable pip command.
It does not submit upstream contributions or run device numerical tests.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

BASE = "71d2a08507ab47dd6b2fa9b66319d17ce2377de4"
CONTRACT_SHA256 = "3bed6c1c84973b2214bb9bd700a48c9ed7644044dff01ea5d7a0de91013f4a03"
PATCH_SHA256 = "07128b3a9bbe39f4e423ef7e038f955c52cbc562d8edb9ed28c4baac59c49b3f"
SOURCES = {
    "torch/headeronly/util/Half.h": (
        "2e74ae13d3fee30371703cefa35a7eaab466cca90c8fd5908e6389e8034a2c90"
    ),
    "torch/headeronly/util/complex.h": (
        "30ae971d71fcc3cbb032bb15507bd965540a31a3c496632582a7d66cafb38a34"
    ),
    "aten/src/ATen/native/IndexingUtils.h": (
        "f80c4d74255e4fb61122aa6e951e4242f1129651ef254f18a3a66ac832ef421a"
    ),
    "aten/src/ATen/native/TensorAdvancedIndexing.cpp": (
        "5b9ecb298cd20325d310956863376374588a5cb91954b90f2eeefb2f549da799"
    ),
    "aten/src/ATen/native/TensorAdvancedIndexingUtils.h": (
        "f9797ba1146e009ec173acc62dac8f8bf2ad08cd4516ef5bd350519560405bc6"
    ),
    "aten/src/ATen/native/mps/kernels/Embedding.metal": (
        "e49bd60d9a396272af948ec4a482987916a9952118447b5c5bc01e41350a5e85"
    ),
    "aten/src/ATen/native/mps/kernels/Indexing.h": (
        "b79813a2f6c08cdd5eb6dfaf53083109d40662595dd2c955e922746e85fac2a3"
    ),
    "aten/src/ATen/native/mps/kernels/Indexing.metal": (
        "cb1442517a5243ab185f638ac1db0b7b0edc3a5b21f79b839e1a1d941534d92f"
    ),
    "aten/src/ATen/native/mps/operations/Convolution.mm": (
        "ba4df4baa3aa61d0edb3f5ec8e0033ecd983dfac3925655fb51eb01cfcfa775f"
    ),
    "aten/src/ATen/native/mps/operations/Embedding.mm": (
        "e7e6fca63cc984cfd61b7c2333b35971446144b9488c9375beda59b905118310"
    ),
    "aten/src/ATen/native/mps/operations/Indexing.mm": (
        "3ed688b52acef54d3429daf500c316574c5e42961d01afd9ed3674cbcb481897"
    ),
    "aten/src/ATen/native/mps/operations/Matmul.mm": (
        "8cfac720468f10145bce549e97071a89a983fc4c8a0cce6b14e8c0aae5bedcc1"
    ),
    "test/test_mps.py": ("a7b6420664a9352dcb1fc7eb070744c4f1f869d7498d036e5d73ef83dab27bc7"),
    "torch/__init__.py": ("a3ac1f3819e4ebddb8d9d71bf1490c335daaa3118ea8ea1300192146f0cf02b2"),
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
    "CFLAGS": "-g0",
    "CXXFLAGS": "-g0",
    "OBJCFLAGS": "-g0",
    "OBJCXXFLAGS": "-g0",
}
NATIVE_EVIDENCE = {
    "fresh_device_validation_required": True,
    "metal_objects": ["Embedding_32.air", "Indexing_32.air"],
    "previous_runtime_evidence_transfer_allowed": False,
    "translation_units": [
        "aten/src/ATen/native/mps/operations/Embedding.mm",
        "aten/src/ATen/native/mps/operations/Matmul.mm",
        "aten/src/ATen/native/mps/operations/Convolution.mm",
        "aten/src/ATen/native/mps/operations/Indexing.mm",
        "aten/src/ATen/native/TensorAdvancedIndexing.cpp",
    ],
    "upstream_clang_format_darwin_arm64_sha256": (
        "f0da3ecf0ab1e9b50e8c27bd2d7ca0baa619e2f4b824b35d79d46356581fa552"
    ),
    "upstream_clang_format_version": "19.1.4",
}
TEST_POLICY = {
    "actual_runtime_collection": "required after native build; AST preservation is not "
    "runtime collection",
    "added_exact_methods": 4,
    "allowed_existing_method_body_change": (
        "TestAdvancedIndexing.test_index_put_accumulate_nondeterministic"
    ),
    "existing_ids_eligibility_tolerances_unchanged": True,
    "inventory_sha256": "453e770813d31b5f45fe07421d7ab0dbbe9c7fe8ab84bb284097a3ce486511aa",
    "new_methods": [
        "TestMPS.test_index_put_half_ordered_layouts_and_chunks",
        "TestMPS.test_index_put_half_ordered_multiple_index_strides",
        "TestMPS.test_index_put_half_ordered_repeated_index_backward",
        "TestMPS.test_index_put_half_ordered_storage_words_and_tail",
    ],
    "original_result_sha256": "eab61f146499b51c30e9cf4ec4b422914f2c89accebd3ef8fbde4dedd68919e2",
    "previous_exact_methods": 10353,
    "previous_expected_failures": 382,
    "previous_skipped": 744,
    "required_exact_methods": 10357,
    "unchanged_test_module_ast_sha256": (
        "9ba8de955cb5b0cb1c8101bde990daf8aba35b63f0fb73566ab88ae7812c672a"
    ),
}
HALF_STORAGE_PROBE = {
    "original_headers": {
        "torch/headeronly/util/Half.h": (
            "6c02317b90c2d1265b7d8d29d3f1f39e28abf41b6d94afce4d91c297623a90c6"
        ),
        "torch/headeronly/util/complex.h": (
            "f28ba904a1087a15f33fadf4b2b7ec8165a061d84a6d2f28f863c75d410ee642"
        ),
    },
    "all_65536_encodings": True,
    "arithmetic_partners_per_encoding": 16,
    "raw_results_and_fenv_exact": True,
    "cxx_standard": 17,
    "platform": "ARM64 CPU",
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


def load_contract(folder=None):
    folder = Path(folder) if folder is not None else Path(__file__).parent
    path = folder / "contract.json"
    if sha256(path) != CONTRACT_SHA256:
        raise RuntimeError("Owned build contract changed")
    contract = json.loads(path.read_text())
    for name, key in (
        ("changes.patch", "combined_patch_sha256"),
        ("requirements-full.txt", "requirements_full_sha256"),
        ("hosted-constraints.txt", "constraints_sha256"),
        ("verify_runtime.py", "runtime_verifier_sha256"),
    ):
        if sha256(folder / name) != contract[key]:
            raise RuntimeError(f"Frozen build input changed: {name}")
    for name, expected in contract["static_input_sha256"].items():
        if sha256(folder / name) != expected:
            raise RuntimeError(f"Frozen build input changed: {name}")
    validate_build_contract(contract, BUILD_ENV, SOURCES)
    return contract


def validate_build_contract(contract, environment, sources):
    if (
        contract.get("schema") != "pytorch-owned-half-hosted-contract-v2"
        or contract.get("source_revision") != BASE
        or contract.get("combined_patch_sha256") != PATCH_SHA256
        or contract.get("formal_eligible") is not False
        or contract.get("admission_stream_opened") is not False
    ):
        raise RuntimeError("Wrong owned candidate identity or qualification")
    if sources != SOURCES or sources != contract.get("source_sha256") or len(sources) != 14:
        raise RuntimeError("Changed or incomplete fourteen-file source inventory")
    if contract.get("half_storage_probe") != HALF_STORAGE_PROBE:
        raise RuntimeError("Half storage probe requirements changed")
    if contract["native_evidence_required"] != NATIVE_EVIDENCE:
        raise RuntimeError("Native compilation evidence requirements changed")
    if contract["test_inventory_policy"] != TEST_POLICY:
        raise RuntimeError("Prospective test policy changed")
    if environment != BUILD_ENV or environment != contract.get("build_environment"):
        raise RuntimeError("Build environment differs from frozen contract")
    if contract.get("budgets") != {
        "workers": 2,
        "initial_free_gib": 40,
        "reserve_gib": 8,
        "maximum_allocation_gib": 60,
        "maximum_source_payload_gib": 8,
        "maximum_runtime_payload_gib": 8,
        "timeout_minutes": 360,
    }:
        raise RuntimeError("Build budget changed")


def validate_wheels(actual, contract):
    if actual != contract["wheel_inventory"]:
        raise RuntimeError("Dependency wheel bytes or complete inventory changed")


def validate_compilation(cache):
    if cache.get("CMAKE_BUILD_TYPE") != "Release":
        raise RuntimeError("Actual native build must be Release")
    for key in ("CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS", "CMAKE_OBJC_FLAGS", "CMAKE_OBJCXX_FLAGS"):
        flags = shlex.split(cache.get(key, ""))
        if "-g0" not in flags or any(flag.startswith("-g") and flag != "-g0" for flag in flags):
            raise RuntimeError(f"Debug information is not disabled in {key}")


def validate_compile_arguments(arguments):
    if "-g0" not in arguments or any(flag.startswith("-g") and flag != "-g0" for flag in arguments):
        raise RuntimeError("Changed translation unit has unexpected debug flags")


class Run:
    def __init__(self, source, output):
        self.source = source
        self.output = output
        self.env = os.environ.copy()
        for name in (
            "PYTHONPATH",
            "PYTHONHOME",
            "DYLD_INSERT_LIBRARIES",
            "CMAKE_ARGS",
            "PYTORCH_MPS_METAL_VERSION",
            "PYTORCH_MPS_PREFER_METAL",
            "USE_CUSTOM_DEBINFO",
        ):
            self.env.pop(name, None)
        self.env.update(BUILD_ENV)
        self.env.update(
            PIP_CACHE_DIR=str(output / "cache/pip"),
            UV_CACHE_DIR=str(output / "cache/uv"),
            UV_PYTHON_INSTALL_DIR=str(output / "cache/uv-python"),
        )
        self.record = {
            "schema": "pytorch-owned-half-hosted-v2",
            "protocol": "gpu-v4-owned-development-v1",
            "patch_revision": "owned-embedding-layout-half-explicit-storage-v2",
            "formal_eligible": False,
            "candidate_qualified": False,
            "admission_stream_opened": False,
            "contract_sha256": CONTRACT_SHA256,
            "component_patches": load_contract()["component_patches"],
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

    def check_half_storage(self, contract):
        """Compile header-only probes on the authorized hosted runner only."""
        root = self.output / "half-probe"
        root.mkdir()
        original = root / "original"
        for name, expected in contract["half_storage_probe"]["original_headers"].items():
            target = original / name
            target.parent.mkdir(parents=True, exist_ok=True)
            content = self.command("original-header", ["git", "show", BASE + ":" + name])
            target.write_text(content)
            if sha256(target) != expected:
                raise RuntimeError("Original Half header differs from the frozen base")
        probe = Path(__file__).with_name("half_probe.cpp")
        artifacts = {}
        for variant, include in (("original", original), ("candidate", self.source)):
            executable = root / variant
            if variant == "original":
                executable = root / "original-probe"
            arguments = [
                "xcrun",
                "clang++",
                "-std=c++17",
                "-O3",
                "-g0",
                "-DC10_USING_CUSTOM_GENERATED_MACROS",
                "-I",
                str(include),
                "-I",
                str(self.source),
                str(probe),
            ]
            self.command(variant + "-half-compile", [*arguments, "-o", str(executable)])
            assembly = root / (variant + ".s")
            self.command(variant + "-half-assembly", [*arguments, "-S", "-o", str(assembly)])
            data = root / (variant + ".bin")
            self.command(variant + "-half-run", [str(executable), str(data)])
            artifacts[variant] = {
                "raw_sha256": sha256(data),
                "raw_bytes": data.stat().st_size,
                "executable_sha256": sha256(executable),
                "assembly_sha256": sha256(assembly),
            }
        self.record["half_storage_probe"] = {
            "artifacts": artifacts,
            "all_65536_encodings": True,
            "arithmetic_partners_per_encoding": 16,
            "raw_result_bits_and_fenv_retained": True,
            "cxx_standard": 17,
            "platform": "hosted ARM64 CPU; CUDA/HIP/SYCL/x86 unvalidated",
            "passed": artifacts["original"]["raw_sha256"] == artifacts["candidate"]["raw_sha256"],
        }
        self.save()
        if not self.record["half_storage_probe"]["passed"]:
            raise RuntimeError("Half storage change altered result bits or floating-point flags")

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

    def retain_source(self, patch, submodules):
        """Archive tracked root/submodule bytes before building; omit Git/build caches."""
        archive = self.output / "pytorch-owned-source.tar.gz"
        partial = archive.with_suffix(archive.suffix + ".partial")
        if archive.exists() or partial.exists():
            raise RuntimeError("Refusing to overwrite retained source")
        names = self.command(
            "tracked-source-inventory",
            ["git", "ls-files", "--recurse-submodules", "-z"],
        ).split("\0")
        names = [name for name in names if name]
        if len(names) != len(set(names)):
            raise RuntimeError("Duplicate tracked source member")
        records = {}
        total = 0
        for name in sorted(names):
            rel = Path(name)
            if rel.is_absolute() or ".." in rel.parts or ".git" in rel.parts:
                raise RuntimeError(f"Unsafe source member: {name}")
            path = self.source / name
            info = path.lstat()
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                target = os.readlink(path)
                records[name] = {"kind": "symlink", "target": target, "mode": mode}
            elif stat.S_ISREG(info.st_mode):
                records[name] = {
                    "kind": "file",
                    "sha256": sha256(path),
                    "bytes": info.st_size,
                    "mode": mode,
                }
                total += info.st_size
            else:
                raise RuntimeError(f"Unsupported tracked source member: {name}")
        if total > 8 * GIB or shutil.disk_usage(self.output).free < total + 9 * GIB:
            raise RuntimeError("Insufficient bounded storage for source retention")
        manifest = {
            "schema": "pytorch-owned-source-archive-v1",
            "source_revision": BASE,
            "patch_sha256": PATCH_SHA256,
            "submodules": submodules,
            "files": records,
            "regular_file_bytes": total,
            "qualification": "source_bytes_only_candidate_unqualified",
        }
        manifest_path = self.output / "source-manifest.json"
        write_json(manifest_path, manifest)
        with tarfile.open(partial, "w:gz", dereference=False) as target:
            for name in sorted(records):
                target.add(self.source / name, arcname=f"source/{name}", recursive=False)
            target.add(manifest_path, arcname="source-manifest.json", recursive=False)
            target.add(patch, arcname="changes.patch", recursive=False)
        # Authenticate the emitted archive, rather than trusting only input files.
        with tarfile.open(partial, "r:gz") as emitted:
            members = emitted.getmembers()
            if {item.name for item in members} != {
                *(f"source/{name}" for name in records),
                "source-manifest.json",
                "changes.patch",
            } or len(members) != len(records) + 2:
                raise RuntimeError("Emitted source member inventory changed")
            for member in members:
                if not member.name.startswith("source/"):
                    expected = manifest_path if member.name == "source-manifest.json" else patch
                    stream = emitted.extractfile(member)
                    if stream is None or hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest() != sha256(expected):
                        raise RuntimeError("Emitted source metadata changed")
                    continue
                record = records[member.name.removeprefix("source/")]
                if member.mode != record["mode"]:
                    raise RuntimeError("Emitted source mode changed")
                if record["kind"] == "symlink":
                    if not member.issym() or member.linkname != record["target"]:
                        raise RuntimeError("Emitted source symlink changed")
                else:
                    stream = emitted.extractfile(member)
                    if (
                        not member.isfile()
                        or member.size != record["bytes"]
                        or stream is None
                        or hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]
                    ):
                        raise RuntimeError("Emitted source bytes changed")
        self.source_guard()
        partial.rename(archive)
        self.record["source_archive"] = {
            "file": archive.name,
            "sha256": sha256(archive),
            "bytes": archive.stat().st_size,
            "manifest_sha256": sha256(manifest_path),
            "members": len(records),
            "qualification": "source_bytes_only_candidate_unqualified",
        }
        self.save()

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
        contract = load_contract()
        validate_build_contract(contract, BUILD_ENV, SOURCES)
        for name in (
            "changes.patch",
            "contract.json",
            "hosted-constraints.txt",
            "requirements-full.txt",
            "verify_runtime.py",
            "hosted_build.py",
            "source_static.py",
            "test-inventory.json",
            "half_probe.cpp",
        ):
            shutil.copy2(Path(__file__).with_name(name), self.output / name)
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
        static_path = Path(__file__).with_name("source_static.py")
        spec = importlib.util.spec_from_file_location("owned_half_source_static", static_path)
        static = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(static)
        self.record["source_static"] = static.validate_source(self.source, contract)
        write_json(self.output / "source-static.json", self.record["source_static"])
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
        self.retain_source(patch, submodules)
        self.check_half_storage(contract)
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
                "--no-deps",
                "--requirement",
                str(Path(__file__).with_name("requirements-full.txt")),
            ],
            timeout=1800,
        )
        self.record["wheel_inventory"] = inventory(wheels)
        validate_wheels(self.record["wheel_inventory"], contract)
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
                "--no-deps",
                "--requirement",
                str(Path(__file__).with_name("requirements-full.txt")),
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
        validate_compilation(cache)
        self.record["cmake_cache"] = cache
        shutil.copy2(build / "CMakeCache.txt", self.output / "CMakeCache.txt")
        shutil.copy2(build / "compile_commands.json", self.output / "compile_commands.json")
        commands = json.loads((build / "compile_commands.json").read_text())
        translation_units = {}
        for relative in contract["native_evidence_required"]["translation_units"]:
            source = (self.source / relative).resolve()
            name = relative
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
                validate_compile_arguments(arguments)
                matches.append(
                    {
                        "compile_command": item,
                        "source_sha256": sha256(source),
                        "object_file": str(object_path),
                        "object_sha256": sha256(object_path),
                    }
                )
            if not matches:
                raise RuntimeError(f"{name} absent from full-framework compile database")
            translation_units[name] = matches
        self.record["changed_translation_units"] = translation_units
        metal_objects = {}
        for name in contract["native_evidence_required"]["metal_objects"]:
            air = list(build.rglob(name))
            if len(air) != 1:
                raise RuntimeError(f"Expected one offline {name} object, found {air}")
            source_name = name.removesuffix("_32.air") + ".metal"
            metal_objects[name] = {
                "path": str(air[0]),
                "sha256": sha256(air[0]),
                "source_sha256": SOURCES["aten/src/ATen/native/mps/kernels/" + source_name],
            }
        self.record["changed_metal_objects"] = metal_objects
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
        formatter = self.source / ".lintbin/clang-format"
        if sha256(formatter) != NATIVE_EVIDENCE["upstream_clang_format_darwin_arm64_sha256"]:
            raise RuntimeError("Authoritative upstream clang-format binary differs")
        self.record["authoritative_clang_format"] = {
            "sha256": sha256(formatter),
            "version": self.command("clang-format-version", [str(formatter), "--version"]).strip(),
        }
        if "19.1.4" not in self.record["authoritative_clang_format"]["version"]:
            raise RuntimeError("Authoritative upstream clang-format version differs")
        self.source_guard()
        self.record.update(
            status="build_lint_import_passed_candidate_unqualified",
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
