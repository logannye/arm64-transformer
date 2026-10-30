"""Check a transported full-framework archive without editable import hooks.

Run with Python -I -S. This script performs import/registration checks only;
real MPS numerical integration tests remain a separate required stage.
"""

import argparse
import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import traceback
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_module_origins(runtime, expected_files, modules, torch_module):
    """Authenticate real files and the two exact registered PyTorch namespaces.

    _Ops/_Classes inherit ModuleType but advertise synthetic relative filenames.
    Do not resolve those labels against the working directory or blanket-exempt
    relative paths: bind their singleton, exact class and constructor to the
    real, manifest-authenticated backing module instead.
    """
    runtime = Path(runtime).resolve(strict=True)
    origins, synthetic = {}, {}

    def real_origin(name, module):
        attributes = vars(module)
        source = attributes.get("__file__", getattr(type(module), "__file__", None))
        spec = attributes.get("__spec__")
        spec_origin = getattr(spec, "origin", None)
        if not source:
            if spec_origin and spec_origin not in {"built-in", "frozen"}:
                source = spec_origin
            else:
                return None
        path = Path(source).resolve(strict=True)
        if not path.is_relative_to(runtime):
            raise RuntimeError(f"Framework module imported outside archive: {name}: {path}")
        relative = path.relative_to(runtime).as_posix()
        expected = expected_files.get(relative)
        if not expected or not path.is_file():
            raise RuntimeError(f"Framework module is absent from runtime manifest: {name}: {path}")
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Framework module bytes differ from manifest: {name}: {path}")
        if spec_origin and spec_origin not in {"built-in", "frozen"}:
            if Path(spec_origin).resolve(strict=True) != path:
                raise RuntimeError(f"Framework module spec/file origins disagree: {name}")
        origins[name] = relative
        return path

    definitions = {
        "torch.ops": ("torch._ops", "_Ops", "ops", "_ops.py"),
        "torch.classes": ("torch._classes", "_Classes", "classes", "_classes.py"),
    }
    if modules.get("torch") is not torch_module:
        raise RuntimeError("Torch module identity changed")
    for name, (backing_name, class_name, singleton_name, label) in definitions.items():
        module = modules.get(name)
        backing = modules.get(backing_name)
        if module is None or backing is None:
            raise RuntimeError(f"Missing registered PyTorch namespace or backing module: {name}")
        cls = vars(backing).get(class_name)
        if (
            type(module) is not cls
            or vars(torch_module).get(singleton_name) is not module
            or vars(backing).get(singleton_name) is not module
            or cls.__module__ != backing_name
            or vars(cls).get("__file__") != label
            or vars(module).get("__file__", label) != label
            or vars(module).get("__name__") != name
            or vars(module).get("__spec__") is not None
            or vars(module).get("__loader__") is not None
        ):
            raise RuntimeError(f"Unrecognized synthetic PyTorch namespace: {name}")
        path = real_origin(backing_name, backing)
        expected_path = runtime / "torch" / label
        constructor = vars(cls).get("__init__")
        code = getattr(constructor, "__code__", None)
        if (
            path != expected_path
            or code is None
            or Path(code.co_filename).resolve(strict=True) != path
        ):
            raise RuntimeError(
                f"Synthetic namespace implementation is not its authenticated backing file: {name}"
            )
        synthetic[name] = {
            "synthetic_file_label": label,
            "backing_module": backing_name,
            "backing_file": origins[backing_name],
            "backing_sha256": expected_files[origins[backing_name]]["sha256"],
            "exact_class": f"{backing_name}.{class_name}",
            "singleton_identity_verified": True,
        }

    for name, module in tuple(modules.items()):
        if name.split(".", 1)[0] not in {"torch", "torchgen", "functorch"}:
            continue
        if name not in definitions and module is not None:
            real_origin(name, module)
    return origins, synthetic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-base", required=True)
    args = parser.parse_args()
    runtime = args.runtime.resolve(strict=True)
    dependencies = args.dependencies.resolve(strict=True)
    record = {"schema": "pytorch-runtime-import-v1", "status": "running"}
    try:
        if not sys.flags.no_site or not sys.flags.isolated:
            raise RuntimeError("Verification requires Python -I -S")
        if sys.version_info[:2] != (3, 12) or platform.machine() != "arm64":
            raise RuntimeError("Archive requires macOS arm64 CPython 3.12")
        if sys.platform != "darwin":
            raise RuntimeError("Archive requires macOS")
        if any("editable" in type(finder).__module__ for finder in sys.meta_path):
            raise RuntimeError("An editable import finder is already active")
        if "torch" in sys.modules:
            raise RuntimeError("Torch was imported before archive verification")
        manifest = json.loads((runtime / "runtime-manifest.json").read_text())
        expected_files = manifest["files"]
        actual_names = {
            str(path.relative_to(runtime))
            for path in runtime.rglob("*")
            if path.is_file() and path.name != "runtime-manifest.json"
        }
        if actual_names != set(expected_files):
            raise RuntimeError("Runtime file inventory differs from its manifest")
        for name, expected in expected_files.items():
            path = runtime / name
            if not path.resolve().is_relative_to(runtime):
                raise RuntimeError(f"Runtime file resolves outside archive: {name}")
            if sha256(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                raise RuntimeError(f"Runtime file hash mismatch: {name}")
        # Appending a site-packages directory directly does not execute .pth
        # files or activate its editable redirect finder.
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(runtime))
        sys.path.append(str(dependencies))
        os.environ["SKBUILD_EDITABLE_REBUILD"] = "false"
        import torch

        importlib.import_module("functorch")
        importlib.import_module("torchgen")
        if torch.version.git_version != args.expected_base:
            raise RuntimeError("Unexpected framework source revision")
        if not torch.backends.mps.is_built():
            raise RuntimeError("Full framework was built without MPS")
        if not torch.backends.openmp.is_available():
            raise RuntimeError("Full framework was built without requested OpenMP")
        if not torch._C._dispatch_has_kernel_for_dispatch_key(
            "aten::embedding_dense_backward", "MPS"
        ):
            raise RuntimeError("The integrated MPS embedding kernel is absent")
        origins, synthetic_origins = verify_module_origins(
            runtime, expected_files, sys.modules, torch
        )
        loaded = sorted(str(path) for path in torch.ops.loaded_libraries)
        if loaded:
            raise RuntimeError(f"Unexpected process-local operator libraries: {loaded}")
        if any("editable" in type(finder).__module__ for finder in sys.meta_path):
            raise RuntimeError("An editable import finder became active")
        dyld = ctypes.CDLL(None)
        dyld._dyld_image_count.argtypes = []
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        native_origins = []
        for index in range(dyld._dyld_image_count()):
            raw = dyld._dyld_get_image_name(index)
            if not raw:
                continue
            path = Path(os.fsdecode(raw)).resolve()
            if path.name.startswith(("libtorch", "libc10", "libomp", "libshm")):
                if not path.is_relative_to(runtime):
                    raise RuntimeError(f"Framework dylib loaded outside archive: {path}")
                native_origins.append(str(path.relative_to(runtime)))
        if not any(Path(name).name == "libtorch_cpu.dylib" for name in native_origins):
            raise RuntimeError("Expected integrated libtorch_cpu.dylib was not loaded")
        record.update(
            status="passed",
            scope="Full archived-framework import and operator registration; no numerical tests",
            runtime=str(runtime),
            python=sys.version,
            platform=platform.platform(),
            torch_version=torch.__version__,
            source_revision=torch.version.git_version,
            torch_file=str(Path(torch.__file__).resolve()),
            extension_file=str(Path(torch._C.__file__).resolve()),
            extension_sha256=sha256(Path(torch._C.__file__)),
            module_origins=origins,
            synthetic_module_origins=synthetic_origins,
            native_library_origins=sorted(native_origins),
            loaded_operator_libraries=loaded,
            mps_built=torch.backends.mps.is_built(),
            mps_available=torch.backends.mps.is_available(),
            openmp_available=torch.backends.openmp.is_available(),
            build_configuration=torch.__config__.show(),
            embedding_dispatch=torch._C._dispatch_dump_table("aten::embedding_dense_backward"),
            metadata_version=importlib.metadata.version("torch"),
            runtime_manifest_sha256=sha256(runtime / "runtime-manifest.json"),
            sys_path=sys.path,
        )
    except BaseException:
        record["status"] = "failed"
        record["exception"] = traceback.format_exc()
        raise
    finally:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
