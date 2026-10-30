"""Import a complete transported Release runtime without executing tensor work."""

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import traceback
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = {"status": "failed", "scope": "import/origin checks only; no tensor execution"}
    try:
        if not sys.flags.isolated or not sys.flags.no_site:
            raise ValueError("Requires Python -I -S")
        if (
            platform.machine() != "arm64"
            or sys.platform != "darwin"
            or sys.version_info[:2] != (3, 12)
        ):
            raise ValueError("Requires native macOS arm64 CPython3.12")
        for key in (
            "MLX_DISABLE_COMPILE",
            "PYTORCH_MPS_PREFER_METAL",
            "DEVICE",
            "PYTHONHOME",
            "MLX_METALLIB_PATH",
            "DYLD_LIBRARY_PATH",
            "DYLD_INSERT_LIBRARIES",
        ):
            if key in os.environ:
                raise ValueError(f"Forbidden inherited runtime control: {key}")
        bundle = args.bundle.resolve(strict=True)
        manifest = json.loads((bundle / "MANIFEST.json").read_text())
        if manifest["scope"] != "mlx_matched_release_operator_build_v1":
            raise ValueError("Wrong build contract")
        files = manifest["files"]
        actual = {str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file()}
        if actual != set(files) | {"MANIFEST.json"}:
            raise ValueError("Bundle inventory differs")
        for name, row in files.items():
            path = bundle / name
            if path.is_symlink() or not path.resolve().is_relative_to(bundle):
                raise ValueError("Bundle contains a symlink or escaping path")
            if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                raise ValueError("Bundle payload differs")
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(bundle / "runtime/python"))
        sys.path.append(str(args.dependencies.resolve(strict=True)))
        for name, version in manifest["runtime_dependencies"].items():
            if importlib.metadata.version(name) != version:
                raise ValueError(f"Dependency differs: {name}")
        import mlx.core as mx

        core = Path(mx.__file__).resolve(strict=True)
        if not core.is_relative_to(bundle / "runtime/python/mlx"):
            raise ValueError("MLX core imported outside bundle")
        if digest(core) != files[str(core.relative_to(bundle))]["sha256"]:
            raise ValueError("Loaded core differs from manifest")
        loader = ctypes.CDLL(None)
        loader._dyld_image_count.restype = ctypes.c_uint32
        loader._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        loader._dyld_get_image_name.restype = ctypes.c_char_p
        native = {}
        for index in range(loader._dyld_image_count()):
            raw = loader._dyld_get_image_name(index)
            if not raw:
                continue
            path = Path(os.fsdecode(raw)).resolve()
            if path.name.startswith(("libmlx", "libjaccl")):
                if not path.is_relative_to(bundle):
                    raise ValueError(f"Native MLX library outside bundle: {path}")
                name = str(path.relative_to(bundle))
                if name not in files or digest(path) != files[name]["sha256"]:
                    raise ValueError("Loaded native library differs from manifest")
                native[name] = digest(path)
        if not any(Path(name).name == "libmlx.dylib" for name in native):
            raise ValueError("Integrated MLX shared library was not loaded")
        record.update(
            status="passed",
            manifest_sha256=digest(bundle / "MANIFEST.json"),
            variant=manifest["variant"],
            mode=manifest["mode"],
            core=str(core),
            core_sha256=digest(core),
            native_libraries=native,
            device_numerics_validated=False,
            performance_measured=False,
        )
        return 0
    except BaseException:
        record["error"] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
