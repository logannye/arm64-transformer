"""Discover the executed Python installation without importing a GPU framework.

The caller cannot certify completeness by supplying a hand-picked hash map.
Discovery walks this interpreter's real base installation, virtual environment,
and every explicitly supplied import directory. Symlink targets are hashed at
their canonical locations, while their logical aliases are separately bound.
Nothing is installed, copied, compiled, or imported from torch/MLX here.
"""

from __future__ import annotations

import importlib.machinery
import os
import platform
import sys
import sysconfig
from pathlib import Path

from transformer.gpu_study import digest, require, safe_relative


def _platform_identity():
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "release": platform.release(),
        "version": platform.version(),
        "macos": platform.mac_ver()[0],
        "python": sys.version,
    }


def _locations(python_paths):
    paths = sysconfig.get_paths()
    # Both prefixes matter: .venv/bin/python commonly points outside .venv.
    scans = [Path(sys.base_prefix), Path(sys.prefix), *map(Path, python_paths)]
    imports = [*map(Path, python_paths), Path(paths["purelib"]), Path(paths["platlib"])]
    return scans, imports


def discover_runtime(root, python_paths=()):
    """Return a complete canonical inventory for this exact installed runtime.

    ``root=/`` can cover a symlinked uv interpreter, project .venv, and MLX add-on
    without relocating anything. All Python bytecode already present is bound;
    formal children disable writing new bytecode to keep the inventory stable.
    System dyld-cache libraries cannot be read as ordinary files: the OS build
    identity is recorded, not represented as an authenticated copy of that cache.
    """
    sys.dont_write_bytecode = True
    root = Path(root).resolve()
    extras = [Path(path).resolve(strict=True) for path in python_paths]
    require(len(set(extras)) == len(extras), "Duplicate supplemental runtime path")
    scan_paths, import_paths = _locations(extras)
    files, aliases, seen = {}, {}, set()

    def relative(path):
        try:
            name = Path(path).resolve(strict=True).relative_to(root).as_posix()
        except (ValueError, FileNotFoundError) as exc:
            raise ValueError(f"Runtime dependency is missing or outside root: {path}") from exc
        return safe_relative(name)

    def alias(path):
        path = Path(path).absolute()
        if path.is_symlink():
            aliases[str(path)] = {
                "link": os.readlink(path),
                "resolved": str(path.resolve(strict=True)),
            }

    def add(path):
        alias(path)
        path = Path(path)
        require(path.is_file(), f"Nonregular runtime artifact: {path}")
        name = relative(path)
        if name not in files:
            files[name] = digest(path.resolve(strict=True))

    add(sys.executable)
    for base in scan_paths:
        require(base.is_dir(), f"Missing installed runtime tree: {base}")
        # Record aliases in prefixes too, e.g. uv's version-independent Python.
        for parent in (base, *base.parents):
            alias(parent)
        for directory, directories, filenames in os.walk(base, followlinks=True):
            alias(directory)
            resolved = Path(directory).resolve(strict=True)
            if resolved in seen:
                directories[:] = []
                continue
            seen.add(resolved)
            directories.sort()
            for name in directories:
                alias(Path(directory) / name)
            for name in sorted(filenames):
                add(Path(directory) / name)
    normalized_imports = list(
        dict.fromkeys(str(path.resolve(strict=True)) for path in import_paths)
    )
    packages = {}
    for name in ("numpy", "torch", "mlx"):
        spec = importlib.machinery.PathFinder.find_spec(name, normalized_imports)
        require(spec is not None, f"Required installed package missing: {name}")
        locations = ([spec.origin] if spec.origin else []) + list(
            spec.submodule_search_locations or []
        )
        require(bool(locations), f"Package has no authenticated location: {name}")
        packages[name] = [str(Path(path).resolve(strict=True)) for path in locations]
        for path in locations:
            target = Path(path)
            prefix = relative(target)
            require(
                prefix in files
                if target.is_file()
                else any(key.startswith(prefix + "/") for key in files),
                f"Required package is outside discovered installations: {name}",
            )
        loaded = sys.modules.get(name)
        if loaded is not None and getattr(loaded, "__file__", None):
            require(
                str(Path(loaded.__file__).resolve()) in packages[name],
                f"Already imported {name} differs from the frozen import resolution",
            )
    layout = {
        "schema": 1,
        "interpreter": str(Path(sys.executable).absolute()),
        "interpreter_resolved": str(Path(sys.executable).resolve(strict=True)),
        "prefix": str(Path(sys.prefix).resolve(strict=True)),
        "base_prefix": str(Path(sys.base_prefix).resolve(strict=True)),
        "scan_roots": sorted({str(Path(path).resolve(strict=True)) for path in scan_paths}),
        "import_roots": normalized_imports,
        "packages": packages,
        "aliases": dict(sorted(aliases.items())),
        "platform": _platform_identity(),
        "system_library_scope": (
            "OS-build identity; dyld shared-cache bytes are not independently copied or hashed"
        ),
    }
    return {
        "files": dict(sorted(files.items())),
        "python_paths": [relative(path) for path in extras],
        "layout": layout,
    }


def verify_runtime(root, inventory):
    require(
        set(inventory) == {"files", "python_paths", "layout"},
        "Incomplete discovered runtime inventory",
    )
    extras = [Path(root) / safe_relative(name) for name in inventory["python_paths"]]
    current = discover_runtime(root, extras)
    require(
        current["layout"] == inventory["layout"],
        "Executed runtime layout, aliases, package resolution, or OS changed",
    )
    require(
        current["files"] == inventory["files"],
        "Installed runtime files changed, appeared, disappeared, or were omitted",
    )
    return current
