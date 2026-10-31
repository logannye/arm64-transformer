"""Authenticate loaded framework modules without importing or executing a framework.

File-backed modules must belong to both the selected full-source payload and the
complete runtime inventory. Fileless modules are accepted only through the
specific native extension or registered namespaces that created them. This is
provenance evidence, not a claim about numerical correctness or repeatability.
"""

from __future__ import annotations

import hashlib
import re
import stat
import sys
from pathlib import Path
from types import ModuleType


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_loaded_framework_modules(name, payload, inventory, root, modules=None):
    """Return authenticated origins; unexplained or foreign modules fail closed.

    ``payload`` maps canonical absolute files to SHA256; ``inventory`` is the
    complete discovered runtime. ``modules`` is a snapshot of sys.modules, with
    an explicit argument provided for CPU-only regression tests.
    """
    require(name in {"pytorch", "mlx"}, "Unknown framework origin scope")
    modules = dict(sys.modules if modules is None else modules)
    root = Path(root).resolve(strict=True)
    require(isinstance(payload, dict) and payload, "Empty framework origin payload")
    require(isinstance(inventory.get("files"), dict), "Missing full runtime inventory")
    prefix = "torch" if name == "pytorch" else "mlx"
    prefixes = {"torch", "torchgen", "functorch"} if name == "pytorch" else {"mlx"}
    origins, synthetic, aliases, namespaces = {}, {}, {}, {}
    verified_files = {}

    def module_at(key):
        module = modules.get(key)
        require(isinstance(module, ModuleType), f"Missing framework module: {key}")
        return module

    def bind_file(value):
        require(isinstance(value, str) and value, "Missing module origin file")
        path = Path(value)
        require(
            path.is_absolute() and path.resolve(strict=True) == path,
            "Noncanonical framework module origin",
        )
        require(not path.is_symlink(), "Symlink framework module origin")
        absolute = str(path)
        require(absolute in payload, "Loaded framework module is outside selected payload")
        expected = payload[absolute]
        require(
            isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            "Malformed framework origin hash",
        )
        require(path.is_relative_to(root), "Framework origin is outside full runtime root")
        require(
            inventory["files"].get(path.relative_to(root).as_posix()) == expected,
            "Framework module origin is absent from full runtime inventory",
        )
        if absolute not in verified_files:
            before = path.stat()
            require(stat.S_ISREG(before.st_mode), "Nonregular framework module origin")
            sha = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    sha.update(block)
            after = path.stat()
            fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
            require(
                all(getattr(before, k) == getattr(after, k) for k in fields),
                "Framework module origin changed during read",
            )
            require(sha.hexdigest() == expected, "Framework module origin bytes changed")
            verified_files[absolute] = expected
        return absolute

    def real_origin(key, module):
        attrs = vars(module)
        source = attrs.get("__file__", vars(type(module)).get("__file__"))
        spec = attrs.get("__spec__")
        spec_origin = getattr(spec, "origin", None)
        if source is None:
            if spec_origin not in (None, "built-in", "frozen"):
                source = spec_origin
            else:
                return None
        actual = bind_file(source)
        if spec_origin is not None:
            require(
                spec_origin not in {"built-in", "frozen"} and spec_origin == actual,
                f"Framework module spec/file origins disagree: {key}",
            )
        origins[key] = actual
        return actual

    top = module_at(prefix)
    core_name = "torch._C" if name == "pytorch" else "mlx.core"
    core = module_at(core_name)
    require(
        vars(top).get("_C" if name == "pytorch" else "core") is core,
        "Native framework extension alias differs",
    )
    core_file = real_origin(core_name, core)
    require(core_file is not None, "Native framework extension has no file origin")

    if name == "pytorch":
        top_file = real_origin("torch", top)
        require(top_file is not None, "Torch package has no file origin")
        package_root = Path(top_file).parent
        definitions = {
            "torch.ops": ("torch._ops", "_Ops", "ops", "_ops.py"),
            "torch.classes": ("torch._classes", "_Classes", "classes", "_classes.py"),
        }
        for key, (backing_name, class_name, singleton, label) in definitions.items():
            module, backing = module_at(key), module_at(backing_name)
            cls = vars(backing).get(class_name)
            require(
                type(module) is cls
                and vars(top).get(singleton) is module
                and vars(backing).get(singleton) is module
                and vars(cls).get("__module__") == backing_name
                and vars(cls).get("__file__") == label
                and vars(module).get("__file__", label) == label
                and vars(module).get("__name__") == key
                and vars(module).get("__spec__") is None
                and vars(module).get("__loader__") is None,
                f"Unrecognized synthetic framework namespace: {key}",
            )
            backing_file = real_origin(backing_name, backing)
            constructor = vars(cls).get("__init__")
            code = getattr(constructor, "__code__", None)
            require(
                backing_file == str(package_root / label)
                and code is not None
                and str(Path(code.co_filename).resolve(strict=True)) == backing_file,
                f"Synthetic namespace constructor origin differs: {key}",
            )
            synthetic[key] = {
                "backing_module": backing_name,
                "backing_file": backing_file,
                "backing_sha256": payload[backing_file],
                "exact_class": f"{backing_name}.{class_name}",
                "singleton_identity_verified": True,
            }
    else:
        top_file = real_origin("mlx", top)
        if top_file is None:
            spec = vars(top).get("__spec__")
            locations = getattr(spec, "submodule_search_locations", None)
            paths = vars(top).get("__path__")
            require(
                getattr(spec, "origin", None) is None
                and locations is not None
                and paths is not None,
                "MLX namespace package is missing explicit import paths",
            )
            locations, paths = list(locations), list(paths)
            expected = inventory.get("layout", {}).get("packages", {}).get("mlx")
            require(
                locations and locations == paths == expected,
                "MLX namespace resolution differs from authenticated runtime",
            )
            for location in locations:
                path = Path(location)
                require(
                    path.is_absolute() and path.resolve(strict=True) == path and path.is_dir(),
                    "Noncanonical MLX namespace directory",
                )
                members = [p for p in payload if Path(p).is_relative_to(path)]
                require(members, "MLX namespace is absent from selected payload")
                require(not (path / "__init__.py").exists(), "Unexpected file-backed MLX namespace")
                for member in members:
                    relative = Path(member).relative_to(root).as_posix()
                    require(
                        inventory["files"].get(relative) == payload[member],
                        "MLX namespace bytes are absent from full runtime",
                    )
            namespaces["mlx"] = {"locations": locations, "extension": core_file}

    for key, module in sorted(modules.items()):
        if key.split(".", 1)[0] not in prefixes:
            continue
        require(isinstance(module, ModuleType), f"Missing framework module: {key}")
        if key in synthetic or key in namespaces or key in origins:
            continue
        if not key.startswith(core_name + "."):
            require(
                real_origin(key, module) is not None,
                f"Unexplained fileless framework module: {key}",
            )
            continue
        # Extension-owned submodules must be the actual attribute reachable from
        # the authenticated extension even if they advertise its shared file.
        # PyTorch's dynamo/init.cpp attaches the module created by
        # python_compiled_autograd.cpp as `compiled_autograd`, while its
        # PyModuleDef and sys.modules name are `autograd_compiler`.
        # Resolve only that exact spelling difference; the actual extension
        # attribute must still be the very same module object.
        attribute_key = (
            "torch._C._dynamo.compiled_autograd"
            if name == "pytorch" and key == "torch._C._dynamo.autograd_compiler"
            else key
        )
        parent = core
        for part in attribute_key[len(core_name) + 1 :].split("."):
            require(isinstance(parent, ModuleType), "Native framework module chain is invalid")
            parent = vars(parent).get(part)
        require(parent is module, f"Native framework submodule identity differs: {key}")
        require(vars(module).get("__name__") == key, "Native framework submodule name differs")
        actual = real_origin(key, module)
        if actual is not None:
            require(actual == core_file, "Native framework submodule file differs from extension")
            del origins[key]
        else:
            spec_origin = getattr(vars(module).get("__spec__"), "origin", None)
            require(spec_origin in (None, "built-in"), "Native framework submodule spec differs")
        aliases[key] = {"extension": core_file, "sha256": payload[core_file]}

    return {
        "schema": "gpu-owned-loaded-framework-origins-v1",
        "framework": name,
        "module_origins": dict(sorted(origins.items())),
        "synthetic_origins": dict(sorted(synthetic.items())),
        "native_aliases": dict(sorted(aliases.items())),
        "namespace_packages": dict(sorted(namespaces.items())),
        "module_count": len(origins) + len(synthetic) + len(aliases) + len(namespaces),
    }
