"""Provenance-only adapters for authenticated project-maintained frameworks.

The wrapped object is the unchanged production MPS/MLX Backend. Its numerical
methods, batches, optimizer and lifetime are delegated without replacement.
Source-build MLX need not have wheel metadata; its native manifest is the
version authority. A capability receipt is never numerical admission.
"""

from __future__ import annotations

import copy
import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import stat
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

from transformer.gpu_study import canonical, digest, identity, read_json, require, safe_relative

SCHEMA = "gpu-owned-runtime-identity-v1"
PROTOCOL = "gpu-v4-owned-five-engine-v1"
BYTECODE_POLICY = {"mode": "fresh_nonexistent_prefix", "writes_disabled": True}
COMPILED_GRAPH = (
    "digraph {\n"
    '{ rank=source; "A"; }\n'
    '{ rank=source; "B"; }\n'
    '{ rank=source; "C"; }\n'
    '{ rank=source; "D"; }\n'
    '{ NODE [label ="CompiledBroadcastMultiplyAddBroadcastMaximum", shape=rectangle]; }\n'
    '"A" -> NODE\n"B" -> NODE\n"C" -> NODE\n"D" -> NODE\n'
    '{ rank=sink; "E"; }\nNODE -> "E"\n}'
)


def validate_compilation_probe(probe):
    require(
        probe.get("status") == "passed"
        and probe.get("device") == "gpu"
        and probe.get("compiled_primitive_present") is True
        and type(probe.get("python_trace_count")) is int
        and probe["python_trace_count"] == 1
        and type(probe.get("executions")) is int
        and probe["executions"] == 4
        and probe.get("shape") == [257]
        and probe.get("dtype") == "float32"
        and probe.get("mode_override_used") is False
        and "mlx_disable_compile" in probe
        and probe["mlx_disable_compile"] is None,
        "Actual compiled GPU graph evidence is required",
    )
    graph = probe.get("graph_dot")
    require(isinstance(graph, str), "Missing raw compiled graph")
    require(
        hashlib.sha256(graph.encode()).hexdigest() == probe.get("graph_sha256"),
        "Compiled graph hash differs",
    )
    identifiers = re.findall(
        r'\{ ([1-9][0-9]{0,19}) \[label ="CompiledBroadcastMultiplyAddBroadcastMaximum", '
        r"shape=rectangle\]; \}",
        graph,
    )
    require(len(identifiers) == 1, "Expected one declared Compiled node")
    node = identifiers[0]
    require(
        int(node) < 2**64 and graph == COMPILED_GRAPH.replace("NODE", node),
        "Compiled graph structure differs",
    )
    expected = []
    for index, offset in enumerate((0.5, 1.0, 2.0, -1.0)):
        values = [max(2.0 * ((i - 128) / 16.0) + offset, 0.0) for i in range(257)]
        expected.append(
            {
                "index": index,
                "offset": offset,
                "python_trace_count": 1,
                "fp32_sha256": hashlib.sha256(struct.pack("<257f", *values)).hexdigest(),
                "exact_dyadic_oracle": True,
            }
        )
    require(
        canonical(probe.get("observations")) == canonical(expected),
        "Compiled replay output proof differs",
    )
    return copy.deepcopy(probe)


def _project_files(project):
    require(set(project) == {"root", "files"}, "Project source authority is incomplete")
    root = Path(project["root"])
    require(
        root.is_absolute() and root.resolve(strict=True) == root and root.is_dir(),
        "Invalid project root",
    )
    files = project["files"]
    require(
        isinstance(files, dict) and "transformer/gpu_owned_backend.py" in files,
        "Backend source is not bound",
    )
    for relative, sha in files.items():
        proof = _file(root / safe_relative(relative))
        require(proof["sha256"] == sha, "Bound project source changed")
    return root, files


def verify_loaded_project_modules(project, modules=None):
    root, files = _project_files(project)
    origins = {}
    for name, module in tuple((sys.modules if modules is None else modules).items()):
        if name != "transformer" and not name.startswith("transformer."):
            continue
        require(module is not None, "Partially imported project module")
        source = vars(module).get("__file__")
        require(isinstance(source, str), "Project module lacks a bound file origin")
        path = Path(source).resolve(strict=True)
        require(path.is_relative_to(root), "Loaded project module is outside bound source")
        relative = path.relative_to(root).as_posix()
        require(files.get(relative) == digest(path), "Loaded project module source differs")
        spec = vars(module).get("__spec__")
        origin = getattr(spec, "origin", None)
        require(
            isinstance(origin, str) and Path(origin).resolve(strict=True) == path,
            "Project module spec/file origins disagree",
        )
        origins[name] = relative
    return origins


def _file(path, expected=None):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path, "Noncanonical authority file")
    require(
        not path.is_symlink() and stat.S_ISREG(path.stat().st_mode), "Nonregular authority file"
    )
    before = path.stat()
    result = {"path": str(path), "sha256": digest(path), "bytes": before.st_size}
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(
        all(getattr(before, k) == getattr(after, k) for k in fields),
        "Authority file changed during read",
    )
    if expected is not None:
        require(result == expected, "Authority file binding changed")
    return result


def _selection(selection):
    require(set(selection) == {"pytorch", "mlx"}, "Both maintained frameworks are required")
    for name, record in selection.items():
        require(
            record.get("kind") == "project_maintained_full_source_build"
            and re.fullmatch(r"[0-9a-f]{40}", record.get("base", ""))
            and re.fullmatch(r"[0-9a-f]{64}", record.get("runtime_manifest_sha256", "")),
            "Malformed maintained framework selection",
        )
        patches = record.get("patches")
        require(
            isinstance(patches, list)
            and patches
            and len(patches) == len(set(patches))
            and all(isinstance(p, str) and re.fullmatch(r"[0-9a-f]{64}", p) for p in patches),
            "Missing or duplicate ordered framework patches",
        )
        require(
            type(record.get("build_run")) is int and record["build_run"] > 0, "Missing build run"
        )
        if name == "pytorch":
            require(
                record.get("engines") == ["cpu_torch", "mps"]
                and record.get("same_distribution_required") is True,
                "CPU and MPS must share one maintained Torch distribution",
            )
        else:
            require(
                record.get("engines") == ["mlx"]
                and record.get("shader_mode") == "jit-on"
                and record.get("variant") == "patched"
                and record.get("graph_compilation_required") is True
                and record.get("combined_patch_state_required") is True,
                "Owned MLX must use the pinned compiled combined patch state",
            )


def _verify_framework(name, row, selected, inventory, runtime_root):
    expected_keys = {"selection", "manifest", "payload_root"}
    if name == "pytorch":
        expected_keys.add("combined_patch_sha256")
    require(
        set(row) == expected_keys and row["selection"] == selected, "Framework selection differs"
    )
    _file(row["manifest"]["path"], row["manifest"])
    require(
        row["manifest"]["sha256"] == selected["runtime_manifest_sha256"],
        "Unselected native manifest",
    )
    manifest = read_json(row["manifest"]["path"])
    base = Path(row["payload_root"])
    require(
        base.is_absolute() and base.resolve(strict=True) == base and base.is_dir(),
        "Invalid payload root",
    )
    if name == "pytorch":
        combined = selected.get("combined_patch_sha256")
        if combined is None:
            require(
                len(selected["patches"]) == 1, "Multiple patches require a bound combined patch"
            )
            combined = selected["patches"][0]
        require(
            row["combined_patch_sha256"] == combined
            and manifest.get("source_revision") == selected["base"]
            and manifest.get("patch_sha256") == combined,
            "Torch source or combined patch differs",
        )
    else:
        require(
            manifest.get("base") == selected["base"]
            and manifest.get("mode") == selected["shader_mode"]
            and manifest.get("variant") == selected["variant"]
            and isinstance(manifest.get("patches"), dict)
            and set(manifest["patches"].values()) == set(selected["patches"])
            and len(manifest["patches"]) == len(selected["patches"]),
            "MLX source, shader mode or combined patch differs",
        )
    records = manifest.get("files")
    require(isinstance(records, dict) and records, "Empty framework manifest")
    actual_files = set()
    for path in base.rglob("*"):
        require(not path.is_symlink(), "Symlink inside native payload")
        if path.is_file() and path != Path(row["manifest"]["path"]):
            actual_files.add(path.relative_to(base).as_posix())
    require(actual_files == set(records), "Native payload inventory changed")
    payload = {}
    for relative, item in records.items():
        path = base / safe_relative(relative)
        expected = {"path": str(path), "sha256": item["sha256"], "bytes": item["bytes"]}
        _file(path, expected)
        if "executable" in item:
            require(
                bool(path.stat().st_mode & 0o111) is item["executable"],
                "Native payload executable mode changed",
            )
        payload[str(path)] = item["sha256"]
    package = "torch" if name == "pytorch" else "mlx"
    locations = inventory["layout"]["packages"].get(package)
    require(isinstance(locations, list) and locations, "Missing framework package resolution")
    for location in locations:
        path = Path(location)
        require(path.is_relative_to(base), "Runtime resolves a different framework package")
        selected_files = {
            p: h for p, h in payload.items() if p == location or Path(p).is_relative_to(path)
        }
        require(selected_files, "Resolved package is absent from native manifest")
        for absolute, sha in selected_files.items():
            relative = str(Path(absolute).relative_to(runtime_root))
            require(
                inventory["files"].get(relative) == sha, "Package bytes differ from full runtime"
            )
    _file(row["manifest"]["path"], row["manifest"])
    return payload


@dataclass(frozen=True)
class RuntimeAuthority:
    """Frozen inputs; revalidation reads actual files, never a caller pass flag."""

    root: str
    inventory_json: bytes
    selection_json: bytes
    receipt_path: str
    receipt_sha256: str

    def verify(self):
        return _authenticate(
            self.root,
            json.loads(self.inventory_json),
            json.loads(self.selection_json),
            self.receipt_path,
            self.receipt_sha256,
        )


def _authenticate(root, inventory, selection, receipt_path, expected_sha256):
    from transformer.gpu_study_runtime import verify_runtime

    _selection(selection)
    require(
        re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None, "Invalid receipt trust anchor"
    )
    receipt_binding = _file(receipt_path)
    require(receipt_binding["sha256"] == expected_sha256, "Owned runtime receipt changed")
    value = read_json(receipt_path)
    require(
        set(value)
        == {
            "schema",
            "protocol",
            "runtime_identity_sha256",
            "frameworks",
            "project",
            "bytecode_policy",
        }
        and value["schema"] == SCHEMA
        and value["protocol"] == PROTOCOL
        and value["runtime_identity_sha256"] == identity(inventory["files"])
        and canonical(value["bytecode_policy"]) == canonical(BYTECODE_POLICY)
        and set(value["frameworks"]) == {"pytorch", "mlx"},
        "Incomplete or foreign owned runtime receipt",
    )
    root = Path(root).resolve(strict=True)
    verify_runtime(root, inventory)
    _project_files(value["project"])
    payloads = {
        name: _verify_framework(name, value["frameworks"][name], selection[name], inventory, root)
        for name in ("pytorch", "mlx")
    }
    _file(receipt_path, receipt_binding)
    return value, payloads


def authenticate_runtime(root, inventory, selection, receipt_path, expected_sha256):
    """Usable before admission, without reading any modeling or admission fixture."""
    authority = RuntimeAuthority(
        str(Path(root).resolve(strict=True)),
        canonical(inventory),
        canonical(selection),
        str(Path(receipt_path).absolute()),
        expected_sha256,
    )
    authority.verify()
    return authority


def _loaded_libraries():
    require(sys.platform == "darwin", "Owned GPU execution requires macOS")
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.argtypes = []
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    return sorted(
        {
            str(Path(os.fsdecode(raw)).resolve(strict=True))
            for index in range(dyld._dyld_image_count())
            if (raw := dyld._dyld_get_image_name(index))
            and Path(os.fsdecode(raw)).name.startswith(
                ("libtorch", "libc10", "libomp", "libshm", "libmlx")
            )
        }
    )


def validate_loaded_origin_proof(proof, *, inventory, runtime_root, receipt):
    """Validate a retained origin proof without accessing its original filesystem.

    The caller must first authenticate the receipt, full runtime inventory and
    selected manifests. This checks the complete saved module/native structure
    against those identities; it does not recreate live object identity from an
    archive or turn provenance evidence into numerical admission.
    """

    def absolute(value):
        require(isinstance(value, str), "Missing retained absolute path")
        path = Path(value)
        require(
            path.is_absolute() and str(path) == value and ".." not in path.parts,
            "Noncanonical retained absolute path",
        )
        return path

    root = absolute(runtime_root)
    require(
        isinstance(inventory, dict)
        and isinstance(inventory.get("files"), dict)
        and inventory["files"]
        and receipt.get("schema") == SCHEMA
        and receipt.get("protocol") == PROTOCOL
        and receipt.get("runtime_identity_sha256") == identity(inventory["files"])
        and set(receipt.get("frameworks", {})) == {"pytorch", "mlx"},
        "Origin proof authority is incomplete or stale",
    )
    fields = {"module_origins", "framework_module_origins", "native_libraries"}
    require(
        isinstance(proof, dict) and set(proof) in (fields, fields | {"project_module_origins"}),
        "Malformed loaded-origin proof",
    )
    frameworks = proof["framework_module_origins"]
    require(
        isinstance(frameworks, dict) and frameworks and set(frameworks) <= {"pytorch", "mlx"},
        "Missing loaded framework module proof",
    )
    bases = {name: absolute(row["payload_root"]) for name, row in receipt["frameworks"].items()}
    require(
        all(base.is_relative_to(root) for base in bases.values())
        and not bases["pytorch"].is_relative_to(bases["mlx"])
        and not bases["mlx"].is_relative_to(bases["pytorch"]),
        "Framework payload roots overlap or escape the runtime",
    )

    def bound(value, framework, expected=None):
        path = absolute(value)
        require(
            path.is_relative_to(bases[framework]) and path.is_relative_to(root),
            "Loaded origin is outside the selected framework payload",
        )
        sha = inventory["files"].get(path.relative_to(root).as_posix())
        require(
            isinstance(sha, str)
            and re.fullmatch(r"[0-9a-f]{64}", sha)
            and (expected is None or expected == sha),
            "Loaded origin is absent from the authenticated runtime",
        )
        return sha

    selected_origins = {}
    for name, value in frameworks.items():
        prefix = "torch" if name == "pytorch" else "mlx"
        core_name = "torch._C" if name == "pytorch" else "mlx.core"
        prefixes = {"torch", "torchgen", "functorch"} if name == "pytorch" else {"mlx"}
        require(
            isinstance(value, dict)
            and set(value)
            == {
                "schema",
                "framework",
                "module_origins",
                "synthetic_origins",
                "native_aliases",
                "namespace_packages",
                "module_count",
            }
            and value["schema"] == "gpu-owned-loaded-framework-origins-v1"
            and value["framework"] == name,
            "Malformed loaded framework origin schema",
        )
        maps = [
            value[key]
            for key in (
                "module_origins",
                "synthetic_origins",
                "native_aliases",
                "namespace_packages",
            )
        ]
        require(
            all(isinstance(item, dict) for item in maps)
            and type(value["module_count"]) is int
            and value["module_count"] > 0
            and value["module_count"] == sum(map(len, maps))
            and len(set().union(*maps)) == value["module_count"]
            and all(
                isinstance(key, str) and key.split(".", 1)[0] in prefixes
                for item in maps
                for key in item
            ),
            "Loaded framework module counts, scopes or identities differ",
        )
        origins, synthetic, aliases, namespaces = maps
        locations = inventory["layout"]["packages"][prefix]
        require(isinstance(locations, list) and locations, "Missing actual package resolution")
        locations = [absolute(path) for path in locations]
        if len(locations) == 2:
            package = locations[1]
            require(
                locations[0] == package / "__init__.py"
                and origins.get(prefix) == str(locations[0])
                and namespaces == {},
                "File-backed package origin differs",
            )
        else:
            require(name == "mlx" and len(locations) == 1, "Unsupported package resolution")
            package = locations[0]
            require(
                prefix not in origins and set(namespaces) == {"mlx"},
                "MLX namespace package proof differs",
            )
        require(package.is_relative_to(bases[name]), "Package is outside selected payload")
        core = absolute(origins.get(core_name))
        stem = "_C" if name == "pytorch" else "core"
        require(
            core.parent == package and core.name.startswith(stem + ".") and core.suffix == ".so",
            "Actual framework native extension is missing",
        )
        core_sha = bound(str(core), name)
        for path in origins.values():
            bound(path, name)
        for module, row in aliases.items():
            require(
                module.startswith(core_name + ".")
                and row == {"extension": str(core), "sha256": core_sha},
                "Native submodule alias is unbound or has the wrong extension",
            )
        if name == "pytorch":
            definitions = {
                "torch.ops": ("torch._ops", "_Ops", "_ops.py"),
                "torch.classes": ("torch._classes", "_Classes", "_classes.py"),
            }
            from transformer.gpu_owned_origins import validate_torch_forwarding_proofs

            require(set(definitions) <= set(synthetic), "Missing Torch synthetic namespaces")
            validate_torch_forwarding_proofs(synthetic, package, lambda p: bound(p, name))
            for module, (backing, class_name, filename) in definitions.items():
                path = str(package / filename)
                require(origins.get(backing) == path, "Synthetic backing module origin differs")
                require(
                    isinstance(synthetic[module], dict)
                    and synthetic[module].get("singleton_identity_verified") is True
                    and synthetic[module]
                    == {
                        "backing_module": backing,
                        "backing_file": path,
                        "backing_sha256": bound(path, name),
                        "exact_class": backing + "." + class_name,
                        "singleton_identity_verified": True,
                    },
                    "Torch synthetic namespace proof differs",
                )
            selected_origins[name] = [origins["torch"], str(core)]
        else:
            require(synthetic == {}, "Unrecognized MLX synthetic namespace")
            if namespaces:
                require(
                    namespaces["mlx"] == {"locations": [str(package)], "extension": str(core)},
                    "MLX namespace paths or native owner differ",
                )
            selected_origins[name] = [str(core)]
    require(
        isinstance(proof["module_origins"], list)
        and sum(proof["module_origins"] == paths for paths in selected_origins.values()) == 1,
        "Selected framework origins are missing or substituted",
    )
    libraries = proof["native_libraries"]
    require(
        isinstance(libraries, list)
        and libraries
        and all(isinstance(path, str) for path in libraries)
        and len(libraries) == len(set(libraries)),
        "Missing or duplicate native library evidence",
    )
    library_names = {name: set() for name in frameworks}
    for path in libraries:
        source = absolute(path)
        owners = [name for name, base in bases.items() if source.is_relative_to(base)]
        require(
            len(owners) == 1 and owners[0] in frameworks, "Foreign or unexplained native library"
        )
        owner = owners[0]
        require(
            source.suffix == ".dylib"
            and source.name.startswith(
                ("libtorch", "libc10", "libomp", "libshm") if owner == "pytorch" else ("libmlx",)
            ),
            "Unexpected framework native library identity",
        )
        bound(path, owner)
        library_names[owner].add(source.name)
    require(
        all(
            ("libtorch_cpu.dylib" if name == "pytorch" else "libmlx.dylib") in names
            for name, names in library_names.items()
        ),
        "Required framework native library was not loaded",
    )
    if "project_module_origins" in proof:
        project = receipt.get("project", {})
        absolute(project.get("root"))
        modules = proof["project_module_origins"]
        require(isinstance(modules, dict) and modules, "Missing project module origin map")
        for module, relative in modules.items():
            require(
                isinstance(module, str)
                and (module == "transformer" or module.startswith("transformer.")),
                "Foreign project module scope",
            )
            relative = safe_relative(relative)
            module_path = module.replace(".", "/")
            require(
                relative in {module_path + ".py", module_path + "/__init__.py"},
                "Project module-to-source layout differs",
            )
            require(
                re.fullmatch(r"[0-9a-f]{64}", project.get("files", {}).get(relative, ""))
                is not None,
                "Project module origin is outside bound sources",
            )
    return copy.deepcopy(proof)


def _loaded_identity(engine, authority, verified):
    from transformer.gpu_owned_origins import verify_loaded_framework_modules

    receipt, payloads = verified
    inventory = json.loads(authority.inventory_json)
    paths = _loaded_libraries()
    selected = "pytorch" if engine in {"mps", "cpu_torch"} else "mlx"
    module = sys.modules.get("torch" if selected == "pytorch" else "mlx.core")
    require(module is not None, "Framework has not been imported")
    origins = [str(Path(module.__file__).resolve(strict=True))]
    if selected == "pytorch":
        require(not module.ops.loaded_libraries, "Custom operator override is not permitted")
        require(
            getattr(module.version, "git_version", None)
            == receipt["frameworks"][selected]["selection"]["base"],
            "Loaded Torch source revision differs",
        )
        origins.append(str(Path(module._C.__file__).resolve(strict=True)))
    all_payloads = {path: sha for files in payloads.values() for path, sha in files.items()}
    for path in paths:
        require(path in all_payloads, "Loaded foreign framework native library")
        relative = str(Path(path).relative_to(authority.root))
        require(
            inventory["files"].get(relative) == all_payloads[path],
            "Native library absent from full runtime",
        )
        require(digest(path) == all_payloads[path], "Loaded native library file changed")
    for path in origins:
        require(path in payloads[selected], "Loaded framework module origin differs")
        require(digest(path) == payloads[selected][path], "Loaded framework module changed")
    required = "libtorch_cpu.dylib" if selected == "pytorch" else "libmlx.dylib"
    require(
        any(Path(p).name == required and p in payloads[selected] for p in paths),
        "Required framework library is not loaded",
    )
    loaded_frameworks = {}
    for name, prefixes in (
        ("pytorch", ("torch", "torchgen", "functorch")),
        ("mlx", ("mlx",)),
    ):
        if any(key.split(".", 1)[0] in prefixes for key in sys.modules):
            loaded_frameworks[name] = verify_loaded_framework_modules(
                name, payloads[name], inventory, authority.root
            )
    require(selected in loaded_frameworks, "Selected framework module proof is missing")
    return {
        "module_origins": origins,
        "framework_module_origins": loaded_frameworks,
        "native_libraries": paths,
    }


def _capabilities(backend, engine, authority):
    from transformer.gpu_environment import MLX_ENVIRONMENT, MPS_ENVIRONMENT, require_environment

    require_environment(MPS_ENVIRONMENT if engine == "mps" else MLX_ENVIRONMENT)
    bytecode = verify_bytecode_policy()
    verified = authority.verify()
    proof = _loaded_identity(engine, authority, verified)
    proof["project_module_origins"] = verify_loaded_project_modules(verified[0]["project"])
    selected = "pytorch" if engine == "mps" else "mlx"
    module = sys.modules.get(f"transformer.gpu_{engine}")
    require(
        module is not None and type(backend) is module.Backend,
        "Unchanged production Backend required",
    )
    require(
        type(backend).__module__ == module.__name__
        and Path(type(backend).__init__.__code__.co_filename).resolve(strict=True)
        == Path(module.__file__).resolve(strict=True),
        "Backend constructor is not from the bound production adapter",
    )
    if engine == "mps":
        result = backend.capabilities()
        require(
            module.torch.are_deterministic_algorithms_enabled(),
            "Strict deterministic algorithms required",
        )
        require(
            not module.torch.is_deterministic_algorithms_warn_only_enabled(),
            "Warn-only mode forbidden",
        )
    else:
        from transformer.gpu_mlx_compile import require_compilation_environment

        require_compilation_environment()
        mx = module.mx
        probe = validate_compilation_probe(backend._compilation_probe)
        for name in module.PARAM_SHAPES:
            require(backend.model[name].dtype == mx.float32, "Parameter storage must be FP32")
            for moment in ("m", "v"):
                require(
                    backend.optimizer.state[name][moment].dtype == mx.float32,
                    "Adam moment storage must be FP32",
                )
        metadata = {}
        for name in ("mlx", "mlx-metal"):
            try:
                metadata[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                metadata[name] = None
        result = {
            "backend": "mlx",
            "mlx": metadata["mlx"],
            "mlx_metal": metadata["mlx-metal"],
            "device": str(mx.gpu),
            "device_info": mx.device_info(),
            "storage_dtype": "float32",
            "cpu_tensor_fallback": False,
            "compiled": True,
            "compilation_probe": probe,
            "compiler_math": "Pinned maintained framework; unchanged project model and optimizer",
            "adam": {"lr": module.LR, "betas": [0.9, 0.999], "eps": 1e-8, "bias_correction": True},
            **{
                name: os.environ.get(name)
                for name in ("MLX_ENABLE_TF32", "MLX_DISABLE_COMPILE", "MLX_SDPA_VJP_FALLBACK")
            },
        }
    result.update(
        baseline_class="project_maintained_full_source_build",
        standard_release_claim=False,
        implementation=f"project-maintained {engine}; unchanged production project adapter",
        framework_identity=copy.deepcopy(verified[0]["frameworks"][selected]["selection"]),
        runtime_identity_sha256=verified[0]["runtime_identity_sha256"],
        runtime_receipt_sha256=authority.receipt_sha256,
        loaded_origin_proof=proof,
        bytecode_policy=bytecode,
        version_authority="source revision, ordered patches and authenticated native manifest",
        numerical_admission_claim=False,
    )
    return result


class _OwnedBackend:
    __slots__ = ("_backend", "_engine", "_authority")

    def __init__(self, backend, engine, authority):
        self._backend, self._engine, self._authority = backend, engine, authority

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def capabilities(self):
        return _capabilities(self._backend, self._engine, self._authority)


def verify_bytecode_policy():
    prefix = sys.pycache_prefix
    require(isinstance(prefix, str) and prefix, "Fresh bytecode prefix is required")
    path = Path(prefix)
    require(
        path.is_absolute()
        and path.parent.resolve(strict=True) == path.parent
        and not path.exists()
        and not path.is_symlink()
        and sys.dont_write_bytecode
        and sys.flags.dont_write_bytecode,
        "Bytecode prefix must remain unused with writes disabled",
    )
    return {**BYTECODE_POLICY, "prefix": str(path)}


def create_runtime_backend(engine, initial, *, authority, max_batch=16, max_length=32):
    require(engine in {"mps", "mlx"}, "Owned framework backend must be MPS or MLX")
    require(type(authority) is RuntimeAuthority, "Authenticated runtime authority required")
    verify_bytecode_policy()
    authority.verify()
    module = importlib.import_module(f"transformer.gpu_{engine}")
    backend = module.Backend(initial, max_batch=max_batch, max_length=max_length)
    try:
        result = _OwnedBackend(backend, engine, authority)
        result.capabilities()
        return result
    except BaseException:
        backend.close()
        raise


def _formal_authority(manifest, roots):
    from transformer.gpu_study import regular, verify_bindings

    require(manifest.get("protocol") == PROTOCOL, "Owned backend rejects another study protocol")
    verify_bindings(manifest, roots)
    selected = manifest["owned_frameworks"]
    name = selected["runtime_receipt"]
    binding = manifest["bindings"]
    path = regular(roots["baseline"], name)
    authority = authenticate_runtime(
        roots["runtime"],
        {
            "files": binding["runtime"],
            "python_paths": binding["runtime_python_paths"],
            "layout": binding["runtime_layout"],
        },
        selected["frameworks"],
        path,
        binding["baseline"][name],
    )
    receipt, _ = authority.verify()
    require(
        receipt["project"]
        == {"root": str(Path(roots["source"]).resolve()), "files": binding["source"]},
        "Runtime authority project differs from formal source binding",
    )
    return authority


def create_backend(engine, initial, *, manifest, roots, max_batch=16, max_length=32):
    return create_runtime_backend(
        engine,
        initial,
        authority=_formal_authority(manifest, roots),
        max_batch=max_batch,
        max_length=max_length,
    )


def backend_capabilities(backend, engine, *, manifest, roots):
    if isinstance(backend, _OwnedBackend):
        require(backend._engine == engine, "Backend engine differs")
        backend = backend._backend
    return _capabilities(backend, engine, _formal_authority(manifest, roots))
