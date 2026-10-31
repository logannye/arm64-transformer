"""Authenticate separate control/math roles and the complete captured runtime."""

import hashlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("library_frozen_contract", HERE / "contract.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)
CONTROL_PREFIX = "integration/library-study-v1/"
METAL_LIBRARY = "native/metal/build/libtransformer_metal.dylib"
METAL_SHA256 = "ec1e5e69005f1dbe672abc059bf2efbbb0e46a7ebb9d13fe7cffd0c2b42d90da"


def relative(name):
    p = PurePosixPath(name)
    C.require(
        isinstance(name, str)
        and name
        and str(p) == name
        and not p.is_absolute()
        and ".." not in p.parts
        and not any(c in name for c in "\\\x00\r\n"),
        "Unsafe relative binding",
    )
    return p


def directory(path):
    path = Path(path).absolute()
    C.require(path.is_dir() and path.resolve(strict=True) == path, "Noncanonical role root")
    return path


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], timeout=60)


def source_role(root, revision, *, control=False):
    root = directory(root)
    C.require(re.fullmatch("[0-9a-f]{40}", revision) is not None, "Invalid source revision")
    C.require(git(root, "rev-parse", "HEAD").decode().strip() == revision, "Source head differs")
    C.require(not git(root, "status", "--porcelain").strip(), "Source role is not clean")
    records = {}
    for entry in (
        git(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD").decode().strip("\0").split("\0")
    ):
        meta, name = entry.split("\t", 1)
        mode, kind, oid = meta.split()
        C.require(kind == "blob" and mode in ("100644", "100755"), "Unsupported source object")
        relative(name)
        if control and not name.startswith(CONTROL_PREFIX):
            continue
        path = root / name
        bound = C.binding(path)
        data = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        C.require(
            blob == oid and hashlib.sha256(data).hexdigest() == bound["sha256"],
            "Source differs from its Git object",
        )
        C.require(bool(path.stat().st_mode & 0o111) == (mode == "100755"), "Source mode differs")
        records[name] = bound["sha256"]
    C.require(
        records and (not control or CONTROL_PREFIX + "contract.py" in records),
        "Incomplete source role",
    )
    if not control:
        C.require(revision == C.MATH_REVISION, "Unselected numerical revision")
        C.require(
            C.binding(root / METAL_LIBRARY)["sha256"] == METAL_SHA256,
            "Unselected retained Metal host binary",
        )
        records[METAL_LIBRARY] = METAL_SHA256
    C.require(
        git(root, "rev-parse", "HEAD").decode().strip() == revision
        and not git(root, "status", "--porcelain").strip(),
        "Source changed during capture",
    )
    return dict(root=str(root), revision=revision, files=dict(sorted(records.items())))


def verify_role(value, *, control=False):
    C.require(set(value) == {"root", "revision", "files"}, "Source role fields differ")
    actual = source_role(value["root"], value["revision"], control=control)
    C.require(C.canonical(actual) == C.canonical(value), "Source role binding changed")
    return actual


def merged_project(roles):
    """Only explicitly bound files under the common root become known sources."""
    C.require(set(roles) == {"control", "math"}, "Both immutable source roles are required")
    roots = [directory(r["root"]) for r in roles.values()]
    root = Path(os.path.commonpath(roots))
    files = {}
    for role in roles.values():
        for name, digest in role["files"].items():
            p = directory(role["root"]) / relative(name)
            key = p.relative_to(root).as_posix()
            C.require(key not in files or files[key] == digest, "Conflicting source role file")
            files[key] = digest
    return dict(root=str(root), files=dict(sorted(files.items())))


def activate(roles, capture_binding):
    """Authenticate before paths/imports; never install, build or generate operands."""
    C.require(
        sys.flags.safe_path
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
        and not sys.flags.optimize
        and sys.flags.hash_randomization == 0,
        "Use canonical Python -P -S -B with fixed hash seed",
    )
    C.require(
        not any(
            k.split(".", 1)[0] in {"torch", "mlx", "numpy", "transformer"} for k in sys.modules
        ),
        "Runtime activation requires a fresh process",
    )
    C.require(
        not any(
            k.startswith("DYLD_")
            or k.startswith("PYTHON")
            and k not in {"PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE"}
            for k in os.environ
        ),
        "Ambient interpreter/native override",
    )
    C.require(
        os.environ.get("PYTHONHASHSEED") == "0"
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        "Interpreter controls differ",
    )
    C.require(
        sys.pycache_prefix is not None and not Path(sys.pycache_prefix).exists(),
        "Fresh unused bytecode prefix required",
    )
    verify_role(roles["control"], control=True)
    verify_role(roles["math"])
    capture_path = C.bound(capture_binding)
    capture = C.read(capture_path, capture_binding["sha256"])
    C.require(
        capture["schema"] == "gpu-owned-runtime-capture-v1"
        and capture["status"] == "identity_captured_not_qualified"
        and capture["formal_eligible"] is False
        and capture["formal_fits"] == 0
        and capture["admission_stream_opened"] is False,
        "Wrong captured runtime identity scope",
    )
    for key in ("runtime_receipt", "runtime_inventory", "framework_selection"):
        C.bound(capture[key])
    runtime = C.read(capture["runtime_inventory"]["path"], capture["runtime_inventory"]["sha256"])
    receipt = C.read(capture["runtime_receipt"]["path"], capture["runtime_receipt"]["sha256"])
    C.require(
        receipt["project"] == dict(root=roles["math"]["root"], files=roles["math"]["files"]),
        "Runtime numerical source differs from the exact qualified role",
    )
    selection = C.read(
        capture["framework_selection"]["path"], capture["framework_selection"]["sha256"]
    )
    C.require(
        C.canonical(selection) == C.canonical(C.comparison()["framework_selection"]),
        "Unselected maintained packages",
    )
    source = Path(roles["math"]["root"])
    # These authority modules import only the standard library. Runtime paths
    # remain inactive until their own complete discovery and hashes authenticate.
    sys.path.insert(0, str(source))
    from transformer.gpu_owned_backend import authenticate_runtime
    from transformer.gpu_study_runtime import verify_runtime

    verify_runtime(capture["runtime_root"], runtime)
    authority = authenticate_runtime(
        capture["runtime_root"],
        runtime,
        selection,
        capture["runtime_receipt"]["path"],
        capture["runtime_receipt"]["sha256"],
    )
    authority.verify()
    paths = [
        str(source),
        *[str(Path(capture["runtime_root"]) / p) for p in runtime["python_paths"]],
    ]
    sys.path[:0] = paths
    from transformer.gpu_environment import require_environment
    from transformer.gpu_study import ENVIRONMENT

    require_environment(ENVIRONMENT)
    return Runtime(roles, capture_binding, capture, runtime, authority, merged_project(roles))


class Runtime:
    def __init__(self, roles, capture_binding, capture, inventory, authority, project):
        self.roles, self.capture_binding, self.capture = roles, capture_binding, capture
        self.inventory, self.authority, self.project = inventory, authority, project

    def origins(self, engine):
        """Check numerical modules in math role and every loaded file in both roles."""
        import numpy as np
        import torch

        from transformer import gpu_owned_backend as backend

        receipt, _ = verified = self.authority.verify()
        C.require(
            np.__version__ == receipt["frameworks"]["pytorch"]["selection"]["numpy_abi_version"]
            and str(Path(np.__file__).resolve()) in self.inventory["layout"]["packages"]["numpy"],
            "NumPy ABI/origin differs",
        )
        C.require(not torch.ops.loaded_libraries, "Process-local operator library loaded")
        result = dict(
            numpy=dict(version=np.__version__, path=str(Path(np.__file__).resolve())),
            pytorch=backend._loaded_identity("cpu_torch", self.authority, verified),
        )
        if engine == "mlx":
            import mlx.core as mx

            from transformer.gpu_mlx_compile import verify_compilation

            C.require(
                mx.default_device() == mx.gpu and mx.metal.is_available(), "Real MLX GPU required"
            )
            result.update(
                mlx=backend._loaded_identity("mlx", self.authority, verified),
                compilation=backend.validate_compilation_probe(verify_compilation(mx)),
            )
        backend.verify_loaded_project_modules(receipt["project"])
        path = Path(self.roles["math"]["root"]) / "integration/gpu-owned/owned_mlx_suites.py"
        spec = importlib.util.spec_from_file_location("frozen_loaded_file_verifier", path)
        suite = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(suite)
        result["loaded_python_files"] = suite.verify_loaded_files(
            self.inventory, self.authority, self.project
        )
        result["runtime_identity_sha256"] = receipt["runtime_identity_sha256"]
        result["source_roles_sha256"] = C.identity(self.roles)
        return result

    def recheck(self):
        C.bound(self.capture_binding)
        verify_role(self.roles["control"], control=True)
        verify_role(self.roles["math"])
        self.authority.verify()
        C.require(not Path(sys.pycache_prefix).exists(), "Bytecode cache appeared")
