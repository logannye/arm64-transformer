"""Immutable namespace selection for the standard and project-owned studies.

Selecting a context never grants admission, generates operands, or imports a
framework. Authorization remains the frozen manifest, runtime and raw evidence.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StudyContext:
    name: str
    protocol: str
    admission_seed: int
    repeatability_contract: str
    owned: bool


STANDARD = StudyContext(
    "standard",
    "gpu-v2-five-engine-v2",
    20261104,
    "gpu-v2-standard-baseline-repeatability-v2",
    False,
)
OWNED = StudyContext(
    "owned",
    "gpu-v4-owned-five-engine-v1",
    20261108,
    "gpu-v4-owned-baseline-repeatability-v1",
    True,
)
CONTEXTS = (STANDARD, OWNED)
RESERVED_SEEDS = frozenset(context.admission_seed for context in CONTEXTS)


def resolve_context(manifest=None, context=None):
    if context is None:
        if manifest is None:
            return STANDARD
        matches = [value for value in CONTEXTS if value.protocol == manifest.get("protocol")]
    elif isinstance(context, str):
        matches = [value for value in CONTEXTS if context in (value.name, value.protocol)]
    else:
        matches = [
            value for value in CONTEXTS if type(context) is StudyContext and value == context
        ]
    if len(matches) != 1:
        raise ValueError("Unknown immutable study context")
    selected = matches[0]
    if manifest is not None and manifest.get("protocol") != selected.protocol:
        raise ValueError("Explicit context disagrees with manifest namespace")
    return selected


def create_backend(engine, initial, *, manifest=None, roots=None, **options):
    context = resolve_context(manifest)
    if context.owned and engine in ("mps", "mlx"):
        from transformer.gpu_owned_backend import create_backend as owned_backend

        return owned_backend(engine, initial, manifest=manifest, roots=roots, **options)
    return importlib.import_module(f"transformer.gpu_{engine}").Backend(initial, **options)


def python_command(manifest, output, *, roots=None, cpu=False):
    """Owned children use the canonical interpreter with no site or ambient import paths.

    The inline stdlib bootstrap authenticates all bound source/runtime bytes
    before adding explicit frozen paths. It accepts only bound study entrypoints.
    Standard launch bytes remain unchanged. Hashing is outside measured work.
    """
    if not resolve_context(manifest).owned:
        return [sys.executable]
    if roots is None:
        raise ValueError("Owned launch needs authenticated source/runtime roots")
    cache = Path(str(Path(output).absolute()) + ".unused-python-cache")
    if cache.exists() or cache.is_symlink():
        raise ValueError("Owned worker bytecode cache must be absent")
    binding = manifest["bindings"]
    if not isinstance(binding.get("runtime"), dict) or not binding["runtime"]:
        raise ValueError("Owned bootstrap needs the complete bound runtime")
    role = "cpu" if cpu else "source"
    source = Path(roots[role]).resolve()
    paths = [
        str(source),
        *[str(Path(roots["runtime"]).resolve() / name) for name in binding["runtime_python_paths"]],
    ]
    entries = {}
    names = {
        "transformer/gpu_study_worker.py",
        "transformer/gpu_study_admission.py",
        "transformer/gpu_admission_checks.py",
        "transformer/gpu_owned_repeatability.py",
        "transformer/gpu_study_replay.py",
        "transformer/gpu_study_cpu_admission.py",
        "transformer/gpu_owned_cpu.py",
        "transformer/python_train.py",
    }
    for selected in ("source", "cpu"):
        if selected not in roots or selected not in binding:
            continue
        for name in names.intersection(binding[selected]):
            entries[str(Path(roots[selected]).resolve() / name)] = binding[selected][name]
    initial = str(source / "transformer/__init__.py")
    initializer_sha = binding[role]["transformer/__init__.py"]
    authority = {
        "schema": "gpu-owned-bootstrap-v1",
        "protocol": OWNED.protocol,
        "cache": str(cache),
        "paths": paths,
        "entries": entries,
        "initializer": {"path": initial, "sha256": initializer_sha},
        "trees": {
            selected: {
                "root": str(Path(roots[selected]).resolve()),
                "files": binding[selected],
            }
            for selected in ("source", "cpu", "runtime")
            if selected in roots and selected in binding
        },
    }
    raw = json.dumps(authority, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    authority_path = Path(str(Path(output).absolute()) + ".owned-bootstrap.json")
    authority_path.parent.mkdir(parents=True, exist_ok=True)
    if authority_path.exists() or authority_path.is_symlink():
        if authority_path.is_symlink() or authority_path.read_bytes() != raw:
            raise ValueError("Owned bootstrap authority collision")
    else:
        with authority_path.open("xb") as stream:
            stream.write(raw)
    authority_sha = hashlib.sha256(raw).hexdigest()
    code = (
        "import hashlib,json,os,pathlib,runpy,sys\n"
        "if not (sys.flags.safe_path and sys.flags.no_site and sys.flags.dont_write_bytecode and "
        "sys.flags.hash_randomization==0 and not sys.flags.optimize): "
        "raise ValueError('Owned interpreter controls differ')\n"
        "if sys.pycache_prefix is None or pathlib.Path(sys.pycache_prefix).exists(): "
        "raise ValueError('Owned bytecode cache must be absent')\n"
        "if os.environ.get('PYTHONHASHSEED')!='0' or "
        "os.environ.get('PYTHONDONTWRITEBYTECODE')!='1': "
        "raise ValueError('Owned Python environment differs')\n"
        "if any(k.startswith('DYLD_') or (k.startswith('PYTHON') and k not in "
        "{'PYTHONHASHSEED','PYTHONDONTWRITEBYTECODE'}) for k in os.environ): "
        "raise ValueError('Unrecognized owned interpreter override')\n"
        f"authority_path=pathlib.Path({str(authority_path)!r}); authority_sha={authority_sha!r}\n"
        "if authority_path.is_symlink() or authority_path.resolve()!=authority_path: "
        "raise ValueError('Noncanonical bootstrap authority')\n"
        "raw=authority_path.read_bytes()\n"
        "if hashlib.sha256(raw).hexdigest()!=authority_sha: "
        "raise ValueError('Owned bootstrap authority changed')\n"
        "authority=json.loads(raw)\n"
        "if authority['schema']!='gpu-owned-bootstrap-v1' or "
        f"authority['protocol']!={OWNED.protocol!r} or authority['cache']!=sys.pycache_prefix: "
        "raise ValueError('Owned bootstrap contract differs')\n"
        "for tree in authority['trees'].values():\n"
        " root=pathlib.Path(tree['root'])\n"
        " if not root.is_absolute() or root.resolve(strict=True)!=root: "
        "raise ValueError('Noncanonical bound root')\n"
        " for name,sha in tree['files'].items():\n"
        "  relative=pathlib.Path(name); path=root/relative\n"
        "  if relative.is_absolute() or relative.as_posix()!=name or '..' in relative.parts "
        "or '.' in relative.parts or path.resolve(strict=True)!=path or path.is_symlink(): "
        "raise ValueError('Unsafe bound bootstrap file')\n"
        "  with path.open('rb') as stream: "
        "actual=hashlib.file_digest(stream,'sha256').hexdigest()\n"
        "  if actual!=sha: raise ValueError('Bound bootstrap bytes changed: '+name)\n"
        "paths=authority['paths']; entries=authority['entries']\n"
        "initial=authority['initializer']['path']; initial_sha=authority['initializer']['sha256']\n"
        "if hashlib.sha256(pathlib.Path(initial).read_bytes()).hexdigest()!=initial_sha: "
        "raise ValueError('Owned package initializer changed')\n"
        "args=sys.argv[1:]\n"
        "if not args: raise ValueError('Missing owned entrypoint')\n"
        "module=args[0]=='-m'\n"
        "entry=str(pathlib.Path(paths[0])/(args[1].replace('.','/')+'.py')) "
        "if module else str(pathlib.Path(args[0]).resolve())\n"
        "if entry not in entries or "
        "hashlib.sha256(pathlib.Path(entry).read_bytes()).hexdigest()!=entries[entry]: "
        "raise ValueError('Unbound or changed owned entrypoint')\n"
        "sys.path[:0]=paths\n"
        "sys.argv=args[1:] if module else args\n"
        "if module: runpy.run_module(args[1],run_name='__main__',alter_sys=True)\n"
        "else: runpy.run_path(entry,run_name='__main__')\n"
        "if pathlib.Path(sys.pycache_prefix).exists(): "
        "raise ValueError('Owned worker wrote bytecode')\n"
    )
    return [
        str(Path(sys.executable).resolve()),
        "-P",
        "-S",
        "-B",
        "-X",
        "pycache_prefix=" + str(cache),
        "-c",
        code,
    ]


def child_controls(manifest, environment):
    """Remove interpreter/native injection paths for owned children only."""
    result = dict(environment)
    if resolve_context(manifest).owned:
        for name in list(result):
            if (
                name.startswith("DYLD_")
                or name.startswith("PYTHON")
                and name
                not in {
                    "PYTHONHASHSEED",
                    "PYTHONDONTWRITEBYTECODE",
                }
            ):
                result.pop(name)
        for name in (
            "PYTHONHOME",
            "PYTHONSTARTUP",
            "PYTHONINSPECT",
            "PYTHONPYCACHEPREFIX",
            "DYLD_INSERT_LIBRARIES",
            "DYLD_LIBRARY_PATH",
            "DYLD_FRAMEWORK_PATH",
        ):
            result.pop(name, None)
        result["PYTHONDONTWRITEBYTECODE"] = "1"
        result["PYTHONHASHSEED"] = "0"
    return result


def write_authority(manifest, roots, path):
    """Snapshot existing authenticated control inputs for an isolated GPU child."""
    from transformer.gpu_study import digest, verify_bindings, write_new

    if not resolve_context(manifest).owned:
        return []
    verify_bindings(manifest, roots)
    write_new(
        path,
        {
            "schema": "gpu-owned-child-authority-v1",
            "manifest": manifest,
            "roots": {key: str(Path(value).resolve()) for key, value in roots.items()},
        },
    )
    return ["--authority", str(Path(path).resolve()), "--authority-sha256", digest(path)]


def read_authority(path, expected_sha256):
    from transformer.gpu_study import digest, read_json, require, validate_manifest, verify_bindings

    require(path is not None and expected_sha256 is not None, "Missing child authority anchor")
    require(digest(path) == expected_sha256, "Child authority changed")
    value = read_json(path)
    require(
        set(value) == {"schema", "manifest", "roots"}
        and value["schema"] == "gpu-owned-child-authority-v1",
        "Malformed child authority",
    )
    manifest = validate_manifest(value["manifest"])
    require(resolve_context(manifest).owned, "Child authority is not the owned namespace")
    verify_bindings(manifest, value["roots"])
    return manifest, value["roots"]
