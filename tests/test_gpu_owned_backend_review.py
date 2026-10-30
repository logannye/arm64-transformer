"""Independent integrated origin checks; all framework modules are CPU fixtures."""

import copy
import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
import test_gpu_owned_origins as origin_tests

from transformer import gpu_owned_backend as owned
from transformer import gpu_owned_origins as origins

origin_fixture = origin_tests.fixture


@contextmanager
def authenticated_process(f):
    native = f.file("libtorch_cpu.dylib" if f.name == "pytorch" else "libmlx.dylib")
    if f.name == "pytorch":
        f.top.ops.loaded_libraries = set()
        f.top.version = SimpleNamespace(git_version="a" * 40)
    payloads = {"pytorch": {}, "mlx": {}}
    payloads[f.name] = f.payload
    authority = SimpleNamespace(root=str(f.root), inventory_json=json.dumps(f.inventory).encode())
    verified = ({"frameworks": {"pytorch": {"selection": {"base": "a" * 40}}}}, payloads)
    with (
        mock.patch.dict(sys.modules),
        mock.patch.object(owned, "_loaded_libraries", return_value=[str(native)]),
    ):
        for key in tuple(sys.modules):
            if key.split(".", 1)[0] in {"torch", "torchgen", "functorch", "mlx"}:
                del sys.modules[key]
        sys.modules.update(f.modules)
        yield "mps" if f.name == "pytorch" else "mlx", authority, verified


def test_backend_binds_all_loaded_framework_origins(origin_fixture):
    f = origin_fixture
    with authenticated_process(f) as args:
        proof = owned._loaded_identity(*args)
    assert proof["framework_module_origins"][f.name]["module_count"] == len(f.modules)
    assert proof["framework_module_origins"][f.name]["native_aliases"]


@pytest.mark.parametrize("suffix", ["nn", "optimizer"])
def test_backend_rejects_foreign_submodules_despite_matching_root_and_native(
    origin_fixture, suffix
):
    f = origin_fixture
    key = (
        ("torch.nn" if suffix == "nn" else "torch.optim")
        if f.name == "pytorch"
        else ("mlx.nn" if suffix == "nn" else "mlx.optimizers")
    )
    foreign = f.root / "foreign.py"
    foreign.write_text("# different imported implementation\n")
    f.modules[key].__file__ = str(foreign)
    with authenticated_process(f) as args, pytest.raises(ValueError, match="payload"):
        owned._loaded_identity(*args)


def test_backend_cannot_swallow_origin_validator_failure(origin_fixture):
    with (
        authenticated_process(origin_fixture) as args,
        mock.patch.object(
            origins, "verify_loaded_framework_modules", side_effect=ValueError("unbound namespace")
        ) as validator,
    ):
        with pytest.raises(ValueError, match="unbound namespace"):
            owned._loaded_identity(*args)
        validator.assert_called_once()


def test_backend_forwards_exact_payload_and_inventory_to_origin_checker(origin_fixture):
    f = origin_fixture
    sentinel = {"schema": "test-only-origin-proof"}
    with (
        authenticated_process(f) as args,
        mock.patch.object(
            origins, "verify_loaded_framework_modules", return_value=sentinel
        ) as validator,
    ):
        result = owned._loaded_identity(*args)
        validator.assert_called_once_with(
            f.name, f.payload, json.loads(args[1].inventory_json), str(f.root)
        )
    assert result["framework_module_origins"][f.name] == sentinel


def test_backend_native_image_must_belong_to_full_runtime(origin_fixture):
    f = origin_fixture
    with authenticated_process(f) as args:
        authority = copy.copy(args[1])
        inventory = json.loads(authority.inventory_json)
        native_name = "libtorch_cpu.dylib" if f.name == "pytorch" else "libmlx.dylib"
        del inventory["files"][f"{f.name}/{native_name}"]
        authority.inventory_json = json.dumps(inventory).encode()
        with pytest.raises(ValueError, match="full runtime"):
            owned._loaded_identity(args[0], authority, args[2])
