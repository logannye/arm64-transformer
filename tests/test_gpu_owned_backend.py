"""CPU-only authority mutations; no framework imports or native compilation."""

import copy
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from transformer import gpu_owned_backend as owned
from transformer.gpu_owned_study import FRAMEWORKS
from transformer.gpu_study import canonical, identity


def save(path, value):
    path.write_bytes(canonical(value))
    return owned._file(path)


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path.resolve()
    selection = copy.deepcopy(FRAMEWORKS)
    inventory = {"files": {}, "python_paths": [], "layout": {"packages": {}}}
    rows = {}
    for name, package in (("pytorch", "torch"), ("mlx", "mlx")):
        base = root / name
        (base / package).mkdir(parents=True)
        package_file = base / package / "__init__.py"
        package_file.write_text("# fixture module\n")
        native_file = base / ("libtorch_cpu.dylib" if name == "pytorch" else "libmlx.dylib")
        native_file.write_bytes(b"fixture-native-bytes")
        records = {}
        for p in (package_file, native_file):
            proof = owned._file(p)
            records[p.relative_to(base).as_posix()] = {
                "sha256": proof["sha256"],
                "bytes": proof["bytes"],
            }
            inventory["files"][p.relative_to(root).as_posix()] = proof["sha256"]
        if name == "pytorch":
            value = {
                "source_revision": selection[name]["base"],
                "patch_sha256": selection[name]["patches"][0],
                "files": records,
            }
        else:
            value = {
                "base": selection[name]["base"],
                "mode": "jit-on",
                "variant": "patched",
                "patches": dict(zip(("embedding", "attention"), selection[name]["patches"])),
                "files": records,
            }
        binding = save(base / "manifest.json", value)
        selection[name]["runtime_manifest_sha256"] = binding["sha256"]
        rows[name] = {"selection": selection[name], "manifest": binding, "payload_root": str(base)}
        if name == "pytorch":
            rows[name]["combined_patch_sha256"] = value["patch_sha256"]
        inventory["layout"]["packages"][package] = [str(package_file), str(package_file.parent)]
    receipt = {
        "schema": owned.SCHEMA,
        "protocol": owned.PROTOCOL,
        "runtime_identity_sha256": identity(inventory["files"]),
        "frameworks": rows,
        "bytecode_policy": copy.deepcopy(owned.BYTECODE_POLICY),
    }
    source = root / "project/transformer/gpu_owned_backend.py"
    source.parent.mkdir(parents=True)
    source.write_text("# source binding fixture\n")
    receipt["project"] = {
        "root": str(source.parent.parent),
        "files": {"transformer/gpu_owned_backend.py": owned.digest(source)},
    }
    receipt_path = root / "owned-runtime.json"
    receipt_sha = save(receipt_path, receipt)["sha256"]
    return SimpleNamespace(
        root=root,
        selection=selection,
        inventory=inventory,
        receipt=receipt,
        path=receipt_path,
        sha=receipt_sha,
    )


def authenticate(f):
    with mock.patch("transformer.gpu_study_runtime.verify_runtime") as discovered:
        result = owned.authenticate_runtime(f.root, f.inventory, f.selection, f.path, f.sha)
        discovered.assert_called_once_with(f.root, f.inventory)
        return result


def rewrite(f):
    f.sha = save(f.path, f.receipt)["sha256"]


def test_runtime_authority_uses_full_discovery_and_frozen_inputs(fixture):
    authority = authenticate(fixture)
    fixture.selection["pytorch"]["base"] = "f" * 40
    assert json.loads(authority.selection_json)["pytorch"]["base"] != "f" * 40


def test_numeric_truth_value_cannot_replace_bytecode_policy(fixture):
    fixture.receipt["bytecode_policy"]["writes_disabled"] = 1
    rewrite(fixture)
    with pytest.raises(ValueError, match="Incomplete or foreign"):
        authenticate(fixture)


def test_authority_module_imports_no_framework():
    script = (
        "import sys; sys.path.insert(0, " + repr(str(Path(__file__).resolve().parents[1])) + "); "
        "import transformer.gpu_owned_backend; "
        "assert 'torch' not in sys.modules and 'mlx.core' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script], check=True)


@pytest.mark.parametrize("mutation", ["receipt_hash", "protocol", "runtime_identity", "extra_pass"])
def test_caller_pass_or_foreign_receipt_cannot_authenticate(fixture, mutation):
    if mutation == "receipt_hash":
        fixture.sha = "0" * 64
    elif mutation == "protocol":
        fixture.receipt["protocol"] = "gpu-v2-five-engine-v2"
        rewrite(fixture)
    elif mutation == "runtime_identity":
        fixture.receipt["runtime_identity_sha256"] = "0" * 64
        rewrite(fixture)
    else:
        fixture.receipt["passed"] = True
        rewrite(fixture)
    with pytest.raises(ValueError):
        authenticate(fixture)


@pytest.mark.parametrize("mutation", ["missing", "extra", "changed", "symlink"])
def test_native_payload_changes_fail(fixture, mutation):
    path = fixture.root / "pytorch/libtorch_cpu.dylib"
    if mutation == "missing":
        path.unlink()
    elif mutation == "extra":
        (path.parent / "extra.py").write_text("# unbound")
    elif mutation == "changed":
        path.write_bytes(b"changed")
    else:
        path.unlink()
        path.symlink_to(fixture.root / "mlx/libmlx.dylib")
    with pytest.raises(ValueError):
        authenticate(fixture)


def test_package_resolution_must_select_bound_native_tree(fixture):
    fixture.inventory["layout"]["packages"]["torch"] = [str(fixture.root / "foreign/torch")]
    with pytest.raises(ValueError, match="different framework"):
        authenticate(fixture)


def test_framework_package_bytes_must_be_in_full_runtime(fixture):
    del fixture.inventory["files"]["pytorch/torch/__init__.py"]
    fixture.receipt["runtime_identity_sha256"] = identity(fixture.inventory["files"])
    rewrite(fixture)
    with pytest.raises(ValueError, match="Package bytes"):
        authenticate(fixture)


def test_cpu_and_mps_share_one_selection(fixture):
    fixture.selection["pytorch"]["engines"] = ["mps"]
    rewrite(fixture)
    with pytest.raises(ValueError, match="share one"):
        authenticate(fixture)


def test_multiple_patches_require_explicit_combined_identity(fixture):
    fixture.selection["pytorch"]["patches"].append("d" * 64)
    rewrite(fixture)
    with pytest.raises(ValueError, match="bound combined patch"):
        authenticate(fixture)


def test_patch_order_is_part_of_receipt_selection(fixture):
    fixture.receipt["frameworks"]["mlx"]["selection"] = copy.deepcopy(fixture.selection["mlx"])
    fixture.receipt["frameworks"]["mlx"]["selection"]["patches"].reverse()
    rewrite(fixture)
    with pytest.raises(ValueError, match="selection differs"):
        authenticate(fixture)


def test_receipt_bytes_are_rechecked(fixture):
    authority = authenticate(fixture)
    fixture.path.write_bytes(fixture.path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="receipt changed"):
        authority.verify()


def test_full_discovery_failure_cannot_be_waived(fixture):
    with mock.patch(
        "transformer.gpu_study_runtime.verify_runtime",
        side_effect=ValueError("omitted interpreter"),
    ):
        with pytest.raises(ValueError, match="omitted interpreter"):
            owned.authenticate_runtime(
                fixture.root, fixture.inventory, fixture.selection, fixture.path, fixture.sha
            )


def test_wrapper_delegates_numerical_methods_and_batch_ownership():
    class Backend:
        step = 0

        def upload_inputs(self, token):
            return SimpleNamespace(owner=self, tokens=token)

        def train_step(self, batch):
            assert batch.owner is self
            self.step += 1
            return batch.tokens

    original = Backend()
    wrapped = owned._OwnedBackend(original, "mps", None)
    batch = wrapped.upload_inputs(object())
    assert wrapped.train_step(batch) is batch.tokens
    assert wrapped.step == original.step == 1
    assert type(original) is Backend


def test_capability_failure_closes_created_backend():
    authority = owned.RuntimeAuthority("/", b"{}", b"{}", "/receipt", "0" * 64)
    backend = mock.Mock()
    module = SimpleNamespace(Backend=mock.Mock(return_value=backend))
    with (
        mock.patch.object(owned, "verify_bytecode_policy"),
        mock.patch.object(owned.RuntimeAuthority, "verify"),
        mock.patch.object(owned.importlib, "import_module", return_value=module),
        mock.patch.object(owned, "_capabilities", side_effect=ValueError("foreign library")),
    ):
        with pytest.raises(ValueError, match="foreign library"):
            owned.create_runtime_backend("mps", object(), authority=authority)
    backend.close.assert_called_once_with()


def test_foreign_protocol_fails_before_formal_backend_execution():
    with mock.patch("transformer.gpu_study.verify_bindings") as verify:
        with pytest.raises(ValueError, match="another study"):
            owned.create_backend("mps", object(), manifest={"protocol": "other"}, roots={})
        verify.assert_not_called()


def test_file_mutation_during_read_is_detected(tmp_path):
    path = tmp_path.resolve() / "input"
    path.write_bytes(b"first")

    def change(p):
        value = hashlib.sha256(p.read_bytes()).hexdigest()
        p.write_bytes(b"changed")
        return value

    with mock.patch.object(owned, "digest", side_effect=change):
        with pytest.raises(ValueError, match="during read"):
            owned._file(path)


def compilation_probe():
    graph = owned.COMPILED_GRAPH.replace("NODE", "123")
    observations = []
    for index, offset in enumerate((0.5, 1.0, 2.0, -1.0)):
        values = [max((i - 128) / 8 + offset, 0.0) for i in range(257)]
        observations.append(
            {
                "index": index,
                "offset": offset,
                "python_trace_count": 1,
                "fp32_sha256": hashlib.sha256(struct.pack("<257f", *values)).hexdigest(),
                "exact_dyadic_oracle": True,
            }
        )
    return {
        "status": "passed",
        "device": "gpu",
        "compiled_primitive_present": True,
        "python_trace_count": 1,
        "executions": 4,
        "shape": [257],
        "dtype": "float32",
        "mode_override_used": False,
        "mlx_disable_compile": None,
        "graph_dot": graph,
        "graph_sha256": hashlib.sha256(graph.encode()).hexdigest(),
        "observations": observations,
    }


def test_compilation_proof_binds_raw_graph_and_exact_output_bytes():
    probe = compilation_probe()
    assert owned.validate_compilation_probe(probe) == probe


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_graph",
        "wrong_hash",
        "changed_node",
        "extra_edge",
        "missing_replay",
        "duplicate_replay",
        "changed_offset",
        "one_ulp",
        "missing_oracle",
        "override",
        "boolean_count",
    ],
)
def test_compilation_receipt_mutations_fail(mutation):
    probe = compilation_probe()
    if mutation == "missing_graph":
        del probe["graph_dot"]
    elif mutation == "wrong_hash":
        probe["graph_sha256"] = "0" * 64
    elif mutation in {"changed_node", "extra_edge"}:
        probe["graph_dot"] = (
            probe["graph_dot"].replace("CompiledBroadcastMultiplyAddBroadcastMaximum", "Add")
            if mutation == "changed_node"
            else probe["graph_dot"] + '\n"A" -> "E"'
        )
        probe["graph_sha256"] = hashlib.sha256(probe["graph_dot"].encode()).hexdigest()
    elif mutation == "missing_replay":
        probe["observations"].pop()
    elif mutation == "duplicate_replay":
        probe["observations"][1] = copy.deepcopy(probe["observations"][0])
    elif mutation == "changed_offset":
        probe["observations"][2]["offset"] = 3.0
    elif mutation == "one_ulp":
        values = [max((i - 128) / 8 + 0.5, 0.0) for i in range(257)]
        changed = bytearray(struct.pack("<257f", *values))
        changed[0] = 1  # Positive zero -> the adjacent positive FP32 subnormal.
        probe["observations"][0]["fp32_sha256"] = hashlib.sha256(changed).hexdigest()
    elif mutation == "missing_oracle":
        del probe["observations"][0]["exact_dyadic_oracle"]
    elif mutation == "override":
        probe["mode_override_used"] = True
    else:
        probe["python_trace_count"] = True
    with pytest.raises(ValueError):
        owned.validate_compilation_probe(probe)


@pytest.mark.parametrize("mutation", ["outside", "missing_origin", "wrong_spec", "unbound"])
def test_loaded_project_module_must_match_bound_source(fixture, mutation):
    project = fixture.receipt["project"]
    path = Path(project["root"]) / "transformer/gpu_owned_backend.py"
    module = SimpleNamespace(__file__=str(path), __spec__=SimpleNamespace(origin=str(path)))
    if mutation == "outside":
        module.__file__ = str(fixture.root / "pytorch/torch/__init__.py")
    elif mutation == "missing_origin":
        del module.__file__
    elif mutation == "wrong_spec":
        module.__spec__.origin = str(fixture.root / "pytorch/torch/__init__.py")
    else:
        path = Path(project["root"]) / "transformer/reference.py"
        path.write_text("# unbound model\n")
        module.__file__ = str(path)
        module.__spec__.origin = str(path)
    with pytest.raises(ValueError):
        owned.verify_loaded_project_modules(project, {"transformer.gpu_owned_backend": module})


def test_source_change_after_authentication_blocks_factory(fixture):
    authority = authenticate(fixture)
    (Path(fixture.receipt["project"]["root"]) / "transformer/gpu_owned_backend.py").write_text(
        "changed"
    )
    with mock.patch("transformer.gpu_study_runtime.verify_runtime"):
        with pytest.raises(ValueError, match="project source changed"):
            authority.verify()
