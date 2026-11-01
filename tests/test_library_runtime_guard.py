"""Immutable-tree guard mutations; no numerical frameworks or operands."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / "integration/library-study-v1"


def module(name):
    spec = importlib.util.spec_from_file_location("test_guard_" + name, ROOT / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


G, C = module("runtime_guard"), module("contract")


def setup(tmp_path, extra_bindings=()):
    runtime, metadata, control, math = (
        tmp_path / n for n in ("runtime", "metadata", "control", "math")
    )
    for directory in (runtime, metadata, control, math):
        directory.mkdir()
    payload = runtime / "payload.bin"
    payload.write_bytes(b"original payload")
    manifest = metadata / "manifest.json"
    C.write(manifest, {"payload": C.binding(payload)})
    receipt = metadata / "receipt.json"
    C.write(
        receipt,
        dict(frameworks=dict(test=dict(payload_root=str(runtime), manifest=C.binding(manifest)))),
    )
    bindings = [C.binding(p) for p in (payload, manifest, receipt)] + list(extra_bindings)

    class Authority:
        receipt_path = str(receipt)
        receipt_sha256 = C.binding(receipt)["sha256"]
        calls = 0

        def verify(self):
            for binding in bindings:
                C.bound(binding)
            self.calls += 1
            return {"identity": "full-byte-authentication", "runtime_identity_sha256": "a" * 64}, {
                "test": {str(payload): bindings[0]["sha256"]}
            }

    authority = Authority()
    roles = {k: dict(root=str(p)) for k, p in (("control", control), ("math", math))}
    inventory = dict(layout=dict(scan_roots=[str(runtime)], aliases={}))
    inventory["files"] = {b["path"].lstrip("/"): b["sha256"] for b in bindings}
    authority.inventory_json = C.canonical(inventory)
    return SimpleNamespace(
        guard=G.RuntimeGuard(authority, inventory, roles, C),
        authority=authority,
        runtime=runtime,
        payload=payload,
        inventory=inventory,
        roles=roles,
        receipt=C.binding(receipt),
    )


def test_guard_reuses_only_process_local_authenticated_bytes_and_copies_proof(tmp_path):
    case = setup(tmp_path)
    first = case.guard.verify()
    first[0]["identity"] = "caller mutation"
    assert case.guard.verify()[0]["identity"] == "full-byte-authentication"
    assert case.authority.calls == 1
    case.guard.verify(full=True)
    assert case.authority.calls == 2
    assert case.guard.witness()["full_authentications"] == 2


@pytest.mark.parametrize(
    "mutation", ["bytes", "restored_mtime", "add", "delete", "replace", "mode", "new_directory"]
)
def test_guard_rejects_file_and_completeness_changes(tmp_path, mutation):
    case = setup(tmp_path)
    case.guard.verify()
    old = case.payload.stat()
    if mutation in ("bytes", "restored_mtime"):
        case.payload.write_bytes(b"changed! payload")
        if mutation == "restored_mtime":
            os.utime(case.payload, ns=(old.st_atime_ns, old.st_mtime_ns))
    elif mutation == "add":
        (case.runtime / "unbound.py").write_text("unbound import")
    elif mutation == "delete":
        case.payload.unlink()
    elif mutation == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(case.payload.read_bytes())
        os.replace(replacement, case.payload)
    elif mutation == "mode":
        case.payload.chmod(0o755)
    else:
        (case.runtime / "new").mkdir()
    with pytest.raises(ValueError, match="inventory or metadata changed"):
        case.guard.verify()
    assert case.authority.calls == 1


@pytest.mark.parametrize("mutation", ["target_bytes", "replace_link", "replace_directory"])
def test_guard_tracks_symlinks_and_their_external_targets(tmp_path, mutation):
    outside = tmp_path / "outside"
    outside.write_bytes(b"external source")
    case = setup(tmp_path, [C.binding(outside)])
    link = case.runtime / "alias"
    link.symlink_to(outside)
    case.inventory["layout"]["aliases"][str(link)] = dict(link=str(outside), resolved=str(outside))
    nested = case.runtime / "nested"
    nested.mkdir()
    case.guard.verify()
    if mutation == "target_bytes":
        outside.write_bytes(b"external change")
    elif mutation == "replace_link":
        other = tmp_path / "other"
        other.write_bytes(outside.read_bytes())
        link.unlink()
        link.symlink_to(other)
    else:
        nested.rmdir()
        nested.symlink_to(tmp_path / "control", target_is_directory=True)
    with pytest.raises(ValueError, match="inventory or metadata changed"):
        case.guard.verify()


def test_guard_rejects_changes_during_initial_full_authentication(tmp_path):
    case = setup(tmp_path)
    original = case.authority.verify

    def changed():
        result = original()
        case.payload.write_bytes(b"changed! payload")
        return result

    case.authority.verify = changed
    with pytest.raises(ValueError, match="during full authentication"):
        case.guard.verify()


def test_guard_cannot_be_reused_by_a_different_process(tmp_path, monkeypatch):
    case = setup(tmp_path)
    case.guard.verify()
    owner = os.getpid()
    with monkeypatch.context() as patch:
        patch.setattr(G.os, "getpid", lambda: owner + 1)
        with pytest.raises(ValueError, match="crossed a process"):
            case.guard.verify()


def test_guard_tracks_aliases_outside_scanned_directories(tmp_path):
    case = setup(tmp_path)
    alias = tmp_path / "prefix-alias"
    alias.symlink_to(case.runtime, target_is_directory=True)
    case.inventory["layout"]["aliases"][str(alias)] = dict(
        link=str(case.runtime), resolved=str(case.runtime)
    )
    case.authority.inventory_json = C.canonical(case.inventory)
    guard = G.RuntimeGuard(case.authority, case.inventory, case.roles, C)
    guard.verify()
    alias.unlink()
    alias.symlink_to(tmp_path / "control", target_is_directory=True)
    with pytest.raises(ValueError, match="inventory or metadata changed"):
        guard.verify()


def test_guard_cannot_watch_a_different_inventory_than_its_byte_authority(tmp_path):
    case = setup(tmp_path)
    changed = dict(case.inventory, layout=dict(scan_roots=[str(tmp_path / "control")], aliases={}))
    guard = G.RuntimeGuard(case.authority, changed, case.roles, C)
    with pytest.raises(ValueError, match="differs from the authenticated authority"):
        guard.verify()


def test_guard_witness_binds_full_postflight_process_and_complete_context(tmp_path):
    import copy

    audit = module("audit")
    case = setup(tmp_path)
    case.roles["control"]["files"] = {"integration/library-study-v1/runtime_guard.py": "b" * 64}
    guard = G.RuntimeGuard(case.authority, case.inventory, case.roles, C)
    guard.verify()
    guard.verify(full=True)
    proof = guard.witness()
    capture = dict(runtime_receipt=case.receipt, runtime_identity_sha256="a" * 64)
    audit.validate_validation(
        proof, case.roles, capture, case.inventory, pid=os.getpid(), final=True
    )
    changes = {
        "pid": os.getpid() + 1,
        "full_authentications": 1,
        "entries": 0,
        "roots": [],
        "runtime_identity_sha256": "b" * 64,
        "inventory_identity_sha256": "b" * 64,
        "source_roles_sha256": "b" * 64,
        "metadata_fingerprint_sha256": "bad",
    }
    for name, value in changes.items():
        altered = copy.deepcopy(proof)
        altered[name] = value
        with pytest.raises(ValueError, match="Invalid process-local"):
            audit.validate_validation(
                altered, case.roles, capture, case.inventory, pid=os.getpid(), final=True
            )
