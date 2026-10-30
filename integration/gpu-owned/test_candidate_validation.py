"""Stdlib-only transport, selection, and failed-evidence controls; no GPU imports."""

import copy
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "candidate_validation", HERE / "candidate_validation.py"
)
candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidate)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.addCleanup(self.temporary.cleanup)

    def archive(self, *, source=True, extra=(), omit=(), corrupt=False):
        payload = b"original source\n"
        row = {"sha256": sha(payload), "bytes": len(payload)}
        if source:
            row.update(kind="file", mode=0o644)
        manifest = {"files": {"test.py": row}}
        prefix = "source/" if source else "runtime/"
        entries = [(prefix + "test.py", b"changed source!\n" if corrupt else payload)]
        if source:
            entries += [
                ("source-manifest.json", json.dumps(manifest).encode()),
                ("changes.patch", b"patch"),
            ]
        else:
            entries += [("runtime/runtime-manifest.json", json.dumps(manifest).encode())]
        entries += list(extra)
        path = self.root / "archive.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for name, value in entries:
                if name in omit:
                    continue
                info = tarfile.TarInfo(name)
                info.mode = 0o644
                if isinstance(value, tarfile.TarInfo):
                    archive.addfile(value)
                else:
                    info.size = len(value)
                    archive.addfile(info, io.BytesIO(value))
        return path, manifest

    def test_source_and_runtime_roundtrip(self):
        for source in (True, False):
            with self.subTest(source=source):
                path, manifest = self.archive(source=source)
                metadata, size = candidate.archive_inventory(path, manifest, source=source)
                self.assertTrue(metadata)
                self.assertEqual(size, len(b"original source\n"))
                output = self.root / str(source)
                output.mkdir()
                candidate.expand_verified(path, output, source=source)
                proof = candidate.verify_tree(
                    output / ("source" if source else "runtime"), manifest, source=source
                )
                self.assertEqual(proof["files"], 1)

    def test_duplicate_missing_extra_and_changed_members_rejected(self):
        changes = [
            dict(extra=[("source/test.py", b"original source\n")]),
            dict(omit=["source/test.py"]),
            dict(extra=[("source/extra.py", b"bad")]),
            dict(corrupt=True),
        ]
        for change in changes:
            with self.subTest(change=change):
                path, manifest = self.archive(**change)
                with self.assertRaises(ValueError):
                    candidate.archive_inventory(path, manifest, source=True)

    def test_traversal_and_hardlink_rejected(self):
        link = tarfile.TarInfo("source/link")
        link.type, link.linkname = tarfile.LNKTYPE, "source/test.py"
        for extra in [[("../escape", b"bad")], [("source/link", link)]]:
            path, manifest = self.archive(extra=extra)
            with self.assertRaises(ValueError):
                candidate.archive_inventory(path, manifest, source=True)

    def test_source_symlink_escape_and_file_directory_collision_rejected(self):
        manifests = [
            {"files": {"a": {"kind": "symlink", "mode": 0o777, "target": "../outside"}}},
            {"files": {"a": {"kind": "symlink", "mode": 0o777, "target": "/absolute"}}},
            {
                "files": {
                    "a": {"sha256": "a" * 64, "bytes": 1},
                    "a/b": {"sha256": "b" * 64, "bytes": 1},
                }
            },
        ]
        for manifest in manifests:
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                candidate.manifest_files(manifest, source=True)

    def test_expanded_corruption_and_extra_file_rejected(self):
        path, manifest = self.archive()
        output = self.root / "expanded"
        output.mkdir()
        candidate.expand_verified(path, output, source=True)
        (output / "source/test.py").write_bytes(b"modified source\n")
        with self.assertRaises(ValueError):
            candidate.verify_tree(output / "source", manifest, source=True)
        (output / "source/test.py").write_bytes(b"original source\n")
        (output / "source/extra.py").write_bytes(b"extra")
        with self.assertRaises(ValueError):
            candidate.verify_tree(output / "source", manifest, source=True)

    def test_symlink_input_and_stale_binding_rejected(self):
        path = self.root / "input"
        path.write_bytes(b"initial")
        identity = candidate.binding(path)
        path.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            candidate.check_binding(identity)
        link = self.root / "link"
        link.symlink_to(path)
        with self.assertRaises(ValueError):
            candidate.binding(link)

    def test_duplicate_and_nonfinite_json_rejected(self):
        path = self.root / "record.json"
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            path.write_text(raw)
            with self.assertRaises(ValueError):
                candidate.json_read(path)

    def test_output_namespace_overwrite_and_dotdot_rejected(self):
        allowed = self.root / "work/gpu-v4-owned"
        allowed.mkdir(parents=True)
        for path in [self.root / "outside", allowed, allowed / ".." / "escape"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                candidate.output_path(self.root, path)
        fresh = allowed / "new"
        self.assertEqual(candidate.output_path(self.root, fresh), fresh)
        fresh.mkdir()
        with self.assertRaises(ValueError):
            candidate.output_path(self.root, fresh)

    def test_stale_plan_rejected_before_backend_or_subprocess(self):
        plan = self.root / "plan.json"
        plan.write_text('{"formal_eligible":true}')
        with mock.patch.object(candidate.subprocess, "Popen") as popen:
            with self.assertRaises(ValueError):
                candidate.run(plan, "0" * 64, self.root / "output", 10)
            popen.assert_not_called()

    def dependency_fixture(self):
        wheels, installed = self.root / "wheels", self.root / "dependencies"
        wheels.mkdir()
        installed.mkdir()
        inventory = {}
        for index in range(45):
            name = f"package{index}"
            files = {
                f"{name}-1.0.dist-info/METADATA": (
                    f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n".encode()
                ),
                f"{name}.py": f"value = {index}\n".encode(),
            }
            if index == 0:
                files["package0_vendor/vendor-1.0.dist-info/METADATA"] = (
                    b"Name: vendored-package\nVersion: 1.0\n"
                )
            wheel = wheels / f"{name}-1.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                for relative, raw in files.items():
                    archive.writestr(relative, raw)
                    target = installed / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
            inventory[wheel.name] = {
                "sha256": candidate.digest(wheel),
                "bytes": wheel.stat().st_size,
            }
        return installed, wheels, {"wheel_inventory": inventory}

    def test_dependency_payload_corruption_rejected_despite_same_metadata(self):
        installed, wheels, contract = self.dependency_fixture()
        proof = candidate.verify_dependencies(installed, wheels, contract)
        self.assertEqual(proof["wheel_count"], 45)
        (installed / "package0.py").write_bytes(b"different bytes\n")
        with self.assertRaises(ValueError):
            candidate.verify_dependencies(installed, wheels, contract)

    def test_unbound_native_or_python_shadow_file_rejected(self):
        installed, wheels, contract = self.dependency_fixture()
        for name in ("package0.so", "unbound.py", "package0.pyc"):
            with self.subTest(name=name):
                extra = installed / name
                extra.write_bytes(b"unbound executable code")
                with self.assertRaises(ValueError):
                    candidate.verify_dependencies(installed, wheels, contract)
                extra.unlink()

    def test_installed_bytecode_is_bound_but_explicitly_not_executed(self):
        installed, wheels, contract = self.dependency_fixture()
        cache = installed / "__pycache__/package0.cpython-312.pyc"
        cache.parent.mkdir()
        cache.write_bytes(b"cache bytes")
        first = candidate.verify_dependencies(installed, wheels, contract)
        self.assertEqual(first["non_wheel_files"], [candidate.binding(cache)])
        self.assertIn("never executed", first["bytecode_policy"])
        cache.write_bytes(b"modified cache")
        second = candidate.verify_dependencies(installed, wheels, contract)
        self.assertNotEqual(first, second)


class CandidateContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(
            (HERE.parents[1] / "validation/owned-pytorch-layout-v1/contract.json").read_text()
        )
        self.head = "f" * 40
        self.receipt = {
            "schema": "pytorch-owned-layout-hosted-v1",
            "contract_sha256": candidate.CONTRACT_SHA,
            "source_revision": candidate.BASE,
            "patch_sha256": self.contract["combined_patch_sha256"],
            "component_patches": copy.deepcopy(self.contract["component_patches"]),
            "github": {
                "GITHUB_REPOSITORY": "logannye/arm64-transformer",
                "GITHUB_RUN_ID": candidate.HOSTED_RUN,
                "GITHUB_SHA": self.head,
            },
            "commands": [
                {"label": label, "status": "passed", "exit_code": 0}
                for label in [
                    "full-framework-build",
                    "normal-editable-import",
                    "transported-runtime-import",
                    "archive-roundtrip-import",
                ]
            ],
            "changed_translation_units": {
                name: [
                    {
                        "source_sha256": self.contract["source_sha256"][
                            "aten/src/ATen/native/mps/operations/" + name
                        ],
                        "object_sha256": "f" * 64,
                    }
                ]
                for name in ("Embedding.mm", "Matmul.mm", "Convolution.mm")
            },
            "embedding_metal_object": {
                "source_sha256": self.contract["source_sha256"][
                    "aten/src/ATen/native/mps/kernels/Embedding.metal"
                ]
            },
            "runtime_archive": {
                "qualification": "import-and-portability-validated; device-numerics-unvalidated"
            },
        }

    def test_valid_build_is_only_development_transport(self):
        candidate.validate_hosted(self.receipt, self.contract, self.head)
        self.assertNotIn("formal_eligible", self.receipt)

    def test_wrong_hosted_origin_or_patch_rejected(self):
        paths = [
            ("github", "GITHUB_RUN_ID"),
            ("github", "GITHUB_SHA"),
            ("github", "GITHUB_REPOSITORY"),
        ]
        for outer, inner in paths:
            bad = copy.deepcopy(self.receipt)
            bad[outer][inner] = "wrong"
            with self.subTest(field=inner), self.assertRaises(ValueError):
                candidate.validate_hosted(bad, self.contract, self.head)
        for field in ("source_revision", "contract_sha256", "patch_sha256"):
            bad = copy.deepcopy(self.receipt)
            bad[field] = "0" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                candidate.validate_hosted(bad, self.contract, self.head)

    def test_missing_failed_duplicate_build_proof_rejected(self):
        cases = []
        missing = copy.deepcopy(self.receipt)
        missing["commands"].pop()
        cases.append(missing)
        failed = copy.deepcopy(self.receipt)
        failed["commands"][0].update(status="failed", exit_code=1)
        cases.append(failed)
        duplicate = copy.deepcopy(self.receipt)
        duplicate["commands"].append(copy.deepcopy(duplicate["commands"][0]))
        cases.append(duplicate)
        source = copy.deepcopy(self.receipt)
        source["changed_translation_units"]["Matmul.mm"][0]["source_sha256"] = "0" * 64
        cases.append(source)
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                candidate.validate_hosted(bad, self.contract, self.head)

    def test_partial_or_unknown_framework_control_rejected(self):
        for env in (
            {"PYTORCH_OPINFO_SAMPLE_INPUT_INDEX": "0"},
            {"PYTORCH_FUTURE_KERNEL_CONTROL": "1"},
            {"DYLD_FUTURE_PATH": "/tmp"},
        ):
            with self.subTest(env=env), self.assertRaises(ValueError):
                candidate.controlled_environment(env)

    def test_controls_pin_normal_dispatch_and_disabled_parent_fallback(self):
        env = candidate.controlled_environment({"PATH": "/bin", "PYTORCH_MPS_PREFER_METAL": "1"})
        self.assertNotIn("PYTORCH_MPS_PREFER_METAL", env)
        self.assertEqual(env["PYTORCH_ENABLE_MPS_FALLBACK"], "0")
        self.assertEqual(env["PYTORCH_MPS_FAST_MATH"], "0")


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f"integrated_upstream_test_mps.TestMPS.test_{index}" for index in range(27)]
        self.failed = [
            {"id": identity, "original_kind": "failures", "original_traceback": "trace"}
            for identity in self.ids
        ]
        self.original = {
            "planned_tests": self.ids + ["retained_skip", "retained_xfail"],
            "tests": {
                "skipped": [{"id": "retained_skip"}],
                "expected_failures": [{"id": "retained_xfail"}],
            },
        }
        self.plan = {
            "focused_inventory": self.failed,
            "workers": [
                {"index": i, "id": identity, "status": "not_started"}
                for i, identity in enumerate(self.ids)
            ],
            "full_suite": {
                "required": True,
                "planned_ids": self.original["planned_tests"],
                "retained_skip_ids": ["retained_skip"],
                "retained_expected_failure_ids": ["retained_xfail"],
                "status": "pending_candidate_stock_conformance_fixture",
                "local_fixture_compilation_enabled": False,
            },
        }
        self.row = self.plan["workers"][0]
        state = {"operator_libraries": [], "host_threads": 1, "interop_threads": 1}
        self.record = {
            "schema": candidate.SCHEMA,
            "protocol": candidate.PROTOCOL,
            "plan_sha256": "a" * 64,
            "pid": 123,
            "index": 0,
            "id": self.ids[0],
            "complete": True,
            "formal_eligible": False,
            "integrity_after": True,
            "outcome": "passed",
            "controlled_environment": dict(candidate.PINS),
            "state_after_bootstrap": dict(state),
            "state_before_test": dict(state),
            "state_after_test": dict(state),
            "tests": {
                "run": 1,
                "executed_ids": [self.ids[0]],
                "failures": [],
                "errors": [],
                "skipped": [],
                "expected_failures": [],
                "unexpected_successes": [],
                "successful": True,
            },
        }

    def test_all_whole_methods_and_retained_full_suite_required(self):
        candidate.validate_inventory(self.plan, self.original, self.failed)

    def test_missing_duplicate_reordered_ids_and_removed_failures_rejected(self):
        variants = []
        for operation in (
            lambda p: p["workers"].pop(),
            lambda p: p["workers"].reverse(),
            lambda p: p["workers"].append(copy.deepcopy(p["workers"][0])),
            lambda p: p["focused_inventory"][0].update(original_traceback=""),
        ):
            bad = copy.deepcopy(self.plan)
            operation(bad)
            variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                candidate.validate_inventory(bad, self.original, self.failed)

    def test_full_suite_or_eligibility_cannot_be_waived(self):
        for field, value in [
            ("required", False),
            ("planned_ids", []),
            ("retained_skip_ids", []),
            ("retained_expected_failure_ids", []),
            ("local_fixture_compilation_enabled", True),
            ("status", "passed"),
        ]:
            bad = copy.deepcopy(self.plan)
            bad["full_suite"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                candidate.validate_inventory(bad, self.original, self.failed)

    def test_passing_and_retained_failed_worker_are_valid_evidence(self):
        candidate.validate_result(self.record, self.row, 123, 0, "a" * 64)
        self.record["tests"]["failures"] = [{"id": self.ids[0], "traceback": "unchanged assertion"}]
        self.record["tests"]["successful"] = False
        self.record["outcome"] = "failed"
        candidate.validate_result(self.record, self.row, 123, 1, "a" * 64)

    def test_failure_removal_wrong_exit_or_new_skip_rejected(self):
        for operation, code in [
            (lambda r: r["tests"].update(skipped=[{"id": self.ids[0], "reason": "new"}]), 0),
            (lambda r: r["tests"].update(executed_ids=[]), 0),
            (lambda r: r.update(outcome="failed"), 0),
            (lambda r: r["tests"].update(failures=[{"id": self.ids[0], "traceback": ""}]), 1),
            (lambda r: r.update(integrity_after=False), 0),
            (lambda r: r.update(formal_eligible=True), 0),
            (lambda r: r.update(plan_sha256="b" * 64), 0),
            (lambda r: r["state_after_test"].update(operator_libraries=["override.dylib"]), 0),
        ]:
            bad = copy.deepcopy(self.record)
            operation(bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                candidate.validate_result(bad, self.row, 123, code, "a" * 64)

    def test_no_numerical_module_imported(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)
        self.assertNotIn("mlx", sys.modules)

    def test_final_audit_rejects_changed_earlier_failure_or_log(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            runtime = root / "runtime"
            runtime.mkdir()
            native = runtime / "libtorch_cpu.dylib"
            native.write_bytes(b"test-only native identity, never loaded")
            imported = root / "runtime-import.json"
            candidate.write(
                imported,
                {
                    "schema": "pytorch-runtime-import-v1",
                    "status": "passed",
                    "runtime": str(runtime),
                    "source_revision": candidate.BASE,
                    "runtime_manifest_sha256": "c" * 64,
                    "loaded_operator_libraries": [],
                    "mps_built": True,
                    "mps_available": True,
                },
            )
            result = copy.deepcopy(self.record)
            result.update(
                normal_import=candidate.binding(imported),
                source_manifest_sha256="b" * 64,
                runtime_manifest_sha256="c" * 64,
                native_origins_before=[candidate.binding(native)],
                native_origins_after=[candidate.binding(native)],
            )
            result["tests"]["failures"] = [{"id": self.ids[0], "traceback": "retained failure"}]
            result["tests"]["successful"] = False
            result["outcome"] = "failed"
            result_path, log = root / "result.json", root / "worker.log"
            candidate.write(result_path, result)
            log.write_text("raw failure log")
            row = {
                **self.row,
                "pid": 123,
                "exit_code": 1,
                "status": "failed",
                "result": candidate.binding(result_path),
                "log": candidate.binding(log),
            }
            plan = {
                "workers": [self.row],
                "runtime": str(runtime),
                "runtime_manifest": {"sha256": "c" * 64},
                "source_manifest": {"sha256": "b" * 64},
            }
            candidate.audit_completed_workers([row], plan, "a" * 64)
            original_result, original_log = result_path.read_bytes(), log.read_bytes()
            result_path.write_text('{"failure_removed":true}')
            with self.assertRaises(ValueError):
                candidate.audit_completed_workers([row], plan, "a" * 64)
            result_path.write_bytes(original_result)
            log.write_text("failure log replaced")
            with self.assertRaises(ValueError):
                candidate.audit_completed_workers([row], plan, "a" * 64)
            log.write_bytes(original_log)
            imported.unlink()
            with self.assertRaises(ValueError):
                candidate.audit_completed_workers([row], plan, "a" * 64)


if __name__ == "__main__":
    unittest.main()
