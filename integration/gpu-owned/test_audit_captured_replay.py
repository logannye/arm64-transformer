"""Captured replay provenance rejection checks; no numerical frameworks imported."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "tested_captured_audit", ROOT / "integration/trained-precision/audit_captured_replay.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.capture = self.root / "capture.json"
        self.sidecar = self.root / "captured-runtime.json"
        self.worker = self.root / "result.json"
        project = dict(root=str(self.root), files={})
        sources = {}
        for key, relative in (
            ("producer", "integration/trained-precision/captured_replay.py"),
            ("worker_source", "transformer/trained_precision.py"),
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# bound source\n")
            sources[key] = AUDIT.SUITE.binding(path)
            project["files"][relative] = sources[key]["sha256"]
        self.receipt = self.root / "runtime-receipt.json"
        self.receipt.write_text(json.dumps(dict(project=project, runtime_identity_sha256="c" * 64)))
        self.runtime = self.root / "inventory.json"
        self.runtime.write_text("{}")
        self.selection = self.root / "selection.json"
        self.selection.write_text("{}")
        self.cap = dict(
            schema="gpu-owned-runtime-capture-v1",
            status="identity_captured_not_qualified",
            formal_eligible=False,
            formal_fits=0,
            admission_stream_opened=False,
            runtime_identity_sha256="c" * 64,
            runtime_inventory=AUDIT.SUITE.binding(self.runtime),
            framework_selection=AUDIT.SUITE.binding(self.selection),
            runtime_receipt=AUDIT.SUITE.binding(self.receipt),
        )
        self.capture.write_text(json.dumps(self.cap))
        self.work = dict(complete=True, passed=True, engine="mps", width=512, pid=123)
        self.worker.write_text(json.dumps(self.work))
        self.value = dict(
            schema="trained-precision-captured-runtime-replay-v1",
            complete=True,
            passed=True,
            postflight_passed=True,
            formal_eligible=False,
            formal_fits=0,
            admission_stream_opened=False,
            capture=AUDIT.SUITE.binding(self.capture),
            runtime_receipt=self.cap["runtime_receipt"],
            runtime_identity_sha256="c" * 64,
            worker_receipt=AUDIT.SUITE.binding(self.worker),
            worker_returncode=0,
            engine="mps",
            width=512,
            pid=123,
            origins_before={},
            origins_after={},
            **sources,
        )
        patch = mock.patch.object(AUDIT, "audit_origins")
        self.origins = patch.start()
        self.addCleanup(patch.stop)

    def run_audit(self):
        self.sidecar.write_text(json.dumps(self.value))
        return AUDIT.audit(
            self.sidecar,
            AUDIT.SUITE.sha256(self.sidecar),
            self.capture,
            AUDIT.SUITE.sha256(self.capture),
        )

    def change_worker(self, **changes):
        self.work.update(changes)
        self.worker.write_text(json.dumps(self.work))
        self.value["worker_receipt"] = AUDIT.SUITE.binding(self.worker)

    def test_valid_chain_requires_two_origin_audits_and_keeps_numerics_separate(self):
        result = self.run_audit()
        self.assertTrue(result["evidence_valid"])
        self.assertFalse(result["numerics_independently_audited"])
        self.assertFalse(result["formal_eligible"])
        self.assertEqual(self.origins.call_count, 2)

    def test_complete_numerical_failure_remains_evidence_valid(self):
        self.change_worker(passed=False)
        self.value.update(passed=False, worker_returncode=1)
        result = self.run_audit()
        self.assertTrue(result["evidence_valid"])
        self.assertFalse(result["numerics_reported_passed"])

    def test_rejects_runtime_or_control_failures(self):
        original = copy.deepcopy(self.value)
        for changes in (
            dict(complete=False),
            dict(postflight_passed=False),
            dict(formal_eligible=True),
            dict(formal_fits=1),
            dict(admission_stream_opened=True),
            dict(error="failed"),
            dict(postflight_error="failed"),
            dict(runtime_identity_sha256="d" * 64),
        ):
            with self.subTest(changes=changes):
                self.value = dict(original, **changes)
                with self.assertRaises(ValueError):
                    self.run_audit()

    def test_rejects_spliced_or_incomplete_worker(self):
        for changes in (
            dict(pid=456),
            dict(engine="mlx"),
            dict(width=1024),
            dict(complete=False),
            dict(owned_runtime={}),
            dict(passed=False),
        ):
            with self.subTest(changes=changes):
                self.work = dict(complete=True, passed=True, engine="mps", width=512, pid=123)
                self.change_worker(**changes)
                with self.assertRaises(ValueError):
                    self.run_audit()

    def test_rejects_changed_numerical_receipt(self):
        self.worker.write_text("{}")
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_rejects_changed_capture_inputs(self):
        self.runtime.write_text('{"changed":true}')
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_rejects_source_outside_captured_project(self):
        self.value["producer"] = self.value["worker_source"]
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_origin_failure_is_not_reported_as_a_valid_audit(self):
        self.origins.side_effect = ValueError("foreign native image")
        with self.assertRaisesRegex(ValueError, "foreign native image"):
            self.run_audit()

    def test_known_file_authenticates_content_and_rejects_outside_authority(self):
        source = self.root / "module.py"
        source.write_text("original")
        runtime = dict(files={"module.py": AUDIT.SUITE.sha256(source)})
        project = dict(root=str(self.root / "project"), files={})
        AUDIT.known_file(source, runtime, self.root, project)
        source.write_text("changed")
        with self.assertRaises(ValueError):
            AUDIT.known_file(source, runtime, self.root, project)
        with self.assertRaises(ValueError):
            AUDIT.known_file(source, dict(files={}), self.root, project)


if __name__ == "__main__":
    unittest.main()
