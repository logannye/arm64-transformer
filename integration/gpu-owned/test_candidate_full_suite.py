"""Pure stdlib full-suite orchestration controls; no framework or compiler runs."""

import copy
import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "candidate_full_suite", HERE / "candidate_full_suite.py"
)
full = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(full)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f"module.Class.test_{index}" for index in range(10353)]
        self.inventory = {
            "test_ids": self.ids,
            "skip_ids": self.ids[:744],
            "expected_failure_ids": self.ids[744:1126],
        }
        self.tests = {
            "run": 10353,
            "executed_ids": list(self.ids),
            "failures": [],
            "errors": [],
            "skipped": [
                {"id": identity, "reason": "unchanged decorator"}
                for identity in self.inventory["skip_ids"]
            ],
            "expected_failures": [
                {"id": identity, "traceback": "retained xfail"}
                for identity in self.inventory["expected_failure_ids"]
            ],
            "unexpected_successes": [],
            "successful": True,
        }

    def test_full_10353_test_inventory_passes_with_retained_eligibility(self):
        self.assertTrue(full.validate_tests(self.tests, self.inventory))

    def test_complete_failed_suite_remains_valid_failed_evidence(self):
        self.tests["failures"] = [{"id": self.ids[-1], "traceback": "actual upstream failure"}]
        self.tests["successful"] = False
        self.assertFalse(full.validate_tests(self.tests, self.inventory))

    def test_missing_duplicate_reordered_tests_rejected(self):
        for operation in (
            lambda t: t["executed_ids"].pop(),
            lambda t: t["executed_ids"].reverse(),
            lambda t: t["executed_ids"].append(t["executed_ids"][0]),
            lambda t: t.update(run=10352),
        ):
            changed = copy.deepcopy(self.tests)
            operation(changed)
            with self.assertRaises(ValueError):
                full.validate_tests(changed, self.inventory)

    def test_new_skip_or_xfail_is_not_a_waiver(self):
        for kind in ("skipped", "expected_failures"):
            for remove in (True, False):
                changed = copy.deepcopy(self.tests)
                if remove:
                    changed[kind].pop()
                else:
                    changed[kind].append(
                        {
                            "id": self.ids[-1],
                            "reason": "new skip",
                            "traceback": "new expected failure",
                        }
                    )
                with self.subTest(kind=kind, remove=remove), self.assertRaises(ValueError):
                    full.validate_tests(changed, self.inventory)

    def test_removed_or_wrong_failure_trace_rejected(self):
        for failure in [
            {"id": self.ids[-1], "traceback": ""},
            {"id": "foreign_id", "traceback": "unrelated failure"},
        ]:
            changed = copy.deepcopy(self.tests)
            changed["errors"] = [failure]
            changed["successful"] = False
            with self.assertRaises(ValueError):
                full.validate_tests(changed, self.inventory)

    def test_failed_suite_cannot_relabel_itself_successful(self):
        self.tests["failures"] = [{"id": self.ids[-1], "traceback": "retained failure"}]
        with self.assertRaises(ValueError):
            full.validate_tests(self.tests, self.inventory)

    def test_missing_skip_reason_rejected(self):
        self.tests["skipped"][0]["reason"] = ""
        with self.assertRaises(ValueError):
            full.validate_tests(self.tests, self.inventory)

    def test_unknown_unexpected_success_rejected(self):
        self.tests["unexpected_successes"] = ["foreign_test"]
        self.tests["successful"] = False
        with self.assertRaises(ValueError):
            full.validate_tests(self.tests, self.inventory)


class FocusedPrerequisiteTests(unittest.TestCase):
    def setUp(self):
        self.summary = {
            "schema": "focused",
            "protocol": full.PROTOCOL,
            "plan": {"sha256": "a" * 64},
            "complete": True,
            "evidence_valid": True,
            "passed": True,
            "formal_eligible": False,
            "formal_fits": 0,
            "admission_stream_opened": False,
            "original_full_suite_failure_unchanged": True,
            "full_suite_required": True,
            "full_suite_executed": False,
            "workers": [{"status": "passed", "exit_code": 0}],
        }
        self.candidate = SimpleNamespace(
            SCHEMA="focused",
            pinned=mock.Mock(return_value={}),
            json_read=mock.Mock(return_value=self.summary),
            check_binding=mock.Mock(),
            audit_completed_workers=mock.Mock(),
        )

    def check(self):
        return full.authenticate_focused(
            self.candidate, {}, "a" * 64, Path("summary.json"), "b" * 64
        )

    def test_success_flag_requires_all_raw_worker_records(self):
        self.candidate.audit_completed_workers.side_effect = ValueError("prior raw failure changed")
        with self.assertRaisesRegex(ValueError, "raw failure"):
            self.check()
        self.candidate.audit_completed_workers.assert_called_once()

    def test_focused_failure_cannot_be_hidden_by_aggregate_boolean(self):
        self.summary["workers"][0].update(status="failed", exit_code=1)
        with self.assertRaisesRegex(ValueError, "cannot be waived"):
            self.check()

    def test_wrong_candidate_and_incomplete_or_formal_receipt_rejected(self):
        for key, value in [
            ("complete", False),
            ("evidence_valid", False),
            ("passed", False),
            ("formal_eligible", True),
            ("admission_stream_opened", True),
            ("full_suite_required", False),
            ("plan", {"sha256": "0" * 64}),
        ]:
            original = self.summary[key]
            self.summary[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check()
            self.summary[key] = original


class ExecutionControls(unittest.TestCase):
    def setUp(self):
        self.candidate = full.candidate_module()
        policy = HERE / "candidate_fixture_policy.py.txt"
        self.candidate.pinned(policy, full.POLICY_SHA)
        loader = importlib.machinery.SourceFileLoader("full_test_stock_fixture_policy", str(policy))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        self.policy = importlib.util.module_from_spec(spec)
        loader.exec_module(self.policy)
        self.plan = {
            "workspace": "/workspace",
            "source": "/source",
            "runtime": "/runtime",
            "runtime_manifest": {"sha256": "a" * 64},
        }
        self.cache = Path("/workspace/work/gpu-v4-owned/full/pytorch-test-extensions")

    def test_environment_is_repeatable_and_excludes_parent_secrets(self):
        parent = {"PATH": "/bin", "HOME": "/home/test", "SECRET_TOKEN": "must-not-be-recorded"}
        with mock.patch.dict(os.environ, parent, clear=True):
            first = full.environment(self.candidate, self.plan, self.policy, self.cache)
        with mock.patch.dict(os.environ, first, clear=True):
            second = full.environment(self.candidate, self.plan, self.policy, self.cache)
        self.assertEqual(first, second)
        self.assertNotIn("SECRET_TOKEN", first)
        self.assertEqual(first["MAX_JOBS"], "2")
        self.assertEqual(first["PYTORCH_ENABLE_MPS_FALLBACK"], "0")
        self.assertEqual(first["TORCH_EXTENSIONS_DIR"], str(self.cache))

    def test_external_compiler_flags_cannot_silently_enable_debug_builds(self):
        with mock.patch.dict(
            os.environ, {"PATH": "/bin", "CXXFLAGS": "-g", "CFLAGS": "-O0"}, clear=True
        ):
            env = full.environment(self.candidate, self.plan, self.policy, self.cache)
        self.assertNotIn("CXXFLAGS", env)
        self.assertNotIn("CFLAGS", env)
        self.assertEqual(env["DEBUG"], "0")
        self.assertEqual(env["REL_WITH_DEB_INFO"], "0")
        self.assertEqual(env["CXX"], "/usr/bin/clang++")

    def test_runtime_specific_fixture_identity_and_limits_required(self):
        fake = SimpleNamespace(
            require_environment=mock.Mock(),
            prepare=mock.Mock(
                return_value={
                    "runtime_manifest_sha256": "a" * 64,
                    "ninja": {
                        "path": str(
                            Path(self.plan["workspace"])
                            / self.candidate.OLD
                            / "runtime-env/bin/ninja"
                        ),
                        "sha256": full.NINJA_SHA,
                    },
                }
            ),
            MAX_CACHE_BYTES=1024**3,
            MAX_CACHE_FILES=64,
            MIN_FREE_BYTES=5 * 1024**3,
            CONTROLS={"MAX_JOBS": "2", "CMAKE_BUILD_PARALLEL_LEVEL": "2"},
        )
        before = dict(os.environ)
        with mock.patch.object(full.subprocess, "Popen") as popen:
            full.fixture_prepare(self.candidate, self.plan, fake, self.cache, {"PATH": "/bin"})
            popen.assert_not_called()
        self.assertEqual(dict(os.environ), before)
        proof = fake.prepare.return_value
        fake.prepare.return_value = {**proof, "runtime_manifest_sha256": "wrong runtime"}
        with self.assertRaisesRegex(ValueError, "another runtime"):
            full.fixture_prepare(self.candidate, self.plan, fake, self.cache, {})
        self.assertEqual(dict(os.environ), before)
        fake.prepare.return_value = proof
        fake.MAX_CACHE_FILES = 65
        with self.assertRaisesRegex(ValueError, "bounds changed"):
            full.fixture_prepare(self.candidate, self.plan, fake, self.cache, {})

    def test_resource_boundaries_reuse_unchanged_stock_policy(self):
        good = {"bytes": 1024**3, "file_count": 64, "free_bytes": 5 * 1024**3}
        self.policy.require_bounds(good)
        for key, value in [
            ("bytes", good["bytes"] + 1),
            ("file_count", 65),
            ("free_bytes", good["free_bytes"] - 1),
        ]:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.policy.require_bounds({**good, key: value})

    def test_cache_comparison_does_not_treat_disk_telemetry_as_content(self):
        original = {"files": {"fixture.so": {"sha256": "a" * 64}}, "free_bytes": 100}
        self.assertEqual(
            full.cache_content(original), full.cache_content({**original, "free_bytes": 99})
        )
        changed = copy.deepcopy(original)
        changed["files"]["fixture.so"]["sha256"] = "b" * 64
        self.assertNotEqual(full.cache_content(original), full.cache_content(changed))

    def test_missing_or_stale_plan_rejected_before_subprocess(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "plan.json"
            path.write_text('{"formal_eligible":true}')
            with mock.patch.object(full.subprocess, "Popen") as popen:
                with self.assertRaises(ValueError):
                    full.run(path, "0" * 64, path.parent / "output")
                popen.assert_not_called()

    def test_frameworks_remain_unimported(self):
        for name in ("torch", "numpy", "mlx"):
            self.assertNotIn(name, sys.modules)


if __name__ == "__main__":
    unittest.main()
