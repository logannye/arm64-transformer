"""CPU-only negative controls for the frozen half-index diagnostic launcher."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("half_launcher", HERE / "run_half_diagnostic.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class FrozenDiagnostic(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.patch_here = patch.object(launcher, "HERE", self.root)
        self.patch_here.start()
        (self.root / "run_half_diagnostic.py").write_text("# fixture driver\n")
        (self.root / "runtime_probe.py").write_text("# fixture runtime\n")
        (self.root / "half_index_diagnostic.py").write_text(
            "REPEATS=20\nOPERATIONS=('combinations',)\nINDICES=((0,0,0,0),)\n"
            "SEED=20261103\ndef case_inventory():\n return [{'dtype':'float16'}]\n"
        )
        self.runtime_pin = patch.object(
            launcher, "RUNTIME_SHA", launcher.digest(self.root / "runtime_probe.py")
        )
        self.runtime_pin.start()
        self.plan = launcher.planned()
        self.path = self.root / "plan.json"
        self.path.write_text(json.dumps(self.plan))
        self.plan_sha = launcher.digest(self.path)

    def tearDown(self):
        self.runtime_pin.stop()
        self.patch_here.stop()
        self.temp.cleanup()

    def test_two_worker_inventory_and_diagnostic_scope(self):
        checked = launcher.verify_plan(self.path, self.plan_sha)
        self.assertEqual(checked["workers"], [0, 1])
        self.assertEqual(checked["processes"], 2)
        self.assertFalse(checked["deterministic_mode_for_diagnosis"])
        self.assertTrue(checked["deterministic_mode_restore_target"])
        self.assertFalse(checked["formal_eligible"])
        self.assertFalse(checked["admission_stream_opened"])

    def test_reordered_or_missing_workers_and_altered_scope_rejected(self):
        for edit in (
            lambda x: x.update(workers=[1, 0]),
            lambda x: x.update(workers=[0]),
            lambda x: x.update(formal_eligible=True),
            lambda x: x.update(deterministic_mode_restore_target=False),
            lambda x: x.update(development_seed=20261108),
        ):
            value = copy.deepcopy(self.plan)
            edit(value)
            self.path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "contract or sources changed"):
                launcher.verify_plan(self.path, launcher.digest(self.path))

    def test_missing_external_hash_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "External plan"):
            launcher.verify_plan(self.path, None)

    def test_stale_plan_hash_rejected(self):
        self.path.write_text(self.path.read_text() + " ")
        with self.assertRaisesRegex(RuntimeError, "plan changed"):
            launcher.verify_plan(self.path, self.plan_sha)

    def test_changed_harness_rejected(self):
        path = self.root / "half_index_diagnostic.py"
        path.write_text(path.read_text() + "# changed\n")
        with self.assertRaisesRegex(RuntimeError, "contract or sources changed"):
            launcher.verify_plan(self.path, self.plan_sha)

    def test_changed_runtime_cannot_be_frozen_as_another_runtime(self):
        (self.root / "runtime_probe.py").write_text("# changed runtime\n")
        with self.assertRaisesRegex(RuntimeError, "Pinned runtime"):
            launcher.planned()

    def test_symlinked_source_rejected(self):
        path = self.root / "half_index_diagnostic.py"
        target = self.root / "same.py"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
        with self.assertRaisesRegex(RuntimeError, "Symlinked"):
            launcher.planned()

    def test_duplicate_json_and_nonfinite_values_rejected(self):
        for text in ('{"passed":true,"passed":false}', '{"x":NaN}'):
            self.path.write_text(text)
            with self.assertRaises((ValueError, RuntimeError)):
                launcher.read_json(self.path)

    def test_raw_mutation_missing_case_and_path_escape_rejected(self):
        cases = self.root / "cases"
        cases.mkdir()
        (cases / "inventory.json").write_text(
            json.dumps(
                {
                    "cases": self.plan["cases"],
                    "operations": self.plan["operations"],
                    "replays": self.plan["replays"],
                }
            )
        )
        raw = cases / "case-00.npz"
        raw.write_bytes(b"array evidence")
        record = {
            "cases": [
                {
                    "case": self.plan["cases"][0],
                    "raw": {
                        "path": raw.name,
                        "bytes": raw.stat().st_size,
                        "sha256": launcher.digest(raw),
                    },
                }
            ]
        }
        self.assertEqual(len(launcher.verify_case_artifacts(self.root, record, self.plan)), 1)
        for changed in ({"cases": []}, copy.deepcopy(record)):
            if changed["cases"]:
                changed["cases"][0]["raw"]["path"] = "../case-00.npz"
            with self.assertRaises(RuntimeError):
                launcher.verify_case_artifacts(self.root, changed, self.plan)
        raw.write_bytes(b"altered arrays")
        with self.assertRaisesRegex(RuntimeError, "arrays changed"):
            launcher.verify_case_artifacts(self.root, record, self.plan)


class NamespaceControls(unittest.TestCase):
    def test_parent_and_worker_reject_output_outside_development_namespace(self):
        helper = launcher.load("runtime_probe")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "old-study-evidence"
            args = SimpleNamespace(workspace=root, output=output)
            for action in (launcher.run, launcher.worker):
                with self.assertRaisesRegex(RuntimeError, "new development namespace"):
                    action(args, helper)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
