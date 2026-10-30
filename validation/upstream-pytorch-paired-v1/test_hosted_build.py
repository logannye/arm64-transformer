import ast
import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("control", HERE / "control.py")
c = importlib.util.module_from_spec(spec)
sys.modules["control"] = c
spec.loader.exec_module(c)
spec = importlib.util.spec_from_file_location("paired_hosted", HERE / "hosted_build.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def methods(path):
    return {
        n.name: ast.dump(n, include_attributes=False)
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef)
    }


def stages(contract):
    result = []
    for variant in ["pristine", "formatted"]:
        result.append(
            {
                "source_variant": variant,
                "native_import_portability_passed": True,
                "toolchain_before": {"tool": "same"},
                "toolchain_after": {"tool": "same"},
                "cmake_cache": contract["cmake_cache"].copy(),
                "embedding_metal_object": {"sha256": variant},
                "embedding_translation_units": [
                    {
                        "compile_command": {"command": "same"},
                        "object_file": "same-tree/object",
                        "object_sha256": variant,
                    }
                ],
                "wheel_inventory": contract["wheel_inventory"],
            }
        )
    return result


class Tests(unittest.TestCase):
    def setUp(self):
        self.contract = c.load_control_contract()

    def test_core_build_command_and_transport_retention_unchanged(self):
        original = methods(HERE.parent / "upstream-pytorch-format-v2/hosted_build.py")
        current = methods(HERE / "hosted_build.py")
        for name in ["command", "retain_runtime", "inventory", "read_cache", "sha256"]:
            self.assertEqual(current[name], original[name], name)
        code = (HERE / "hosted_build.py").read_text()
        self.assertEqual(code.count('"--no-build-isolation"'), 1)
        self.assertIn('"formatted_incremental"', code)
        self.assertNotIn('shutil.rmtree(self.source / "build")', code)
        self.assertEqual(
            (HERE / "verify_runtime.py").read_bytes(),
            (HERE.parent / "upstream-pytorch-format-v2/verify_runtime.py").read_bytes(),
        )

    def test_pair_accepts_only_matched_complete_rebuilt_variants(self):
        self.assertTrue(h.validate_pair_records(stages(self.contract), self.contract))
        mutations = [
            lambda r: r.reverse(),
            lambda r: r[0].update(native_import_portability_passed=False),
            lambda r: r[1]["toolchain_after"].update(tool="other"),
            lambda r: r[1]["cmake_cache"].update(CMAKE_CXX_FLAGS="-ffast-math"),
            lambda r: r[1]["embedding_metal_object"].update(sha256="pristine"),
            lambda r: r[1]["embedding_translation_units"][0].update(object_sha256="pristine"),
            lambda r: r[1]["embedding_translation_units"][0].update(object_file="another-tree"),
            lambda r: r[1].update(wheel_inventory={}),
        ]
        for change in mutations:
            rows = copy.deepcopy(stages(self.contract))
            change(rows)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                h.validate_pair_records(rows, self.contract)

    def test_shared_missing_or_null_cmake_setting_cannot_compare_equal(self):
        for mode in ["missing", "null"]:
            rows = stages(self.contract)
            for row in rows:
                if mode == "missing":
                    row["cmake_cache"].pop("CMAKE_CXX_FLAGS")
                else:
                    row["cmake_cache"]["CMAKE_CXX_FLAGS"] = None
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                h.validate_pair_records(rows, self.contract)

    def test_history_drift_is_explicit_and_not_a_within_pair_substitute(self):
        old = self.contract["host"]
        now = {
            "python": old["python"],
            "platform": old["platform"],
            "image": old["image"].copy(),
            "tools": {k: v["normalized_stdout"] for k, v in self.contract["tool_checks"].items()},
        }
        now["tools"]["libomp-package"] = self.contract["libomp"]
        self.assertEqual(h.historical_drift(now, self.contract), {})
        now["image"]["ImageVersion"] = "new-image"
        self.assertEqual(set(h.historical_drift(now, self.contract)), {"image/ImageVersion"})
        pair = stages(self.contract)
        pair[1]["toolchain_before"] = now
        with self.assertRaises(RuntimeError):
            h.validate_pair_records(pair, self.contract)

    def test_shared_environment_and_source_guards(self):
        code = (HERE / "hosted_build.py").read_text()
        self.assertEqual(code.count('"create-environment"'), 1)
        self.assertEqual(code.count('"patch-apply"'), 1)
        self.assertIn("stage.env = setup.env.copy()", code)
        self.assertIn('"dependencies-before-build"', code)
        self.assertIn('"dependencies-after-build"', code)
        self.assertIn('"--exclude", "torch"', code)
        self.assertIn('"Retained variant archive changed"', code)
        self.assertIn("require(sys.dont_write_bytecode", code)
        self.assertEqual(h.BUILD_ENV, self.contract["build_environment"])

    def test_hosted_main_rejects_before_filesystem_mutation(self):
        with (
            patch.dict(h.os.environ, {}, clear=True),
            patch.object(h.sys, "argv", ["x", "--source", ".", "--output", "unused"]),
        ):
            with patch.object(Path, "mkdir") as mkdir, self.assertRaises(RuntimeError):
                h.main()
            mkdir.assert_not_called()

    def test_workflow_serial_and_failure_archives_complete(self):
        text = (HERE.parents[1] / ".github/workflows/upstream-pytorch-paired-v1.yml").read_text()
        self.assertIn("group: pytorch-upstream-integration", text)
        self.assertIn("python -B study/", text)
        self.assertIn("if: always()", text)
        for variant in ["pristine", "formatted"]:
            self.assertIn("integration-paired-v1-result/" + variant + "/*.tar.gz", text)
            self.assertIn("integration-paired-v1-result/" + variant + "/*.json", text)
        self.assertIn("integration-paired-v1-result/wheels/", text)


if __name__ == "__main__":
    unittest.main()
