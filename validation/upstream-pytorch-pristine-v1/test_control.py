import ast
import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("control", HERE / "control.py")
c = importlib.util.module_from_spec(spec)
sys.modules["control"] = c
spec.loader.exec_module(c)
spec = importlib.util.spec_from_file_location("pristine_hosted", HERE / "hosted_build.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def functions(path):
    result = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef):
            result[node.name] = ast.dump(node, include_attributes=False)
    return result


class Tests(unittest.TestCase):
    def setUp(self):
        self.contract = c.load_control_contract()

    def test_exact_environment_and_no_patch(self):
        self.assertEqual(h.BUILD_ENV, self.contract["build_environment"])
        self.assertEqual(h.SOURCES, self.contract["pristine_source_sha256"])
        self.assertIsNone(h.PATCH_SHA256)
        text = (HERE / "hosted_build.py").read_text()
        self.assertNotIn('"git", "apply"', text)
        self.assertNotIn("--patch", text)

    def test_native_build_transport_methods_preserved(self):
        old = functions(HERE.parent / "upstream-pytorch-format-v2/hosted_build.py")
        new = functions(HERE / "hosted_build.py")
        changed = {name for name in old if old[name] != new[name]}
        self.assertEqual(changed, {"__init__", "source_guard", "run", "main"})
        self.assertEqual(
            (HERE / "verify_runtime.py").read_bytes(),
            (HERE.parent / "upstream-pytorch-format-v2/verify_runtime.py").read_bytes(),
        )

    def test_tools_exact_and_mount_location_only_normalized(self):
        for label, row in self.contract["tool_checks"].items():
            c.check_tool(label, row["normalized_stdout"], self.contract)
            with self.assertRaises(RuntimeError):
                c.check_tool(label, row["normalized_stdout"] + "different\n", self.contract)
        value = "Apple metal version1\nTarget: air64\nInstalledDir: /one\n"
        self.assertEqual(c.normalize_tool(value), c.normalize_tool(value.replace("/one", "/two")))
        with self.assertRaises(RuntimeError):
            c.check_tool("unknown", "anything", self.contract)

    def test_host_drift_rejected(self):
        host = self.contract["host"]
        record = {"build_environment": h.BUILD_ENV}
        with (
            patch.object(c.sys, "version", host["python"]),
            patch.object(c.platform, "platform", return_value=host["platform"]),
            patch.dict(c.os.environ, host["image"]),
        ):
            c.check_host(self.contract, record)
            for key in host["image"]:
                with patch.dict(c.os.environ, {key: "changed"}), self.assertRaises(RuntimeError):
                    c.check_host(self.contract, record)
            altered = copy.deepcopy(record)
            altered["build_environment"]["MAX_JOBS"] = "9"
            with self.assertRaises(RuntimeError):
                c.check_host(self.contract, altered)

    def test_complete_pinned_wheel_and_dependency_inventory(self):
        self.assertEqual(len(self.contract["wheel_inventory"]), 45)
        self.assertEqual(
            (HERE / "requirements-full.txt").read_text(), self.contract["dependency_freeze"]
        )
        for name, row in self.contract["wheel_inventory"].items():
            self.assertTrue(name.endswith(".whl"))
            self.assertEqual(len(row["sha256"]), 64)
            self.assertGreater(row["bytes"], 0)
        self.assertIn(
            '"Downloaded wheels differ from formatted control"',
            (HERE / "hosted_build.py").read_text(),
        )
        self.assertIn(
            '"Installed dependency freeze differs"', (HERE / "hosted_build.py").read_text()
        )

    def test_submodule_commits_are_exact_but_descriptive_refs_are_not_identity(self):
        text = self.contract["submodule_identities"]
        rows = c.submodule_identities(text)
        self.assertEqual(len(rows), 61)
        lines = [line.split(" (", 1)[0] + " (another-described-ref)" for line in text.splitlines()]
        self.assertEqual(c.submodule_identities("\n".join(lines)), rows)
        with self.assertRaises(RuntimeError):
            c.submodule_identities(text.replace(" ", "+", 1))
        altered = text.replace(rows[0][1], "0" * 40, 1)
        self.assertNotEqual(c.submodule_identities(altered), rows)

    def test_canonical_ast_preserves_empty_fields_and_special_constants(self):
        node = ast.parse("def f(): return x[..., 0.5], b'x', 2j").body[0]
        value = c.canonical_ast(node)
        self.assertEqual(set(value[1]), set(node._fields))
        self.assertEqual(value[1]["args"][1]["args"], [])
        encoded = json.dumps(value, sort_keys=True, allow_nan=False)
        self.assertIn('"constant_type": "ellipsis"', encoded)
        self.assertIn('"constant_type": "float"', encoded)
        self.assertIn('"constant_type": "bytes"', encoded)
        self.assertIn('"constant_type": "complex"', encoded)
        altered = ast.parse("def f(): return x[..., 0.6], b'x', 2j").body[0]
        self.assertNotEqual(value, c.canonical_ast(altered))

    def test_unchanged_method_hash_inventory(self):
        expected = self.contract["unchanged_test_method_ast_sha256"]
        self.assertEqual(len(expected), 9)
        with self.assertRaises(RuntimeError):
            c.method_hashes("class TestMPS: pass")

    def test_openmp_bottle_version_and_options_are_bound(self):
        base = copy.deepcopy(self.contract["libomp"])
        c.check_tool("libomp-package", json.dumps({"formulae": [base]}), self.contract)
        for key in ["revision", "name"]:
            changed = copy.deepcopy(base)
            changed[key] = "different"
            with self.assertRaises(RuntimeError):
                c.check_tool("libomp-package", json.dumps({"formulae": [changed]}), self.contract)
        changed = copy.deepcopy(base)
        changed["installed"][0]["built_as_bottle"] = False
        with self.assertRaises(RuntimeError):
            c.check_tool("libomp-package", json.dumps({"formulae": [changed]}), self.contract)

    def test_workflow_is_additive_serial_and_retains_failure_evidence(self):
        text = (HERE.parents[1] / ".github/workflows/upstream-pytorch-pristine-v1.yml").read_text()
        self.assertIn("group: pytorch-upstream-integration", text)
        self.assertIn("ref: " + h.BASE, text)
        self.assertIn("if: always()", text)
        self.assertIn("pytorch-pristine-test-source.tar.gz", text)
        self.assertNotIn("--patch", text)
        self.assertIn("python-version: '3.12.10'", text)


if __name__ == "__main__":
    unittest.main()
