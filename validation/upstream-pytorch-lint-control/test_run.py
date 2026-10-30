"""CPU-only synthetic orchestration controls; never executes upstream lint."""

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lint_control", HERE / "run.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class Tests(unittest.TestCase):
    def test_contract_preserves_variants_and_original_evidence(self):
        value = json.loads((HERE / "contract.json").read_text())
        self.assertEqual(value["variants"], ["base", "original", "formatted"])
        self.assertEqual(len(value["wheel_inventory"]), 45)
        for item in value["patches"].values():
            self.assertEqual(r.sha(HERE / item["file"]), item["sha256"])
        self.assertEqual(r.sha(HERE / "requirements-full.txt"), value["requirements_sha256"])
        self.assertEqual(
            value["patches"]["original"]["sha256"],
            "3dd6bf6567284d798074e7d6102ff7f8b2a051a3a4ff2824f86184c120eb0ce9",
        )
        self.assertEqual(
            value["patches"]["formatted"]["sha256"],
            "3834dca844ea3de6b40467ce214b99bce83fa06dae2bc454623516dafc0fdd71",
        )
        self.assertEqual(set(value["sources"]["base"]), set(value["sources"]["original"]))
        self.assertEqual(set(value["sources"]["base"]), set(value["sources"]["formatted"]))

    def test_finding_multisets_keep_duplicates_and_errors(self):
        block = """>>> Lint for torch/headeronly/util/complex.h:

  Error (CLANGTIDY) [narrowing]
    narrowing conversion

    >>> 555  |  value.real()

"""
        rows = r.parse_findings(block + block)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["line"], 555)
        result = r.compare_findings(rows, [rows[0]])
        self.assertEqual(result["shared"][0]["count"], 1)
        self.assertEqual(result["base_only"][0]["count"], 1)
        self.assertEqual(result["candidate_only"], [])
        self.assertEqual(result["shared"][0]["finding"]["severity"], "Error")

    def test_format_warning_remains_candidate_only(self):
        value = r.parse_findings(""">>> Lint for aten/src/Embedding.metal:

  Warning (CLANGFORMAT) format
    replacement diff

$ uvx --python 3.10 lintrunner@0.12.7 next
""")
        result = r.compare_findings([], value)
        self.assertEqual(result["candidate_only"][0]["finding"]["code"], "CLANGFORMAT")
        self.assertNotIn("$ uvx", value[0]["block"])
        self.assertIsNone(value[0]["line"])

    def test_missing_file_identity_fails_closed(self):
        with self.assertRaises(ValueError):
            r.parse_findings("  Error (CLANGTIDY) no source\n")

    def test_unchanged_bare_spin_command_and_failed_return_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp).resolve()
            (source / "sample.h").write_text("original header")
            run = r.Run(source, source, {"schema": "test", "historical_run": {}})
            run.environment = source / ".venv"
            run.transition = lambda variant: {"head": "a", "diff_names": []}
            run.source_identity = lambda label: {"head": "a", "diff_names": []}
            text = (
                "$ uvx --python 3.10 lintrunner@0.12.7 --take FAST --all-files\n"
                ">>> Lint for sample.h:\n\n  Error (CLANGTIDY) narrowing\n    problem\n"
                "$ uvx --python 3.10 lintrunner@0.12.7 --take SLOW\nLint failed!\n"
            )
            calls = []

            def command(label, argv, **kwargs):
                calls.append((label, argv, kwargs))
                return {"exit_code": 1}, text

            run.command = command
            result = run.lint("compare", "base")
            self.assertEqual(calls[0][1], [str(source / ".venv/bin/spin"), "lint"])
            self.assertFalse(result["lint_passed"])
            self.assertTrue(result["eligible_for_comparison"])
            self.assertFalse(calls[0][2]["check"])

    def test_setup_failure_never_qualifies_as_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = r.Run(root, root, {"schema": "test", "historical_run": {}})
            run.transition = lambda variant: {}
            run.source_identity = lambda label: {}
            run.command = lambda *args, **kwargs: (
                {"exit_code": 1},
                "Failed to provision lint tools\n",
            )
            self.assertFalse(run.lint("compare", "base")["eligible_for_comparison"])

    def test_inventory_detects_dependency_payload_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "module.py").write_text("original")
            first = r.inventory(root)
            (root / "module.py").write_text("changed!")
            self.assertNotEqual(first, r.inventory(root))
            (root / "__pycache__").mkdir()
            (root / "__pycache__/temporary.pyc").write_bytes(b"cache")
            self.assertEqual(set(r.inventory(root)), {"module.py"})

    def test_zero_findings_and_mixed_runtime_error_fail_closed(self):
        text = (
            "$ uvx --python 3.10 lintrunner@0.12.7 --take FAST --all-files\n"
            "$ uvx --python 3.10 lintrunner@0.12.7 --take SLOW\nLint failed!\n"
        )
        self.assertFalse(r.lint_execution_eligible(text, 1, [])["eligible"])
        self.assertTrue(r.lint_execution_eligible(text, 1, [{"code": "CLANGTIDY"}])["eligible"])
        for failure in (
            "Traceback (most recent call last):",
            "RuntimeError: failure",
            "command-failed",
        ):
            self.assertFalse(
                r.lint_execution_eligible(text + failure, 1, [{"code": "CLANGTIDY"}])["eligible"]
            )

    def test_own_repository_guard_and_ambient_override_removal(self):
        for env in ({}, {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "another/repo"}):
            with patch.dict(r.os.environ, env, clear=True), self.assertRaises(ValueError):
                r.hosted_guard()
        with patch.dict(
            r.os.environ,
            {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "logannye/arm64-transformer"},
            clear=True,
        ):
            r.hosted_guard()
        with tempfile.TemporaryDirectory() as tmp:
            overrides = {
                "UV_INDEX_URL": "secret",
                "PIP_EXTRA_INDEX_URL": "secret",
                "PYTHONPATH": "override",
                "CC": "custom",
                "LINTRUNNER_CONFIG": "custom",
            }
            with patch.dict(r.os.environ, overrides):
                run = r.Run(Path(tmp), Path(tmp), {"schema": "test", "historical_run": {}})
            self.assertTrue(all(key not in run.env for key in overrides))
            self.assertNotIn("secret", json.dumps(run.record))

    def test_no_framework_import_or_build_entry(self):
        tree = ast.parse((HERE / "run.py").read_text())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(item.name.split(".")[0] for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        self.assertFalse(imports & {"numpy", "torch", "mlx"})
        text = (HERE / "run.py").read_text()
        self.assertNotIn('"spin"), "develop"', text)
        self.assertNotIn('"--skip"', text)
        self.assertNotIn('"--take"', text)
        self.assertNotIn('"--apply-patches"', text)
        self.assertNotIn("cmake --build", text)


if __name__ == "__main__":
    unittest.main()
