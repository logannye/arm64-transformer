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
        previous = HERE.parent / "upstream-pytorch-lint-control-v2"
        predecessor = json.loads((previous / "contract.json").read_text())
        self.assertEqual(value["schema"], "pytorch-pristine-lint-control-v3")
        self.assertEqual(value["previous_failed_controls"][-1]["run_id"], 40993204538)
        for filename, key in (("run.py", "helper_sha256"), ("contract.json", "contract_sha256")):
            self.assertEqual(r.sha(previous / filename), value["previous_failed_controls"][-1][key])
        for key in (
            "sources",
            "patches",
            "wheel_inventory",
            "requirements_sha256",
            "lint_binaries",
        ):
            self.assertEqual(value[key], predecessor[key])
        self.assertEqual(
            set(value["lint_configuration"]) - set(predecessor["lint_configuration"]),
            {
                "tools/linter/clang_tidy/generate_build_files.py",
                "tools/setup_helpers/generate_code.py",
                ".github/scripts/lintrunner.sh",
                "cmake/EnvVarForwarding.cmake",
            },
        )
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
            self.assertEqual(result["classification"], "source_diagnostics")
            self.assertTrue(result["diagnostics_parsed"])

    def test_setup_failure_never_qualifies_as_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            run = r.Run(root, root, {"schema": "test", "historical_run": {}})
            run.transition = lambda variant: {}
            run.source_identity = lambda label: {}
            run.command = lambda *args, **kwargs: (
                {"exit_code": 1},
                "Failed to provision lint tools\n",
            )
            self.assertFalse(run.lint("compare", "base")["eligible_for_comparison"])

    def generated_fixture(self, tmp, mutation=None):
        root = Path(tmp).resolve()
        source, output = root / "source", root / "output"
        source.mkdir()
        output.mkdir()
        contract = json.loads((HERE / "contract.json").read_text())
        run = r.Run(source, output, contract)
        run.env["CMAKE_PREFIX_PATH"] = str(root / "libomp")
        run.source_identity = lambda label: {
            "head": "base",
            "generated_inputs": run.generated_identity(),
        }
        calls = []

        def command(label, argv, **kwargs):
            calls.append((label, argv, kwargs))
            self.assertEqual(argv, [str(source / ".venv/bin/spin"), "regenerate-clangtidy-files"])
            for name in contract["configure_prerequisite"]["generated_roots"]:
                folder = source / name
                folder.mkdir(parents=True)
                (folder / "generated.h").write_text("synthetic generated header")
            (source / "build/CMakeCache.txt").write_text(
                "CMAKE_BUILD_TYPE:STRING=Release\nCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON\n"
                "USE_MPS:BOOL=ON\nUSE_CUDA:BOOL=OFF\nUSE_DISTRIBUTED:BOOL=OFF\n"
                "USE_OPENMP:BOOL=ON\nUSE_NCCL:BOOL=OFF\nUSE_PRECOMPILED_HEADERS:BOOL=ON\n"
            )
            (source / "build/env_forwarded.txt").write_text(
                "".join(
                    line.split(":", 1)[0] + "=" + line.split("=", 1)[1] + "\n"
                    for line in (source / "build/CMakeCache.txt").read_text().splitlines()
                )
            )
            (source / "build/compile_commands.json").write_text(
                json.dumps(
                    [{"file": "test.cpp", "directory": str(source), "command": "clang++ test.cpp"}]
                )
            )
            if mutation:
                mutation(source)
            return {"exit_code": 0}, "official command synthetic output"

        run.command = command
        return run, calls

    def test_official_configure_command_and_all_generated_roots_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, calls = self.generated_fixture(tmp)
            run.prepare_generated_inputs()
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], "regenerate-clangtidy-files")
            expected = set(run.contract["configure_prerequisite"]["generated_roots"])
            self.assertEqual(set(run.generated_inputs), expected)
            self.assertEqual(
                run.generated_identity()["inventory_sha256"],
                r.sha(run.output / "generated-inputs.json"),
            )
            self.assertFalse(run.record["configure_prerequisite"]["full_framework_build_executed"])
            self.assertTrue(
                run.record["configure_prerequisite"]["compiler_configuration_probes_possible"]
            )
            self.assertEqual(run.record["configure_prerequisite"]["compile_commands_count"], 1)
            self.assertEqual(
                run.record["configure_prerequisite"]["effective_forwarded"]["USE_MPS"], "ON"
            )
            self.assertEqual(
                run.record["configure_prerequisite"]["effective_forwarded_sha256"],
                r.sha(run.output / "env_forwarded.txt"),
            )

    def test_each_generated_tree_mutation_blocks_every_variant(self):
        names = json.loads((HERE / "contract.json").read_text())["configure_prerequisite"][
            "generated_roots"
        ]
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                run, _ = self.generated_fixture(tmp)
                run.prepare_generated_inputs()
                (run.source / name / "generated.h").write_text("changed input")
                for variant in r.VARIANTS:
                    with (
                        self.subTest(variant=variant),
                        self.assertRaisesRegex(ValueError, "Generated lint inputs changed"),
                    ):
                        run.transition(variant)

    def test_generated_source_roots_missing_or_aliased_fail_closed(self):
        def missing(source):
            path = source / "torch/csrc/lazy/generated"
            (path / "generated.h").unlink()
            path.rmdir()

        def aliased(source):
            path = source / "torch/csrc/autograd/generated"
            path.rename(path.with_name("generated-original"))
            path.symlink_to(path.with_name("generated-original"), target_is_directory=True)

        for mutation in (missing, aliased):
            with self.subTest(mutation=mutation.__name__), tempfile.TemporaryDirectory() as tmp:
                run, _ = self.generated_fixture(tmp, mutation)
                with self.assertRaisesRegex(ValueError, "Missing or aliased"):
                    run.prepare_generated_inputs()

    def test_missing_database_wrong_effective_flag_and_failed_command_block(self):
        def missing_database(source):
            (source / "build/compile_commands.json").unlink()

        def wrong_flag(source):
            path = source / "build/CMakeCache.txt"
            path.write_text(path.read_text().replace("USE_MPS:BOOL=ON", "USE_MPS:BOOL=OFF"))

        def missing_effective(source):
            (source / "build/env_forwarded.txt").unlink()

        def contradictory_effective(source):
            path = source / "build/env_forwarded.txt"
            path.write_text(path.read_text().replace("USE_MPS=ON", "USE_MPS=OFF"))

        for mutation in (missing_database, wrong_flag, missing_effective, contradictory_effective):
            with self.subTest(mutation=mutation.__name__), tempfile.TemporaryDirectory() as tmp:
                run, _ = self.generated_fixture(tmp, mutation)
                with self.assertRaises(ValueError):
                    run.prepare_generated_inputs()
                self.assertIsNone(run.generated_inputs)
        with tempfile.TemporaryDirectory() as tmp:
            run, _ = self.generated_fixture(tmp)
            run.command = lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError("configure failed")
            )
            with self.assertRaisesRegex(ValueError, "configure failed"):
                run.prepare_generated_inputs()
            self.assertIsNone(run.generated_inputs)

    def test_effective_setting_parser_and_empty_tool_policy_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "effective.txt"
            for text in ("", "USE_MPS\n", "USE_MPS=ON\nUSE_MPS=OFF\n", "../USE_MPS=ON\n"):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    path.write_text(text)
                    r.read_effective_forwarded(path)
            path.write_text("USE_MPS=ON\nCMAKE_PREFIX_PATH=/declared/path\n")
            self.assertEqual(r.read_effective_forwarded(path)["USE_MPS"], "ON")
        names = ("venv", "managed_python", "host_python", "lintbin", "libomp", "uv/archive-v0")
        policy = {"": {"kind": "inventory_root"}}
        payload = {"payload": {"bytes": 1, "sha256": "a" * 64}}
        r.require_tool_payloads({name: {**policy, **payload} for name in names})
        for missing in names:
            result = {name: {**policy, **payload} for name in names}
            result[missing] = dict(policy)
            with (
                self.subTest(root=missing),
                self.assertRaisesRegex(ValueError, "root policy is not payload"),
            ):
                r.require_tool_payloads(result)

    def test_requested_fullbuild_flags_are_exact_and_ambient_flags_removed(self):
        contract = json.loads((HERE / "contract.json").read_text())
        owner = HERE.parent / "upstream-pytorch-format-v2/hosted_build.py"
        self.assertEqual(
            r.sha(owner), contract["configure_prerequisite"]["fullbuild_helper_sha256"]
        )
        tree = ast.parse(owner.read_text())
        env = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "BUILD_ENV" for target in node.targets
            )
        )
        self.assertEqual(env, contract["configure_prerequisite"]["fullbuild_environment"])
        self.assertEqual(
            contract["configure_prerequisite"]["official_generator_overrides"],
            {"CC": "clang", "CXX": "clang++", "USE_NCCL": "0", "USE_PRECOMPILED_HEADERS": "1"},
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                r.os.environ,
                {"USE_FAKE_ENGINE": "1", "CMAKE_ARGS": "secret", "BUILD_FAKE": "1", "DEBUG": "1"},
            ),
        ):
            run = r.Run(Path(tmp), Path(tmp), contract)
            for key in ("USE_FAKE_ENGINE", "CMAKE_ARGS", "BUILD_FAKE"):
                self.assertNotIn(key, run.env)
            self.assertEqual({key: run.env[key] for key in env}, env)

    def test_general_linter_failure_precedes_source_parser(self):
        text = (
            "$ uvx --python 3.10 lintrunner@0.12.7 --take FAST --all-files\n"
            ">>> General linter failure:\n"
            "  Error (CLANGTIDY_EXECUTORCH_COMPATIBILITY) Linter failed\n"
            "    Traceback (most recent call last):\n"
            "    FileNotFoundError: source/build\n"
            "$ uvx --python 3.10 lintrunner@0.12.7 --take SLOW\nLint failed!\n"
        )
        for prefix in ("", ">>> Lint for sample.h:\n  Error (CLANGTIDY) narrowing\n"):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                run = r.Run(root, root, {"schema": "test", "historical_run": {}})
                run.transition = lambda variant: {}
                run.source_identity = lambda label: {}
                run.command = lambda *args, **kwargs: ({"exit_code": 1}, prefix + text)
                with patch.object(
                    r, "parse_findings", side_effect=AssertionError("must not parse")
                ):
                    result = run.lint("provision", "base")
                self.assertEqual(result["classification"], "tool_runtime_failure")
                self.assertTrue(result["runtime_failure_detected"])
                self.assertFalse(result["eligible_for_comparison"])
                self.assertFalse(result["diagnostics_parsed"])
                self.assertFalse(result["lint_passed"])
                self.assertEqual(result["findings"], [])
                self.assertIsNone(result["diagnostic_parse_error"])
                self.assertIn(
                    "general_linter_failure",
                    {row["kind"] for row in result["tool_failure_markers"]},
                )
                self.assertEqual(json.loads((root / "provision-base.json").read_text()), result)

    def test_parser_failure_retains_explicit_unqualified_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            run = r.Run(root, root, {"schema": "test", "historical_run": {}})
            run.transition = lambda variant: {}
            run.source_identity = lambda label: {}
            run.command = lambda *args, **kwargs: (
                {"exit_code": 1},
                "  Error (CLANGTIDY) unknown source\n",
            )
            result = run.lint("provision", "base")
            self.assertEqual(result["classification"], "diagnostic_parse_failure")
            self.assertEqual(result["diagnostic_parse_error"]["type"], "ValueError")
            self.assertFalse(result["diagnostics_parsed"])
            self.assertFalse(result["eligible_for_comparison"])
            self.assertTrue((root / "provision-base.json").is_file())

    def test_synthetic_success_does_not_accept_runtime_failure_or_boolean_exit(self):
        normal = (
            "$ uvx --python 3.10 lintrunner@0.12.7 --take FAST --all-files\n"
            "$ uvx --python 3.10 lintrunner@0.12.7 --take SLOW\n"
        )
        self.assertTrue(r.lint_execution_eligible(normal, 0, [])["eligible"])
        self.assertFalse(r.lint_execution_eligible(normal, False, [])["eligible"])
        self.assertFalse(
            r.lint_execution_eligible(normal + ">>> General linter failure:\n", 0, [])["eligible"]
        )

    def test_inventory_detects_dependency_payload_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
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
