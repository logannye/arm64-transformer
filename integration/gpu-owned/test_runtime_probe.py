"""CPU-only negative controls for the independently owned runtime probe."""

import copy
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("owned_runtime_probe", HERE / "runtime_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def row(path):
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class InputAuthentication(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.file = self.root / "input.bin"
        self.file.write_bytes(b"original")
        self.expected = row(self.file)

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_original_hash_passes(self):
        result = probe.verify_file(self.file, self.expected["sha256"], self.expected["bytes"])
        self.assertEqual(result["sha256"], self.expected["sha256"])

    def test_one_byte_mutation_rejected(self):
        self.file.write_bytes(b"Original")
        with self.assertRaisesRegex(RuntimeError, "original authority"):
            probe.verify_file(self.file, self.expected["sha256"], self.expected["bytes"])

    def test_truncated_input_rejected(self):
        self.file.write_bytes(b"x")
        with self.assertRaisesRegex(RuntimeError, "Size differs"):
            probe.verify_file(self.file, self.expected["sha256"], self.expected["bytes"])

    def test_file_symlink_rejected(self):
        alias = self.root / "alias"
        alias.symlink_to(self.file)
        with self.assertRaisesRegex(RuntimeError, "Noncanonical"):
            probe.verify_file(alias, self.expected["sha256"])

    def test_exact_tree_passes(self):
        self.assertEqual(
            set(probe.verify_tree(self.root, {"input.bin": self.expected})), {"input.bin"}
        )

    def test_missing_member_rejected(self):
        self.file.unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing"):
            probe.verify_tree(self.root, {"input.bin": self.expected})

    def test_additional_member_rejected(self):
        (self.root / "injected.py").write_text("pass")
        with self.assertRaisesRegex(RuntimeError, "Unrecorded"):
            probe.verify_tree(self.root, {"input.bin": self.expected})

    def test_tree_symlink_rejected(self):
        (self.root / "injected").symlink_to(self.file)
        with self.assertRaisesRegex(RuntimeError, "Symlink"):
            probe.verify_tree(self.root, {"input.bin": self.expected})

    def test_executable_bit_rejected(self):
        self.file.chmod(0o755)
        with self.assertRaisesRegex(RuntimeError, "Mode differs"):
            probe.verify_tree(self.root, {"input.bin": {**self.expected, "executable": False}})

    def test_postflight_content_mutation_rejected(self):
        baseline = probe.verify_file(self.file, self.expected["sha256"])
        auth = {"files": {str(self.file): baseline}, "trees": {}}
        self.file.write_bytes(b"Original")
        with self.assertRaises(RuntimeError):
            probe.postflight(auth)

    def test_postflight_replacement_same_bytes_rejected(self):
        baseline = probe.verify_file(self.file, self.expected["sha256"])
        replacement = self.root / "replacement"
        replacement.write_bytes(b"original")
        replacement.replace(self.file)
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            probe.postflight({"files": {str(self.file): baseline}, "trees": {}})

    def test_output_namespace_preserves_prior_evidence(self):
        valid = self.root / "work/gpu-v4-owned/attempt-v1"
        self.assertEqual(probe.checked_output(self.root, valid), valid)
        for invalid in (
            self.root / "outputs/previous",
            self.root / "work/upstream-integration-v2/changed",
        ):
            with self.assertRaisesRegex(RuntimeError, "new development namespace"):
                probe.checked_output(self.root, invalid)

    def test_existing_output_is_never_reused(self):
        output = self.root / "work/gpu-v4-owned/attempt-v1"
        output.mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            probe.checked_output(self.root, output)

    def test_result_file_cannot_be_overwritten(self):
        output = self.root / "result.json"
        probe.write_json_new(output, {"passed": False})
        with self.assertRaises(FileExistsError):
            probe.write_json_new(output, {"passed": True})
        self.assertFalse(json.loads(output.read_text())["passed"])


class RuntimeControls(unittest.TestCase):
    def test_environment_clears_presence_controls_and_loaders(self):
        bad = {
            "MLX_DISABLE_COMPILE": "0",
            "PYTORCH_MPS_PREFER_METAL": "0",
            "DYLD_INSERT_LIBRARIES": "evil",
            "MLX_SOMETHING": "1",
            "PYTHONPATH": "other",
        }
        with patch.dict(os.environ, bad, clear=True):
            actual = probe.controlled_env()
        self.assertFalse(probe.ABSENT & set(actual))
        self.assertFalse(any(key.startswith("DYLD_") for key in actual))
        self.assertEqual({key: actual[key] for key in probe.ENVIRONMENT}, probe.ENVIRONMENT)

    def test_owned_identity_pins_are_specific(self):
        self.assertEqual(probe.TORCH_BASE, "71d2a08507ab47dd6b2fa9b66319d17ce2377de4")
        self.assertEqual(probe.MLX_BASE, "a21f892a6e8e83afb117cef1d1ceae3e82f48c32")
        self.assertNotEqual(probe.TORCH_MANIFEST_SHA, probe.MLX_MANIFEST_SHA)
        self.assertEqual(len(probe.PINNED_FILES), 6)
        self.assertTrue(
            all(len(sha) == 64 and size > 0 for size, sha in probe.PINNED_FILES.values())
        )

    def test_bootstrap_cannot_import_before_authentication(self):
        with patch.object(probe, "require_environment") as controls:
            with patch.object(
                probe, "authenticate_runtime", side_effect=RuntimeError("archive rejected")
            ):
                original = list(probe.sys.path)
                with self.assertRaisesRegex(RuntimeError, "archive rejected"):
                    probe.bootstrap("/invalid")
                self.assertEqual(probe.sys.path, original)
                controls.assert_called_once()

    def test_missing_authentication_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                probe.authenticate_runtime(Path(directory))

    def test_numpy_two_version_rejected(self):
        np = SimpleNamespace(__version__="2.2.6", __file__="/untrusted/numpy/__init__.py")
        with self.assertRaisesRegex(RuntimeError, "NumPy version"):
            probe.verify_numpy_origin(np, {"workspace": "/workspace"})

    def test_wrong_numpy_origin_rejected(self):
        np = SimpleNamespace(__version__="1.26.4", __file__="/untrusted/numpy/__init__.py")
        with self.assertRaisesRegex(RuntimeError, "Foreign NumPy"):
            probe.verify_numpy_origin(np, {"workspace": "/workspace"})

    def test_numpy_metadata_substitution_rejected(self):
        workspace = Path("/workspace")
        np = SimpleNamespace(
            __version__="1.26.4", __file__=str(workspace / probe.NUMPY_SOURCE / "__init__.py")
        )
        with patch.object(
            probe.importlib.metadata, "distribution", return_value=SimpleNamespace(version="2.2.6")
        ):
            with self.assertRaisesRegex(RuntimeError, "metadata version"):
                probe.verify_numpy_origin(np, {"workspace": str(workspace)})


class NumPyOverlay(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name).resolve()
        source = self.workspace / probe.NUMPY_SOURCE
        source.mkdir(parents=True)
        (source / "__init__.py").write_text("# existing package\n")
        self.auth = {
            "workspace": str(self.workspace),
            "trees": {},
            "numpy_overlay_metadata": {"METADATA": b"Version: 1.26.4\n"},
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_overlay_reuses_exact_existing_package(self):
        overlay = probe.prepare_numpy_overlay(self.auth)
        self.assertTrue((overlay / "numpy").is_symlink())
        self.assertEqual((overlay / "numpy").resolve(), self.workspace / probe.NUMPY_SOURCE)
        self.assertFalse(self.auth["numpy_overlay"]["native_binaries_copied"])
        before = dict(self.auth["numpy_overlay"])
        probe.prepare_numpy_overlay(self.auth)
        self.assertEqual(self.auth["numpy_overlay"], before)

    def test_overlay_extra_pth_rejected(self):
        overlay = probe.prepare_numpy_overlay(self.auth)
        (overlay / "injected.pth").write_text("import evil")
        with self.assertRaisesRegex(RuntimeError, "Unexpected overlay member"):
            probe.prepare_numpy_overlay(self.auth)

    def test_overlay_retarget_rejected(self):
        overlay = probe.prepare_numpy_overlay(self.auth)
        (overlay / "numpy").unlink()
        (overlay / "numpy").symlink_to(self.workspace)
        with self.assertRaisesRegex(RuntimeError, "target differs"):
            probe.prepare_numpy_overlay(self.auth)

    def test_overlay_metadata_mutation_rejected(self):
        overlay = probe.prepare_numpy_overlay(self.auth)
        (overlay / probe.NUMPY_METADATA / "METADATA").write_bytes(b"Version: 2.2.6!\n")
        with self.assertRaises(RuntimeError):
            probe.prepare_numpy_overlay(self.auth)


class CompiledGraphComparison(unittest.TestCase):
    GRAPH = (
        'digraph {\n{ rank=source; "A"; }\n{ rank=source; "B"; }\n'
        '{ rank=source; "C"; }\n{ rank=source; "D"; }\n'
        '{ 45553443352 [label ="CompiledBroadcastMultiplyAddBroadcastMaximum", '
        'shape=rectangle]; }\n"A" -> 45553443352\n"B" -> 45553443352\n'
        '"C" -> 45553443352\n"D" -> 45553443352\n{ rank=sink; "E"; }\n'
        '45553443352 -> "E"\n}'
    )

    def record(self, graph=None):
        graph = self.GRAPH if graph is None else graph
        return {
            "core": "/unchanged/core.so",
            "compilation": {
                "graph_dot": graph,
                "graph_sha256": hashlib.sha256(graph.encode()).hexdigest(),
                "observations": [
                    {"offset": 0.5, "fp32_sha256": "a" * 64, "exact_dyadic_oracle": True}
                ],
                "python_trace_count": 1,
                "device": "gpu",
            },
        }

    def test_numeric_node_id_only_change_passes_and_raw_records_stay_unchanged(self):
        first = self.record()
        second = self.record(self.GRAPH.replace("45553443352", "35023210008"))
        original_first, original_second = copy.deepcopy(first), copy.deepcopy(second)
        result = probe.compare_probes("mlx", first, second)
        self.assertTrue(result["passed"])
        self.assertFalse(result["raw_probe_json_equal"])
        self.assertEqual(first, original_first)
        self.assertEqual(second, original_second)
        self.assertNotEqual(
            result["graph_comparisons"][0]["raw_graph_sha256"],
            result["graph_comparisons"][1]["raw_graph_sha256"],
        )
        self.assertEqual(
            result["graph_comparisons"][0]["canonical_graph_sha256"],
            result["graph_comparisons"][1]["canonical_graph_sha256"],
        )

    def test_changed_compiled_operation_is_rejected_even_in_both_records(self):
        wrong = self.record(self.GRAPH.replace("BroadcastMaximum", "BroadcastMinimum"))
        with self.assertRaisesRegex(RuntimeError, "Compiled graph node"):
            probe.compare_probes("mlx", wrong, wrong)

    def test_changed_edge_is_rejected(self):
        wrong = self.record(self.GRAPH.replace('"D" ->', '"A" ->'))
        with self.assertRaisesRegex(RuntimeError, "structure differs"):
            probe.compare_probes("mlx", self.record(), wrong)

    def test_added_constant_node_is_rejected(self):
        wrong = self.record(self.GRAPH.replace("digraph {", 'digraph {\n{ X [label="2.0"]; }'))
        with self.assertRaisesRegex(RuntimeError, "structure differs"):
            probe.compare_probes("mlx", self.record(), wrong)

    def test_inconsistent_node_identifier_is_rejected(self):
        wrong = self.record(self.GRAPH.replace('"A" -> 45553443352', '"A" -> 35023210008'))
        with self.assertRaisesRegex(RuntimeError, "structure differs"):
            probe.compare_probes("mlx", self.record(), wrong)

    def test_raw_graph_hash_must_match_before_normalization(self):
        wrong = self.record()
        wrong["compilation"]["graph_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "Raw compiled graph hash"):
            probe.compare_probes("mlx", self.record(), wrong)

    def test_constant_input_and_tensor_hashes_are_not_normalized(self):
        for key, value in (("offset", 1.0), ("fp32_sha256", "b" * 64)):
            wrong = self.record()
            wrong["compilation"]["observations"][0][key] = value
            with self.assertRaisesRegex(RuntimeError, "probe fields differ"):
                probe.compare_probes("mlx", self.record(), wrong)

    def test_missing_or_changed_other_fields_are_rejected(self):
        wrong = self.record()
        wrong["core"] = "/substituted/core.so"
        with self.assertRaisesRegex(RuntimeError, "probe fields differ"):
            probe.compare_probes("mlx", self.record(), wrong)
        del wrong["compilation"]
        with self.assertRaisesRegex(RuntimeError, "Missing compilation"):
            probe.compare_probes("mlx", self.record(), wrong)

    def test_torch_probe_remains_exact(self):
        first = {"dyadic_roundtrips": {"cpu": {"fp32_sha256": "a" * 64}}}
        self.assertTrue(probe.compare_probes("torch", first, first)["raw_probe_json_equal"])
        second = copy.deepcopy(first)
        second["dyadic_roundtrips"]["cpu"]["fp32_sha256"] = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "probe fields differ"):
            probe.compare_probes("torch", first, second)


if __name__ == "__main__":
    unittest.main()
