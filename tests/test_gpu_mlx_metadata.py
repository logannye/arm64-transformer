"""Exercise the real capability-report method without importing a GPU framework."""

import ast
import copy
import importlib.metadata
import unittest
from pathlib import Path
from types import SimpleNamespace


def capability_method(version):
    source = Path(__file__).resolve().parents[1] / "transformer/gpu_mlx.py"
    tree = ast.parse(source.read_text())
    backend = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Backend")
    method = next(
        n for n in backend.body if isinstance(n, ast.FunctionDef) and n.name == "capabilities"
    )
    namespace = dict(
        importlib=SimpleNamespace(
            metadata=SimpleNamespace(
                version=version, PackageNotFoundError=importlib.metadata.PackageNotFoundError
            )
        ),
        mx=SimpleNamespace(gpu="gpu", device_info=lambda: {"test": True}),
        os=SimpleNamespace(environ={"MLX_ENABLE_TF32": "0", "MLX_SDPA_VJP_FALLBACK": "0"}),
        LR=0.001,
        copy=copy,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            str(source),
            "exec",
        ),
        namespace,
    )
    return namespace["capabilities"]


class SourceBundleMetadataTests(unittest.TestCase):
    def test_missing_wheel_versions_are_reported_honestly_without_changing_numerical_capabilities(
        self,
    ):
        for present in (set(), {"mlx"}, {"mlx", "mlx-metal"}):
            with self.subTest(present=present):
                calls = []

                def version(name):
                    calls.append(name)
                    if name not in present:
                        raise importlib.metadata.PackageNotFoundError(name)
                    return "0.32.4"

                result = capability_method(version)(
                    SimpleNamespace(_compilation_probe={"status": "passed"})
                )
                self.assertEqual(calls, ["mlx", "mlx-metal"])
                self.assertEqual(result["mlx"], "0.32.4" if "mlx" in present else None)
                self.assertEqual(result["mlx_metal"], "0.32.4" if "mlx-metal" in present else None)
                self.assertIn("authenticated source/native manifests", result["version_authority"])
                self.assertTrue(result["compiled"])
                self.assertEqual(result["storage_dtype"], "float32")
                self.assertFalse(result["cpu_tensor_fallback"])
                self.assertEqual(
                    result["adam"],
                    dict(lr=0.001, betas=[0.9, 0.999], eps=1e-8, bias_correction=True),
                )

    def test_other_metadata_errors_are_not_converted_to_missing_versions(self):
        def invalid(name):
            raise RuntimeError("Corrupt installed metadata")

        with self.assertRaisesRegex(RuntimeError, "Corrupt"):
            capability_method(invalid)(SimpleNamespace(_compilation_probe={"status": "passed"}))


if __name__ == "__main__":
    unittest.main()
