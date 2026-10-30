"""Provenance controls without importing PyTorch or loading native libraries."""

import hashlib
import importlib.util
import os
import tempfile
import types
import unittest
from pathlib import Path

VERIFIER = Path(__file__).resolve().parents[1] / "validation/upstream-pytorch/verify_runtime.py"
SPEC = importlib.util.spec_from_file_location("upstream_runtime_verifier", VERIFIER)
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


class TestRuntimeOrigins(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        self.modules, self.manifest = {}, {}
        self.torch = self.real_module("torch", "torch/__init__.py")
        self.real_module("torch._C", "torch/_C.so")
        self.real_module("torchgen", "torchgen/__init__.py")
        self.real_module("functorch", "functorch/__init__.py")
        # These ModuleType subclass definitions mirror the literal __file__
        # labels and singleton registrations in the pinned upstream source.
        for name, class_name, singleton, label in (
            ("torch._ops", "_Ops", "ops", "_ops.py"),
            ("torch._classes", "_Classes", "classes", "_classes.py"),
        ):
            source = (
                "import types\n"
                f"class {class_name}(types.ModuleType):\n"
                f"    __file__ = {label!r}\n"
                "    def __init__(self):\n"
                f"        super().__init__('torch.{singleton}')\n"
                f"{singleton} = {class_name}()\n"
            )
            module = self.real_module(name, "torch/" + label, source)
            exec(compile(source, module.__file__, "exec"), vars(module))
            namespace = vars(module)[singleton]
            self.modules["torch." + singleton] = namespace
            setattr(self.torch, singleton, namespace)

    def real_module(self, name, relative, source="# file-backed fixture\n"):
        path = self.runtime / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        self.manifest[relative] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        module = types.ModuleType(name)
        module.__file__ = str(path)
        module.__spec__ = importlib.util.spec_from_file_location(name, path)
        self.modules[name] = module
        return module

    def verify(self):
        return V.verify_module_origins(self.runtime, self.manifest, self.modules, self.torch)

    def test_only_exact_registered_namespaces_are_synthetic(self):
        origins, synthetic = self.verify()
        self.assertEqual(set(synthetic), {"torch.ops", "torch.classes"})
        self.assertEqual(origins["torch._ops"], "torch/_ops.py")
        self.assertEqual(origins["torch._C"], "torch/_C.so")
        self.assertNotIn("torch.ops", origins)
        self.assertEqual(synthetic["torch.classes"]["synthetic_file_label"], "_classes.py")

    def test_working_directory_does_not_resolve_synthetic_labels(self):
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.root)
        (self.root / "_ops.py").write_text("unrelated cwd file")
        (self.root / "_classes.py").write_text("unrelated cwd file")
        self.assertEqual(len(self.verify()[1]), 2)

    def test_fake_or_subclass_namespace_rejected(self):
        original = self.modules["torch.ops"]
        for replacement in (
            types.ModuleType("torch.ops"),
            type("Derived", (type(original),), {})(),
        ):
            with self.subTest(type=type(replacement).__name__):
                self.modules["torch.ops"] = replacement
                with self.assertRaisesRegex(RuntimeError, "Unrecognized synthetic"):
                    self.verify()
        self.modules["torch.ops"] = original

    def test_singleton_module_spec_and_label_mutations_rejected(self):
        ops = self.modules["torch.ops"]
        mutations = (
            (vars(self.torch), "ops", object()),
            (vars(self.modules["torch._ops"]), "ops", object()),
            (vars(ops), "__file__", "/foreign/_ops.py"),
            (vars(ops), "__name__", "torch.fake"),
            (vars(ops), "__spec__", object()),
            (vars(ops), "__loader__", object()),
        )
        missing = object()
        for target, key, value in mutations:
            with self.subTest(key=key):
                old = target.get(key, missing)
                target[key] = value
                with self.assertRaisesRegex(RuntimeError, "Unrecognized synthetic"):
                    self.verify()
                if old is missing:
                    del target[key]
                else:
                    target[key] = old

    def test_backing_file_missing_unmanifested_or_modified_rejected(self):
        path = self.runtime / "torch/_ops.py"
        original = path.read_bytes()
        path.write_bytes(original + b"# change\n")
        with self.assertRaisesRegex(RuntimeError, "bytes differ"):
            self.verify()
        path.write_bytes(original)
        del self.manifest["torch/_ops.py"]
        with self.assertRaisesRegex(RuntimeError, "absent from runtime manifest"):
            self.verify()

    def test_namespace_constructor_from_another_file_rejected(self):
        foreign = self.root / "foreign.py"
        foreign.write_text("def replacement(self): pass\n")
        namespace = {}
        exec(compile(foreign.read_text(), str(foreign), "exec"), namespace)
        type(self.modules["torch.ops"]).__init__ = namespace["replacement"]
        with self.assertRaisesRegex(RuntimeError, "authenticated backing file"):
            self.verify()

    def test_real_module_outside_archive_still_rejected(self):
        module = self.real_module("torch.real", "torch/real.py")
        outside = self.root / "outside.py"
        outside.write_text("outside")
        module.__file__ = str(outside)
        with self.assertRaisesRegex(RuntimeError, "outside archive"):
            self.verify()

    def test_real_module_spec_cannot_disagree_or_hide_external_origin(self):
        module = self.real_module("torch.real", "torch/real.py")
        outside = self.root / "outside.py"
        outside.write_text("outside")
        module.__spec__ = importlib.util.spec_from_file_location("torch.real", outside)
        with self.assertRaisesRegex(RuntimeError, "origins disagree"):
            self.verify()
        del module.__file__
        with self.assertRaisesRegex(RuntimeError, "outside archive"):
            self.verify()

    def test_unknown_inherited_relative_filename_is_not_exempt(self):
        class Unknown(types.ModuleType):
            __file__ = "unknown.py"

        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.root)
        (self.root / "unknown.py").write_text("outside")
        self.modules["torch.other"] = Unknown("torch.other")
        with self.assertRaisesRegex(RuntimeError, "outside archive"):
            self.verify()

    def test_real_module_symlink_escape_rejected(self):
        module = self.real_module("torch.real", "torch/real.py")
        path = Path(module.__file__)
        path.unlink()
        outside = self.root / "outside.py"
        outside.write_text("outside")
        path.symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "outside archive"):
            self.verify()

    def test_fileless_vf_style_proxy_does_not_trigger_dynamic_lookup(self):
        class VFModule(types.ModuleType):
            def __getattr__(self, name):
                raise AssertionError(f"Provenance inspection triggered proxy lookup: {name}")

        self.modules["torch._VF"] = VFModule("torch._VF")
        self.modules["torch._C._autograd"] = types.ModuleType("torch._C._autograd")
        origins, _ = self.verify()
        self.assertNotIn("torch._VF", origins)
        self.assertNotIn("torch._C._autograd", origins)
        self.assertIn("torch._C", origins)

    def test_real_module_missing_file_rejected(self):
        module = self.real_module("torch.real", "torch/real.py")
        Path(module.__file__).unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify()


if __name__ == "__main__":
    unittest.main(verbosity=2)
