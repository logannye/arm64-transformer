"""Archive-safe provenance mutations; no runtime files or frameworks are opened."""

import hashlib
import unittest
from unittest import mock

from transformer import gpu_owned_backend as owned


class LoadedProofTests(unittest.TestCase):
    def fixture(self, selected="mlx", both=True, project=True):
        root = "/unmounted/original-runtime"
        bases = {"pytorch": root + "/torch-runtime", "mlx": root + "/mlx-bundle"}
        packages = {
            "pytorch": bases["pytorch"] + "/torch",
            "mlx": bases["mlx"] + "/runtime/python/mlx",
        }
        inventory = {"files": {}, "layout": {"packages": {}}}

        def file(path):
            digest = hashlib.sha256(path.encode()).hexdigest()
            inventory["files"][path.removeprefix(root + "/")] = digest
            return digest

        frameworks, libraries, selected_origins = {}, [], {}
        for name in ("pytorch", "mlx") if both else (selected,):
            package = packages[name]
            prefix = "torch" if name == "pytorch" else "mlx"
            core_name = "torch._C" if name == "pytorch" else "mlx.core"
            core = package + (
                "/_C.cpython-312-darwin.so" if name == "pytorch" else "/core.cpython-312-darwin.so"
            )
            core_sha = file(core)
            origins = {core_name: core, prefix + ".nn": package + "/nn/__init__.py"}
            file(origins[prefix + ".nn"])
            synthetic, namespaces = {}, {}
            if name == "pytorch":
                origins["torch"] = package + "/__init__.py"
                file(origins["torch"])
                inventory["layout"]["packages"]["torch"] = [origins["torch"], package]
                for module, backing, cls, filename in (
                    ("torch.ops", "torch._ops", "_Ops", "_ops.py"),
                    ("torch.classes", "torch._classes", "_Classes", "_classes.py"),
                ):
                    path = package + "/" + filename
                    origins[backing] = path
                    synthetic[module] = {
                        "backing_module": backing,
                        "backing_file": path,
                        "backing_sha256": file(path),
                        "exact_class": backing + "." + cls,
                        "singleton_identity_verified": True,
                    }
                selected_origins[name] = [origins["torch"], core]
            else:
                inventory["layout"]["packages"]["mlx"] = [package]
                namespaces["mlx"] = {"locations": [package], "extension": core}
                selected_origins[name] = [core]
            alias = {core_name + ".child": {"extension": core, "sha256": core_sha}}
            frameworks[name] = {
                "schema": "gpu-owned-loaded-framework-origins-v1",
                "framework": name,
                "module_origins": origins,
                "synthetic_origins": synthetic,
                "native_aliases": alias,
                "namespace_packages": namespaces,
                "module_count": len(origins) + len(synthetic) + len(alias) + len(namespaces),
            }
            library = (
                package + "/lib/" + ("libtorch_cpu.dylib" if name == "pytorch" else "libmlx.dylib")
            )
            file(library)
            libraries.append(library)
        receipt = {
            "schema": owned.SCHEMA,
            "protocol": owned.PROTOCOL,
            "runtime_identity_sha256": owned.identity(inventory["files"]),
            "frameworks": {name: {"payload_root": path} for name, path in bases.items()},
            "project": {
                "root": "/unmounted/project",
                "files": {"transformer/gpu_owned_backend.py": "a" * 64},
            },
        }
        proof = {
            "module_origins": selected_origins[selected],
            "framework_module_origins": frameworks,
            "native_libraries": libraries,
        }
        if project:
            proof["project_module_origins"] = {
                "transformer.gpu_owned_backend": "transformer/gpu_owned_backend.py"
            }
        return proof, {"inventory": inventory, "runtime_root": root, "receipt": receipt}

    def test_both_selected_frameworks_and_single_framework_proofs(self):
        for selected in ("pytorch", "mlx"):
            for both in (False, True):
                for project in (False, True):
                    proof, inputs = self.fixture(selected, both, project)
                    self.assertEqual(owned.validate_loaded_origin_proof(proof, **inputs), proof)

    def test_declared_forwarding_proofs_are_bound_in_retained_runtime(self):
        from transformer.gpu_owned_origins import TORCH_FORWARDERS

        for name, (relative, cls, target) in TORCH_FORWARDERS.items():
            with self.subTest(name=name):
                proof, inputs = self.fixture("pytorch", both=False)
                framework = proof["framework_module_origins"]["pytorch"]
                package = framework["module_origins"]["torch"].removesuffix("/__init__.py")
                path = package + "/" + relative
                sha = hashlib.sha256(path.encode()).hexdigest()
                key = path.removeprefix(inputs["runtime_root"] + "/")
                inputs["inventory"]["files"][key] = sha
                inputs["receipt"]["runtime_identity_sha256"] = owned.identity(
                    inputs["inventory"]["files"]
                )
                record = {
                    "backing_module": name,
                    "backing_file": path,
                    "backing_sha256": sha,
                    "exact_class": name + "." + cls,
                    "singleton_identity_verified": True,
                    "source_code_verified": True,
                    "forwarding_target": target,
                }
                framework["synthetic_origins"][name] = record
                framework["module_count"] += 1
                self.assertEqual(owned.validate_loaded_origin_proof(proof, **inputs), proof)
                for flag in ("source_code_verified", "singleton_identity_verified"):
                    record[flag] = 1
                    with self.assertRaises(ValueError):
                        owned.validate_loaded_origin_proof(proof, **inputs)
                    record[flag] = True

    def test_validation_is_archive_safe_and_returns_independent_copy(self):
        proof, inputs = self.fixture()
        with (
            mock.patch("builtins.open", side_effect=AssertionError("Live file opened")),
            mock.patch.object(
                owned.Path, "resolve", side_effect=AssertionError("Original path resolved")
            ),
            mock.patch.object(owned, "digest", side_effect=AssertionError("Original bytes read")),
        ):
            result = owned.validate_loaded_origin_proof(proof, **inputs)
        result["native_libraries"].clear()
        self.assertTrue(proof["native_libraries"])

    def rejects(self, mutate, *, inputs_mutation=False):
        proof, inputs = self.fixture()
        mutate(inputs if inputs_mutation else proof)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            owned.validate_loaded_origin_proof(proof, **inputs)

    def test_missing_extra_duplicate_and_wrong_framework_schema(self):
        for mutation in (
            lambda p: p.pop("framework_module_origins"),
            lambda p: p.update(foreign=True),
            lambda p: p.update(framework_module_origins={}),
            lambda p: p["framework_module_origins"]["mlx"].update(schema="wrong"),
            lambda p: p["framework_module_origins"]["mlx"].update(framework="pytorch"),
            lambda p: p["framework_module_origins"]["mlx"].update(module_count=999),
            lambda p: p["framework_module_origins"]["mlx"].update(module_count=True),
            lambda p: p["framework_module_origins"]["mlx"]["native_aliases"].update(
                {"mlx.core": {}}
            ),
        ):
            self.rejects(mutation)

    def test_foreign_missing_or_replaced_core_and_module_origins(self):
        for mutation in (
            lambda p: p.update(module_origins=[]),
            lambda p: p["module_origins"].append(p["module_origins"][0]),
            lambda p: p["module_origins"].__setitem__(0, "/foreign/core.so"),
            lambda p: p["framework_module_origins"]["mlx"]["module_origins"].update(
                {"mlx.core": "/foreign/core.so"}
            ),
            lambda p: p["framework_module_origins"]["mlx"]["module_origins"].update(
                {"mlx.nn": p["framework_module_origins"]["pytorch"]["module_origins"]["torch.nn"]}
            ),
        ):
            self.rejects(mutation)

    def test_native_alias_requires_selected_core_bytes_and_scope(self):
        for mutation in (
            lambda p: p["framework_module_origins"]["mlx"]["native_aliases"][
                "mlx.core.child"
            ].update(sha256="f" * 64),
            lambda p: p["framework_module_origins"]["mlx"]["native_aliases"][
                "mlx.core.child"
            ].update(extension="/foreign/core.so"),
            lambda p: p["framework_module_origins"]["mlx"]["native_aliases"].update(
                {
                    "mlx.nn.alias": p["framework_module_origins"]["mlx"]["native_aliases"].pop(
                        "mlx.core.child"
                    )
                }
            ),
        ):
            self.rejects(mutation)

    def test_torch_synthetic_backing_class_singleton_and_hash_are_exact(self):
        for key, value in (
            ("backing_module", "torch._classes"),
            ("backing_file", "/foreign.py"),
            ("backing_sha256", "f" * 64),
            ("exact_class", "torch._ops.Foreign"),
            ("singleton_identity_verified", False),
            ("singleton_identity_verified", 1),
        ):
            self.rejects(
                lambda p: p["framework_module_origins"]["pytorch"]["synthetic_origins"][
                    "torch.ops"
                ].update({key: value})
            )
        self.rejects(
            lambda p: p["framework_module_origins"]["pytorch"]["synthetic_origins"].pop(
                "torch.classes"
            )
        )

    def test_namespace_requires_exact_selected_paths_and_owner(self):
        for key, value in (("locations", ["/foreign/mlx"]), ("extension", "/foreign/core.so")):
            self.rejects(
                lambda p: p["framework_module_origins"]["mlx"]["namespace_packages"]["mlx"].update(
                    {key: value}
                )
            )
        self.rejects(lambda p: p["framework_module_origins"]["mlx"].update(namespace_packages={}))

    def test_all_loaded_frameworks_require_actual_selected_native_library(self):
        for mutation in (
            lambda p: p.update(native_libraries=[]),
            lambda p: p["native_libraries"].pop(0),
            lambda p: p["native_libraries"].pop(),
            lambda p: p["native_libraries"].append(p["native_libraries"][0]),
            lambda p: p["native_libraries"].append("/foreign/libmlx.dylib"),
            lambda p: p["framework_module_origins"].pop("pytorch"),
        ):
            self.rejects(mutation)

    def test_project_map_is_optional_but_bound_when_present(self):
        for value in (
            {},
            {"foreign": "transformer/gpu_owned_backend.py"},
            {"transformer.model": "missing.py"},
            {"transformer.model": "../outside.py"},
        ):
            self.rejects(lambda p: p.update(project_module_origins=value))

    def test_project_package_layout_and_wrong_existing_source_mapping(self):
        proof, inputs = self.fixture()
        inputs["receipt"]["project"]["files"]["transformer/__init__.py"] = "b" * 64
        proof["project_module_origins"]["transformer"] = "transformer/__init__.py"
        self.assertEqual(owned.validate_loaded_origin_proof(proof, **inputs), proof)
        proof["project_module_origins"]["transformer.spec"] = "transformer/gpu_owned_backend.py"
        with self.assertRaisesRegex(ValueError, "module-to-source"):
            owned.validate_loaded_origin_proof(proof, **inputs)

    def test_inventory_receipt_and_payload_roots_cannot_be_substituted(self):
        for mutation in (
            lambda x: x["receipt"].update(schema="wrong"),
            lambda x: x["receipt"].update(runtime_identity_sha256="f" * 64),
            lambda x: x["inventory"]["files"].clear(),
            lambda x: x["receipt"]["frameworks"]["mlx"].update(payload_root="/foreign"),
            lambda x: x["receipt"]["frameworks"]["mlx"].update(payload_root=x["runtime_root"]),
            lambda x: x.update(runtime_root=x["runtime_root"] + "/../other"),
        ):
            self.rejects(mutation, inputs_mutation=True)

    def test_missing_bound_file_fails_even_with_recomputed_inventory_identity(self):
        proof, inputs = self.fixture()
        inputs["inventory"]["files"].pop(
            proof["module_origins"][0].removeprefix(inputs["runtime_root"] + "/")
        )
        inputs["receipt"]["runtime_identity_sha256"] = owned.identity(inputs["inventory"]["files"])
        with self.assertRaisesRegex(ValueError, "authenticated runtime"):
            owned.validate_loaded_origin_proof(proof, **inputs)


if __name__ == "__main__":
    unittest.main()
