"""Portable MPS evidence roundtrip and mutations, without framework imports/builds."""

import copy
import importlib.util
import json
import shlex
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


M = module("mps_producer", HERE / "owned_pytorch_suites.py")
P = module("mps_fixture", HERE / "owned_pytorch_fixture.py")
A = module("mps_archive", HERE / "owned_pytorch_archive.py")
T = module("mps_existing_tests", HERE / "test_owned_pytorch_suites.py")


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.live = self.root / "original"
        self.live.mkdir()
        self.output = self.live / "output"
        self.output.mkdir()
        self.archived = self.root / "relocated"
        self.source = self.live / "source"
        self.cache = self.live / P.CACHE_NAME
        self.cache.mkdir()
        native = T.NativeProofTests()
        native.setUp()
        self.runtime = copy.deepcopy(native.runtime)
        self.native = copy.deepcopy(native.proof)
        self.runtime["files"]["deps/numpy/__init__.py"] = "b" * 64
        self.runtime["files"]["python"] = "f" * 64
        self.runtime["layout"] = {
            "packages": {"numpy": ["/runtime/deps/numpy/__init__.py"]},
            "interpreter_resolved": "/runtime/python",
        }
        self.imported = {
            n: dict(path=p, sha256="a" * 64)
            for n, p in self.native["framework_module_origins"]["pytorch"]["module_origins"].items()
        }
        self.imported["numpy"] = dict(path="/runtime/deps/numpy/__init__.py", sha256="b" * 64)
        project = dict(
            root=str(ROOT),
            files={n: M.digest(ROOT / n) for n in (M.PRODUCER, M.POLICY, M.INVENTORY, A.SELF)},
        )
        self.runtime_receipt = dict(project=project)
        self.selected = dict(
            pytorch=dict(base="a" * 40, patches=["b" * 64], runtime_manifest_sha256="c" * 64)
        )
        authorities = {
            "inventory": self.runtime,
            "selection": self.selected,
            "receipt": self.runtime_receipt,
        }
        authority_rows = {
            n: self.json_file(self.live / (n + ".json"), v) for n, v in authorities.items()
        }
        source_files = {}
        for name in P.SOURCE_INPUTS:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                (HERE / "stock_mps_fixture_fragments.py.txt").read_text()
                if name == "test/test_mps.py"
                else name
            )
            source_files[name] = {k: M.binding(path)[k] for k in ("sha256", "bytes")}
        source_manifest = self.json_file(
            self.live / "source.json",
            dict(base="a" * 40, patch_sha256="b" * 64, files=source_files),
        )
        helper_rows, helper_pins = {}, {}
        for name, (filename, _) in M.HELPERS.items():
            path = self.live / filename
            path.write_text("# synthetic retained " + name)
            helper_rows[name] = M.binding(path)
            helper_pins[name] = (filename, helper_rows[name]["sha256"])
        self.addCleanup(patch.stopall)
        patch.dict(M.HELPERS, helper_pins, clear=True).start()
        header = self.live / "Python.h"
        header.write_text("synthetic header")
        identity = dict(
            schema=P.SCHEMA,
            runtime_manifest_sha256="c" * 64,
            source_inputs={n: r["sha256"] for n, r in source_files.items()},
            conformance_ast_sha256=P.CONFORMANCE_AST_SHA,
            controls=P.CONTROLS,
            ninja=dict(path="/tools/ninja", sha256="d" * 64),
            compiler=dict(path="/usr/bin/clang++", sha256="e" * 64),
            include_inputs={str(header): M.digest(header)},
            toolchain_environment={},
        )
        marker = self.json_file(self.cache / "cache-identity.json", identity)
        fixture_before = self.snapshot()
        fixture = dict(identity, before=fixture_before)
        self.plan = dict(
            schema=M.PLAN_SCHEMA,
            inventory=M.binding(ROOT / M.INVENTORY),
            protocol=M.PROTOCOL,
            producer=M.binding(ROOT / M.PRODUCER),
            runtime_root="/runtime",
            authority_inputs=authority_rows,
            source_root=str(self.source),
            source_manifest=source_manifest,
            limits=M.LIMITS,
            formal_fits=0,
            formal_admission_opened=False,
            retained_helpers=helper_rows,
            cache=str(self.cache),
            cache_identity=marker,
            fixture_preparation=fixture,
            subprocess_contract=M.subprocess_contract(self.source / "test/test_mps.py"),
            environment={**M.PINS, **P.CONTROLS, "TORCH_EXTENSIONS_DIR": str(self.cache)},
        )
        self.plan_path = self.live / "plan.json"
        self.json_file(self.plan_path, self.plan)
        plan_sha = M.digest(self.plan_path)
        ext = self.cache / "mps_iter_conformance/mps_iter_conformance.so"
        ext.parent.mkdir()
        ext.write_bytes(b"synthetic-native-fixture")
        ninja = ext.with_name("build.ninja")
        ninja.write_text("cflags = -O2 -g0\npost_cflags = -std=c++17\n")
        process = dict(
            pid=100,
            started_unix=1,
            ended_unix=9,
            returncode=0,
            timed_out=False,
            command=M.worker_command(
                self.plan, self.plan_path, plan_sha, self.output, self.runtime
            ),
        )
        self.record = dict(
            plan_sha256=plan_sha,
            runtime_receipt_sha256=authority_rows["receipt"]["sha256"],
            source_manifest=source_manifest,
            test_source_sha256={"test/test_mps.py": source_files["test/test_mps.py"]["sha256"]},
            fixture_before=fixture_before,
            fixture_after=self.snapshot(),
            stock_fixture_loaded=dict(
                path=str(ext),
                sha256=M.digest(ext),
                build_ninja_sha256=M.digest(ninja),
                build_ninja=ninja.read_text(),
            ),
            worker_pid=100,
            started_unix=1.5,
            process=process,
            bytecode_prefix=str(self.output / "unused-bytecode-cache"),
            native_origin_verification=dict(before=self.native, after=self.native),
            loaded_python_files_before=self.imported,
            loaded_python_files_after=self.imported,
            descendants=[],
        )
        cmd = M.child_command(
            self.plan, self.plan_path, plan_sha, self.output, 100, runtime=self.runtime
        )
        (self.output / "owned-suite-python").write_text(
            "#!/bin/sh\nexec " + shlex.join(cmd) + ' "$@"\n'
        )
        for index, invocation in enumerate(self.plan["subprocess_contract"]["invocations"]):
            folder = self.output / ("descendant-" + invocation["name"])
            folder.mkdir()
            child = dict(
                schema="gpu-owned-mps-suite-descendant-v1",
                protocol=M.PROTOCOL,
                producer=M.PRODUCER,
                plan_sha256=plan_sha,
                parent_pid=100,
                pid=101 + index,
                complete=True,
                returncode=0,
                invocation=invocation,
                formal_fits=0,
                formal_admission_opened=False,
                runtime_receipt_sha256=authority_rows["receipt"]["sha256"],
                bytecode_prefix=str(folder / "unused-bytecode-cache"),
                native_origin_verification=self.native,
                loaded_python_files=self.imported,
                started_unix=2 + index * 2,
                ended_unix=3 + index * 2,
            )
            self.record["descendants"].append(self.json_file(folder / "receipt.json", child))
        self.manifest = dict(
            bindings=dict(
                source=project["files"],
                runtime=self.runtime["files"],
                runtime_layout=self.runtime["layout"],
            ),
            owned_frameworks=dict(frameworks=self.selected),
        )
        self.record["portable_evidence"] = A.retain(
            self.plan, self.plan_path, self.output, self.record, M.binding, M.LIMITS
        )
        shutil.copytree(self.output, self.archived)
        shutil.rmtree(self.live)

    def json_file(self, path, value):
        path.write_text(json.dumps(value, indent=2) + "\n")
        return M.binding(path)

    def snapshot(self):
        files = {
            str(p.relative_to(self.cache)): {k: M.binding(p)[k] for k in ("bytes", "sha256")}
            for p in self.cache.rglob("*")
            if p.is_file()
        }
        return dict(
            path=str(self.cache),
            bytes=sum(r["bytes"] for r in files.values()),
            file_count=len(files),
            free_bytes=10 * 1024**3,
            max_cache_bytes=P.MAX_CACHE_BYTES,
            min_free_bytes=P.MIN_FREE_BYTES,
            max_cache_files=P.MAX_CACHE_FILES,
            files=files,
        )

    def check(self):
        return A.audit(self.archived, self.record, self.manifest, self.runtime_receipt, M, P)

    def change_child(self, field, value, index=0):
        row = self.record["portable_evidence"]["descendants"][index]
        path = self.archived / row["path"]
        child = M.read(path)
        child[field] = value
        path.write_text(json.dumps(child))
        changed = M.binding(path)
        row.update({k: changed[k] for k in ("sha256", "bytes")})
        self.record["descendants"][index].update({k: changed[k] for k in ("sha256", "bytes")})

    def qualification_fixture(self):
        sys.path.insert(0, str(ROOT))
        from transformer import gpu_owned_eligibility as gate

        contract = gate.suite_contract()
        contract["pytorch"]["test_source_sha256"] = self.record["test_source_sha256"][
            "test/test_mps.py"
        ]
        expected = contract["pytorch"]
        tests = dict(
            run=len(expected["test_ids"]),
            executed_ids=expected["test_ids"],
            skipped=[dict(id=n, reason="synthetic eligible skip") for n in expected["skip_ids"]],
            expected_failures=[
                dict(id=n, traceback="synthetic expected failure")
                for n in expected["expected_failure_ids"]
            ],
            failures=[],
            errors=[],
            unexpected_successes=[],
            successful=True,
        )
        paths = {"tests": self.archived / "tests.json", "log": self.archived / "tests.log"}
        self.json_file(paths["tests"], tests)
        paths["log"].write_text("synthetic complete-suite log")
        raw = {k: dict(M.binding(p), path=p.name) for k, p in paths.items()}
        runtime_identity = gate.identity(self.runtime["files"])
        self.record.update(
            schema=M.SCHEMA,
            protocol=M.PROTOCOL,
            name="pytorch_mps",
            complete=True,
            integrity_after=True,
            formal_admission_opened=False,
            formal_fits=0,
            producer=M.PRODUCER,
            source_sha256_before=self.manifest["bindings"]["source"],
            source_sha256_after=self.manifest["bindings"]["source"],
            runtime_identity_sha256_before=runtime_identity,
            runtime_identity_sha256_after=runtime_identity,
            framework_manifest_sha256=self.selected["pytorch"]["runtime_manifest_sha256"],
            execution_scope="full_gpu_default_suite",
            no_filters=True,
            no_test_source_modifications=True,
            native_artifacts=self.native["files"],
            executable=dict(runtime_relative="python", sha256="f" * 64),
            raw=raw,
        )
        self.record["native_origin_verification"]["method"] = "loaded_images"
        self.manifest["bindings"]["baseline"] = {
            "owned-runtime.json": self.record["runtime_receipt_sha256"]
        }
        self.manifest["owned_frameworks"]["runtime_receipt"] = "owned-runtime.json"
        # Synthetic helper payloads have explicit test-only trust pins. All
        # portable validation still runs; no audit result or validator is mocked.
        original = gate.runpy.run_path

        def load(path):
            return vars(M) if Path(path) == ROOT / M.PRODUCER else original(path)

        return gate, contract, load

    def test_production_eligibility_rechecks_complete_portable_evidence(self):
        gate, contract, load = self.qualification_fixture()
        with patch.object(gate.runpy, "run_path", side_effect=load):
            result = gate.validate_suite(
                self.record,
                self.archived,
                self.manifest,
                contract,
                "pytorch_mps",
                self.runtime_receipt,
            )
        self.assertEqual(result["cases"], 10353)
        self.assertFalse(any(n.split(".")[0] in ("torch", "mlx", "numpy") for n in sys.modules))

    def test_passing_counts_cannot_hide_a_missing_child_archive(self):
        gate, contract, load = self.qualification_fixture()
        self.record["portable_evidence"]["descendants"].pop()
        with patch.object(gate.runpy, "run_path", side_effect=load), self.assertRaises(ValueError):
            gate.validate_suite(
                self.record,
                self.archived,
                self.manifest,
                contract,
                "pytorch_mps",
                self.runtime_receipt,
            )

    def test_relocated_complete_evidence_without_original_files(self):
        self.assertFalse(self.live.exists())
        result = self.check()
        self.assertTrue(result["archive_valid"])
        self.assertEqual(result["descendants"], 2)

    def test_missing_and_changed_raw_members(self):
        for key in ("plan", "source_manifest", "launcher"):
            row = self.record["portable_evidence"][key]
            path = self.archived / row["path"]
            old = path.read_bytes()
            path.write_bytes(old + b" ")
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check()
            path.write_bytes(old)

    def test_unbound_archive_member(self):
        (self.archived / "portable/extra.py").write_text("unexpected")
        with self.assertRaises(ValueError):
            self.check()

    def test_symlink_member(self):
        p = self.archived / "portable/plan.json"
        data = p.read_bytes()
        p.unlink()
        target = self.archived / "elsewhere"
        target.write_bytes(data)
        p.symlink_to(target)
        with self.assertRaises(ValueError):
            self.check()

    def test_duplicate_member_binding(self):
        self.record["portable_evidence"]["headers"]["extra"] = self.record["portable_evidence"][
            "plan"
        ]
        with self.assertRaises(ValueError):
            self.check()

    def test_missing_helper(self):
        self.record["portable_evidence"]["helpers"].pop("driver")
        with self.assertRaises(ValueError):
            self.check()

    def test_missing_source_input(self):
        self.record["portable_evidence"]["source_inputs"].pop("c10/metal/common.h")
        with self.assertRaises(ValueError):
            self.check()

    def test_old_evidence_cannot_be_relabeled_as_half_v3(self):
        evidence = self.record["portable_evidence"]
        row = evidence["plan"]
        path = self.archived / row["path"]
        plan = M.read(path)
        plan["suite_contract"] = "half-v3"
        path.write_text(json.dumps(plan))
        changed = M.binding(path)
        for target in (row, evidence["original_plan"]):
            target.update({key: changed[key] for key in ("sha256", "bytes")})
        self.record["plan_sha256"] = changed["sha256"]
        with self.assertRaisesRegex(ValueError, "Plan suite inventory differs"):
            self.check()

    def test_missing_generated_fixture(self):
        self.record["portable_evidence"]["cache"].pop(
            "mps_iter_conformance/mps_iter_conformance.so"
        )
        with self.assertRaises(ValueError):
            self.check()

    def test_changed_loaded_fixture(self):
        self.record["stock_fixture_loaded"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.check()

    def test_fixture_resource_observation(self):
        self.record["fixture_after"]["free_bytes"] = 1
        with self.assertRaises(RuntimeError):
            self.check()

    def test_forged_actual_parent_command(self):
        self.record["process"]["command"].remove("-S")
        with self.assertRaises(ValueError):
            self.check()

    def test_changed_child_invocation(self):
        self.change_child("invocation", self.plan["subprocess_contract"]["invocations"][1])
        with self.assertRaises(ValueError):
            self.check()

    def test_duplicate_child_pid(self):
        self.change_child("pid", 101, index=1)
        with self.assertRaises(ValueError):
            self.check()

    def test_overlapping_children(self):
        self.change_child("started_unix", 2.5, index=1)
        with self.assertRaises(ValueError):
            self.check()

    def test_changed_child_import_authority(self):
        imported = copy.deepcopy(self.imported)
        imported["numpy"]["sha256"] = "0" * 64
        self.change_child("loaded_python_files", imported)
        with self.assertRaises(ValueError):
            self.check()

    def test_wrong_external_runtime_authority(self):
        self.manifest["bindings"]["runtime"] = dict(self.runtime["files"], foreign="0" * 64)
        with self.assertRaises(ValueError):
            self.check()

    def test_noncanonical_member_names(self):
        for name in (".", "..", "a/../b", "/absolute", "a//b", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                A.relative(name)

    def test_full_source_selection_cannot_be_replaced(self):
        row = self.record["portable_evidence"]["source_manifest"]
        source = M.read(self.archived / row["path"])
        for field in ("base", "patch_sha256"):
            changed = copy.deepcopy(source)
            changed[field] = "0" * len(changed[field])
            with self.subTest(field=field), self.assertRaises(ValueError):
                A.source_authority(changed, self.selected["pytorch"])

    def test_full_source_manifest_rejects_unsafe_member(self):
        row = self.record["portable_evidence"]["source_manifest"]
        source = M.read(self.archived / row["path"])
        source["files"]["../escape"] = next(iter(source["files"].values()))
        with self.assertRaises(ValueError):
            A.source_authority(source, self.selected["pytorch"])


if __name__ == "__main__":
    unittest.main()
