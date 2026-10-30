"""Tiny temporary-tree controls; never inspect installed tool or framework trees."""

import hashlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("tested_tree_inventory", HERE / "tree_inventory.py")
t = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(t)


class TreeInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.a = self.root / "a"
        self.b = self.root / "b"
        self.a.mkdir()
        self.b.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_invalid_root_names_are_rejected(self):
        for name in ("", ".", "..", "../escape", "/absolute", "uv//archive", "a\\b", "a\x7f"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                t.inventory_forest({name: self.a})
        self.assertIn("uv/archive", t.inventory_forest({"uv/archive": self.a}))

    def test_regular_flat_files_and_empty_directory_topology(self):
        data = b"fixture only"
        file = self.a / "payload"
        file.write_bytes(data)
        file.chmod(0o755)
        (self.a / "empty").mkdir()
        rows = t.inventory(self.a)
        self.assertEqual(
            rows["payload"],
            {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "executable_bits": 0o111,
            },
        )
        self.assertEqual(rows["empty"]["kind"], "directory")
        self.assertNotIn("", rows)
        policy = t.inventory_forest({"one": self.a})["one"][""]
        self.assertEqual(policy["exclusions"], t.EXCLUSIONS)
        self.assertEqual(policy["bounds"]["expanded_entries"], t.MAX_ENTRIES)

    def test_file_and_directory_aliases_expand_all_declared_payloads(self):
        (self.b / "nested").mkdir()
        (self.b / "nested/payload").write_bytes(b"abc")
        (self.a / "external").symlink_to(self.b, target_is_directory=True)
        (self.a / "file").symlink_to(self.b / "nested/payload")
        rows = t.inventory_forest({"a": self.a, "b": self.b})
        alias = rows["a"]["external"]
        self.assertEqual(alias["kind"], "directory_alias")
        self.assertEqual(alias["target_root"], "b")
        self.assertEqual(alias["target_relative"], ".")
        self.assertEqual(alias["link"], str(self.b))
        self.assertEqual(alias["resolved"], str(self.b))
        self.assertEqual(rows["a"]["external/nested/payload"], rows["b"]["nested/payload"])
        self.assertEqual(rows["a"]["file"]["sha256"], rows["b"]["nested/payload"]["sha256"])
        flat = t.inventory(self.a, {"a": self.a, "b": self.b})
        self.assertEqual(flat["external/nested/payload"], rows["b"]["nested/payload"])

    def test_parent_components_follow_real_alias_resolution(self):
        (self.a / "subdir").mkdir()
        (self.b / "subdir").mkdir()
        (self.a / "payload").write_bytes(b"wrong lexical target")
        (self.b / "payload").write_bytes(b"actual OS target")
        (self.a / "aliasdir").symlink_to(self.b / "subdir", target_is_directory=True)
        (self.a / "link").symlink_to("aliasdir/../payload")
        forest = t.inventory_forest({"a": self.a, "b": self.b})
        self.assertEqual(forest["a"]["link"]["resolved"], str(self.b / "payload"))
        self.assertEqual(forest["a"]["link"]["sha256"], forest["b"]["payload"]["sha256"])
        (self.a / "sibling").symlink_to("../b/payload")
        forest = t.inventory_forest({"a": self.a, "b": self.b})
        self.assertEqual(forest["a"]["sibling"]["resolved"], str(self.b / "payload"))
        (self.a / "invalid-prefix").symlink_to("payload/../subdir")
        with self.assertRaisesRegex(ValueError, "Non-directory"):
            t.inventory_forest({"a": self.a, "b": self.b})

    def test_internal_alias_chain_and_hardlinks_hash_physical_bytes_once(self):
        (self.a / "one").write_bytes(b"abcd")
        os.link(self.a / "one", self.a / "hardlink")
        (self.a / "alias").symlink_to("one")
        (self.a / "chain").symlink_to("alias")
        original = os.open
        opens = []

        def recording(path, flags, *args, **kwargs):
            opens.append(Path(path))
            return original(path, flags, *args, **kwargs)

        with patch.object(t.os, "open", recording):
            result = t.inventory(self.a, max_unique_bytes=4)
        self.assertEqual(len(opens), 1)
        self.assertEqual(len({row["sha256"] for row in result.values()}), 1)
        self.assertEqual(result["chain"]["link"], "alias")
        self.assertEqual(result["chain"]["resolved"], str(self.a / "one"))

    def test_unknown_escapes_broken_links_and_linked_roots_rejected(self):
        (self.b / "payload").write_bytes(b"outside")
        for target in (self.b, self.b / "payload", self.a / "missing"):
            with self.subTest(target=target):
                link = self.a / "alias"
                link.symlink_to(target)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    t.inventory(self.a)
                link.unlink()
        root_link = self.root / "root-link"
        root_link.symlink_to(self.a, target_is_directory=True)
        with self.assertRaises(ValueError):
            t.inventory(root_link)
        with self.assertRaises(ValueError):
            t.inventory_forest({"one": self.a, "duplicate": self.a})

    def test_file_link_cycles_and_directory_ancestor_cycles_rejected(self):
        (self.a / "x").symlink_to("y")
        (self.a / "y").symlink_to("x")
        with self.assertRaises(ValueError):
            t.inventory(self.a)
        (self.a / "x").unlink()
        (self.a / "y").unlink()
        (self.a / "child").mkdir()
        (self.a / "child/back").symlink_to("..", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "cycle"):
            t.inventory(self.a)

    def test_cross_root_directory_cycle_rejected(self):
        (self.a / "to-b").symlink_to(self.b, target_is_directory=True)
        (self.b / "to-a").symlink_to(self.a, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "cycle"):
            t.inventory_forest({"a": self.a, "b": self.b})

    def test_excluded_bytecode_cannot_hide_other_payload_or_alias(self):
        cache = self.a / "__pycache__"
        cache.mkdir()
        (cache / "test.pyc").write_bytes(b"bytecode")
        (self.a / "other.pyc").write_bytes(b"bytecode")
        self.assertEqual(t.inventory(self.a), {})
        for name, make in (
            ("hidden.so", lambda p: p.write_bytes(b"native")),
            ("alias.pyc", lambda p: p.symlink_to(cache / "test.pyc")),
            ("nested", lambda p: p.mkdir()),
        ):
            with self.subTest(name=name):
                value = cache / name
                make(value)
                with self.assertRaises(ValueError):
                    t.inventory(self.a)
                value.rmdir() if value.is_dir() else value.unlink()
        for target in (cache, cache / "test.pyc", self.a / "other.pyc"):
            with self.subTest(target=target):
                link = self.a / "visible"
                link.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "excluded"):
                    t.inventory(self.a)
                link.unlink()
        (self.a / "alias.pyc").symlink_to("other.pyc")
        with self.assertRaisesRegex(ValueError, "excluded"):
            t.inventory(self.a)

    def test_special_nodes_and_invalid_limits_rejected(self):
        fifo = self.a / "pipe"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, "Special"):
            t.inventory(self.a)
        fifo.unlink()
        for key in ("max_entries", "max_unique_bytes", "max_depth"):
            for value in (0, -1, True, 1.5):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    t.inventory(self.a, **{key: value})

    def test_entry_unique_byte_and_depth_bounds(self):
        (self.a / "one").write_bytes(b"abcd")
        with self.assertRaisesRegex(ValueError, "entry bound"):
            t.inventory(self.a, max_entries=1)
        with self.assertRaisesRegex(ValueError, "Unique-byte"):
            t.inventory(self.a, max_unique_bytes=3)
        (self.a / "nested/deep").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "depth bound"):
            t.inventory(self.a, max_depth=1)

    def test_final_recheck_detects_file_change_after_cached_hash(self):
        first, last = self.a / "a", self.a / "z"
        first.write_bytes(b"old")
        last.write_bytes(b"last")
        original = t._Scanner.hash_file

        def mutate(scanner, path, before):
            row = original(scanner, path, before)
            if path == last:
                prior = first.stat()
                first.write_bytes(b"new")
                os.utime(first, ns=(prior.st_atime_ns, prior.st_mtime_ns))
            return row

        with (
            patch.object(t._Scanner, "hash_file", mutate),
            self.assertRaisesRegex(ValueError, "changed"),
        ):
            t.inventory(self.a)

    def test_final_recheck_detects_alias_retarget_and_directory_mutation(self):
        for kind in ("alias", "directory"):
            with self.subTest(kind=kind):
                (self.a / "payload").write_bytes(b"same")
                (self.a / "z").write_bytes(b"last")
                (self.a / "alias").symlink_to("payload")
                original = t._Scanner.hash_file

                def mutate(scanner, path, before):
                    row = original(scanner, path, before)
                    if path.name == "z":
                        if kind == "alias":
                            (self.a / "alias").unlink()
                            (self.a / "alias").symlink_to("z")
                        else:
                            (self.a / "added").write_bytes(b"new")
                    return row

                with (
                    patch.object(t._Scanner, "hash_file", mutate),
                    self.assertRaisesRegex(ValueError, "changed"),
                ):
                    t.inventory(self.a)
                for path in self.a.iterdir():
                    path.unlink()

    def test_mutation_while_reading_and_alias_explosion_are_rejected(self):
        (self.a / "payload").write_bytes(b"1234")
        original = t.os.fdopen

        class ChangedRead:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def fileno(self):
                return self.stream.fileno()

            def read(self, count):
                data = self.stream.read(count)
                if data:
                    (self.a / "payload").write_bytes(b"xxxx")
                return data

        owner = self

        def wrapped(fd, mode):
            obj = ChangedRead(original(fd, mode))
            obj.a = owner.a
            return obj

        with patch.object(t.os, "fdopen", wrapped), self.assertRaisesRegex(ValueError, "changed"):
            t.inventory(self.a)
        for index in range(5):
            (self.b / str(index)).symlink_to(self.a, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "entry bound"):
            t.inventory_forest({"a": self.a, "b": self.b}, max_entries=10)


if __name__ == "__main__":
    unittest.main()
