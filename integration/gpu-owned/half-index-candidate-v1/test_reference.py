import ctypes
import hashlib
import importlib.util
import itertools
import json
import random
import unittest
from pathlib import Path

PATH = Path(__file__).with_name("reference.py")
SPEC = importlib.util.spec_from_file_location("half_candidate_reference", PATH)
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


class CandidateDesignTests(unittest.TestCase):
    def test_scalar_abi_matches_declared_host_offsets(self):
        class Params(ctypes.Structure):
            _fields_ = [
                ("ndim", ctypes.c_uint32),
                ("nindices", ctypes.c_uint32),
                ("count", ctypes.c_uint32),
                ("merge_width", ctypes.c_uint32),
                ("chunk_begin", ctypes.c_uint64),
                ("output_base_offset", ctypes.c_uint64),
                ("output_storage_bytes", ctypes.c_uint64),
            ]

        self.assertEqual(ctypes.sizeof(Params), 40)
        self.assertEqual(ctypes.alignment(Params), 8)
        self.assertEqual(Params.chunk_begin.offset, 16)
        self.assertEqual(Params.output_base_offset.offset, 24)
        self.assertEqual(Params.output_storage_bytes.offset, 32)

    def test_prospective_patch_inventory_and_capability_scope_are_bound(self):
        here = Path(__file__).parent
        contract = json.loads((here / "contract.json").read_text())
        data = (here / "changes.patch").read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), contract["patch_sha256"])
        self.assertEqual(len(contract["source_files"]), 8)
        self.assertIn("torch/__init__.py", contract["source_files"])
        self.assertEqual(
            contract["strict_indirect_callers_retained_unsupported"], ["index_add_", "put_"]
        )
        self.assertFalse(contract["native_compiled"])
        self.assertFalse(contract["device_numerics_validated"])
        self.assertFalse(contract["formal_eligible"])

    def test_scratch_payload_bound(self):
        self.assertEqual(reference.MAX_SCRATCH_BYTES, 2 * 1024**2)

    def test_merge_ranks_match_independent_sort_for_tails_and_duplicate_keys(self):
        rng = random.Random(20261103)
        for count in (0, 1, 2, 3, 5, 17, 31, 32, 33, 127, 129):
            pairs = [(rng.choice([0, 2, 4, 1 << 42, reference.SENTINEL]), i) for i in range(count)]
            self.assertEqual(reference.merge_rank_sort(pairs), sorted(pairs))

    def test_single_index_value_strides_choose_cpu_traversal(self):
        self.assertEqual(reference.iterator_permutation((5, 4), [(0, 0), (8, 2), (32, 8)]), [1, 0])
        self.assertEqual(reference.iterator_permutation((5, 4), [(0, 0), (2, 10), (32, 8)]), [0, 1])

    def test_mixed_index_strides_must_not_be_normalized_before_ordering(self):
        shape = (2, 3)
        original = reference.iterator_permutation(shape, [(0, 0), (0, 0), (8, 16), (24, 8)])
        normalized = reference.iterator_permutation(shape, [(0, 0), (0, 0), (24, 8), (24, 8)])
        self.assertEqual(original, [0, 1])
        self.assertNotEqual(original, normalized)

    def test_global_ordinals_survive_reordering_and_large_values(self):
        shape = (1 << 32, 3)
        ordinal = (1 << 32) + 7
        self.assertEqual(reference.coordinates(shape, [0, 1], ordinal), [7, 1])

    def test_destination_keeps_64_bit_offsets_and_negative_aliases(self):
        base = (1 << 40) + 2
        args = (base, 0, [-1, 1], [4, 2], [4, 2], base + 20)
        value = reference.physical_destination(*args)
        self.assertEqual(value, base + 14)
        self.assertNotEqual(value, value & 0xFFFFFFFF)
        self.assertEqual(
            value, reference.physical_destination(base, 0, [3, -1], [4, 2], [4, 2], base + 20)
        )

    def test_invalid_indices_and_logical_tail_rejected(self):
        for indices in ([4], [-5]):
            with self.assertRaises(IndexError):
                reference.physical_destination(0, 0, indices, [4], [2], 8)
        self.assertEqual(reference.physical_destination(0, 0, [2], [3], [2], 6), 4)
        with self.assertRaises(IndexError):
            reference.physical_destination(0, 0, [3], [4], [2], 6)

    def test_word_ownership_preserves_guards_odd_tail_and_signed_zero(self):
        initial = {0: -0.0, 2: 3.0, 4: -2.0}
        updates = [(2, 0.25), (4, 0.5), (2, -0.25), (4, 0.125)]
        got = reference.candidate(initial, updates, 3)
        self.assertEqual(
            reference.byte_state(got), reference.byte_state(reference.serial(initial, updates))
        )
        self.assertEqual(reference.half_bytes(got[0]), b"\x00\x80")

    def test_empty_updates_do_not_touch_storage(self):
        initial = {2: -0.0}
        self.assertEqual(
            reference.byte_state(reference.candidate(initial, [])), reference.byte_state(initial)
        )

    def test_exhaustive_small_chunks_match_global_serial_rounding(self):
        initial = {0: 0.5, 2: -0.0, (1 << 40) + 4: -1.0}
        for destinations in itertools.product(initial, repeat=4):
            updates = list(zip(destinations, (2048.0, 1.0, -2048.0, 0.5), strict=True))
            expected = reference.byte_state(reference.serial(initial, updates))
            for chunk in (1, 2, 3, 4):
                self.assertEqual(
                    reference.byte_state(reference.candidate(initial, updates, chunk)), expected
                )

    def test_unstable_equal_key_order_is_detected(self):
        initial = {0: 0.0}
        updates = [(0, 2048.0), (0, 1.0), (0, -2048.0)]
        good = reference.candidate(initial, updates)
        reordered = reference.serial(initial, [updates[0], updates[2], updates[1]])
        self.assertNotEqual(reference.byte_state(good), reference.byte_state(reordered))

    def test_reversed_chunks_and_round_once_are_detected(self):
        initial = {0: 0.0}
        updates = [(0, 2048.0), (0, 1.0), (0, -2048.0)]
        good = reference.candidate(initial, updates, 2)
        wrong_chunks = reference.serial(initial, [updates[2], *updates[:2]])
        wrong_precision = {0: reference.half(sum(value for _, value in updates))}
        self.assertNotEqual(reference.byte_state(good), reference.byte_state(wrong_chunks))
        self.assertNotEqual(reference.byte_state(good), reference.byte_state(wrong_precision))

    def test_boundary_crossing_duplicate_runs_keep_prior_half_state(self):
        updates = [(0, 0.125), (2, -0.125)] * 65 + [(0, 2048.0), (0, 0.5), (0, -2048.0)]
        initial = {0: 1.0, 2: -0.5}
        self.assertEqual(
            reference.byte_state(reference.candidate(initial, updates, 64)),
            reference.byte_state(reference.serial(initial, updates)),
        )


if __name__ == "__main__":
    unittest.main()
