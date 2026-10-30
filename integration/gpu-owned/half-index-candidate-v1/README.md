# Prospective FP16 indexed-accumulation candidate

**Uncompiled and unqualified.** This is a source patch and scalar design model,
not a selected framework, backend override, completed repair or study admission.
The existing MPS broad-suite failure remains a failure. No second hosted build
is launched by these tools.

The purpose is to make general MPS FP16 `index_put_(accumulate=True)` follow the
CPU TensorIterator's per-addition half rounding. It applies to all supported
shapes in ordinary and deterministic modes. It has no branch for combinations,
two-element inputs, or the retained failing sample. Better CPU compatibility is
not a claim of better FP64 accuracy; the [diagnostic](../../../experiments/gpu-v4-owned/half-report-v1/README.md)
shows why those are different questions.

The native candidate has four stages:

1. Preserve original index strides before GPU-specific normalization. A new
   opt-in overload leaves the original two-argument constructor/helper and every
   other dtype/device path unchanged. TensorIterator then selects the same
   stride-based traversal used by the CPU path.
2. For each consecutive chunk of at most 65,536 global iterator ordinals,
   compute `(64-bit storage byte offset, 64-bit original ordinal)` pairs on GPU.
   Each index has its own byte strides. Negative indices are normalized only
   after bounds checks. Invalid updates retain a sentinel and use the existing
   asynchronous error buffer; they do not access output storage.
3. Stable-sort the pairs with parallel integer merge-rank passes. Two reusable
   buffers hold at most **2 MiB of numeric scratch**, independent of the total
   update count. This excludes driver command-buffer metadata and input/output
   storage. The implementation does not use `with_32bit_indexing()` partitions,
   whose traversal does not define the required global order.
4. Give one writer each aligned four-byte storage word. It processes only the
   one or two half lanes with updates, in their original per-destination order,
   starting from each lane's existing value and explicitly rounding to half
   after each addition. Typed half accesses avoid explicit four-byte reads past
   an odd logical storage tail; untouched lanes retain their original bits.
   Explicit GPU buffer barriers order merge passes and successive chunks.

Word ownership removes races between adjacent packed halves without atomic
arithmetic. Serializing successive chunks retains update order even when a
duplicate destination spans chunks. Full-width storage offsets support views
whose offsets exceed 32 bits; no new limit is imposed on total iterator elements.
These are source-design properties awaiting native validation, not device claims.

The tradeoff is deliberate: up to 16 merge passes per full chunk, additional
integer searches, scratch traffic, and serial processing of hot destinations.
Word ownership offers at most half as many independent owners as half-element
ownership. The simple merge-rank implementation is not a performance-selected
sorting kernel. Its actual complete-operation cost, memory usage, driver command
growth and GPU watchdog behavior require measurements before adoption.

The [contract](contract.json) binds eight source files and the exact patch.
FP32, BF16, complex and integer accumulation keep their existing paths, as does
non-accumulating indexing. Explicit guards retain the previous strict-mode
rejection and warn-only warning for FP16 `index_add_` and accumulating `put_`;
their delegation to `index_put_` must not silently grant a new public capability.
In particular, `index_add_` rejects negative indices that direct `index_put_`
accepts, and its non-unit-alpha arithmetic needs separate qualification. The
existing parameterized nondeterminism test retains all IDs and rejection checks
for FP32/BF16; its FP16 direct-index-put branch now checks exact results and
repeats, while strict index-add/put rejection and warn-only diagnostics remain
asserted. Public determinism documentation changes only the FP16 index-put case.
The previously observed loss of the `serial` argument
in recursive 32-bit dispatch is a separate unchanged bug; this new path never
calls that dispatcher. This patch can follow the retained formatted embedding
patch and the disjoint layout repair, but that combined state needs a new build
identity and validation.

The scalar tests check merge ranks against independent Python sorting; mixed
index strides and CPU traversal; negative aliases; offsets/ordinals beyond
32 bits; odd tails and untouched negative zeros; duplicate runs spanning chunks;
and mutations that reverse chunks, reorder equal keys, or round only once.
They do not execute a GPU kernel. Host static assertions and the scalar ABI
control require a 40-byte, eight-byte-aligned parameter structure with 64-bit
fields at offsets 16, 24 and 32; the Metal ABI remains a native validation gate.
Four proposed native regression methods cover
chunk boundaries, repeated backward indexing, mismatched index strides,
broadcast values, subnormals, storage offsets and guard elements. They have not
run. True large-offset device tests, invalid-index asynchronous-error tests,
nonfinite inputs, fresh-process repetition, the unchanged broad suite, model
gates and matched performance/memory measurements remain required.

Static reproduction uses the retained formatted source and a fresh small copy:

```sh
python3 -I -B -m unittest discover \
  -s integration/gpu-owned/half-index-candidate-v1 -p 'test_reference.py'
python3 -I -B integration/gpu-owned/half-index-candidate-v1/verify_patch.py \
  --source /ABSOLUTE/WORKSPACE/work/upstream-integration-v2/pytorch/formatted-source-v2 \
  --output /ABSOLUTE/WORKSPACE/work/gpu-v4-owned/half-candidate-static-fresh
```

The verification tool copies only the eight bound files, checks and applies the
patch there, verifies resulting hashes, and checks the originals remain unchanged.
It does not install, compile, import or replace a framework. Metal compilation,
the host/shader ABI and actual half-rounding behavior remain unverified until a
new isolated full-framework build and serial device checks succeed.
