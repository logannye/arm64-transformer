# Half-precision indexing: compatibility and accuracy differ

The focused diagnostic finds an accumulation-order disagreement, not a new
deterministic-gradient pass. All observations use ordinary framework operations;
the harness temporarily disables deterministic algorithms to exercise the same
atomic path as the retained upstream test, then restores strict mode.

Two fresh processes each execute 40 cases: FP16/FP32, five incoming-gradient
patterns, row-major/column-major gradients, and contiguous/strided input views.
Each runs combinations backward, explicit-index backward and accumulating
`index_put_` 20 times on CPU and MPS. The test seed is development-only.

The [independent audit](independent-audit.json) rehashed all **80 raw archives**,
parsed **1,200 saved arrays**, and recomputed the scalar derivative, CPU's
per-addition rounded reference, forward outputs, repeat comparisons and original
CPU/MPS tolerances using only the Python standard library. All retained raw
arrays are byte-identical across the two fresh processes. Each process has:

- Zero observed repeated-execution differences in these small cases.
- Zero disagreements between CPU and the reconstructed serial CPU order.
- Exact agreement between the scalar FP64 derivative and saved CPU FP64 autograd.
- **120 CPU/MPS tolerance failures**, all FP16 cancellation with column-major
  incoming gradients: two input layouts × three operations × 20 executions.

The same cancellation values produce these gradients; input layout does not
change them:

| Stored dtype and incoming-gradient layout | Exact FP64 derivative | CPU result | MPS result |
|---|---|---|---|
| FP16, row-major | `[2049, -2046.75]` | `[2048, -2046]` | `[2048, -2046]` |
| FP16, column-major | `[2049, -2046.75]` | `[2046, -2047]` | `[2050, -2048]` |
| FP32, either layout | `[2049, -2046.75]` | `[2049, -2046.75]` | `[2049, -2046.75]` |

CPU and MPS both accumulate rounding error in FP16. In the column-major case,
CPU's coordinate errors are `[3, .25]` and MPS's are `[1, 1.25]`; neither result
is more accurate in both coordinates. Matching CPU's sequential half rounding
could improve compatibility without improving mathematical accuracy. A wider
accumulator could improve accuracy without matching CPU. Neither change can be
selected merely because it makes this example pass.

The source inspection confirms that this single-index case preserves the same
TensorIterator stride permutation on CPU and MPS. The column-major difference
therefore does not require an implicit contiguous copy: GPU atomic completion
order need not follow iterator thread indices. The repeatability observed here
does not make that parallel-atomic algorithm deterministic in general.

The original failing OpInfo cotangent is **not recoverable from the retained
receipts**. The earlier 81 state snapshots contain execution settings, not RNG
bytes or incoming gradients. The test does reseed each sample and creates its
cotangent with CPU double-precision `rand_like` before casting; that supports a
future source-based reconstruction, which must be labeled and verified as such.
This new diagnostic does not reproduce or clear the original sample-11 failure.

No half-precision kernel patch, tolerance waiver, formal fit, speed measurement
or admission pass follows. The broad framework suite remains blocking. See the
[implementation analysis](../../../integration/gpu-owned/MPS-REPAIR-NOTES.md)
for the general repair design and required regression coverage.

The [copy catalog](copy-catalog.json) binds byte-preserving copies of the frozen
[plan](plan.json), [execution summary](execution-summary.json), both worker
receipts, and audit/helper. Full case summaries and raw NPZ arrays remain in the
separate development evidence archive. The copied [audit helper](independent-audit.py.txt)
expects its original location `work/gpu-v4-owned/half-index-audit-v1.py` in the
reconstructed workspace and refuses to overwrite its report. This audit checks
saved evidence; it does not rerun GPU work or independently rehash the complete
native runtime. Those pre/postflight checks remain bound by the worker receipts.
