# V5 numerical admission: failures, diagnosis, and revised contract

These findings were recorded before any matched v5 benchmark. The initial wider
implementation **failed the original admission contract**. The prospective revised
criteria below were developed after examining those failures. They have not yet
passed all required gates at the time of this record and must be frozen before
measurement. This changes the admission contract; it is not a retrospective pass.

| Initial candidate | Passed / total original checks | Failed |
|---|---:|---:|
| [Width 512, automatic SME](initial/width-512.json) | 1,105 / 1,108 | 3 |
| [Width 512, forced NEON](initial/width-512-neon.json) | 1,106 / 1,106 | 0 |
| [Width 1024, automatic SME](initial/width-1024.json) | 1,093 / 1,108 | 15 |
| [Width 1024, forced NEON](initial/width-1024-neon.json) | 1,100 / 1,106 | 6 |

Original sources, binaries, fixtures, failed checks, and diagnostic scripts are
preserved. No coordinate, fixture, backend failure, or unfavorable result is
discarded. These are synthetic correctness experiments, not language-model
quality or performance results.

## Why small gradient differences caused larger update differences

At width 512, all three failed assertions concerned one ordinary feed-forward
weight, `ff2_weight[2,378]`. It is not a redundant key-bias parameter. At identical
initial weights, its native and PyTorch gradients were respectively
`+2.94244e-9` and `−2.15871e-9`; a full float64 reference gave `+1.50717e-10`.
The derivative sums products whose absolute magnitudes total approximately
`0.00554`. Their near cancellation makes the small remainder sensitive to rounding.

Adam's first update is `−learning_rate * g / (abs(g) + epsilon)`.
With learning rate `0.001` and epsilon `1e-8`, the observed gradients produce a
parameter gap of `4.04892e-4`, exceeding the original allowance `2.44149e-4`.
The gap persists through subsequent steps. [The compact width-512 evidence](width-512-summary.json)
retains the exact values, common-operand reductions and optimizer replays.

Holding each implementation's captured operands fixed reproduced its gradient
using the same native sequential FMA routine. Cross-products identified upstream
ReLU activation rounding as the larger contributor; the active masks agreed.
Replaying each captured gradient sequence through both Adam implementations gave
bit-identical parameters across all three steps. Some second moments differed
by one FP32 ULP. This isolates the observed update discrepancy from an Adam
formula error; it does not prove every upstream kernel correct.

At width 1024, both backends first failed on the ordinary V-projection weight
`qkv_weight[2370,779]`. Its automatic-SME, NEON, PyTorch and float64 first
gradients were respectively `5.58804e-9`, `7.21327e-9`, `7.78788e-10` and
`4.67974e-9`. A small first-update difference preceded a broader divergence.
In the automatic backend's three-step numerical trajectory, differing
ReLU masks increased from zero to one to three. Four-update runtime traces also
showed later mask differences under both SME and NEON, and exactly reproduced
each implementation's saved checkpoint. Later cross-implementation gradients
were therefore being compared at different weights and activation branches.
The automatic backend had 1, 20 and 289 failing parameter coordinates over the
three numerical updates; NEON had 1, 2 and 12, with no numerical-trajectory mask
differences. In the separate four-update runtime smoke, final parameter failure
counts were 460 and 160, and failing test-logit counts were 63 and 46.
[The width-1024 summary](width-1024-summary.json) keeps each backend and workload
separate, with counts, first-state float64 comparisons and artifact hashes.

At either identical saved checkpoint, native and PyTorch logits passed the
original stricter forward tolerance: maximum differences were `3.5763e-6`
for automatic SME and `1.00136e-5` for NEON. Same-gradient Adam replays passed
the unchanged optimizer tolerance for every affected coordinate. Float64 helps
characterize this conditioning; it does not establish general framework instability.

## A general accuracy change, with remaining failures retained

The wider SME forward now uses four interleaved FP32 accumulator chains per
output and combines them as `(sum0 + sum1) + (sum2 + sum3)`, matching NEON's
reduction order. It computes a 32-by-32 output panel as four 16-by-16 subtiles,
using the four ZA tiles as partial-sum chains for each subtile in turn.
Packing bounds and backward arithmetic stay unchanged; additional operand loads
are an execution tradeoff, not an asserted speedup. Smaller compiled widths
retain their previous forward path.

The isolated candidate [matched NEON bitwise](striped/primitive-parity.json) in
81 declared synthetic shape/case combinations. [All 33 cases with input width at least 512](striped/fp64-accuracy.json)
improved both RMS and maximum error against float64 dot products. Its unchanged
numerical checks passed
1,035/1,035 at width 512. At width 1024, the three original parameter-trajectory
assertions still failed. Accuracy improvement did not establish the old contract.
These 1,035-check results exclude standalone runtime admission; the initial full gates above had
1,106 or 1,108 checks. [The candidate summary](striped/summary.json) retains the
remaining failures and points to the complete numerical records.

## Revised admission and its limits

The new contract retains the original free-running comparisons, thresholds, and
failure records as diagnostics. Admission instead requires common-state forward
and every-parameter gradient comparisons at **both implementations' evolving
states**, plus optimizer comparisons with identical incoming parameters, moments,
gradients, and step. Existing strict forward, gradient, and optimizer tolerances
remain unchanged. Runtime predictions must also be independently recomputed from
each implementation's own saved checkpoint; exact within-implementation repeats
remain required.

This tests local numerical and execution correctness without claiming that
independently rounded training trajectories remain close. Dataset, model,
float32 training, optimizer settings, and update budget are unchanged. All
required revised checks must actually pass before a matched run is admitted.

The [final admission record](final-admission.json) contains **21,734 passing
blocking checks** across all six widths and three backend policies. The wider
gates retain 438 original diagnostic comparisons separately, including **18 failed
comparisons** at width 1024 (six per policy). Width 512's retained diagnostics
pass after the accumulation change. These failures are excluded from passing
admission counts; width 1024 has not passed the old free-running contract.

The complete test suite passed 153 tests at each of the six widths. An independent
four-update runtime replay also reproduced each engine's own best/last weights,
first/second moments, and training/validation histories bit-for-bit. Every native
replay update passed same-input PyTorch Adam checks at the original optimizer
limits. Fault injection confirms that incorrect partial-batch normalization,
gradient accumulation, schedule, and Adam updates are rejected.

The four initial gate JSONs are byte-identical copies. [Original identity](initial/identity.json)
and [the provenance index](provenance.json) record SHA-256 identities and exact
raw-artifact paths. Full evidence, including coordinate tables omitted from Git,
is reserved for these release development extras:

- `development/numerical-diagnosis/`, from `work/v5-diagnosis/`.
- `development/sme-striped/`, from `work/v5-sme-striped/`.
- `development/initial-admission/`, from `work/v5/initial-admission/`.

The release archive is pending at this stage. Runtime diagnostic reports did not
independently capture import-time source hashes; the index states that limit.
Their recorded library identity matches the original snapshot, and their native
and Python final checkpoints each reproduce the saved original runs bit-for-bit.
The [v5 protocol](../PROTOCOL-v5.md) defines final admission requirements; this
report does not replace them.
