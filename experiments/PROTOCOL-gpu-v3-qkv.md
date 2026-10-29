# GPU v3 QKV development protocol

This phase optimizes the precision-protected QKV forward and input-gradient
contractions against the retained GPU v2 selected policies. It does not open
formal admission or authorize the five-engine campaign. The standard framework
baseline, numerical, repeatability, workload, and checkpoint requirements stay
unchanged. Admission stream 20261104 remains unopened.

The model, widths 32/64/128/256/512/1024, FP32 stored state, and original seed-42
initialization bytes are unchanged. Performance selection uses only the first
16 original training sequences: B16/T32 complete training steps and B1/T32
resident forward. Original validation/test arrays are not opened. The historical
GPU v2 source, policies, failures, measurements and releases remain separate.

## Fixed candidate inventory

The retained protected reference evaluates each dot product in ascending inner
dimension order using `ff_product` and `ff_add`, with the same bias initialization
and final FP32 rounding. The candidate retains those arithmetic operations and
their order exactly. Cooperative loads stage input tiles and padded/transposed
weight tiles. Each thread owns two rows and two or four output columns.

Each of forward and dX has exactly eight schedules: M={16,32}, N={32,64},
K={16,32}. Threads per group are 8M; outputs per thread are 2×(N/16).
Static threadgroup storage is `4*K*(M+N+1)` bytes, at most 12,416 bytes.
The same inventory applies to every width. dX is prospectively excluded from
inference. dW retains its selected v2 dense implementation. No new arithmetic
candidate, tile, or fusion may be added after performance scoring begins.

Schema-2 policies and their hashes remain unchanged. Schema 3 adds independent
QKV forward/dX schedules; null retains the protected reference. Old native ABI
entries retain their contracts; a new entry checks its 54-word control count.
For matched timing, the incumbent is lifted to schema 3 with both QKV schedules
null, so candidate and incumbent have the same pipeline inventory. Exact
forward, loss, gradient and first-Adam-state comparisons against the original
schema-2 policy must pass before scoring and again for the selected bundle.

## Qualification and selection

Commit the protocol, candidate ledger, source and tests before scoring. Record
source, binary, shader, compiler/device capabilities, runtime, fixture and
incumbent-policy hashes. Compile/resource checks for all applicable isolated
candidates precede timing. Composed policies receive another resource preflight
before their family is scored. Rejected candidates remain visible and unranked.
Run widths in the prospective order 512, 1024, 32, 64, 128, 256, with one fresh
process per width. A failed attempt is retained; any repaired attempt receives
a new directory and source freeze rather than overwriting observations.

Use the existing development stream 20261103. All 23 attention and 22 joint
LayerNorm/QKV stress families remain required at each width. The new direct-QKV
family uses SeedSequence family 53 and 16 declared cases: random rows
1,3,9,16,17,31,32,33,48,96,511,512; forward cancellation; dX cancellation;
large common bias; and alternating exponent products. Tests require independent
FP64 equations, oracle crosschecks, exact reference-versus-candidate output and
gradient bytes, input immutability, repeatability, finite values and guards.

Forward tolerances remain 2e-5/2e-4; gradients 3e-5/3e-3; probability/math
2e-7/2e-6; independent FP64 oracles 1e-10/1e-10. The original optimizer
tolerance and trajectory scopes remain unchanged. The full B1/T32 and B16/T32
logits, loss and every parameter gradient are checked before a candidate is
timed. Complete all-family qualification and combined-policy checks again after
selection, then rerun the declared selected-Metal repeatability subset in two
fresh processes at every width. These remain development checks, not admission.

Profile the v2 incumbent under the new build, separately from production timing.
Visit forward and dX once in descending measured stage cost, breaking ties by
family name. Start from the corresponding v2 selected training/inference policy.
Each correct candidate receives 20 warmups and 100 synchronized observations
in five alternating paired blocks. Restore exact state after warmup and before
each block; require exact within-engine block replay. Qualification instances
must be closed before fresh, allocation-matched timing instances are created.
Accept only a median paired complete-step/forward improvement of at least 3%;
choose the best qualified ratio and break ties by candidate ID. Retain the
incumbent otherwise. Inference is selected independently.

Measure final selected bundles directly against the original v2 incumbent with
fresh matching allocations. Report raw samples, p50/p95, five paired block
ratios, and separate intrusive stage profiles. Do not multiply intermediate
greedy ratios. Require AC power and reject known thermal throttling. Run one
numerical process at a time, with fixed host numerical threading; no concurrent
builds, validation, profiling or numerical project work during timing. Actual
hardware counters remain unavailable unless independently collected. Report
their absence rather than infer occupancy or bandwidth from latency.

## Delivery and formal-study boundary

Publish code, policies, all candidate outcomes, raw correctness/replay evidence,
timings, profiles, failures, source/runtime identities, static figures, and
archive verification. No speedup is required for completion. No GPU/framework
or CPU/GPU superiority claim follows from a custom-Metal development comparison.

Complete five-engine results still require eligible standard releases passing
the full baseline prerequisite, a new source/binary/policy/runtime freeze,
formal admission, 108 GPU plus 72 fresh CPU fits, and all-checkpoint five-engine
and FP64 audits. Local upstream patches are not substituted into that study.
