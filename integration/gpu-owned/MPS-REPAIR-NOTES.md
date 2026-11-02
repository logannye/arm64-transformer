# Candidate repairs for the retained broad MPS failures

This combines source diagnosis, fresh development controls and a **candidate
without a passing native validation receipt**, not a qualification result.
The earlier broad suite remains failed: 25 failures and two errors. All
27 IDs also fail in the separate matched pristine/patched control. The new owned
study keeps these failures blocking. No test is removed, skipped or relaxed.

The reviewed source is the retained formatted embedding-patch checkout on
PyTorch base `71d2a08507ab47dd6b2fa9b66319d17ce2377de4`. The device environment
under investigation is macOS 26.2, build 25C56, on the M4 Max. Assertions below
separate source facts from proposed explanations. The fresh device controls
below localize the failures; a rebuilt framework still has to establish a repair.

## Strided matrix outputs: an inconsistent version guard

`aten/src/ATen/native/mps/operations/Matmul.mm:588` chooses the Metal `mm` path
when the output is noncontiguous only on macOS older than 26.0. The equivalent
`bmm` guard at line 1127 has the same threshold. The retained regression test
comment at `test/test_mps.py:1984` states that MPSGraph miswrites noncontiguous
outputs before macOS 26.4. macOS 26.2 therefore takes a path the regression test
itself describes as affected.

The retained failures comprise five strided `mm` cases, six strided `bmm` cases
and one GEMV case whose non-unit output stride sends it through the `mm`
fallback. Their large discrepancies and NaNs are incompatible with an
explanation limited to ordinary floating-point reduction drift.

The [candidate patch](patches/mps-layout-candidate.patch) changes both guards
from 26.0 to 26.4 and leaves all other dispatch conditions intact. It selects
existing stride-aware GPU kernels. It does not force Metal globally, alter
contiguous-output dispatch, or introduce CPU fallback. The source inconsistency
strongly motivates this change; the patch still needs compiled device tests.

## Fourteen convolution failures may share one forward path

Twelve failed channels-last-3D tests first compare contiguous and channels-last
outputs at `test/test_mps.py:11559`. They cover small/production 3x3x3 kernels,
three dtypes and bias on/off. The comparison fails before the CPU reference
assertion, so those old logs alone do not establish which output is wrong.

The source provides a useful dispatch distinction:

- `Convolution.mm:875` routes 3D forward into pointwise, im2col or direct Metal.
- For the failing cases, direct Metal can select MPP on macOS 26.2 through
  `has_mpp()` and the bundled catalog.
- `Convolution.metal:815–874` instantiates the relevant MPP catalog only for
  NCDHW outputs. An NDHWC name is absent, so a channels-last output uses the
  existing SIMD-group fallback.
- The factorized kernel-depth-one and grouped controls pass in the historical
  suite. The grouped 3x3x3 shape is absent from the MPP catalog and falls back.
- Both failed transposed-convolution tests fail on the input gradient. That
  gradient calls `mps_convolution_transpose_backward_input`, which calls
  `_mps_convolution_impl` at `Convolution.mm:1441`. It can therefore exercise
  the same direct 3D forward path; a separate transposed-backward defect is not
  established by the test names.

This makes multi-depth MPP convolution the strongest shared hypothesis for
these fourteen failures. MPP's kernel-depth loop calls `op.run` repeatedly into
one cooperative accumulator. The available evidence does not distinguish
operand interpretation, accumulation, compiler behavior or runtime support as
the underlying cause. It does not establish that any later macOS fixes it.

The candidate restricts MPP selection for kernel depth greater than one on the
**macOS 26.2 major/minor family only**, selecting the existing GPU SIMD kernel
there. It queries `NSProcessInfo` because this source has no `MACOS_26_3` enum.
Other OS families and kernel-depth-one cases retain their prior dispatch. This
is a conservative candidate workaround for the reproduced environment, not a
claim of hardware-wide or later-OS correctness.

A useful development control is a fresh process with
`PYTORCH_MPS_METAL_VERSION=3.2`: the retained `has_mpp()` implementation then
returns false. Comparing the same CPU, contiguous-MPS and channels-last-MPS
operands under ordinary versus MPP-disabled execution can localize the problem
without rebuilding. Such a process must be labeled a diagnostic configuration;
it cannot silently replace the declared study runtime or qualify this patch.

## Fresh layout controls and independent saved-array audit

The closed `layout-diagnostic-v1` run used the retained embedding-patched
framework, two ordinary fresh processes and two Metal 3.2 diagnostic processes.
Each process ran 76 matrix cases and six Conv3d cases using development seed
`20261103`. All four execution-integrity/postflight checks passed; no admission
operands or formal fits were used.

| Check | Ordinary, each process | Metal 3.2 diagnostic, each process |
|---|---:|---:|
| Strided matrix output failures / cases | 12/76 | 12/76 |
| Contiguous GPU temporary then strided copy failures / cases | 0/76 | 0/76 |
| Matrix output guard failures / cases | 7/76 | 7/76 |
| NCDHW Conv3d failures against saved CPU FP64 / cases | 6/6 | 0/6 |
| Channels-last Conv3d failures against saved CPU FP64 / cases | 0/6 | 0/6 |

Thus the channels-last/contiguous discrepancy comes from the **contiguous**
output in these controlled shapes. Disabling MPP selection removes the six
observed convolution failures but leaves all matrix failures and guard failures
unchanged. The contiguous-GPU-temporary control preserves both the matrix
answer and output guards. Together these results support the two separate
dispatch changes; they do not validate the candidate compiled implementation or
establish whether the MPP defect lies in source arithmetic, compilation or the
OS runtime. The transposed-convolution backward hypothesis still needs its
original failing tests rerun on the rebuilt candidate.

The independent NumPy-only auditor authenticated all 328 NPZ archives,
recomputed all 680 recorded comparisons, checked guards and recorded layout and
dtype metadata, and verified that the 82 operand/reference sets were byte-equal
across all four workers. It checks saved logical array values; physical GPU
strides and execution provenance are retained metadata, not independently
measured again. Matrix references are same-dtype PyTorch CPU, not independent
FP64. Convolution references are saved PyTorch CPU FP64 results, not an
independent equation newly evaluated by the auditor.

The diagnostic tolerances remain explicit in every case: matrix FP32 absolute
and relative `1e-4`, FP16/BF16 `1e-2`; convolution FP32 `1e-5`/`1e-5`, FP16
`1e-3`/`1e-3`, BF16 `1e-3`/`1.6e-2`. They do not replace the study's unchanged
admission tolerances. See the [figure and complete evidence](../../experiments/gpu-v4-owned/README.md#what-the-independent-development-checks-found),
[raw-array audit](../../experiments/gpu-v4-owned/layout-report-v1/array-audit.json),
and [receipt catalog](../../experiments/gpu-v4-owned/layout-report-v1/receipt-catalog.json).
The original layout summary SHA-256 is
`08f98e4ea6adf7b1257d6dcf598205fd0414333d5f286c3177f64b6f81ad6f48`.

## Half-precision combinations: a separate accumulation problem

The remaining failure is
`TestConsistencyMPS.test_output_grad_match_combinations_mps_float16`, sample 11:
two input elements, `r=4`, `with_replacement=True`. It differs by 0.0078125 in
one of two gradients, exceeding the unchanged relative limit of .001.

The source path is `combinations` in `Itertools.cpp`, indexed gather, autograd
`index_backward`, and `_index_put_impl_(accumulate=true)`. MPS dispatches
`index_put_accumulate` through `AtomicType<half>::atomic_add`; its CAS loop rounds
each update into a half value. CPU uses serial half additions in
`native/cpu/IndexKernel.cpp:166–190`. Duplicate contributions therefore have a
defined serial order on CPU and a schedule-dependent atomic order on MPS.

That is a concrete source-level order/rounding difference, but it does not prove
the cause of this exact retained mismatch. The new
[focused diagnostic and independent raw-array audit](../../experiments/gpu-v4-owned/half-report-v1/README.md)
retain 40 cases in each of two fresh processes, with 20 repeats of three ordinary
operations per device. All 80 raw archives authenticate; all saved arrays agree
exactly between processes. CPU agrees with the reconstructed per-addition
rounded iterator order. No within-process difference is observed in these small
cases; that does not establish general determinism for atomic accumulation.

The cancellation example makes the distinction concrete:

| FP16 incoming-gradient layout | Exact FP64 derivative | CPU | MPS |
|---|---|---|---|
| Row-major | `[2049, -2046.75]` | `[2048, -2046]` | `[2048, -2046]` |
| Column-major | `[2049, -2046.75]` | `[2046, -2047]` | `[2050, -2048]` |

FP32 controls equal the exact derivative in both layouts. The column-major FP16
cases fail the original CPU/MPS tolerance on both input layouts and all three
operations: 120 failed comparisons per process. Both engines have half-rounding
error, and neither dominates both coordinates. Serializing to CPU order may
improve compatibility while worsening one coordinate's mathematical accuracy;
promoting accumulation may improve accuracy without reproducing CPU's answer.
The two requirements must remain separate.

For this one-index example, `AdvancedIndex` does not perform its GPU-specific
index-contiguity conversion: that conversion requires at least two index
tensors with differing strides. `make_index_put_iterator` adds the restrided
zero-stride destination, then incoming values, then indices. CPU and MPS share
that iterator construction. The first nonzero value strides select row-major
or column-major order; Metal's `pos_from_thread_index` uses the same fastest
dimension first. The parallel atomic kernel does not promise that thread-index
order is the order in which duplicate additions complete. A contiguous-copy
explanation is therefore unnecessary for the observed difference.

A general compatible repair would compute each update's physical destination
and original CPU-iterator ordinal, stable-sort by destination while preserving
that ordinal, and assign one writer to each group. The writer starts from the
existing destination value and rounds after each addition in the stored dtype.
Parallelism remains across independent destinations. For multiple differently
strided indices, the CPU permutation must be retained before GPU-specific
contiguous normalization; sorting by logical row order alone is insufficient.
Packed-half storage, odd offsets, negative aliases, broadcast values, duplicate
destinations across partitions and asynchronous bounds errors require explicit
coverage. The existing recursive 32-bit dispatch also drops its `serial` flag;
a serial candidate must preserve both that flag and global update order across
partitions. These are design requirements, not a selected implementation.

Regressions must cover FP16/BF16/FP32, positive and cancellation-heavy values,
initial nonzero destinations, single/multiple indices, both value layouts,
strided/offset outputs, negative and invalid indices, empty/tail cases, duplicate
runs across tile boundaries, packed-half neighbors, and same/fresh-process
repeats. Compare mathematical accuracy with independent FP64 equations separately
from CPU compatibility, retain the unchanged original broad suite, and measure
the complete cost of sorting and hot-destination serialization. A special branch
for this test shape or relaxed tolerance would not establish a general repair.

The original failing cotangent is not recoverable from retained evidence. The
81 historical state snapshots record deterministic/autocast/default settings,
thread counts and operator libraries, not RNG state or gradient arrays. The
upstream sample iterator reseeds by test-function identity and the test draws
CPU FP64 `rand_like` cotangents before conversion, so a source-based
reconstruction is possible in principle. It would need independent verification
and must not be represented as the original saved tensor. This diagnostic uses
new declared cotangents and does not clear the historical failure.

The current layout patch **does not address this case**. Even if it fixes the
other 26 cases, it cannot establish a passing broad suite by itself. No new
low-precision accumulation implementation is selected in this note.

An [uncompiled general FP16 candidate](half-index-candidate-v1/README.md) now
implements that design prospectively: preserved CPU iterator ordering, full
64-bit destination/ordinal keys, consecutive global chunks, stable integer
sorting and one writer per packed storage word. It uses at most 2 MiB of numeric
scratch. Explicit guards preserve strict `index_add_` and `put_` rejection and
warn-only diagnostics; direct `index_put_` support does not qualify those separate
public operations. The eight-file patch applies to an isolated source copy,
15 scalar/ABI/negative controls pass, and original test-method names are retained
with four new native methods. Native compilation, real-device tests, the complete
broad suite and performance remain unvalidated. These static results do not
clear any gate or replace a framework runtime.

## Validation before adopting any repair

The static review applies the patch to an isolated two-file fixture and compares
the resulting bytes with the intended changes. It verifies the original source
files remain unchanged. See [the review receipt](patches/mps-layout-candidate-review.json).
This establishes patch application only; native compilation and numerical
validation are explicitly false in that receipt.

Before selecting the repaired runtime:

1. Build the exact combined embedding and candidate patch state in the authorized
   hosted environment, retaining the old runtime and source evidence.
2. Rerun all 27 original failing IDs unchanged. Include successful neighboring
   shapes, dtypes, layouts, offsets, grouped/factorized/depthwise convolution,
   kernel-depth-one cases and transposed-convolution gradients.
3. Run the complete original broad suite with its existing eligibility rules,
   plus targeted embedding and full matched-transformer accuracy/repeatability
   inventories. Preserve all failures and interrupted attempts.
4. Revalidate PyTorch CPU and MPS together under that exact distribution, then
   complete the new study's remaining prerequisite and admission checks.

A small hosted framework patch is worth testing: the focused diagnostics now
support these dispatch explanations. It has a narrow change surface and reuses
existing GPU implementations. It has no demonstrated speed benefit and must
not be presented as validated before the rebuilt device checks pass.

## Source identities

All paths below are relative to the pinned PyTorch source root. Hashes were read
from the retained checkout; this table is not a substitute for its complete
source/runtime authentication.

| Path | SHA-256 |
|---|---|
| `aten/src/ATen/native/mps/operations/Matmul.mm` | `c0446d315b6914e9f5bca70339c81f4c267ed5c3c6ffca536e2c287efc015570` |
| `aten/src/ATen/native/mps/operations/Convolution.mm` | `95749bfffbad7687070b02a448c9b5b9a033b42c691f62a46aa28dba36f1b960` |
| `aten/src/ATen/native/mps/kernels/Convolution.metal` | `29ac7ee39591885e48eceff4ae5682ea6b6c58676208a48d40446ee7dc8c94c3` |
| `aten/src/ATen/native/mps/operations/Indexing.mm` | `ae9ce032e96856e1161f140829e06986ef7776015f3105c6685b8d2bfd411b15` |
| `aten/src/ATen/native/mps/kernels/Indexing.metal` | `283362996d9c86188510dc5be2e522de9bc59ebea448ecff5b46bd2c4233ff58` |
| `aten/src/ATen/native/cpu/IndexKernel.cpp` | `e41bf8bd9c66e7fe65b84886c54af82bea79295bc12fb9cfa6d381073b22b31a` |
| `c10/metal/atomic.h` | `ae4c8a6d92d314178b3046f8e66d333d267121987354be39a02efea68fa0067b` |
| `test/test_mps.py` | `7345b6e229aed240921460a3b598ffd1a47148ab4816aead0d405539a154a396` |

The retained `failure-replay-v2/failure-localization.json` has SHA-256
`9de50394b76b7bb2622c3b848a1dd6512f8f855190c1be2d688f9341edbe63d0`.
The layout candidate patch has SHA-256
`4e9c501353522821773cf8eafc92386b142ba24edca39a4d5a4c61cb68bc59a7`.
