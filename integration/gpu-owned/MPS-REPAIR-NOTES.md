# Candidate repairs for the retained broad MPS failures

This is a source diagnosis and an **uncompiled candidate**, not a qualification
result. The earlier broad suite remains failed: 25 failures and two errors. All
27 IDs also fail in the separate matched pristine/patched control. The new owned
study keeps these failures blocking. No test is removed, skipped or relaxed.

The reviewed source is the retained formatted embedding-patch checkout on
PyTorch base `71d2a08507ab47dd6b2fa9b66319d17ce2377de4`. The device environment
under investigation is macOS 26.2, build 25C56, on the M4 Max. Assertions below
separate source facts from proposed explanations. Fresh device diagnostics and
a newly compiled framework are needed to establish a repair.

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
the cause of this exact retained mismatch. A focused diagnostic should retain
the full incoming cotangent, compare many exact repeats, reconstruct the CPU
iteration order and compare both engines with FP64 equations. A stable grouped
or serial single-writer reference is a repair candidate. Simply promoting to
FP32 does not guarantee agreement with CPU's sequential half-rounded answer;
the numerical and compatibility checks must remain separate.

The current layout patch **does not address this case**. Even if it fixes the
other 26 cases, it cannot establish a passing broad suite by itself. No new
low-precision accumulation implementation is selected in this note.

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

A small hosted framework patch is worth testing once the focused diagnostics
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
