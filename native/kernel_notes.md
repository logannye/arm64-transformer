# Numerical implementation

The `.S` files contain handwritten Darwin AArch64 implementations of every
numerical primitive declared under **Numerical primitives** in `ABI.md`.
There are no calls to C numerical routines, BLAS, Accelerate, PyTorch, or libm.
Cross-entropy and attention call the handwritten scalar/vector exponentials;
cross-entropy also calls the handwritten logarithm.
Model buffers and tensor dimensions come from the caller. Numerical kernels use
registers and private stack scratch, with no heap allocation or shared numerical
scratch. `backend.S` separately owns an immutable initialized policy and atomic,
mutable diagnostic counters. Its system calls detect capabilities and read the
policy; they do not compute model values. The optional ABI diagnostic allocates
its own temporary test buffers, outside ordinary model execution.

## Layout and caller requirements

All arrays are contiguous row-major float32; shapes and targets are int32.
The row and feature dimensions must be positive. Attention additionally requires
`width % heads == 0` and positive heads. Linear, LayerNorm, and attention support
non-multiple-of-four feature dimensions. Linear SIMD loops include scalar tails.
ReLU and Adam also support zero-length arrays and arbitrary scalar tails.
Memory arguments must refer to correctly sized, validated buffers; these low-level
kernels do not perform runtime bounds or finiteness checks. Distinct input and
output buffers are the contract, except that ReLU supports its natural in-place
destination. `dw/db`, `dgamma/dbeta`, and Adam state must be initialized by the
caller before their first use.

Darwin callee-saved x19-x28 and the low 64 bits of v8-v15 are saved/restored where
used. The reserved x18 register is never used. Stack frames are 16-byte aligned.
All public interfaces require non-streaming entry, PSTATE.SM=0. See the backend
and ZA-state contract in `ABI.md`; no streaming-compatible public ABI is provided.

## SME dispatch and matrix execution

The library constructor initializes `ARM64_TRANSFORMER_BACKEND` once: `auto`
selects the available backend, `neon` forces NEON, and `sme` requires SME
capability. Invalid policy or unavailable forced SME returns an initialization
error. Changing the environment afterward does not change a loaded library's
policy. Capability detection checks Darwin's FEAT_SME and FEAT_SME2 selectors
before any SME instruction, then requires RDSVL to report 64 bytes. At this
streaming vector length, each FP32 ZA tile is 16 by 16. The implementation uses
non-widening FP32 FMOPA; it does not convert operands to lower precision.

The public linear dispatchers require 32 rows, input width
64/128/256/512/1024/2048, and output width divisible by 32 between 64 and 3072
for the SME path. They also
require the current vector length to remain 64 bytes, ZA to be off, and
TPIDR2_EL0 to be null. Under the public SM=0 precondition, live ZA or pending
lazy-save state takes the NEON path without changing that state. Unsupported
shapes also take NEON, whose existing generic kernels cover scalar tails.
Width 32 therefore stays on NEON; all five linear layers at widths 64, 128,
256, 512, and 1024 are eligible for SME. Selection and actual dispatch counts are recorded
separately, so selecting SME does not imply every primitive used it.

`linear_sme.S` first packs the 32 input rows into contiguous vectors along the
reduction dimension. It then packs one 32-output weight panel at a time, reusing
the packed input for every panel. Horizontal ZA loads followed by vertical ZA
stores perform these transposes. Bias is added after the dot products, when
storing the result.

For compiled widths 32 through 256, the existing forward holds a 32-by-32 result
in four FP32 ZA tiles. Each reduction step loads two 16-lane vectors from each
operand and issues four outer products. Those compiled paths retain their
previous numerical order.

For compiled widths 512 and 1024, forward divides each 32-by-32 output panel into
four 16-by-16 subtiles. Within each subtile, ZA0 through ZA3 accumulate input
indices modulo four, giving each output four independent FP32 partial sums.
The final combination is `(sum0 + sum1) + (sum2 + sum3)`, then bias, matching
the NEON forward's four-lane reduction order on the validated finite inputs.
This shortens each accumulation chain by four. The same number of FMOPA
instructions performs the matrix products, but visiting four subtiles requires
additional operand loads and partial-sum additions. Packing and scratch bounds
are unchanged. This is a numerical-accuracy tradeoff, not a claimed speedup.
Selection depends on the compiled width, never on fixture contents or parameter
values. Backward retains its existing reduction order at every width.

Backward packs dY once for dX, then streams contiguous weight-row segments while
overwriting each dX tile. For dW it loads the caller's existing gradient tile into
ZA and accumulates outer products of contiguous dY and input rows. Bias gradients
are added in token-row order. dX reduction visits output channels in order; dW
visits token rows in order. This preserves the contract of overwriting dX while
adding to dW/db across the runtime's sequence-by-sequence batch accumulation.

Forward scratch is `256 * input_width` bytes, at most **512 KiB**, plus a
160-byte saved-register frame. Backward scratch is `128 * output_width` bytes,
at most **384 KiB**, plus a 192-byte frame. The validated shapes make every
scratch allocation a positive multiple of 4 KiB. The assembly lowers the stack
and touches memory at each 4 KiB step, then restores it exactly on return.
The caller must supply enough thread stack for that scratch, saved registers,
and its own active frames; a 512 KiB thread stack cannot run the largest forward
shape. The standalone executable uses the main thread. Page touching respects
stack guards; it does not create additional stack capacity.
Packing is repeated inside each call; there is no persistent packed-weight cache
to invalidate after Adam updates. Packing, page touches, and the single
SMSTART/SMSTOP pair per call are included in measured execution time.

The symbol `tf_linear_pack32_sme` names the transpose helper for sampled profiling.
It is an internal streaming entry, not a normal public function: it requires
SM=1, ZA enabled, a 64-byte streaming vector length, and p0.s all true. Its private
arguments are source, destination, source stride in bytes, and column count in
x0-x3; it packs exactly 32 source rows, with columns a positive multiple of 16.
It clobbers scratch GPRs and ZA0/ZA1. Public kernels establish these conditions
and save d8-d15 before switching modes; d8-d15 are restored after SMSTOP.

These rules follow the distinctions between ordinary and streaming interfaces
and between active/dormant ZA state in the
[Arm procedure-call standard](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)
and [LLVM's SME ABI documentation](https://www.llvm.org/docs/AArch64SME.html).

## Tiled and vectorized execution

- `linear.S` computes four token rows by four output columns together, using
  sixteen independent NEON dot-product accumulators. Loaded input/weight vectors
  contribute to multiple outputs. Its backward kernels retain four-by-sixteen
  gradient tiles in registers, reducing repeated loads and stores of partial
  sums. dW uses scalar-lane FMLAs directly from each loaded four-channel dY
  vector, avoiding four separate broadcasts. dX loads four consecutive
  coefficients for each token row and unrolls four output-channel steps, using
  each lane in its original order. Its per-accumulator FMA sequence is unchanged.
  Forward requires row/input/output counts divisible by four; backward
  additionally requires input width divisible by sixteen. Other positive shapes
  tail-call the preserved v1 generic implementations in `kernels.S`.
- `attention.S` specializes four-head models of width 32, 64, 128, 256, 512,
  and 1024 for 1..32 tokens. Heads of 8, 16, 32, or 64 elements retain their
  existing paths with 2, 4, 8, or 16 NEON
  vectors. QK, weighted values, and their derivatives are vectorized. Wider
  paths retain query/output-gradient vectors in registers, use four independent
  dot-product accumulators, and unroll the feature loops. Backward caches one
  row of attention derivatives in a 32-float stack buffer instead of recomputing
  those dot products. Heads of 128 or 256 elements use register blocks: forward
  retains 32 query channels converted to FP64 across causal keys, accumulates
  four FP64 dot-product chains, and adds each block's result to a private
  32-double score row. Scale, maximum selection and subtraction remain FP64.
  Only the centered scores are converted to FP32 before the existing vector
  exponential, probability normalization and 64-channel value accumulation.
  Conversion after centering avoids discarding the small differences between
  large, nearly tied scores. Q/K conversion uses ordinary NEON FP64 instructions,
  requiring no optional SME FP64 capability. This trades additional conversion
  and FP64 arithmetic for score accuracy; elapsed cost must be measured.
  Backward computes FP64 dA dot products alongside unchanged FP32 dV updates
  in 64-channel blocks. It reuses a 32-double row for FP64 dScore after the FP64
  softmax contraction, retaining the supplied probabilities exactly. Blocks of
  16 converted Q channels and 16 FP64 dQ accumulators keep the complete causal
  dQ sum in registers until its final float32 store. Each dK contribution is
  added in FP64 before rounding to its existing float32 destination. This
  addresses cancellation in gradients when keys share a large component without
  renormalizing probabilities, clipping gradients or changing the model.
  These larger-head paths use 400-byte forward and 384-byte backward frames;
  the paths for widths 32 through 256 are unchanged.
  Other shapes use the original generic implementation.
- `softmax.S` evaluates four exponentials at a time, using pairs of float64 lanes
  for range reduction and a degree-nine polynomial, then rounding to float32.
  Inputs outside the finite interval [-104,88] use the original scalar behavior.
  Attention tails mask inactive lanes and avoid out-of-bounds reads. The
  ASCII-128 cross-entropy path vectorizes maximum, exponential, and gradient
  calculations; other vocabulary sizes use the generic path.
- `normalization.S` vectorizes the two-pass mean/variance calculation and
  normalization derivatives for widths divisible by four. Scalar-width tails
  use the original implementation. The two-pass variance avoids the cancellation
  risk of computing variance as mean-square minus squared-mean.
- `model.S` checks finiteness four values at a time with integer exponent-bit
  operations and a scalar remainder. The same checks remain in the runtime.

The native runtime still accumulates complete per-sequence gradients within each
training batch. The model workspace holds one sequence, supplemented by private
per-call SME packing scratch. Batch-wide execution remains a separate measured
question.

## Width specialization and identity

`ARM64_TRANSFORMER_WIDTH` selects 32, 64, 128, 256, 512, or 1024 before Python imports the
model specification. Four heads, a 32-token context, and a feed-forward width of
twice the model width remain fixed. `make widths` builds separate native binaries
under `native/build/width-W`; generated `layout.inc` files contain parameter and
workspace offsets. The arithmetic kernels remain handwritten assembly.

Model code derives embedding strides, loop bounds, parameter counts, and buffer
sizes from that layout. Each library exports its width and parameter count; the
Python validator checks both before allocating or calling through the ABI. The
standalone executable requires a version-2 fixture header with the matching width
before it allocates model state or creates an output directory. This prevents
accidentally loading a wider fixture into a smaller compiled model.

At width 1024, some parameter byte offsets exceed the 12-bit page immediate of
one AArch64 shifted ADD. The address macros split those offsets into bounded
page additions; offsets below that boundary retain their original two-ADD
sequence. Parameter counts and byte sizes use two 16-bit immediate pieces and
remain below 2^32 for every supported model. Embedding row offsets use 64-bit
register shifts derived from `WIDTH_SHIFT`, including the 4096-byte rows at
width 1024. Loop counters and pointer arithmetic remain 64-bit where their
range depends on tensor size.

## Arithmetic

- NEON linear forward uses four-lane fused multiply-add and horizontal reduction,
  then bias addition. Its tiled and generic paths preserve the same per-output
  arithmetic order. At compiled widths up to 256, SME forward accumulates one
  FP32 fused multiply-add per reduction element into ZA, changing the grouping
  relative to NEON's four partial sums. At compiled widths 512 and 1024, SME
  forward uses the same four partial-sum chains and combination order as NEON.
  SME and NEON backward visit the same output-channel/token-row
  reduction orders. Every backend overwrites input gradients and adds to the
  caller's parameter/bias gradients.
- LayerNorm uses float32 two-pass mean and biased variance, epsilon `1e-5`, and
  scalar or vector arithmetic. It retains normalized inputs and inverse standard deviations.
  Backward applies `inv * (dy*gamma - mean(dy*gamma) - norm*mean(dy*gamma*norm))`.
  Scale and bias gradients are added to the caller's accumulators.
- Attention stores Q, K, and V consecutively in each token row. Each query attends
  only to keys through its own position. Masked probabilities are explicitly zero.
  Scores use `1/sqrt(head_width)`, max-subtracted softmax, and fixed-order reductions.
  The optimized 128/256-channel heads keep scores in FP64 through centering;
  stored tensors, probabilities and value mixing remain FP32. The corresponding
  backward keeps dA, its softmax contraction, dScore and dQ sums in FP64 before
  rounding output gradients to their float32 destinations.
  The implementation therefore uses mixed accumulation precision, while preserving
  the model's real-valued equations and float32 parameter/activation interfaces.
  Backward overwrites all Q/K/V gradients, computes the softmax row contraction,
  and accumulates Q/K/V contributions. The specialized path caches one row;
  the generic path recomputes it. No quadratic backward tensor is allocated.
- ReLU uses NEON comparisons and masked gradients. Its derivative at zero is zero.
- Cross-entropy uses `log(sum(exp(logits-max))) - (target_logit-max)`, preserving
  finite losses even when the target probability underflows. Returned gradients
  are divided by the number of token rows. Individual token losses are float32;
  their reporting sum uses float64 before conversion of the mean to float32.
- Adam uses float32 moments
  and updates, float64 scalar bias-correction powers, and float32-rounded derived
  coefficients. Four parameters are processed per NEON iteration, with scalar tails.

The scalar exponential uses float64 range reduction and a degree-12 polynomial before
rounding to float32. The logarithm normalizes the exactly converted float32 input
in float64 and evaluates an atanh series through the 23rd power. Their documented
nonfinite/domain behavior is covered by the numerical admission checks. These methods are
not a claim of a correctly rounded general-purpose math library.

Fused operations, reduction order, LayerNorm variance evaluation, and reporting
accumulation may differ from PyTorch. Numerical parity must be established by the
independent audit for the exact linked binary; successful assembly is only a syntax
check and does not establish numerical correctness or speed.

The kernels leave FPCR unchanged. FP32 FMOPA uses a single rounding for each
multiply-add, but its ZA arithmetic forces default NaNs and suppresses floating-point
exceptions internally, as specified by FPMulAdd_ZA in the
[Arm SME architecture supplement, section E2.1.21](https://documentation-service.arm.com/static/6526e1bd9e189a266cef8412).
Consequently this project does not claim identical NaN payloads or exception-status
behavior across SME and NEON. Matched arithmetic claims concern the validated
finite inputs under the study's floating-point environment and fixed tolerances.
The runtime retains its nonfinite parameter, gradient, loss, and logit checks.
