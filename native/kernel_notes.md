# Numerical implementation

The `.S` files contain handwritten Darwin AArch64 implementations of every
numerical primitive declared under **Numerical primitives** in `ABI.md`.
There are no calls to C numerical routines, BLAS, Accelerate, PyTorch, or libm.
Cross-entropy and attention call the handwritten scalar/vector exponentials;
cross-entropy also calls the handwritten logarithm.
All buffers and tensor dimensions come from the caller; no dynamic allocation,
mutable global state, or shared scratch storage is used.

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

## V2 execution changes

- `linear.S` computes four token rows by four output columns together, using
  sixteen independent NEON dot-product accumulators. Loaded input/weight vectors
  contribute to multiple outputs. Its backward kernels retain four-by-sixteen
  gradient tiles in registers, reducing repeated loads and stores of partial
  sums. Forward requires row/input/output counts divisible by four; backward
  additionally requires input width divisible by sixteen. Other positive shapes
  tail-call the preserved v1 generic implementations in `kernels.S`.
- `attention.S` specializes the actual width-32/four-head model for 1..32 tokens.
  Each eight-element head uses two NEON vectors. QK, weighted values, and their
  derivatives are vectorized. Backward caches one row of attention derivatives
  in a 32-float stack buffer instead of recomputing those dot products. Other
  shapes use the original generic implementation.
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
training batch. V2 changes kernel execution, not this scheduling rule. It does
not add batch-wide model workspaces, mixed precision, a KV cache, or a new model.
This keeps the one-sequence workspace while making its hot operations efficient.
Whether batch-wide execution helps further remains a separate measured question.

## Arithmetic

- Linear forward uses four-lane NEON fused multiply-add and horizontal reduction,
  then bias addition. The tiled and generic implementations preserve the same
  per-output arithmetic order. Linear backward overwrites input gradients and
  accumulates parameter/bias gradients in fixed row order.
- LayerNorm uses float32 two-pass mean and biased variance, epsilon `1e-5`, and
  scalar or vector arithmetic. It retains normalized inputs and inverse standard deviations.
  Backward applies `inv * (dy*gamma - mean(dy*gamma) - norm*mean(dy*gamma*norm))`.
  Scale and bias gradients are added to the caller's accumulators.
- Attention stores Q, K, and V consecutively in each token row. Each query attends
  only to keys through its own position. Masked probabilities are explicitly zero.
  Scores use `1/sqrt(head_width)`, max-subtracted softmax, and fixed-order reductions.
  Backward overwrites all Q/K/V gradients, computes the softmax row contraction,
  and accumulates Q/K/V contributions. The specialized path caches one row;
  the generic path recomputes it. No quadratic backward tensor is allocated.
- ReLU uses NEON comparisons and masked gradients. Its derivative at zero is zero.
- Cross-entropy uses `log(sum(exp(logits-max))) - (target_logit-max)`, preserving
  finite losses even when the target probability underflows. Returned gradients
  are divided by the number of token rows. Individual token losses are float32;
  their reporting sum uses float64 before conversion of the mean to float32.
- Adam matches the former CNN implementation's operation order: float32 moments
  and updates, float64 scalar bias-correction powers, and float32-rounded derived
  coefficients. Four parameters are processed per NEON iteration, with scalar tails.

The scalar exponential uses float64 range reduction and a degree-12 polynomial before
rounding to float32. The logarithm normalizes the exactly converted float32 input
in float64 and evaluates an atanh series through the 23rd power. Their documented
nonfinite/domain behavior follows the historical CNN routines. These methods are
not a claim of a correctly rounded general-purpose math library.

Fused operations, reduction order, LayerNorm variance evaluation, and reporting
accumulation may differ from PyTorch. Numerical parity must be established by the
independent audit for the exact linked binary; successful assembly is only a syntax
check and does not establish numerical correctness or speed.
