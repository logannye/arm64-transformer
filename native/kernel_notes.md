# Numerical implementation

`kernels.S` contains handwritten Darwin AArch64 implementations of every
numerical primitive declared under **Numerical primitives** in `ABI.md`.
There are no calls to C numerical routines, BLAS, Accelerate, PyTorch, or libm.
Only cross-entropy and attention call the handwritten `tf_exp` and `tf_log`.
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

Darwin callee-saved x19-x28 are saved/restored where used. The reserved x18 register
and callee-saved SIMD v8-v15 are not used. Non-leaf stack frames are 16-byte aligned.

## Arithmetic

- Linear forward uses four-lane NEON fused multiply-add and a final horizontal
  reduction, then scalar tails and bias addition. Linear backward overwrites each
  input-gradient row and accumulates parameter/bias gradients in fixed row order.
- LayerNorm uses float32 two-pass mean and biased variance, epsilon `1e-5`, and
  scalar arithmetic. It retains normalized inputs and inverse standard deviations.
  Backward applies `inv * (dy*gamma - mean(dy*gamma) - norm*mean(dy*gamma*norm))`.
  Scale and bias gradients are added to the caller's accumulators.
- Attention stores Q, K, and V consecutively in each token row. Each query attends
  only to keys through its own position. Masked probabilities are explicitly zero.
  Scores use `1/sqrt(head_width)`, max-subtracted softmax, and fixed-order reductions.
  Backward overwrites all Q/K/V gradients. It computes the softmax row contraction
  once, recomputes attention-probability gradients in a second pass, and accumulates
  Q/K/V contributions. No temporary quadratic backward tensor is allocated.
- ReLU uses NEON comparisons and masked gradients. Its derivative at zero is zero.
- Cross-entropy uses `log(sum(exp(logits-max))) - (target_logit-max)`, preserving
  finite losses even when the target probability underflows. Returned gradients
  are divided by the number of token rows. Individual token losses are float32;
  their reporting sum uses float64 before conversion of the mean to float32.
- Adam matches the former CNN implementation's operation order: float32 moments
  and updates, float64 scalar bias-correction powers, and float32-rounded derived
  coefficients. Four parameters are processed per NEON iteration, with scalar tails.

The exponential uses float64 range reduction and a degree-12 polynomial before
rounding to float32. The logarithm normalizes the exactly converted float32 input
in float64 and evaluates an atanh series through the 23rd power. Their documented
nonfinite/domain behavior follows the historical CNN routines. These methods are
not a claim of a correctly rounded general-purpose math library.

Fused operations, reduction order, LayerNorm variance evaluation, and reporting
accumulation may differ from PyTorch. Numerical parity must be established by the
independent audit for the exact linked binary; successful assembly is only a syntax
check and does not establish numerical correctness or speed.
