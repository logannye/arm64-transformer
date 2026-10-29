# Handwritten Metal backend

`backend.mm` owns the device, command queue, cached compute pipelines and shared
buffers. `kernels.metal` implements every numerical operation in the model,
including backward propagation and Adam. The host uploads exact FP32 parameter
bytes and integer inputs; it performs shape/range validation, dispatch, state
copies and scalar Adam step/bias-correction bookkeeping. No framework computes
model tensors on behalf of this backend.

Build the small Objective-C++ host explicitly:

```sh
make -C native/metal
```

This uses one reusable `build/` directory. Xcode's Metal command-line compiler
is not required. Initialization calls `newLibraryWithSource:options:error:` and
creates pipelines on the actual Metal device. It selects MSL 3.1,
`MTLMathModeSafe` and `MTLMathFloatingPointFunctionsPrecise`; it does not enable
unsafe reassociation or reduced-precision storage. The host embeds its source
SHA-256 during compilation and the Python adapter rejects a stale host binary.
The capability receipt retains the binary/shader hashes, compiler identity,
device families, thread execution widths and resource limits.

The fixed transformer uses four heads, at most 32 tokens per sequence, widths
32–1024 and batches of at most 16. Dense layers flatten batch and token axes into
one matrix; causal attention keeps sequence boundaries separate. The 16×16
threadgroup matrix tiles cover forward, input gradients and weight gradients,
with guarded tails and ascending inner reductions. Each output has one writer.
Bias, LayerNorm parameter and embedding gradients use deterministic reductions;
repeated token IDs never require floating-point atomics. The sole atomic
operation combines integer finite-error flags.

All tensors and saved optimizer state use FP32. Development stress exposed loss
of small score differences, normalization cancellation and propagated QKV input
gradient errors in the initial reference. The selected precision path uses
temporary pairs of FP32 values to retain rounding residuals in sensitive work:

- Attention keeps FMA product residuals and compensated sums through QK scaling,
  maximum selection and score centering. Exponentials still return FP32, while
  their denominator sum and quotient correction use two-float arithmetic before
  the final FP32 probability store. Backward retains expansions through dA,
  the softmax contraction, dScore and the Q/K/V gradient contractions.
- LayerNorm retains compensated means and centered variances, with compensated
  affine and backward contractions. Saved normalized activations, inverse
  standard deviations, outputs and gradients remain FP32.
- QKV projection and its input-gradient matrix contraction use compensated
  accumulators. Its weight/bias gradients and other dense layers retain their
  existing FP32 kernels because their declared development checks passed.

An error-free sum decomposition preserves addition residuals; explicit FMA
recovers each rounded product's residual. Safe compiler math is required for
these formulas. The host computes the scalar attention scale `1/sqrt(head_width)`
once in double precision and splits it into two FP32 constants. All tensor
operations execute on the GPU. Two-float arithmetic retains FP32 dynamic range
and does not provide native FP64 or general FP64 equivalence.

The bounded sequence is declared in
[GPU-PRECISION-CANDIDATES-v1.md](../../experiments/GPU-PRECISION-CANDIDATES-v1.md).
The initial attention candidate still failed two width-64 mathematical-gradient
coordinates because a plain FP32 softmax sum produced row mass `0.99999970055`.
Compensating that sum and the quotient resolved the failure without clipping or
renormalizing saved probabilities during backward. P2 resolved normalization
errors but left five width-1024 joint input-gradient failures; compensating QKV
forward and dX resolved those in P3. Each superseded candidate and its failures
are retained in the development evidence.

Forward, backward and optimizer dispatches for a step share one command buffer.
Metal command buffers are single-use; the queue, pipelines, parameter buffers
and workspaces persist. Resident datasets and schedules support GPU gather.
Shared storage is GPU execution memory, not a CPU numerical fallback. Completion
is explicit before host loss/state readback and benchmark boundaries.

`transformer/gpu_metal.py` implements the shared device-batch contract. Its device
results borrow a workspace until the next computation; reading an expired result
raises an error. Gathered batches borrow reusable staging buffers and must be
consumed before the next gather. State export returns independent host arrays.
State restoration validates every array before writing and supports exact
warmup restoration. The caller serializes access to one backend instance.

Every buffer includes 256-byte sentinels before and after its logical capacity.
`check_guards()` verifies them after GPU completion. `check_finite()` reduces
the last logits/loss, gradients, parameters and both moments on the GPU, returning
only a scalar flag. Command-buffer errors propagate to Python. Malformed
dimensions, token IDs, gather indices and state layouts are rejected before
dispatch; uploaded host inputs are copied and remain unmodified.

Primitive inspection methods expose the same attention, normalization and dense
pipelines used in production. Actual forward and backward workspace traces are
available for independent FP64 comparisons. Development records live under
`work/gpu-v1/`; no development result by itself qualifies a formal benchmark.

The final P3 implementation passed 23 attention cases and 22 LayerNorm/QKV cases
at each of the six widths, plus full-model gradients, Adam and lifecycle checks.
The following tracked drivers reproduce the development checks. Give each run
a fresh label/output path; existing output JSON is never overwritten. These
commands require GPU access, an explicitly built host and the existing Python
environment. The LayerNorm/QKV diagnostic bridges primitives with host copies;
it is not the production training path or a timing benchmark.

```sh
ARM64_TRANSFORMER_WIDTH=1024 PYTHONPATH=. .venv/bin/python scripts/check_metal_attention.py --label reproduced-p3
ARM64_TRANSFORMER_WIDTH=1024 PYTHONPATH=. .venv/bin/python scripts/check_metal_norm_qkv.py --width 1024 --output work/gpu-v1/reproduced-p3/ln-qkv-1024.json
ARM64_TRANSFORMER_WIDTH=1024 PYTHONPATH=. .venv/bin/python scripts/check_metal_development.py --label reproduced-p3
```

Run separately for widths 32, 64, 128, 256, 512 and 1024. The recorded runs pin
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and `VECLIB_MAXIMUM_THREADS=1` for CPU
oracle work. Admission fixtures and formal timing remain separate contracts.
