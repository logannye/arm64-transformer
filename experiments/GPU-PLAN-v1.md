# Mapping the same transformer to the Apple GPU

The question is whether the same model can be expressed around GPU execution,
memory movement, and numerical constraints more effectively than standard GPU
frameworks. This is a new engineering arm of the project. It preserves the model,
data, optimization budget, and FP32 stored state from the CPU study. A handwritten
shader is useful only if the complete implementation is correct, reproducible,
and competitive under matched measurements.

This is the authorized eight-stage development plan. The associated
[prospective protocol](PROTOCOL-gpu-v1.md) is explicitly incomplete until capability
discovery, the finite candidate grid, complete admission inventory, execution
order, reporters, and source identities are committed. No GPU result or speedup
is claimed by this document.

## 1. Close CPU work and freeze the GPU experiment

The CPU phase is closed. Preserve its source, protocols, reports, failures, and
[published release](https://github.com/logannye/arm64-transformer/releases/tag/transformer-attention-precision-v6)
without reinterpretation. Its measured revision is
`2c5191047068c16b12e3312652b118d48a5f3744`, publication revision
`7504ae04684eb7e5ec89904471360c11930fabb7`, and merged main revision
`04a69ba7b13e4c0800a58be400c62d8095d5acae`; publication CI is
[40815358560](https://github.com/logannye/arm64-transformer/actions/runs/40815358560).
The recorded CPU outcome is `native_precision_limit`. Its historical timings
provide context; a paired CPU/GPU claim requires separately frozen CPU rerun
anchors under the new machine conditions.

Use the separate `codex/metal-gpu-v1` branch and new GPU source, result, input,
development, and report namespaces. Discover and record actual device identity,
GPU family support, SIMD execution widths, threadgroup limits, memory limits,
compiler settings, operating system, dependencies, power, and thermal information.
The initial host is an M4 Max with 40 GPU cores, macOS 26.2, and PyTorch 2.14.1
reporting MPS available. Record those observations in executable discovery
evidence. They do not establish support for any particular matrix or Metal 4
shader feature. The current host has clang 17 but no standalone Metal compiler;
runtime MSL compilation through the Metal API is a valid path to test.

Keep the six widths, context 32, batch 16, fixed corpus partitions, exact initial
weights, schedules, and eight-epoch/1,024-update budget. No mixed-precision storage,
larger context, larger batch, additional data, or additional training is included.
Use one bounded build/cache location and pinned dependencies; avoid Rust.

## 2. Working GPU baselines before optimization

Implement fixture-compatible PyTorch MPS and standard compiled MLX paths first.
Use their ordinary GPU linear algebra, normalization, attention, differentiation,
and Adam facilities where those preserve the declared equations and state layout.
The MLX comparison must include its [documented compilation path](https://ml-explore.github.io/mlx/build/html/usage/compile.html), evaluated lazy
results, and completion synchronization; do not compare custom fused execution
only against an intentionally uncompiled MLX implementation.

Load and export the existing flat FP32 parameters, moments, step count, token
rows, and schedules exactly. Expose the common `DeviceBatch`/state interface and
separate device upload, execution, synchronization, and host export boundaries.
Verify actual GPU dispatch, resident tensor types, and unsupported-operation
behavior. CPU tensor arithmetic fallback is disallowed. If a baseline cannot
implement the workload, retain the limitation and stop or prospectively narrow
the comparison; do not silently substitute a CPU operation.

Before custom optimization, run primitive, common-state, full-gradient, Adam,
checkpoint, and small end-to-end development checks. Distinguish framework fused
production outputs from expanded diagnostic intermediates. A diagnostic
reconstruction of probabilities must never be represented as the probabilities
actually used by a framework's fused attention implementation.

## 3. Custom Metal runtime and a correct reference path

Build a minimal Objective-C++/C++ or Swift host with handwritten Metal Shading
Language kernels for the complete forward pass, backward pass, and Adam update.
Host code owns allocation, dispatch, synchronization, checkpoint I/O, and
bookkeeping. All model tensor arithmetic stays on the GPU. Any use of framework
numerical operators makes that variant a labeled hybrid, not a wholly custom
Metal implementation.

Start with transparent reference kernels and explicit saved intermediates. Then
map the real workload to batch-wide dense operations over `B*T = 512` rows, with
separate forward, input-gradient, and weight-gradient matrix kernels. Implement
short-context causal attention and backward, three LayerNorms, ReLU, residuals,
cross-entropy, token/position embedding gradients, and complete Adam state.
Preserve deterministic reductions for shared parameters and repeated token IDs.
Validate partial batches and tails without making them a different measured
workload. Preallocate workspaces, cache pipeline objects, and batch commands only
after a correct path exists. Retain every unsuccessful development variant.

## 4. Numerical precision before throughput

The CPU solution's selective FP64 accumulation cannot simply be copied into a
Metal kernel: the GPU path cannot assume native FP64 arithmetic. Independent
NumPy and PyTorch FP64 calculations remain CPU validation oracles outside the
production GPU path. A CPU FP64 numerical fallback cannot support a GPU-only
claim. Parameters, activations, saved probabilities, gradients, and optimizer
state remain stored as FP32.

Start with controlled reduction order and safe compiler settings. Investigate
multiple FP32 partial sums, pairwise reductions, and targeted compensated or
two-float arithmetic where development evidence warrants them. Extra precision
has a cost and is not guaranteed to satisfy the existing limits. Preserve
precision through the operation that removes a large common score, rather than
rounding a large uncentered score and hoping that softmax subtraction recovers
lost differences. A key-centered expression such as
`q dot (k_j - k_0) / sqrt(head_width)` is an unproven candidate: derive all
anchor-dependent backward contributions and validate them before considering it.

Use the new independent development stream 20261103 and reserve 20261104 for
admission. Cover independently varied common scores and near-tied margins,
contrasting values, cancellation, all head widths, causal prefixes, future
interventions, nonzero cotangents, and epsilon-sensitive normalization. Add
upstream LayerNorm and QKV stress: a small normalization/projection discrepancy
can become a much larger downstream score discrepancy. Validate input and
parameter gradients. Evaluate the mathematical FP64 derivative and the
derivative using the exact rounded saved probabilities separately. The unchanged
limits are forward 2e-5/2e-4, gradient 3e-5/3e-3, probability/math 2e-7/2e-6,
and independent FP64-oracle agreement 1e-10/1e-10 (absolute/relative).

## 5. Profile-guided bounded candidate grid

Profile the correct development implementation before choosing performance work.
Determine whether complete steps are limited by dense arithmetic, reductions,
memory traffic, allocation, synchronization, or dispatch overhead. Focus on tiled
matrix forward/dX/dW, fused T32 attention and backward, normalization/reductions,
elementwise and Adam fusion, pipeline reuse, and command batching.

After hardware discovery and before evaluated candidate search, commit the exact
finite grid and candidate-selection rule described in the protocol. Legal
feature filtering must use discovery/compile constraints, not measured speed or
test-set outcomes. Keep a complete ledger of attempted, unsupported, failed,
and successful candidates, compiler settings, correctness results, timings,
and selection decisions. Development operands may guide this search; admission
and corpus test outcomes may not. Do not turn a failed candidate into an
unreported omission or expand the grid after seeing unfavorable results.

## 6. Freeze and pass GPU admission

Before formal timing, commit the model/runtime code, shaders, compiler flags,
dependency lock, selected kernels, complete fixtures and named gates, protocol,
three-engine order, runner, auditor, reporter, and packaging rules. Authenticate
all relevant source, binaries or compiled shader artifacts, runtime pipeline
identities, inputs, and configuration. Open the reserved admission stream only
after this complete contract is frozen.

Cover all widths and engines: primitives, full parameter gradients, common-state
forward/backward, identical-gradient Adam moments/steps, retained width-specific
trajectory rules, causal invariance, repeated embeddings, tails, bounds, races,
barriers, input immutability, malformed input, output collisions, finite states,
partial batches, checkpoint selection/replay, and exact repeatability. Exact
replay is the target; allclose is not an acceptable relabeling of a failed exact
check. Keep integrity, repeatability, per-engine FP64 accuracy, mutual FP32
compatibility, and predictive quality separate.

Fail closed when a required custom correctness/accuracy gate cannot be met.
Do not automatically launch 108 fits after failed admission. A descriptive
campaign with known failures would require an explicit prospective amendment
and unambiguous outcome labels committed before any formal fit.

## 7. Matched 108-fit campaign

The planned matrix is six widths × three seeds × two repeats × three GPU
engines: **108 fits** and **884,736 retained test token predictions** if complete.
Every planned cell remains in the denominator, including errors, skips, and
blocked runs. Preserve evidence; retries receive new paths and remain linked to
the original attempt.

Run one measured process at a time on AC power, with fixed host threading,
predeclared cooldowns, a balanced three-engine order, and recorded before/after
conditions. No builds, profiling, tests, or other project numerical work run
concurrently. Time completion, not asynchronous enqueue. Preserve the existing
training-timer semantics, including validation, checkpointing, state updates,
finite checks, and bookkeeping; separate compilation, setup, and warmup.
Restore exact parameters, moments, step, and relevant mutable runtime state after
warmup. Primary inference is B1/T32 forward with 20 warmups and 100 samples.
Any batched throughput or device-only kernel timing is a separately named
secondary metric. Report backend-specific memory definitions without pretending
that RSS, allocated tensors, and driver memory measure the same quantity.

## 8. Audit and publish

At every selected checkpoint from every training engine, evaluate all validation
and test rows using each GPU implementation and independent FP64 equations with
the same exact saved parameters. Authenticate own-engine exact replay, histories,
selection, optimizer states, repeat equality, inputs, shaders, and sources.
Retain complete error records, failing coordinates, probability/NLL differences,
and argmax changes. No good aggregate NLL erases a failed coordinate contract.

Report paired custom/MPS and custom/compiled-MLX timing ratios by width, latency
p50/p95, throughput, quality by seed, memory, variability, and qualification.
Average timing repeats within seed for quality summaries; repeats are not extra
independent seeds. Keep unsuccessful and blocked outcomes visible. A speedup is
not required to publish the engineering result.

Use static plots and dataflow diagrams without Mermaid controls. Keep the README
idea-first, with explanatory results later and the project standalone. Preserve
the closed CPU arm. Archive complete raw states, predictions, gates, failures,
candidate ledger, source/configuration identities, and reproducibility tools;
verify GitHub publication assets before reporting completion. Claims stay within
this short-context, FP32-storage model family and the tested GPU baselines.

Relevant primary documentation: [PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html)
describes its Apple GPU backend; [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
documents shader dispatch and differentiation integration. Apple's
[Metal resources](https://developer.apple.com/metal/resources/) and
[feature tables](https://developer.apple.com/metal/feature-sets/) guide capability
checks. [FlashAttention](https://arxiv.org/abs/2205.14135) motivates considering
memory traffic and fusion; it does not establish a speedup for this T32 workload.
