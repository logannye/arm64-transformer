# Project-maintained framework study, GPU v4

On October 30, 2026, the user requested an independent route that does not require
upstream submission or cooperation. This prospective contract permits pinned,
project-maintained full-source PyTorch and MLX builds as comparison engines. It
changes baseline provenance and the scope of the resulting claim. Accuracy,
exact repeatability, workload matching and checkpoint auditing remain mandatory.

The protocol identifier is `gpu-v4-owned-five-engine-v1`. Reports must identify
the framework engines as **project-maintained patched builds**, including the
PyTorch CPU anchor. Results cannot be described as performance against standard
released PyTorch or MLX. Upstream acceptance is not a prerequisite for this
study; published patches do not become upstream-supported packages.

## Preserve the prior study

The standard-release study remains at zero of 180 fits. Its protocol, failures,
release artifacts and reserved stream `20261104` are unchanged. Historical CPU
v6 precision limitations and GPU development results keep their original scope.
No existing receipt is edited or relabeled to qualify the new contract.

The new admission stream is **20261108, reserved and unopened**. Development
continues using the already exposed training-only stream `20261103`. A bounded
search of 33,620 project/work code, documentation and JSON files up to 10 MB,
excluding expanded framework, build and environment trees, found earlier use of
20261105, 20261106 and 20261107 and no use of 20261108. This search records the
selection basis; it is not a claim about uninspected external files.

New plans, runtime receipts, prerequisite evidence, admission, fits and reports
use the `gpu-v4-owned` namespace. The old standard-study admission command must
reject the new plan. A plan or successful preflight cannot authorize formal
execution. A dedicated implementation must bind the new protocol and stream at
every boundary before admission can open.

## Fixed modeling and measurement contract

| Component | Requirement |
|---|---|
| Model | One causal decoder block, four heads, context 32, FFN 2W, ReLU, three LayerNorms, ASCII-128 vocabulary |
| Widths | 32, 64, 128, 256, 512, 1024 |
| Stored numerical state | FP32 parameters, activations, probabilities, gradients and Adam moments |
| Inputs | Exact original CPU v6 corpus partitions, initialization bytes and schedules |
| Partition sizes | 65,536 training targets; 8,192 validation; 8,192 test |
| Training | B16/T32, eight epochs, 1,024 updates, 524,288 processed targets per fit |
| Seeds and repeats | Seeds 42, 43, 44; two timing repeats per seed |
| Adam | Learning rate .001, betas .9/.999, epsilon 1e-8 |
| LayerNorm | Population variance, epsilon 1e-5 |
| Checkpoint | Lowest own-engine epoch validation NLL; earliest epoch wins ties |
| Denominator | 180 fits: 72 CPU and 108 GPU; 1,474,560 retained test-token predictions if complete |

The five engines are native ARM64 CPU, project-maintained PyTorch CPU, custom
Metal, project-maintained PyTorch MPS and project-maintained compiled MLX.
Internal engine keys remain `cpu_arm64`, `cpu_torch`, `metal`, `mps` and `mlx`;
the protocol and explicit display labels distinguish their provenance. Both
PyTorch engines must load the **same selected distribution and binary identity**.
The immutable CPU model/native source remains
`2c5191047068c16b12e3312652b118d48a5f3744`; rerun its checks under the selected
environment. The existing custom Metal production policy remains selected unless
a separately declared development experiment qualifies a replacement before the
study freeze.

Preserve the 180-cell rotation and full denominator from GPU v2. The base order
is native CPU, PyTorch CPU, Metal, MPS, MLX, rotated by
`(width_index + 2 * seed_index + repeat_index) % 5`. The first repeat traverses
widths ascending; the second descending. Run one measured process at a time on
AC power, fixed one-thread host numerical libraries, with 15-second inter-fit
cooldowns and recorded power, thermal and load conditions. No concurrent builds,
profiling or project numerical work is permitted during measurements.

Training timing includes forward/backward, reductions, Adam, finite checks,
validation, checkpoint writes, packing, synchronization and bookkeeping. Report
setup, compilation, warmup and final test evaluation separately. Primary inference
uses resident B1/T32 inputs, 20 warmups and 100 synchronized observations, retaining
output on the device until completion. Restore exact parameters, moments and step
after warmup. Do not equate process RSS, driver allocation and GPU allocation.

This remains a reused engineering benchmark. All development operands and data
come from declared synthetic fixtures or the original training partition. Test
outcomes cannot guide implementation or policy selection. Additional data, larger
models, longer contexts, cached decoding, mixed-precision storage and longer
training remain outside this contract.

## Pin and qualify the maintained frameworks

Start from the existing authenticated full-source builds and patches, preserving
all failed and successful historical evidence:

| Framework | Initial source and patch identities |
|---|---|
| PyTorch | Base `71d2a08507ab47dd6b2fa9b66319d17ce2377de4`; formatted embedding patch SHA-256 `3834dca844ea3de6b40467ce214b99bce83fa06dae2bc454623516dafc0fdd71` |
| MLX | Base `a21f892a6e8e83afb117cef1d1ceae3e82f48c32`; embedding patch SHA-256 `9aecf0c9b0131fe0673493035fd3c47fba3e4ffb1cfc04d00a506702e1d1067c`; attention patch SHA-256 `b9787a423920641034f42e611f0d8529f6dd49022d094e280fa5ce52a38b7cc3` |

These identities select a starting implementation, not an admitted runtime.
Historical source, binary, shader-mode, dependency and platform receipts remain
distinct. In particular, the paired PyTorch failure control does not substitute
for the selected model-validation runtime. Reuse compatible authenticated native
artifacts when possible; full native rebuilds use the authorized hosted route,
one build tree and bounded caches. Record any repair as a new exact source and
binary identity and rerun its affected checks.

Before numerical work, verify archive members, full source/patch state, loaded
module and dylib origins, interpreter, dependencies, compiler settings, hardware,
OS and device capabilities. Bind the actual execution environment, including
normal imports and complete optimizer state. Assembling previously qualified
components together creates a new runtime identity; no old pass automatically
qualifies that combination.

Use ordinary MPS dispatch and actual MLX graph compilation.
`PYTORCH_MPS_PREFER_METAL` and `MLX_DISABLE_COMPILE` must be absent, including
absence of the string `"0"`. Retain disabled CPU fallback, fast-math/TF32 controls
and the required MLX compile-trace, Compiled-primitive and changing-input proof.
The initial owned-build plan selects the combined patched MLX Release build with
shader JIT on; bind that exact runtime before formal measurement. Shader JIT and
graph compilation are separate controls. No monkeypatch, hybrid embedding,
alternative project-level backward, or silent CPU tensor fallback is allowed.
The repaired operations remain inside the normal full-source frameworks.

The complete required framework integration checks remain prerequisites.
**PyTorch's retained 25 failures and two errors in the broad MPS suite remain
blocking for owned-build eligibility.** Reproduction in the pristine control is
useful attribution evidence; it neither clears those failures nor changes their
status to diagnostic. The existing skip/expected-failure eligibility cannot be
expanded to obtain a pass. Repair and revalidate independently where feasible.
Missing required tests, changed contracts or incomplete evidence cannot qualify.

Fresh baseline checks must cover the complete existing repeatability inventory:
all widths, batches 1/3/16, full/partial sequences, repeated/alternating/skewed and
eligible unique tokens, zero/random/cancelling cotangents, ten executions in each
of two fresh processes, full-model gradients, six-step parameters/moments and
saved checkpoint replay. CPU anchors must be checked under the same selected
PyTorch distribution used by MPS. Historical targeted passes establish useful
development evidence, not passage of this new prerequisite.

## Admission stays strict

Freeze the complete source, runtime, binary/shader, corpus, schedule, selected
policy, prerequisite evidence and 180-cell inventory before admission. The
machine-enumerable admission inventory retains all 2,088 gate groups and 9,858
members of the corrected GPU v2 implementation unless an explicitly versioned
prospective extension adds coverage. No required check may be removed.

Retain primitive forward/dX/dW, all 23 attention and 22 LayerNorm/QKV stress
families at every width, actual B1/T32 and B16/T32 logits/loss/all gradients,
common-state comparisons, identical-gradient optimizer replay, causal and
sequence-isolation controls, partial batches/tails/guards, immutability, lifecycle
and malformed-input controls, warmup restoration, same/fresh-process repeats,
checkpoint continuation and tie selection. Preserve the existing small-width
trajectory and wide-width replay rules. The standard-framework stress diagnostic
policy remains exactly its existing scope; it does not absorb blocking runtime,
common-state, optimizer, finite-state, exact-repeatability or integration failures.

| Numerical comparison | Absolute tolerance | Relative tolerance |
|---|---:|---:|
| Forward | 2e-5 | 2e-4 |
| Gradients | 3e-5 | 3e-3 |
| Probability/math | 2e-7 | 2e-6 |
| Independent FP64 oracle agreement | 1e-10 | 1e-10 |
| Exact repeatability/restoration | Byte equality | Byte equality |

Every expected check needs its identity, operand hashes, shape, tolerance and raw
result. Missing, duplicate or reordered evidence fails. Admission must recompute
results from the retained arrays and authenticate the exact execution identities.
A boolean flag, old receipt or diagnostic subset cannot open formal fits. After
admission operands are opened, an implementation repair requires another declared
contract and a fresh reserved stream. Changed artifacts or invalid measurement
conditions stop execution and preserve the failed attempt.

## Audit and report the actual claim

At every selected checkpoint, evaluate all validation and test sequences through
all five engines and independent FP64 equations using identical saved parameters.
Require authenticated states/histories, correct update counts and selection,
exact originating-engine replay and exact within-engine repeat equality.
Cross-check the independent FP64 implementations at the prescribed locations.
Keep integrity, repeatability, each engine's FP64 accuracy, cross-engine FP32
compatibility and predictive quality separate. Post-training precision failures
remain failures even if initial-state admission or aggregate NLL passes.

Report per-width paired training/inference ratios, p50/p95, throughput, quality,
variability and memory with their exact qualification. Average timing repeats
within seed before three-seed quality summaries. Never pool widths into a
headline speedup. Publish source/patch/runtime identities, raw arrays and timing
samples, failures, admission/audit evidence, static figures and reproduction
commands. Identify project-maintained frameworks throughout captions and tables.

Completion is either an audited owned-build 180-fit study or a truthful
engineering release identifying remaining failures. Neither outcome resolves
the original standard-release campaign or historical CPU precision limitations.
