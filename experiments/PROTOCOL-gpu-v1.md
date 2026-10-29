# Prospective GPU comparison, v1

**Status: development plan; formal admission and measurement are closed.**
This document fixes the scientific scope and unchanged numerical limits. It is
not yet a complete frozen execution contract. Capability discovery, the exact
finite candidate list, all named check counts, dependency/compiler identities,
three-engine order, and audited runner/report schemas must be committed before
admission. The code currently exposes a development fixture inventory and rejects
generation from the reserved admission stream. The companion
[eight-stage plan](GPU-PLAN-v1.md) records the implementation sequence.

Development MPS and compiled MLX probes pass initial forward/gradient FP64 checks
but do not reproduce training states exactly, despite exact state restoration
after warmup. Two-step controls localize observed gradient differences to repeated
token embeddings; six-update runs at widths 32 and 1024 also fail exact repeats.
Custom Metal repeats those development runs exactly. These results do not open
admission or qualify a formal campaign. The baseline repeatability failures and
the resulting contract decision remain open. Formal timing stays blocked until
this behavior and every other admission requirement are resolved under the rules
below; no stochastic waiver is implied. MLX also rejects a short-sequence T3
attention-backward development probe at head width 256; the T32 model path works.

## Question and preserved CPU evidence

Compare complete custom Metal training and inference with standard PyTorch MPS
and standard compiled MLX on the same Apple GPU. The intervention is hardware
mapping, reduction/precision design, fusion, and execution scheduling; the model
equations and experiment budget remain fixed. No speedup is required.

CPU v6 remains immutable: measured revision
`2c5191047068c16b12e3312652b118d48a5f3744`; publication
`7504ae04684eb7e5ec89904471360c11930fabb7`; merged main
`04a69ba7b13e4c0800a58be400c62d8095d5acae`; CI run `40815358560`;
[release](https://github.com/logannye/arm64-transformer/releases/tag/transformer-attention-precision-v6);
status `native_precision_limit`. Historical CPU timings are contextual only.
Paired CPU/GPU claims require prospectively frozen CPU rerun anchors.

## Fixed model, data, and budget

| Item | Fixed value |
| --- | --- |
| Widths | 32, 64, 128, 256, 512, 1024 |
| Model | One decoder block; four heads; T32; feed-forward 2W; ReLU; three LayerNorms; ASCII-128 vocabulary |
| Stored numerical state | FP32 parameters, activations, probabilities, gradients, and Adam moments |
| Training | B16; eight epochs; 1,024 Adam steps; 524,288 processed targets per fit |
| Fixed partition sizes | 65,536 training targets; 8,192 validation; 8,192 test |
| Adam | Learning rate .001; betas .9/.999; epsilon 1e-8; existing update/correction semantics |
| LayerNorm | Epsilon 1e-5; existing population-variance equation |
| Seeds/repeats | Seeds 42, 43, 44; two timing repeats per seed |
| Engines | PyTorch MPS; standard compiled MLX; custom Metal |
| Planned denominator | 108 fits; 884,736 saved test token predictions if complete |
| Checkpoint selection | Lowest own-engine epoch validation NLL; first epoch wins ties |

Reuse the exact existing corpus bytes, partition rows, initial parameter bytes,
and deterministic training schedules. Copy or reference them under new GPU
namespaces with hashes; do not modify CPU inputs. The data and earlier test
results are already exposed, so this is a reused engineering benchmark, not
fresh independent model assessment. Test results cannot guide kernels, candidate
selection, seeds, thresholds, hyperparameters, or training duration. No larger
context/batch/data budget, mixed-precision storage, or model change is included.

## Device and engine contract

Record actual device/OS, GPU families, thread execution width, memory and
threadgroup limits, runtime/compiler versions and settings, and dependency
versions/hashes. Initial discovery concerns an M4 Max 40-core GPU, macOS 26.2,
PyTorch 2.14.1 with MPS available, and clang 17. A missing standalone Metal CLI
does not itself block runtime MSL compilation. No optional Metal feature is
assumed until the real device and runtime compile/dispatch checks establish it.

All tensor/model arithmetic for all three engines runs on the GPU. Disable
silent CPU numerical fallback and verify actual dispatch. CPU FP64 work is
permitted only in separately identified validation/audit oracles, outside GPU
production execution and timing. Ordinary host dispatch, scalar configuration,
I/O, and bookkeeping do not change this boundary. A custom path using framework
numerical primitives must be labeled hybrid and cannot replace the declared
fully custom engine without a prospective protocol amendment.

The shared interface separates device batches, forward/diagnostic traces,
loss/gradients, training steps, exact state export/load, synchronization, and
host conversion. QKV weight storage remains `[3W,W]`; its equation is
`LayerNorm(x) @ weight.T + bias`. Standard compiled MLX is a required baseline;
record the exact compilation boundary and actual graph evaluation. Framework
expanded diagnostics are labeled separately from fused production outputs.

## Precision, independent oracles, and development fixtures

GPU arithmetic may use controlled FP32 partial sums, pairwise/compensated
reductions, or explicitly documented two-float intermediates. Native GPU FP64
is not assumed or supplied through a CPU fallback. Disable unsafe fast-math
transformations initially and record contraction/reassociation choices. Any
changed numerical algorithm, including key centering, requires its complete
forward and backward derivation and development validation. No probability
clamping, hidden renormalization, or changed equations can be introduced solely
to suppress failures.

| Comparison | Absolute tolerance | Relative tolerance |
| --- | ---: | ---: |
| Forward/output accuracy | 2e-5 | 2e-4 |
| Gradient accuracy | 3e-5 | 3e-3 |
| Probability/math checks | 2e-7 | 2e-6 |
| Independent FP64 oracle cross-check | 1e-10 | 1e-10 |

All existing optimizer, trajectory, and finite-difference tolerances remain
unchanged. Widths 32–256 retain the strict free-running trajectory contract;
512/1024 retain the existing common-state/identical-gradient replay contract,
with explicit legacy trajectory results. Each engine's gradients must be
tested through the identical-gradient optimizer replay, including moments and
multiple steps. Keep shared-checkpoint accuracy distinct from independent
training-trajectory agreement.

`transformer/gpu_precision.py` uses development seed **20261103** and reserves
**20261104** for admission, disjoint from CPU v6 streams. Family/case-separated
random streams prevent one fixture family from consuming another's draws.
The initial development inventory contains 23 direct-attention patterns at
every width and 22 LayerNorm/QKV patterns: random row counts 1/3/32/33/511/512;
common offsets 0/64/4096 independently crossed with spreads 0/2^-12/.01/1;
signed or zero affine channels; large common QKV bias; and projection
cancellation. All operands and nonzero incoming gradients are stored as FP32.
The oracle converts these exact values to FP64; construction targets do not
substitute for the stored operands.

Attention covers common-score scale independently of near-tied gaps,
contrasting V, cancellation, causal prefixes, future interventions, head widths
8 through 256, and nonzero derivatives. Retain two distinct backward checks:
the full mathematical FP64 softmax derivative and the derivative evaluated using
the exact saved rounded probabilities. Upstream checks compare normalized
values, affine LayerNorm, QKV projections, and gradients of x/gamma/beta/weight/
bias. Zero variance must remain finite through epsilon without incorrectly
erasing meaningful input gradients. Independent NumPy analytic gradients are
cross-checked by PyTorch FP64 autograd and selected finite differences.

This inventory is **development-only**, not an assertion that full admission is
implemented. Batched interventions, complete check names and counts, guards,
malformed cases, and end-to-end/optimizer fixtures must be defined and committed
prospectively before admission generation is enabled. Admission failures must
remain failures; changes afterward require a new explicitly versioned contract
and a new reserved stream, not reuse disguised as fresh evaluation.

## Bounded development candidate search

First profile the correct development reference path. Then freeze a finite,
explicit candidate ledger after capability discovery and before any evaluated
candidate search. The following bounds limit that ledger; they do not assert
that any listed tile or execution feature is supported:

| Family | Proposed finite dimensions | Maximum signatures per width |
| --- | --- | ---: |
| Each dense direction: forward, dX, dW | M tile {8,16}; N tile {16,32}; K tile {8,16}; SIMD groups {1,2}; arithmetic {fixed partial sums, compensated/two-float} | 32 per direction |
| T32 attention forward/backward pair | Query rows/group {1,2}; score staging {register, threadgroup}; arithmetic {fixed partial sums, compensated/two-float} | 8 |
| LayerNorm forward/backward pair | SIMD groups {1,2}; reduction {two-pass fixed tree, pairwise}; arithmetic {ordinary, compensated} | 8 |
| Whole-step fusion/scheduling | Explicit unfused reference; residual/bias elementwise fusion; optimizer elementwise fusion; both | 4 |

These are candidate definitions, not promised implementation features. Unsupported
or intentionally unimplemented entries must be marked before search. The final
ledger can be a smaller legal subset chosen from capability and implementation
constraints, with reasons recorded before scoring. Actual thread counts derive
from discovered execution width and limits. Record exact tile/layout/arithmetic/
compiler/pipeline identifiers; a macro name alone is insufficient. The family
caps do not authorize an unbounded cross-product of complete models.

Every evaluated candidate first passes unchanged development correctness tests.
Failed/unsupported candidates remain in the ledger and receive no performance
ranking. Rank passing candidates using completion-synchronized development
microbenchmarks and complete B16/T32 step measurements, with initial state
restored after warmup; the final ledger must freeze sample counts, objective,
tie-breaking, and traversal order before scoring. B1/T32 forward may select a
separately declared inference dispatch policy. Test-set quality and admission
data never enter selection. No data-dependent expansion after an unfavorable
result is permitted within this contract.

## Admission and stop rule

Before the first formal fit, commit the final source/shaders, dependency lock,
compiler settings, selected dispatch policies, protocol, complete gates, fixtures,
counterbalanced order, runner, auditor, reporter, and evidence packaging logic.
Authenticate current sources, compiled artifacts/pipelines, configuration, and
all fixture bytes. Open admission only after this freeze.

Admission covers every width and engine: primitive shapes/values; common-state
forward/backward and all parameter gradients; multiple Adam steps/moments;
retained width-specific trajectory checks; causal prefixes/future interventions;
repeated embeddings and shared gradients; tails and partial batches; bounds,
races, barriers, input immutability and finite states; malformed input and output
collisions; checkpoint selection/replay; exact within-engine repeated execution.
Every required check has an authenticated unique name, tolerance, operand
identity, shape, and result. Missing checks cannot become implicit passes.

Custom Metal's added FP64 accuracy checks are blocking. Standard MPS/MLX FP32
stress disagreements are retained as explicit diagnostics, as in CPU v6;
baseline shape/finite/runtime/common-state/optimizer requirements still block.
Post-run FP64 accuracy for each engine remains a separate qualification, never
implied by admission or good aggregate quality. Exact replay/within-engine
repeatability is required and is not replaced with allclose.

If any required admission check fails, retain evidence and stop before the
108-fit campaign. This version does not authorize a descriptive campaign with
failed blocking admission. Such a campaign requires a new prospective amendment
that identifies the failed contract and outcome categories before any formal fit.

## Matched measurement and planned denominator

Freeze the balanced three-engine order and 15-second inter-fit cooldown before
timing. One measured process runs at a time on AC power, with fixed one-thread
host numerical libraries and recorded power/thermal/load information. No builds,
profiling, validation, or other project numerical work runs concurrently. A
changed source/input/artifact or invalid execution condition stops the campaign
and preserves its partial evidence. Every one of the planned 108 cells remains
in the report; failed, blocked, and skipped cells are explicit. Retrying a run
uses a new path and never erases the original attempt.

Training time includes forward/backward, gradient reductions, Adam, finite
checks, epoch validation, checkpoint selection/writes, and bookkeeping. Required
packing/repacking or synchronization cannot be shifted outside the timer.
Setup, compilation, warmup, final test evaluation, and inference samples are
reported separately. Warmup must restore exact parameters, moments and step.
No warmed gradient or accumulator may contribute to a measured update; buffers
must be reset or completely overwritten before use. Compiled pipeline caches
remain warm by design, with their setup cost recorded separately. Warmup covers
the exact training/evaluation shapes and finite-check paths used by the fit.

Every timed interval completes GPU work before stopping its host clock. Primary
forward latency is B1/T32 after 20 warmups, with 100 retained samples, excluding
loss, output-probability softmax, sampling, token loading, and I/O. Freeze the
input upload and output materialization boundaries identically across engines;
any device-only timer is labeled separately. Secondary B16 throughput does not
replace primary latency. Report allocated tensor memory, runtime/driver memory,
and process RSS with their distinct definitions and available peak-reset semantics.

## Post-run audit, analysis, and publication

For each selected checkpoint from all three origins, evaluate every validation
and test row through every GPU engine and independent CPU FP64 equations using
the exact stored parameters. Cross-check the independent FP64 oracles at
prespecified locations. Require exact originating-engine replay with original
batch semantics, selected history/epoch consistency, optimizer/state counts,
file/source/pipeline identity, finite states, and repeat equality. Separately
retain per-engine FP64 accuracy and all mutual FP32 compatibility results.

Preserve complete false checks, failing coordinate counts/examples, raw-logit
error, probability error, NLL delta, and argmax changes. Reports separately expose
evidence integrity, replay/repeatability, custom/MPS/MLX FP64 accuracy, mutual
compatibility, and saved-prediction quality. Full numerical qualification requires
all declared accuracy and integrity conditions; a precise failing-engine status
must survive even when language-model quality appears unaffected. Final status
names and complete expected record counts must be frozen with the reporter
before admission; this development document does not invent completed statuses.

Report by width the geometric mean of six matched MPS/custom training ratios
and six compiled-MLX/custom ratios, with raw paired measurements. Forward
comparisons use matched per-fit medians and retain p50/p95/sample distributions.
Average repeats within seed before quality mean and sample SD across three
seeds; no pooled-width speedup or treatment of repeats as independent seeds.
Report throughput, memory, variability, qualification, and all failures alongside
speed. Preserve the 884,736 planned test-token denominator with actual completion
counts, not merely a count of successful rows.

Use static plots/dataflow and an idea-first explanatory README, with results
later and no Mermaid controls or references that make the project dependent on
earlier unrelated experiments. Keep CPU evidence intact. Package all raw states,
predictions, gates, candidates, failures, timing samples, conditions, source/
compiler/dependency identities, and reproducibility tools; authenticate the
full release archive and public assets. Scope conclusions to this model family,
machine, FP32 storage, and tested standard GPU baselines. No long-context,
cached-decoding, energy, general GPU superiority, or production-serving claim
follows from these measurements.
