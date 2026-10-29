# ARM64 Transformer

[![Validation](https://github.com/logannye/arm64-transformer/actions/workflows/validate.yml/badge.svg)](https://github.com/logannye/arm64-transformer/actions/workflows/validate.yml)

**How much can a transformer gain when its computation is designed around the
machine that executes it?** This project implements a small language model's
complete forward pass, backward pass, and Adam optimizer in handwritten ARM64
CPU assembly and custom Metal GPU kernels. Standard PyTorch and MLX provide
comparison implementations.

## The idea: make the hardware shape the computation

Transformer equations leave many engineering decisions open. Which outputs
share an input load? Where do partial sums live? Which threads cooperate? When
do values round to 32 bits? Those choices determine how much useful work a
processor performs and whether the answer remains accurate.

On the CPU, tiles of outputs share packed operands and accumulate in Arm's
**Scalable Matrix Extension (SME)** registers. NEON vector instructions handle
attention, normalization, exponentials, and the remaining arithmetic. The
native numerical path calls no BLAS, Accelerate, PyTorch, or external math
library. Backward and the optimizer receive the same attention as inference.

![Packing operands for matrix accumulation](docs/sme-dataflow.png)

On the Apple GPU, the useful unit of work becomes a cooperating group of
threads. Our **Metal** implementation assigns matrix tiles, rows, and reductions
to those groups, reuses data in registers and threadgroup memory, and combines
operations to avoid unnecessary launches and global-memory reads. The model
keeps the same equations and FP32 stored state.

LayerNorm gives a concrete example. The reference assigns an entire row to one
GPU thread, which walks its channels serially. The replacement assigns a row to
a cooperating group: adjacent lanes process adjacent channels, then combine
their partial sums in a fixed, compensated reduction. The equations stay the
same while much more work can proceed in parallel.

| Computation | Hardware mapping |
|---|---|
| Dense forward, input gradients, weight gradients | Cooperative FP32 SIMD-group matrix operations; guarded tiles and explicit output ownership |
| LayerNorm and shared gradients | Parallel row/channel reductions with fixed trees and compensated sums |
| Short-context attention | Fused score formation, centering, softmax, and value mixing; fused parts of backward |
| Repeated-token embedding gradients | Stable integer grouping followed by a deterministic floating-point reduction |
| Elementwise work | Selected matrix-output activation/residual fusion and Adam/finite-scan fusion |
| Execution | Persistent pipelines and buffers, resident inputs, separate training and batch-one policies |

A faster primitive is only a candidate. We select schedules by the complete
training step or synchronized forward pass, after checking numerical accuracy.
A wide matrix tile can help a training batch and hurt a single-context request.
The [GPU kernel notes](native/metal/OPTIMIZED-KERNELS.md) describe the layouts,
barriers, precision, and retained reference paths.

## Precision is part of the architecture

Attention can depend on the tiny difference between two large query–key scores.
Subtracting the maximum before softmax prevents overflow, but cannot recover
bits already lost while forming those scores. Normalization and backward
reductions can expose similar cancellation.

The CPU uses selective FP64 intermediates for sensitive wide attention. The GPU
uses **two-float arithmetic**: a pair of FP32 values carries a main value and its
rounding residual through important contractions. Scores retain that information
through centering and softmax; normalization and sensitive QKV paths also use
compensated arithmetic. Parameters, saved activations, probabilities, gradients,
and Adam moments remain FP32. Extra intermediate precision has a measurable cost.

Accuracy, exact repeatability, and predictive quality are tested separately.
Small aggregate loss differences do not waive a failed logit or gradient check.
The [CPU numerical investigation](experiments/numerics-v6/README.md) records
historical failures that remain unresolved.

The latest QKV experiment asks whether we can reuse more operands **without
changing even the output bytes**. A cooperating group loads a tile of inputs
and weights; each thread maintains four or eight independent two-float sums.
Padded, transposed shared-memory tiles improve access layout, and every output
still visits its products in the original order. This avoids atomics and
cross-thread floating-point sums. Larger tiles also need more live accumulators
and barrier coordination, so their value must be established by complete-step
measurements. The [QKV kernel notes](native/metal/QKV-KERNELS.md) explain the
ownership, guarded tails, precision, and versioned dispatch policy.

## The model and matched workload

One causal decoder block, four heads, context 32, feed-forward width 2W with
ReLU, three LayerNorms, and an ASCII-128 vocabulary predict the next character.
Widths range from 32 to 1024. Pinned Tiny Shakespeare partitions contain 65,536
training targets and 8,192 targets each for validation and test.

The formal workload uses batch 16, eight epochs, and 1,024 Adam updates, with
identical initialization bytes and schedules for seeds 42, 43, and 44. Each seed
has two timing repeats. Lowest own-engine validation NLL selects the checkpoint;
the earliest epoch wins ties. This is a reused engineering benchmark with a
fixed training budget, not a fresh generalization assessment or convergence study.

## GPU engineering results

The [QKV v3 protocol](experiments/PROTOCOL-gpu-v3-qkv.md) freezes eight forward
and eight input-gradient schedules at each width. Candidates must pass the
existing FP64 gates and match the retained implementation byte for byte before
timing. A policy changes only after at least a 3% improvement in the complete
B16/T32 training step or B1/T32 forward pass. Training and inference are selected
separately. The search uses original training sequences, with no validation or
test outcomes used for optimization.

**All 144 candidates passed their numerical checks; none met the 3% adoption
threshold.** Every width retains its previous kernel choices. The target widths
illustrate the result:

| Width | Best training candidate | Best batch-one candidate | Decision |
|---:|---:|---:|---|
| 512 | 1.63% lower latency | 8.90% higher latency | Retain incumbent |
| 1024 | 2.41% lower latency | 1.03% higher latency | Retain incumbent |

![Every QKV schedule and its complete-operation effect](experiments/gpu-v3-qkv/results/qkv-candidate-grid.png)

The wider tiles reuse more operands but keep the same compensated arithmetic
work and need more live accumulators. They also create fewer independent
groups, a potential disadvantage for small batches. These are structural
tradeoffs; hardware counters were unavailable, so register pressure and
occupancy are not claimed as measured causes.

Final FP64 checks and exact replay pass in two fresh processes per width,
including full gradients, six-step parameters and moments, and checkpoint
predictions. The [QKV report](experiments/gpu-v3-qkv/README.md) includes all
candidate outcomes, p50/p95, paired timing variation and separate stage profiles.
The final selected-versus-incumbent measurements compare unchanged kernel
choices; their small differences around 1× are not a new speedup.

### Retained GPU v2 results

The earlier cooperative dense, normalization, reduction and fusion work did
produce substantial gains against the original Metal reference: individual
widths show **1.84–2.40× training-step** and **1.22–2.76× forward** improvements.
At width 1024, training-step latency fell from 25.079 to 13.641 ms, and forward
latency from 2.062 to 0.818 ms. These development measurements include backward,
Adam, finite checks and completion, but exclude full-fit validation and checkpoint
work. Their [full results and figures](experiments/gpu-v2/README.md) and
[selected policies](experiments/gpu-v2/policies/selected.json) remain unchanged.

### Full-study status

The formal five-engine CPU/GPU study remains **blocked: 0 of 180 planned fits**.
The historical PyTorch MPS and MLX runs fail the required byte-exact training
repeatability checks at every width. A later environment audit found that those
runs used forced-Metal MPS dispatch and graph-disabled MLX: both presence-based
controls had incorrectly been set to `"0"`. The [configuration correction](experiments/gpu-compile-control-v1/CORRECTION.md)
preserves the evidence and requires fresh checks with ordinary MPS dispatch and
verified MLX compilation. The first divergent full-gradient case
examined at each width localizes the initial difference to token embeddings;
subsequent updates propagate it into parameters, optimizer moments, and predictions.
The historical baseline FP64 accuracy comparisons pass, but MLX also fails T3
full-model/training cases at widths 256–1024. Those missing results remain errors.

The baseline algorithms, gate, and tolerances remain unchanged. No replacement
embedding implementation or descriptive formal-campaign waiver was introduced.
The reserved admission stream remains unopened. There are no new qualified
CPU/GPU crossover, Metal/framework speedup, or model-quality results.
The [GPU v2 report](experiments/gpu-v2/README.md) separates development measurements,
failed attempts, baseline limitations, and work that remains unexecuted.

| Study component | Current evidence |
|---|---|
| Complete forward, backward and Adam implementations | Native CPU, PyTorch CPU, custom Metal, PyTorch MPS and MLX are implemented |
| Matched full CPU fits | 72 historical v6 fits; wide-model precision limitations remain |
| Custom GPU kernel development | Six widths, full-step/forward comparisons, FP64 checks and exact replay; no full-fit quality claim |
| Standard GPU baseline admission | Fresh qualification with corrected controls is required; historical failures remain recorded |
| Fresh five-engine study and checkpoint audit | 0/180 fits; admission remains unopened |

Official stable-source inspection found no eligible repair in PyTorch 2.15.0 or
MLX 0.32.4. The separately prepared framework patches remain development work;
they do not qualify the released standard baselines. Unblocking the full study
requires suitable supported releases, successful baseline checks, a new bound
source/runtime/policy freeze, admission, and the 180 fits with all-checkpoint
audits. The [feasibility receipt](experiments/gpu-v3-qkv/baseline-feasibility.json)
records the inspected tagged sources and distinguishes source inspection from
runtime validation.

## Retained CPU results

The completed CPU v6 study contains 72 fits. Widths **32–256 satisfy both
implementations' FP64 accuracy requirements**: native training is **1.29–1.79×
faster**, and full-context forward inference is **1.43–3.53× faster** than eager
PyTorch on one CPU compute thread. At width 32, forward latency is 15.1 µs versus
53.2 µs.

![Historical CPU time ratios and numerical qualification](experiments/width-sweep-v6/speedup.png)

Widths 512 and 1024 are not fully numerically qualified and favor PyTorch on time.
The retained audit finds native logit failures at width 1024, and PyTorch failures
at widths 512 and 1024. These historical limitations have not been relabeled as
resolved. See the [full CPU report](experiments/width-sweep-v6/README.md) for
learning curves, memory, paired timings, and prediction-level effects. These
historical CPU numbers are separate from the new GPU development measurements.

## Reproduce and inspect

The [QKV v3 reproduction guide](experiments/gpu-v3-qkv/REPRODUCE.md) covers the
fixed candidate sweep, exact replay, report generation and evidence packaging.
Its [evidence release](https://github.com/logannye/arm64-transformer/releases/tag/transformer-gpu-qkv-v3)
contains the new source snapshots, raw states and logits, timing samples,
candidate records and archive verification.

The [GPU v2 protocol](experiments/PROTOCOL-gpu-v2.md) fixes the workload, candidate
search, numerical gates, and measurement boundaries. The
[formal-study guide](experiments/gpu-v2/FORMAL-STUDY.md) documents the authenticated
runner and full-checkpoint auditor. Its 180-cell plan is committed; an ordinary
flag or a self-reported pass cannot enable formal measurement.

The [release guide](experiments/gpu-v2/RELEASE-TOOLS.md) describes report generation,
complete evidence retention, and split-archive verification. The
[GPU v2 evidence release](https://github.com/logannye/arm64-transformer/releases/tag/transformer-gpu-engineering-v2)
contains raw states, predictions, candidate ledgers, timing samples, failures,
runtime identities, source snapshots, and verification tools. The
[CPU reproduction guide](docs/REPRODUCE.md) and [native ABI](native/ABI.md) cover the
assembly implementation. Historical
[CPU v6](https://github.com/logannye/arm64-transformer/releases/tag/transformer-attention-precision-v6)
and [GPU development v1](https://github.com/logannye/arm64-transformer/releases/tag/transformer-gpu-development-v1)
releases remain unchanged.

The measurements concern short-context, full-context forward computation and a
small training workload. They do not establish long-context, cached decoding,
energy, production-serving, or larger-data performance.
