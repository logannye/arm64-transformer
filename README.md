# ARM64 Transformer

[![Validation](https://github.com/logannye/arm64-transformer/actions/workflows/validate.yml/badge.svg)](https://github.com/logannye/arm64-transformer/actions/workflows/validate.yml)

**One small transformer, five implementations, and a question: how much does
execution design matter when the learning algorithm stays the same?** This
project builds a character-level language model in handwritten ARM64 CPU
assembly, custom Metal GPU kernels, and Python frameworks. Each implementation
trains the model end to end, including backward propagation and Adam updates.

The experiment investigates speed, numerical accuracy and repeatability together.
It asks how memory layout, parallel work, reduction order and arithmetic precision
change the cost of the same computation, and which apparent optimizations survive
a complete training or inference measurement.

## How it works

The model predicts the next ASCII character using one causal decoder block:
four attention heads, context length 32, a ReLU feed-forward layer with twice the
model width, three LayerNorms and a 128-character vocabulary. The five engines
implement the same equations, fixtures and training schedule.

| Engine | Hardware | Implementation used in the accepted comparison |
|---|---|---|
| ARM64 | CPU | Handwritten SME/NEON assembly; no BLAS, Accelerate or external math library |
| PyTorch CPU | CPU | Project-maintained PyTorch with selected precision operators |
| Metal | GPU | Custom Metal kernels, using the frozen **reference kernel policy** |
| PyTorch MPS | GPU | Project-maintained PyTorch with selected precision operators |
| MLX | GPU | Project-maintained MLX with a compiled graph and selected precision operators |

The framework arms also execute compiled native kernels. This comparison measures
whole implementations, including their dispatch and arithmetic choices; it does
not isolate the overhead of the Python language.

On the CPU, packed matrix tiles feed Arm's **Scalable Matrix Extension (SME)**
registers. NEON vectors handle attention, normalization and other arithmetic.
Keeping partial sums in registers and reusing loaded operands reduces memory
traffic. The width-32 model uses NEON; larger models use SME where supported.
Backward and optimizer kernels receive the same attention as inference.
The [matrix dataflow diagram](docs/sme-dataflow.png) shows the packing and
accumulation paths.

The GPU investigations explore cooperative matrix tiles, fixed reduction trees,
stable embedding-gradient accumulation and operation fusion. A faster individual
kernel can still lose overall through extra launches or intermediate storage.
The [optimized GPU experiments](experiments/gpu-v2/README.md) are separate from
the reference-policy Metal arm measured below.

Precision is another execution choice. Rounding can alter the differences between
attention scores or flip a value across ReLU's zero boundary, changing gradients.
The project investigates selective FP64 and compensated two-float intermediates
while keeping model state in FP32 storage. Each arm's selected arithmetic is
[declared explicitly](experiments/gpu-v5-library-phase1/results/summary.json), and
its cost remains inside the timing boundary. These are **project-maintained
framework builds**, not measurements of unmodified standard releases.

## The matched experiment

Pinned Tiny Shakespeare partitions contain 65,536 training targets and 8,192 each
for validation and test. Each fit uses batch 16, eight epochs and 1,024 Adam
updates. Three paired seeds (42–44) and two repeats give six fits per engine and
width. Each fit selects its lowest-validation-loss checkpoint, breaking ties by
the earliest epoch. This is a reused engineering benchmark with a fixed budget.

Qualification and admission precede fitting. Timed fits run serially. Both the
best and last checkpoint from every fit are replayed through all five engines
and checked against independent FP64 calculations, with frozen tolerances.
Same-engine repeatability and predictive quality are assessed separately.

## Accepted results

The first phase **passed all required gates and audits for 120 fits at widths
32, 64, 128 and 256** on an Apple M4 Max with 128 GiB memory, running macOS 26.2.
All **1,200 engine replays, 240 independent FP64 oracles, 240 checkpoint audits
and 60 exact repeat pairs** passed, covering 983,040 audited test-token predictions.

![Training time and resident batch-one latency for all five engines](experiments/gpu-v5-library-phase1/results/performance.png)

Each point is the median of six fits; both vertical axes are logarithmic and
lower is better. Training includes forward, backward, Adam, finite checks,
validation, checkpoint writes, packing, synchronization and bookkeeping.
Inference measures resident batch-one forward execution and completion, using
20 warmups and 100 samples per fit.
[Full results](experiments/gpu-v5-library-phase1/results/README.md) ·
[Numeric data](experiments/gpu-v5-library-phase1/results/measurements.csv) ·
[Vector figure](experiments/gpu-v5-library-phase1/results/performance.svg).

ARM64 CPU trained **2.38–2.70× faster** than the selected PyTorch CPU arm and
delivered **3.99–7.52× faster resident batch-one inference**. The paired-seed
speedups are:

| Width | ARM64 training speedup | ARM64 batch-one speedup |
|---:|---:|---:|
| 32 | 2.53× | 7.52× |
| 64 | 2.70× | 7.28× |
| 128 | 2.48× | 5.16× |
| 256 | 2.38× | 3.99× |

ARM64 had the lowest median training time at widths 32–128. Metal was fastest
at width 256: **3.593 s**, versus **3.981 s** for ARM64. ARM64 had the lowest
batch-one latency at every included width. The selected MPS and MLX arms were
slower than both CPU arms on this short-context workload.

Test loss and accuracy vary across engines. Passing the numerical checks does
not mean identical training trajectories or identical final model quality;
[all 20 engine/width summaries](experiments/gpu-v5-library-phase1/results/README.md)
remain available alongside the timing results.

## Takeaways and open questions

Hardware-aware execution can substantially reduce the cost of this small
transformer. The winning implementation depends on model width and whether the
work is training or batch-one inference. Arithmetic precision is part of that
cost, and correctness must be established on trained states as well as at
initialization.

Optimizations also produce useful negative results. A separate
[144-candidate QKV search](experiments/gpu-v3-qkv/README.md) passed correctness
for every candidate, but none met the 3% complete-operation adoption threshold.
The incumbent kernels were retained.

**Widths 512 and 1024, and their 60 fits, remain deferred. The original 180-fit
study is incomplete.** The [scope amendment](experiments/gpu-v5-library-phase1/AMENDMENT.md)
was approved after development and partial admission, before any formal fits.
Earlier wide-model failures, ordinary MLX diagnostic crashes and standard-release
repeatability failures remain outside this phase's acceptance.

There were no failed formal fits or final numerical audits in the accepted
phase. Two post-run execution failures were recovered with the original
checkpoints and failure evidence preserved; the
[results report](experiments/gpu-v5-library-phase1/results/README.md#failures-retained)
records both. These measurements establish no long-context, cached-decoding,
energy, serving or larger-data performance claim.

## Explore and reproduce

- [Accepted report, scope and evidence identities](experiments/gpu-v5-library-phase1/results/README.md).
  Public summaries link to retained evidence by hash; the full current raw
  campaign archive remains local.
- [CPU implementation and reproduction](docs/REPRODUCE.md),
  [GPU kernel design](native/metal/OPTIMIZED-KERNELS.md) and
  [QKV reproduction](experiments/gpu-v3-qkv/REPRODUCE.md).
- [Historical CPU study](experiments/width-sweep-v6/README.md) and
  [framework integration results](experiments/upstream-integration-v2/README.md),
  including retained broad-suite failures and their controls.
- [Framework reproduction guide](experiments/upstream-integration-v2/REPRODUCTION.md)
  and [public evidence catalog](experiments/upstream-integration-v2/PUBLICATION.md).
  Historical evidence packages remain distinct from the accepted phase above.
