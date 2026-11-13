# ARM64 Transformer

[![Validation](https://github.com/logannye/arm64-transformer/actions/workflows/validate.yml/badge.svg)](https://github.com/logannye/arm64-transformer/actions/workflows/validate.yml)

**One small transformer, five implementations, and a question: how much does
execution design matter when the learning algorithm stays the same?** This
project builds a character-level language model in handwritten ARM64 CPU
assembly, custom Metal GPU kernels, and Python frameworks. Each implementation
trains the model end to end, including backward propagation and Adam updates.

We investigate speed, numerical accuracy and repeatability together. Memory
layout, parallel work, reduction order and arithmetic precision all affect the
cost of the same equations. The experiment asks which choices improve complete
training and inference, and which apparent optimizations fail under measurement.

## Results at a glance

Measured on one Apple M4 Max, for models of 18 thousand to 600 thousand
parameters (widths 32–256):

- **The handwritten assembly trained 2.4–2.7× faster than the study's PyTorch
  CPU engine** and ran single-sample inference 4.0–7.5× faster, at 16–82
  microseconds per 32-character context. That PyTorch engine is a
  project-maintained build with extra-precision operators, not a stock release.
  Against stock PyTorch in an
  [earlier two-engine study](experiments/width-sweep-v6/README.md), the assembly
  trained 1.3–1.8× faster at these widths and was slower at widths 512 and 1024.
  [Both comparisons](#accepted-results) are set out below.
- **The fastest engine depends on model size.** The assembly had the lowest
  training time at widths 32–128; the custom Metal GPU kernels led at width 256.
- **Every checkpoint was audited, not only timed.** All 120 training runs
  passed: 1,200 cross-engine replays and 240 checkpoint audits against
  independent 64-bit calculations, and all 60 same-engine repeat pairs matched
  exactly.
- **Passing loss checks did not guarantee correct gradients.** In one
  development case outside the accepted runs, a rounding difference below one
  millionth flipped the sign of a ReLU input. Logits and loss passed; 12 weight
  gradients were wrong. See
  [what the investigations revealed](#what-the-investigations-revealed).

![Training time and resident batch-one latency for all five engines; lower is better](experiments/gpu-v5-library-phase1/results/performance.png)

Each point is the median of six training runs; both vertical axes are
logarithmic.
[Full results](experiments/gpu-v5-library-phase1/results/README.md) ·
[Numeric data](experiments/gpu-v5-library-phase1/results/measurements.csv) ·
[Vector figure](experiments/gpu-v5-library-phase1/results/performance.svg).

Widths 512 and 1024 are deferred, so 120 of the 180 planned training runs are
complete. See [scope and remaining work](#scope-and-remaining-work).

## Why this exists

This follows an earlier exploration from the
[low-latency neural decoding](https://github.com/logannye/low-latency-neural-decoding)
project, a brain-computer interface testbed that measures streaming decoders
against real-time deadlines. The question carried over from that work: when a
model is small and must run inside a fixed time and memory budget, how much is
gained by controlling the implementation down to the instruction level, and
what does it cost in numerical trust? A one-block transformer is a small model
that still exercises attention, normalization and a complete training loop.

The model here is a character-level language model trained on public text. It
is not a neural decoder, and these measurements make no claim about embedded or
implanted hardware.

## Quick start

Requires an Apple silicon Mac, recent Xcode Command Line Tools (the assembler
must accept SME instructions) and Python 3.11 or newer.

```sh
git clone https://github.com/logannye/arm64-transformer.git
cd arm64-transformer
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --only-binary=:all: -e '.[dev,report]'
make -C native widths PYTHON=python
ARM64_TRANSFORMER_WIDTH=64 python -m pytest -q
ARM64_TRANSFORMER_WIDTH=64 python -m transformer.validate --runtime \
  --output results/quickstart/width-64.json \
  --work-dir results/quickstart/run-64
```

`make` assembles the native library and executable for all six widths in a few
seconds, and the test suite takes under a minute on an M4 Max. The last command
audits the assembly kernels, their gradients, the optimizer and the standalone
native executable against independent references, then prints `"passed": true`
with its check count. It refuses to overwrite earlier output, so give it fresh
paths on a rerun. On chips with SME, such as the M4 family, widths 64 and above
use the SME matrix kernels; other Apple silicon falls back to NEON.
[Hosted CI](.github/workflows/validate.yml) runs the same steps at every width.

```sh
python -m transformer.fetch_data --output data/tinyshakespeare.txt
python -m transformer.demo
```

These two commands download the 1.1 MB Tiny Shakespeare text, check it against
a pinned SHA-256, then train the width-32 model once in assembly and once in
stock PyTorch on one thread and print both training times and their ratio. It
is an informal single run against stock PyTorch, not the accepted measurement
reported below; `ARM64_TRANSFORMER_WIDTH` selects another width, and at 512 and
1024 the assembly is the slower engine.

## How it works

The model predicts the next ASCII character using one causal decoder block:
four attention heads, context length 32, a ReLU feed-forward layer with twice the
model width, three LayerNorms and a 128-character vocabulary. Every engine uses
the same equations, fixtures and training schedule.

![The shared transformer computation and its two CPU and three GPU implementations](docs/figures/model-and-engines.png)

On the CPU, packed matrix tiles feed Arm's **Scalable Matrix Extension (SME)**
registers, while NEON vectors handle attention, normalization and other
arithmetic. Reusing operands and keeping partial sums in registers reduces
memory traffic. The width-32 model uses NEON; larger models use SME where
supported. The native numerical path calls no BLAS, Accelerate or external math
library. See the [matrix dataflow](docs/sme-dataflow.png) for the detailed design.

The GPU investigations explore cooperative tiles, fixed reduction trees,
deterministic embedding gradients and operation fusion. The accepted comparison
runs the Metal engine on its **reference kernels**, the plain versions, with a
precision refinement in normalization. Results for the optimized kernels are
development measurements, reported separately below.

**The three framework engines are not stock releases.** PyTorch CPU, PyTorch MPS
and MLX are [built from source by this project](experiments/gpu-v5-library/framework-selection.json),
with its patches and an added precision API. In the accepted comparison they
compute selected stages with extra precision. PyTorch CPU uses FP64
intermediates in the first LayerNorm, the query/key/value (QKV) projection,
attention and the first feed-forward projection (FF1). MLX uses two-float FP32
arithmetic for the same four stages and MPS for the first three; two-float
arithmetic carries each value as a pair of floats so that rounding error is
kept, and assumes no GPU FP64. The protocol allows no width-specific framework
arithmetic, so these operators run at every width. The ARM64 engine is compiled
per width and runs its ordinary kernels, with no added refinement, at widths
32–256.

All model state is stored in FP32; the arithmetic of every engine is
[declared in the summary](experiments/gpu-v5-library-phase1/results/summary.json).
The frameworks also execute compiled native kernels, so this comparison measures
complete implementations rather than isolating Python-language overhead.

## The matched experiment

Pinned Tiny Shakespeare partitions contain 65,536 training targets and 8,192 each
for validation and test. Each fit, meaning one complete training run, uses
batch 16, eight epochs and 1,024 Adam updates. Three paired seeds (42–44) and
two repeats give six fits per engine and width. Each fit selects its
lowest-validation-loss checkpoint, breaking ties by the earliest epoch. This is
a reused engineering benchmark with a fixed budget.

Before any timed fit, every engine must pass a fixed set of numerical checks,
which the protocol calls qualification and admission. Timed fits then run one
at a time on one numerical thread. Both the best and last checkpoint from every
fit are replayed through all five engines and checked against independent FP64
calculations, with tolerances fixed before the runs. Same-engine repeatability
and predictive quality are assessed separately.

## Accepted results

The first phase **passed all required gates and audits for 120 fits at widths
32, 64, 128 and 256** on an Apple M4 Max with 128 GiB memory, running macOS 26.2.
All **1,200 engine replays, 240 independent FP64 oracles, 240 checkpoint audits
and 60 exact repeat pairs** passed, covering 983,040 audited test-token predictions.

In the [chart above](#results-at-a-glance), training time includes forward,
backward, Adam, finite checks, loss reads, validation, checkpoint writes,
packing, synchronization and bookkeeping. Inference is resident batch-one: one
32-character context whose input is already in the engine's memory, timed
through forward execution and completion, with 20 warmups and 100 samples per
fit.

**The CPU gains survive complete training runs.** ARM64 trained **2.38–2.70×
faster** than the study's PyTorch CPU engine and delivered **3.99–7.52× faster
resident batch-one inference**. The median paired-seed speedups are:

| Width | ARM64 training speedup | ARM64 batch-one speedup |
|---:|---:|---:|
| 32 | 2.53× | 7.52× |
| 64 | 2.70× | 7.28× |
| 128 | 2.48× | 5.16× |
| 256 | 2.38× | 3.99× |

**What that baseline is.** These speedups are over the PyTorch CPU engine as
configured for this study: a project-maintained build on one numerical thread,
with FP64 intermediates at four stages. They are not speedups over a stock
PyTorch release. The earlier
[two-engine study](experiments/width-sweep-v6/README.md) measured stock
PyTorch 2.14.1 on an Apple M4 Max with the same model, data partitions and
training schedule, also on one thread, under its own protocol:

| Width | Training speedup | Batch-one forward speedup | Peak training memory, PyTorch → ARM64 | Numerical audit |
|---:|---:|---:|---:|---|
| 32 | 1.79× | 3.53× | 276 → 2.3 MiB | Passed |
| 64 | 1.76× | 3.20× | 278 → 3.1 MiB | Passed |
| 128 | 1.47× | 2.12× | 288 → 5.7 MiB | Passed |
| 256 | 1.29× | 1.43× | 314 → 14.6 MiB | Passed |
| 512 | 0.93× | 0.79× | 392 → 47 MiB | PyTorch failed |
| 1024 | 0.72× | 0.72× | 654 → 173 MiB | Both failed |

Values above 1 favor the assembly. At widths 512 and 1024 the assembly was
slower, and neither width passed that study's numerical qualification, so
those two rows are descriptive. Memory is the whole process's peak resident
size, including the framework and data; it is not model memory. The two studies
ran under different protocols and numerical gates, so their ratios should not
be combined.

**The fastest engine depends on the workload.** ARM64 had the lowest median
training time at widths 32–128. Metal led at width 256: **3.593 s**, versus
**3.981 s** for ARM64. ARM64 had the lowest batch-one latency at every included
width. The study's MPS and MLX engines were slower than both CPU engines here;
this short-context experiment does not establish a general CPU-over-GPU
advantage.

Those two engines carry a cost the other engines do not. Their selected stages
use two-float arithmetic assembled from ordinary tensor operations, so one
matrix product becomes a broadcast elementwise product followed by a pairwise
compensated sum. The study did not measure how much of their time this accounts
for, and this repository holds no accepted timing for standard MPS or MLX
releases on this workload.

Test loss and accuracy vary across engines. Passing the numerical checks does
not imply identical training trajectories or identical final model quality.
[All 20 engine/width summaries](experiments/gpu-v5-library-phase1/results/README.md)
remain available alongside the timing results.

## What the investigations revealed

### Tiny rounding differences can change gradients

Attention can depend on small differences between large scores. Subtracting the
maximum before softmax cannot recover bits already lost while forming those
scores. ReLU adds another sensitivity: rounding across zero changes whether a
gradient passes through at all.

In one historical MLX development case, a preactivation in the first
feed-forward projection (FF1) was about **+3.65 × 10⁻⁷**, versus
**−1.26 × 10⁻⁷** in FP64. Logit and loss checks passed, but 12 weight-gradient
coordinates failed. A controlled gate substitution isolated the mismatch to
that branch.

![A historical ReLU sign disagreement switches the backward multiplier from zero to one](docs/figures/precision-gate.png)

Selective FP64 and compensated two-float arithmetic corrected specific failures
while retaining FP32 model storage. Their cost stays inside the timing boundary.
The [precision investigation](experiments/trained-precision-v1/STATUS.md) separates
passing development repairs from formal qualification: neither good average
loss nor passing initialization checks can replace trained-state gradient audits.

### Repeatability is an implementation property

Embedding backward combines many contributions to the same output. Floating-point
atomic updates can accumulate them in different orders. The framework repairs
use stable grouping and defined accumulation order; attention-dispatch fixes
also preserve the state backward needs. Determinism has an explicit computational
cost. The [integration report](experiments/upstream-integration-v2/README.md)
retains targeted successes, broad-suite failures and their controls.

### A plausible optimization still has to win overall

Separate Metal development experiments achieved **1.84–2.40× faster training
steps** and **1.22–2.76× faster forward passes** versus the original Metal
implementation. Parallel reductions, cooperative matrix tiles and selected
fusion improved that workload. These are development measurements, separate
from the reference-kernel Metal engine in the accepted full-fit comparison.
[GPU results](experiments/gpu-v2/README.md).

A subsequent [144-candidate search](experiments/gpu-v3-qkv/README.md) over
kernels for the QKV projection passed correctness for every candidate, yet
**none met the 3% complete-operation improvement threshold fixed in advance**.
The incumbent kernels were retained. More operand reuse did not reliably reduce
the cost of the complete operation.

## Scope and remaining work

**Widths 512 and 1024, and their 60 fits, remain deferred. The original 180-fit
study is incomplete.** The [scope amendment](experiments/gpu-v5-library-phase1/AMENDMENT.md)
was approved after development and partial admission, before any formal fits.
Earlier wide-model failures, ordinary MLX diagnostic crashes and standard-release
repeatability failures remain outside this phase's acceptance.

There were no failed formal fits or final numerical audits in the accepted
phase. Two post-run execution failures were recovered with checkpoints and
failure evidence preserved; the [results report](experiments/gpu-v5-library-phase1/results/README.md#failures-retained)
records both. These measurements establish no long-context, cached-decoding,
energy, serving or larger-data performance claim.

## Explore and reproduce

- [Accepted report, scope and evidence identities](experiments/gpu-v5-library-phase1/results/README.md).
  Public summaries link to retained evidence by hash; the full current raw
  campaign archive remains local.
- [Reproduction guide for the two-engine CPU study](docs/REPRODUCE.md),
  [CPU kernel notes](native/kernel_notes.md),
  [GPU kernel design](native/metal/OPTIMIZED-KERNELS.md) and
  [QKV reproduction](experiments/gpu-v3-qkv/REPRODUCE.md).
- [Historical CPU study](experiments/width-sweep-v6/README.md),
  [framework reproduction](experiments/upstream-integration-v2/REPRODUCTION.md)
  and [public evidence catalog](experiments/upstream-integration-v2/PUBLICATION.md).
- [Figure sources and vector downloads](docs/figures/README.md).
  The visuals read retained summaries without rerunning the experiment.
