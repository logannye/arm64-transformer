# ARM64 Transformer

[![Validation](https://github.com/logannye/arm64-transformer/actions/workflows/validate.yml/badge.svg)](https://github.com/logannye/arm64-transformer/actions/workflows/validate.yml)

A complete small language-model training stack, implemented in ordinary PyTorch
and handwritten Apple ARM64/NEON assembly. The central experiment is how much
performance comes from organizing a model's execution around the processor's
registers, vector instructions, and memory accesses.

The latest experiment scales model width through **32, 64, 128, and 256**, from
**17,952 to 601,472 parameters**, with matched data and training rules. Across
48 fits on one Apple M4 Max, native training wins at width 32, and native
full-context forward latency wins at widths 32 and 64. PyTorch is faster on both
at widths 128 and 256. Native process memory remains much lower throughout.

**The useful finding is the crossover:** an implementation optimized for a tiny
model does not keep its advantage automatically as the model grows. The complete
[width-sweep report](experiments/width-sweep-v3/README.md) preserves every width,
seed, repeat, and unfavorable result.

![Implementation time ratios across all four model widths](experiments/width-sweep-v3/speedup-publication.png)

## The width experiment

Each model has one pre-LayerNorm causal decoder block, four attention heads,
learned token/position embeddings, residual connections, a ReLU feed-forward
network of width `2 × model_width`, final LayerNorm, and an untied output head.
It predicts the next ASCII character from a 32-character context. PyTorch uses
built-in scaled-dot-product attention, autograd, and `torch.optim.Adam`; its tensor
operations already execute in optimized native code.

Pinned Tiny Shakespeare excerpts supply **65,536 training, 8,192 validation, and
8,192 test target characters** from disjoint contiguous partitions. These are the
same known benchmark used in the earlier transformer versions. The sweep makes
no fresh, untouched test-set claim.

All widths use eight epochs, batch size 16, 1,024 Adam updates, learning rate
0.001, and one CPU compute thread. For each width and seed, both implementations
receive identical initial weights, token rows, and batch schedules. The lowest
validation NLL selects the checkpoint before test evaluation. Three seeds and two
repeats per implementation produce 48 fits, with width and implementation order
reversed on the second pass. See the frozen [v3 protocol](experiments/PROTOCOL-v3.md).

| Width | Parameters | PyTorch training (s) | ARM64 training (s) | PyTorch / ARM64 |
|---:|---:|---:|---:|---:|
| 32 | 17,952 | 1.296 | 0.754 | 1.719× |
| 64 | 52,160 | 1.649 | 1.846 | 0.893× |
| 128 | 169,728 | 2.432 | 5.596 | 0.435× |
| 256 | 601,472 | 5.232 | 19.758 | 0.265× |

Training includes validation, finite checks, checkpoint selection/writes, and
history bookkeeping; it excludes setup and final test evaluation. Times are means
of three seed means, with repeats averaged within seed. The primary training ratio
is the geometric mean of six paired time ratios. **Above one favors assembly;
below one favors PyTorch.** The full report includes sample SD and every run.

| Width | PyTorch forward p50 (µs) | ARM64 forward p50 (µs) | PyTorch / ARM64 |
|---:|---:|---:|---:|
| 32 | 53.66 | 14.55 | 3.688× |
| 64 | 63.54 | 34.58 | 1.837× |
| 128 | 78.40 | 102.10 | 0.768× |
| 256 | 121.67 | 359.74 | 0.338× |

Forward ratios use the implementation means of p50 latency, a different statistic
from the training ratio. Each run takes 100 samples after 20 warmups. This is a
batch-one forward pass over a complete 32-character context, excluding loss,
softmax, sampling, token loading, and I/O. It is not cached next-token generation.

![Training and forward cost as model width grows](experiments/width-sweep-v3/scaling.png)

At width 256, native training takes **3.78× as long** and its forward pass takes
**2.96× as long** as PyTorch. The same benchmark that demonstrates the small-model
wins also establishes this larger-model limitation.

## Why the advantage changes with size

For these fixed-context models, dense projection and feed-forward arithmetic grow
roughly with the square of width. Dispatch and other fixed costs become a smaller
fraction of the work as those matrices grow. PyTorch can amortize overhead and use
its optimized matrix kernels on larger operations. During training it also handles
the full batch together; the native runtime accumulates gradients one sequence at
a time, retaining a much smaller workspace.

The handwritten kernels reuse data within small register tiles. They do not yet
have batch-wide execution or a larger matrix-packing and cache-blocking strategy.
That gives a concrete direction for further engineering: profile the wider
projection/backward operations, then test larger-scale data reuse and batching.
The forward crossover shows that training batching alone cannot explain the
whole gap. These are implementation-based explanations consistent with the data;
the sweep does not measure cache misses or isolate individual bottlenecks.

## The engineering idea

A network's equations specify what must be calculated. They leave many choices
about which calculations run together, how often inputs are loaded, and where
partial results live. **This project reorganizes execution around the hardware
while checking that predictions, gradients, and optimizer updates still implement
the intended learning calculation.**

Writing assembly was the starting point. The gains came from changing the work
done between memory accesses and making more calculations available together.

**Tiled matrix kernels reuse inputs and keep partial sums in registers.** The
original forward kernel completed one output dot product at a time. The tiled
implementation computes four token rows and four output channels together. For each four-feature chunk,
eight vector loads feed sixteen `FMLA` instructions. Each instruction multiplies
and accumulates four float32 lanes, so a loaded input vector contributes to four
outputs and a loaded weight vector contributes to four token rows.

Sixteen vector registers hold the tile's partial dot products until the reduction
finishes. This reduces repeated loads and provides independent accumulator chains
that the processor can overlap, instead of making all progress wait on one sum.
Backward uses four-by-sixteen tiles for input and weight gradients. It loads an
existing weight-gradient tile once, accumulates contributions across token rows,
then writes the tile once. Input-gradient tiles likewise stay in registers across
the output-channel reduction. The original code repeatedly loaded and stored
those partial gradients inside the reduction loops.

![How matrix tiles share loads and retain partial results in NEON registers](docs/kernel-tiling.png)

**Attention specializes the head width, with vectorized forward and backward.**
Model widths 32, 64, 128, and 256 have heads of 8, 16, 32, and 64 values.
Those heads occupy 2, 4, 8, or 16 NEON registers. The wider paths unroll the
feature loops, keep query and output-gradient vectors in registers across key
visits, and use four independent dot-product accumulators. Weighted values and
Q/K/V derivatives also use vector arithmetic, while causal masking remains exact.
Backward computes each dot product between an output gradient and a value
vector once, stores it in a 32-float, **128-byte** stack row, and reuses it for
the softmax derivative and Q/K gradients. Only one derivative row is cached;
this adds no quadratic backward buffer.

**Softmax evaluates four exponentials per call.** The original scalar routine
used a degree-12 polynomial for one value. The vector helper range-reduces four
values together, evaluates a degree-nine polynomial in two pairs of float64 lanes, and rounds back
to float32. Subtracting the row maximum before exponentiation preserves stable
softmax normalization. The ASCII-128 cross-entropy kernel also vectorizes its
maximum, exponential sum, and logit-gradient calculations. The shared `exp4`
helper benefits both attention and loss. Out-of-range or nonfinite inputs fall
back to the original scalar exponential behavior.

**LayerNorm vectorizes both statistics and derivatives.** The first pass computes
the mean and the second sums squared deviations from that mean, four values per
instruction. Keeping the centered two-pass variance avoids the cancellation risk
of subtracting squared means from mean squares. Normalization and backward's
reductions, input gradients, and accumulated scale/bias gradients also use vectors.
Finiteness checks use integer exponent-bit masks on four floats at once, followed
by a scalar remainder. Every original runtime check remains in place.

Specialization has explicit boundaries. Linear tiles require compatible row and
feature dimensions; attention specializes the four declared model widths, four
heads, and up to 32 tokens. Other supported shapes use the original generic kernels. Partial softmax
vectors load only valid lanes and mask unused lanes. The original numerical
tolerances are unchanged, with additional tests for tails, extreme values,
causality and accumulated gradients. See the [kernel notes](native/kernel_notes.md).

These mechanisms are visible in the code. Their combined performance effect is
measured in the width sweep; this experiment does not isolate each change in a separate
ablation. Native training still accumulates gradients one sequence at a time.
The tiles share work within a sequence; batch-wide native execution is not yet
implemented.

**Width is an authenticated build configuration.** A shared specification derives
parameter offsets, embedding strides, workspace sizes, and loop bounds. Each
width gets its own generated layout and native binary. Only constants are
generated; the numerical instructions and training runtime remain handwritten.
The library exports its width and parameter count, and the executable checks a
versioned fixture header before loading model buffers. Separate Python processes
select their width before imports, preventing accidental configuration mixing.

## Memory and learning quality

| Width | Implementation | Peak process RSS (MiB) | Test NLL | Accuracy (%) |
|---:|---|---:|---:|---:|
| 32 | PyTorch | 277.09 | 2.4585 | 35.046 |
| 32 | ARM64 | 2.34 | 2.4621 | 35.042 |
| 64 | PyTorch | 280.93 | 2.4145 | 36.869 |
| 64 | ARM64 | 3.11 | 2.4122 | 36.877 |
| 128 | PyTorch | 287.15 | 2.3862 | 37.964 |
| 128 | ARM64 | 5.64 | 2.3901 | 37.455 |
| 256 | PyTorch | 314.36 | 2.3962 | 37.203 |
| 256 | ARM64 | 14.45 | 2.3984 | 36.788 |

RSS is measured through training and includes framework, runtime, and data. It
is not isolated parameter storage. At width 256, the native process used **14.45
MiB versus 314.36 MiB** for PyTorch, despite having the same parameter count.

![Memory and quality at each width, including seed variation](experiments/width-sweep-v3/memory-quality.png)

Lower NLL means more probability assigned to the correct next character. Both
implementations improve on the training-only unigram baseline's NLL of **3.3623**.
Mean NLL improves through width 128, with no further improvement observed at 256
under this fixed training budget. This does not establish an optimal width or
convergence; the larger models received the same small dataset and 1,024 updates.

| Width | Mean ARM64 minus PyTorch NLL | Top-one prediction disagreement |
|---:|---:|---:|
| 32 | +0.003609 | 8.610% |
| 64 | -0.002369 | 15.055% |
| 128 | +0.003875 | 28.918% |
| 256 | +0.002282 | 19.832% |

The implementations have similar aggregate NLL here, but their trained predictions
are not identical. Floating-point grouping can alter long training trajectories;
width 128 has about 29% top-one disagreement despite a small mean NLL difference.
Numerical admission checks and these descriptive observations do not establish
statistical equivalence of prediction quality. Per-seed differences, perplexity,
bits per character, accuracy, and learning curves are retained in each width's
report.

## Correctness and retained evidence

Each exact native binary passed **923 numerical/runtime checks**, for **3,692**
across the four widths, with all original tolerances unchanged. The gates cover
primitive derivatives, every model parameter gradient, Adam states, finite
differences, causal masking, partial batches, malformed input, width identity,
vector tails, extreme scores, and buffer guards.

The independent post-run audit passed **14,113 checks**: 3,399 per width plus
517 sweep consistency checks. It recomputed all **393,216 retained test prediction
rows** from saved checkpoints and checked selected-checkpoint validation loss
against selection history. Within each implementation/width/seed, repeat
checkpoints, optimizer states, predictions, and non-timing histories matched exactly.

Read the [complete report](experiments/width-sweep-v3/README.md),
[summary](experiments/width-sweep-v3/summary.json), and
[all audit records](experiments/width-sweep-v3/audit.json). The
[v3 release](https://github.com/logannye/arm64-transformer/releases/tag/matched-transformer-width-sweep-v3)
contains raw fixtures, schedules, initial/selected/final weights, optimizer states,
predictions, latency samples, all four binaries and generated layouts, and the
exact measured source revision `e11243f454d7b42bfb044883089e2df23c5a59e0`.
The archive has an internal payload checksum manifest, an external SHA256, and a
verification record. Publication changes are separate from the measured source.

## Reproduce

Native execution requires an Apple silicon Mac and Xcode Command Line Tools.
Python reference tests also run on Linux. Use Python 3.12 and prebuilt dependencies;
a compatible existing environment can be reused.

```sh
git clone https://github.com/logannye/arm64-transformer.git
cd arm64-transformer
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: -e '.[dev,report]'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0
python -m transformer.fetch_data --output data/tinyshakespeare.txt
make -C native widths PYTHON=python
for width in 32 64 128 256; do
  export ARM64_TRANSFORMER_WIDTH=$width
  python -m pytest -q
  python -m transformer.validate --output results/preflight-v3/width-$width.json \
    --runtime --work-dir results/smoke-v3/width-$width
  python -m transformer.prepare --source data/tinyshakespeare.txt \
    --output results/inputs-v3/width-$width
done
unset ARM64_TRANSFORMER_WIDTH
python -m transformer.sweep --inputs results/inputs-v3 \
  --preflight-dir results/preflight-v3 --output results/width-sweep-v3
python -m transformer.sweep_report --inputs results/inputs-v3 \
  --results results/width-sweep-v3 --output results/report-v3
```

The source must be committed and clean, the Mac must use AC power, and all four
binaries must match their passing preflights. The coordinator checks source and
input hashes around every child, runs only one fit at a time, pauses 15 seconds
between fits, and retains failures. Output directories refuse overwrites.
CI runs synthetic correctness checks at every width; it does not run the full
benchmark. [requirements-benchmark.txt](requirements-benchmark.txt) records the
measured package versions.

## Earlier transformer versions

The original native implementation took **2.23× as long** as PyTorch to train.
Tiled matrix kernels, vectorized attention/normalization, and shared vector
exponentials changed the width-32 result to a **1.751× training speedup** in v2.
The fresh v3 width-32 comparison above is **1.719×**. These are separate campaigns
whose machine conditions may differ; each primary ratio uses its own paired runs.

The [v1 report](experiments/matched-v1/README.md),
[v2 report and kernel diagnostics](experiments/matched-v2/README.md),
[v1 release](https://github.com/logannye/arm64-transformer/releases/tag/matched-transformer-cpu-v1),
and [v2 release](https://github.com/logannye/arm64-transformer/releases/tag/matched-transformer-cpu-v2)
preserve the slower starting point and the optimization evidence. Individual
kernel diagnostics are distinct from full-model and end-to-end training timings.

## Code map

| Path | Purpose |
|---|---|
| `transformer/spec.py` | Process-local width, parameters, workspace, generated constants |
| `transformer/reference.py`, `python_train.py` | Standard PyTorch model and training runtime |
| `transformer/prepare.py` | Fixed data partitions, weights, schedules, versioned fixtures |
| `native/linear.S` | Tiled linear forward and backward |
| `native/attention.S`, `softmax.S`, `normalization.S` | Specialized vector kernels |
| `native/kernels.S` | Generic fallbacks, ReLU, Adam, and scalar math |
| `native/model.S`, `runtime.S` | Complete native model, training, files, and measurements |
| `transformer/validate.py`, `audit.py` | Numerical gates and independent checkpoint audit |
| `transformer/sweep.py`, `sweep_report.py` | Matched width study, consistency audit, and charts |
| `scripts/package_sweep_evidence.py` | Verified streaming release archive |

All native numerical work and training control are handwritten assembly. macOS/libc
provides files, memory, and clocks; no Torch, BLAS, Accelerate, or compiled C
numerical implementation is linked into the native executable. See the
[ABI](native/ABI.md), [kernel notes](native/kernel_notes.md), and
[static model diagram](docs/architecture.png).

The measured scope is this short-context float32 family on one CPU machine against
eager PyTorch. GPU performance, graph-compiled PyTorch, long contexts, energy,
cached decoding, and production serving remain outside the experiment.
