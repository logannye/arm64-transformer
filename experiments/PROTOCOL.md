# Prespecified matched transformer experiment, v1

The experimental variable is the implementation of one fixed small transformer:
ordinary eager, batched PyTorch with built-in scaled-dot-product attention and
autograd versus handwritten Apple ARM64/NEON numerical kernels and a standalone
assembly training runtime. No speedup is assumed or required. This is a bounded
single-machine implementation comparison, not a claim about programming languages
or general-purpose language-model capability.

## Model and data

One pre-LayerNorm decoder block; context 32; width 32; four heads of width 8;
feed-forward width 64 with ReLU; learned token and position embeddings; final
LayerNorm and untied vocabulary head. All biases included, 17,952 parameters.
ASCII-128 vocabulary, float32 tensors, LayerNorm epsilon 1e-5, causal attention.
No dropout, weight decay, scheduler, cached decoding, or pretrained weights.

Tiny Shakespeare source: karpathy/char-rnn commit
`370cbcd448eb7daf32f21a6be560b70e0b33c4e3`, file
`data/tinyshakespeare/input.txt`, 1,115,394 bytes, SHA256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
The source is first partitioned contiguously at floor(0.8*N), floor(0.9*N).
Use only the first 65,537 bytes of the train partition and the first 8,193 bytes
of each validation/test partition. Each row contains 33 consecutive characters:
32 inputs and their 32 next-character targets. Start rows at multiples of 32,
reset context between rows, and never cross partition boundaries. This gives
2,048 training sequences, 256 validation and 256 test sequences: 65,536/8,192/8,192
predicted characters. Excerpts represent a deliberately modest engineering
benchmark, not the entire corpus or independent draws of natural language.

Seeds 42/43/44 initialize weights independently with normal standard deviation
.02, biases zero, LayerNorm scales one. Export exact float32 parameters and
all per-epoch shuffled training sequence IDs before either implementation runs.
No random operations occur inside either measured training path.

## Training and selection

Eight epochs, batch 16, Adam lr .001, betas (.9,.999), epsilon 1e-8. No clipping.
Each update is the mean loss over actual batch tokens. Partial batches remain
supported and tested even though this study's 2,048 rows divide evenly by 16.
Exactly 1,024 optimizer steps per run. Evaluate validation after every epoch;
select the lowest validation mean token negative log-likelihood (NLL), first
epoch on exact ties. Test is evaluated only after checkpoint selection.
Preflight tests use synthetic text, never the study's held-out text. No model,
seed, optimizer or epoch tuning from test results is allowed. Report all outcomes.

## Compute and order

One Apple ARM64 Mac, CPU only, one compute thread in both implementations;
PyTorch interop and numerical-library thread budgets also one. Native processes
only call libc/macOS for memory, files, process operations, and clocks. All
model math, gradients, Adam, batching and selection execute handwritten .S.
PyTorch uses its compiled native tensor/attention kernels and standard optimizer.
Both load owned inputs and initialize optimizer memory before training timers.
Native accumulates per-sequence gradients; PyTorch performs normal batched tensor
operations. Their operation grouping, layout, reduction order and dispatch are
intentional implementation differences. Small floating-point differences can
produce different long-run trajectories; same model does not mean identical bits.

Run one child process at a time, on AC power, with a fixed 15-second pause after
each run except the last. Retain power/thermal/load snapshots before and after.
No CPU affinity, fixed frequency, or controlled laboratory temperature is claimed.

| Seed | Repeat 1 | Repeat 2 |
|---|---|---|
| 42 | PyTorch, ARM64 | ARM64, PyTorch |
| 43 | ARM64, PyTorch | PyTorch, ARM64 |
| 44 | PyTorch, ARM64 | ARM64, PyTorch |

There are 12 fits, three independent initializations and two timing repeats each.
Report every child, including failures. Never replace an inconvenient run.

## Admission, measurements, and evidence

Before the full experiment, require passing primitive/full-model numerical
parity, every parameter gradient, Adam moments/updates, finite differences,
causality, repeated-token gradients, short standalone partial-batch trajectories,
malformed-file rejection, overwrite protection and deterministic native repeats.
Numerical tolerances are declared in transformer/validate.py before measurement.
The preflight authenticates source files and binary hashes. Measured source is
committed and clean. The runner records source revision, SHA256 values, exact
runtime/compiler/platform versions, commands, timings and all process outcomes.

- Training wall/CPU: all training, finite checks, epoch validation, selection and
  best-checkpoint writes, epoch history bookkeeping. Excludes setup, final test,
  final exports and inference microbenchmarks.
- Parent process wall: launch through successful exit, including Python startup,
  manifest validation, setup, final evaluation and all exports. Keep separate.
- Memory: lifetime peak process RSS before training, after training and at end;
  includes framework/runtime and data. Never interpret peak differences as
  isolated allocation sizes or model-only memory.
- Inference: full 32-character context, batch one, same first test sequence,
  selected checkpoint, 20 warmups and 100 samples. Forward logits only, excluding
  token loading, softmax, loss, sampling and I/O. This is not cached per-token
  generation latency or end-to-end application latency.
- Quality: independent float64 NumPy token NLL, bits/character, perplexity and
  top-one next-character accuracy from every saved held-out logit row. Also report
  train-only unigram and uniform baselines for context, and learning histories.

Primary time ratio: geometric mean of six paired PyTorch/ARM64 training wall
ratios (>1 favors assembly). Report mean and sample SD of per-seed timing means;
repeats are not additional independent seeds. Quality uses three seeds after
exact repeat-output checks. Report paired quality differences and prediction
disagreement descriptively, without asserting statistical equivalence.

Recompute every selected-checkpoint test prediction with an independent functional
implementation; verify selected validation loss and artifact lengths/finiteness.
Retain initial inputs, schedules, checkpoints, optimizer states, raw logits,
histories, individual latency samples, source snapshot and executable. A failed
gate remains a failed gate; do not relax thresholds after seeing discrepancies.

## Scope

This study isolates a small, fixed architecture and short-context CPU workload.
It does not evaluate GPU tensor cores, long context, KV caching, production serving,
energy, modern large-language-model quality, or compiled PyTorch graph execution.
An assembly performance loss is a valid result and must be reported plainly.
