# Optimized ARM64 matched transformer experiment, v2

This protocol extends the frozen [v1 protocol](PROTOCOL.md) with optimized native
kernels and stronger boundary checks. The v1 protocol, result report, raw evidence,
and release remain unchanged. V2 reruns the full twelve-fit comparison from the
same exported initial weights and schedules. An implementation speedup is a
measurement outcome, not an admission requirement.

## Purpose and evaluation status

The question is whether hardware-aware implementation changes improve the native
transformer's measured training or full-context forward performance against the
same ordinary eager, batched PyTorch baseline. The baseline uses built-in scaled
dot-product attention, autograd, and Adam. Both use one CPU compute thread.

The v1 test results have already been inspected. V2 therefore reuses a fixed
engineering benchmark; its test partition is not a newly untouched scientific
assessment set. Reuse supports controlled implementation comparisons. It does
not establish fresh generalization evidence. No model architecture, training
hyperparameter, seed, data subset, or stopping-rule choice may be tuned from
test performance during this phase.

## Unchanged computational and data contract

The complete architecture, vocabulary, precision and training rules remain those
in v1: one pre-LayerNorm causal decoder block, context 32, width 32, four heads,
feed-forward width 64 with ReLU, learned token/position embeddings, final LayerNorm
and untied vocabulary head, 17,952 parameters, ASCII-128, float32, epsilon 1e-5.
There is no dropout, clipping, weight decay, scheduler, pretrained initialization,
or KV cache. Adam uses lr .001, betas (.9,.999), epsilon 1e-8.

The same Tiny Shakespeare source, source SHA256, contiguous partition boundaries,
selected byte intervals, sequence rows, and fixed vocabulary are reused. Their
identities are recorded in each fixture manifest. The study uses 65,536 training,
8,192 validation and 8,192 test target characters. Seeds 42/43/44, eight epochs,
batch 16, and 1,024 updates per fit are unchanged. Each update averages over its
actual batch tokens. The lowest epoch validation NLL selects the checkpoint,
with the first epoch winning exact ties. Test evaluation follows selection.

## Allowed implementation changes

The native implementation may tile linear forward/backward operations, reuse data
in registers, vectorize attention contractions and normalization, and evaluate
softmax-related exponentials in vector groups. Generic paths remain available
for supported shapes outside specialized fast paths. All numerical work and
training control remain handwritten Apple ARM64 assembly; no BLAS, Accelerate,
PyTorch, compiled C math implementation, or external numerical runtime is used.

The implementations retain their actual operation grouping. Native batching may
continue to accumulate per-sequence gradients, while PyTorch uses batched tensor
operations. A claim that native training uses batched matrix kernels requires
implementation and separate verification; this protocol does not assume it.
Changes to arithmetic reduction order may affect floating-point trajectories.
Mathematical fidelity is admitted under the original v1 numerical tolerances.

## Admission before measured runs

All original primitive, model, gradient, Adam, finite-difference, causality,
standalone partial-batch, malformed-input and repeatability checks remain. Their
original random fixture stream and tolerance values are preserved. V2 appends
synthetic boundary probes from a separate declared random seed, 20261028:

- Tiled linear fast paths and generic fallbacks, including row remainders,
  scalar feature tails, and nonzero parameter-gradient accumulators reused twice.
- Vector normalization and scalar fallbacks, constant/zero/near-constant inputs,
  input gradients, and accumulated scale/bias gradients.
- Attention token counts around vector boundaries, the maximum specialized
  context, a longer generic-fallback context, and a non-eight head width.
  Masked probabilities and future gradients are required to be exactly zero.
- Stable cross-entropy with tied large logits, exponentially small probabilities,
  underflow boundaries, low-probability targets, and generic vocabulary sizes.
- Finite sentinels around output tensors detect writes beyond declared buffers;
  read-only inputs must remain unchanged.
- Finiteness scans exercise lengths 0/1/3/4/5/17 and NaN/positive infinity/negative
  infinity at every position, checking vector blocks, scalar tails and unchanged
  input bits. The schedule of runtime finite checks remains unchanged.

No thresholds are relaxed after observing v2 discrepancies. Failures remain in
their original evidence files. Development probes and timing checks use synthetic
inputs. The native source authentication includes `kernels.S`, `linear.S`,
`attention.S`, `softmax.S`, `normalization.S`, `model.S`, `runtime.S`, and
`layout.inc`, together with the reference, fixture, training and validator sources.
After the final edits, rebuild once and pass the complete exact-binary preflight
before freezing the measured source revision. Measured source must be committed
and clean; the runner rejects changed source, binary or power conditions.

## Execution, statistics and reporting

Run three seeds, two repeats and both implementations, twelve complete fits.
The reversed order, one-child-at-a-time execution, AC-power requirement, fixed
15-second inter-run pause, thermal/load snapshots and single-thread settings are
unchanged from v1. Use a new result directory and never replace an inconvenient
or failed measured child. A stopped run is retained and reported as stopped.

Training wall/CPU, whole-process time, peak process RSS, full-context batch-one
forward timing, and prediction-quality boundaries remain exactly those in v1.
The inference sample is the full 32-character forward pass with 20 warmups and
100 observations; it is not cached single-token generation or application latency.
Quality includes NLL, bits per character, perplexity, top-one accuracy, and the
same training-only unigram/uniform baselines.

The primary ratio is the geometric mean of the six paired PyTorch/ARM64 training
wall-time ratios. Values greater than one favor assembly. Average timing repeats
within seed, then report mean and sample SD across the three independent seeds.
Exact within-implementation repeat outputs are required before pooling quality
results. Report every per-run observation and paired quality difference. Similar
metrics do not establish statistical equivalence.

Independently recompute every selected-checkpoint test logit and selected
validation NLL using the unchanged mathematical reference and original tolerances.
Retain raw outputs, all input manifests, source and binary identities, checkpoints,
optimizer states, histories, latency samples, audits, and figures under distinct
v2 paths. Direct v1/v2 timing changes are historical comparisons across separate
campaigns, with potentially different machine conditions. The primary controlled
v2 claim comes from its newly paired PyTorch/ARM64 runs.

The scope remains this small fixed architecture on one CPU machine. This study
does not evaluate GPU execution, larger models, long contexts, energy, cached
decoding, production serving, or graph-compiled PyTorch.
