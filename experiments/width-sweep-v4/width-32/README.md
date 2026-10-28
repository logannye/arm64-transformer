# Matched transformer CPU comparison, v4, width 32

This optimization study reuses v1's fixed train/validation/test fixtures. It is an engineering regression benchmark, not a fresh untouched test set. Kernel development used synthetic correctness and timing inputs; the model, optimizer, seeds and stopping rule were unchanged.

ARM64 training was **1.819× faster** than PyTorch, using the geometric mean of six paired training-time ratios.
This is the same 17,952-parameter causal transformer implemented with ordinary
batched PyTorch (built-in scaled-dot-product attention and autograd) and handwritten
Apple ARM64 assembly. The experiment contains **12 complete fits**: three
initialization seeds, two timing repeats, and both implementations.

Measured source revision: `276c6d091a7ebdc5f7831be2169683db4715f748`. Hardware: Apple M4 Max.
Platform: `macOS-26.2-arm64-arm-64bit`. PyTorch: `2.14.1`.

Native processes requested automatic dispatch. Hardware SME capability: True; streaming vector length: 64 bytes. Actual dense calls per native fit were SME 0 forward / 0 backward, and NEON 94,040 forward / 81,920 backward. These authenticated counters cover the whole process, including evaluation and inference samples.

The auto-policy preflight contains 1,039 checks; forced-policy gates add 2,078, for 3,117 admission checks.

| Measurement | PyTorch | ARM64 assembly |
|---|---:|---:|
| Test NLL (lower is better) | 2.4585 ± 0.0292 | 2.4621 ± 0.0211 |
| Test bits / character | 3.5469 ± 0.0422 | 3.5521 ± 0.0304 |
| Test perplexity | 11.691 ± 0.341 | 11.731 ± 0.246 |
| Next-character accuracy (%) | 35.046 ± 0.532 | 35.042 ± 0.671 |
| Training + validation (s) | 1.295 ± 0.012 | 0.712 ± 0.003 |
| Training CPU time (s) | 1.289 ± 0.011 | 0.711 ± 0.003 |
| Whole child process (s) | 2.373 ± 0.054 | 0.724 ± 0.003 |
| Forward p50, full context (ms) | 0.0544 ± 0.0003 | 0.0150 ± 0.0003 |
| Forward p95, full context (ms) | 0.0563 ± 0.0008 | 0.0162 ± 0.0002 |
| Forward p99, full context (ms) | 0.0582 ± 0.0011 | 0.0172 ± 0.0010 |
| Peak RSS through training (MiB) | 277.52 ± 0.26 | 2.34 ± 0.00 |

Entries are mean ± sample SD of three seed means. Timing repeats are averaged
within seed; they are not six independent quality seeds. The primary ratio is
Python/ARM64 training time, **1.818772**, where values above one favor assembly.
All within-implementation repeat checkpoints, optimizer states, held-out logits,
and non-timing learning histories matched exactly.

## Prediction quality and baselines

The mean assembly-minus-PyTorch test NLL difference was
**+0.003609**, and the accuracy difference was
**-0.004069 percentage points**.
Mean paired next-character prediction disagreement was **8.610%**.
These are descriptive observations, not a statistical equivalence claim.

A uniform ASCII-128 predictor has NLL **4.8520** and perplexity
**128.000**. An add-one-smoothed unigram predictor fitted only
to the 65,536 training target characters has held-out NLL **3.3623**,
perplexity **28.856**, and accuracy **15.344%**.
Its training counts are retained in [summary.json](summary.json).

The source corpus is Tiny Shakespeare. Contiguous 80/10/10 partitions were made
before selecting the first 65,537/8,193/8,193 bytes for this bounded experiment.
There are 2,048 training and 256 validation/test sequences, each predicting 32
characters. Eight epochs and batch 16 give 1,024 optimizer updates per run.
Validation NLL selects the checkpoint; test data are used only after selection.

## Timing and memory scope

![Training and whole-process times](training.png)

Training includes forward/backward, Adam, finite checks, epoch validation,
checkpoint selection/writes and history bookkeeping. Setup, final evaluation,
last-state export and inference timing are outside this timer. Whole-process
time is independently measured by the parent from child launch through exit.

![Full-context forward latency](latency.png)

Latency is a **batch-one forward pass over a complete 32-character context**,
with 20 warmups and 100 measured samples per run. It excludes token loading,
softmax, sampling, loss, and I/O. It is not cached single-token generation or
end-to-end application latency. The p99 values describe these 100-sample
microbenchmarks and do not establish rare-event deployment guarantees.

![Process memory](memory.png)

Memory is process lifetime peak RSS through each boundary, including runtime,
framework and loaded data. It is not model-only memory; subtracting these peaks
does not isolate individual allocations.

![Training and validation learning curves](learning.png)

## Every measured run

| Seed | Repeat | Implementation | Train (s) | Test NLL | Accuracy (%) | Best epoch |
|---:|---:|---|---:|---:|---:|---:|
| 42 | 1 | PyTorch | 1.269 | 2.46377 | 34.912 | 8 |
| 42 | 1 | ARM64 assembly | 0.722 | 2.47330 | 34.790 | 8 |
| 43 | 1 | ARM64 assembly | 0.729 | 2.47524 | 34.534 | 8 |
| 43 | 1 | PyTorch | 1.333 | 2.48474 | 34.595 | 8 |
| 44 | 1 | PyTorch | 1.329 | 2.42697 | 35.632 | 8 |
| 44 | 1 | ARM64 assembly | 0.719 | 2.43776 | 35.803 | 8 |
| 42 | 2 | ARM64 assembly | 0.702 | 2.47330 | 34.790 | 8 |
| 42 | 2 | PyTorch | 1.293 | 2.46377 | 34.912 | 8 |
| 43 | 2 | PyTorch | 1.271 | 2.48474 | 34.595 | 8 |
| 43 | 2 | ARM64 assembly | 0.701 | 2.47524 | 34.534 | 8 |
| 44 | 2 | ARM64 assembly | 0.698 | 2.43776 | 35.803 | 8 |
| 44 | 2 | PyTorch | 1.273 | 2.42697 | 35.632 | 8 |

## Controls and retained evidence

Both implementations used the same float32 initial parameters, token rows,
per-epoch schedules, Adam settings, one-thread CPU budget and stopping rule.
Run order was reversed between repeats, one child ran at a time, and fixed
15-second pauses separated children. Power, thermal and load snapshots are
retained in the raw runner manifest. Operating-system scheduling, temperature
and clock frequencies were not experimentally fixed.

The independent post-run audit passed **3409 checks**.
All **98,304 held-out prediction rows**
were recomputed from their selected checkpoints with a separate functional
PyTorch reference. Selected-checkpoint validation loss was independently
recomputed and matched the selection history. NumPy float64 calculations
produced the reported quality metrics. See [audit.json](audit.json) for every
check and [summary.json](summary.json) for per-run metrics and hashes.

The result applies to this small, short-context CPU workload on one machine.
It does not establish a general language-speed advantage, large-language-model
quality, GPU performance, energy use, cached decoding speed, or production
serving behavior. Eager PyTorch is the baseline; graph-compiled PyTorch is not
evaluated. Float32 operation grouping differs, so long training trajectories
need not be bitwise identical between implementations.

The compact report contains metrics, checks, hashes, and figures. Full inputs,
schedules, raw logits, checkpoints, optimizer states, latency samples, the source
snapshot and executable belong to the accompanying raw evidence bundle.
