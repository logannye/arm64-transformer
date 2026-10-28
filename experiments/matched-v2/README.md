# Matched small-transformer CPU comparison, v2

This optimization study reuses v1's fixed train/validation/test fixtures. It is an engineering regression benchmark, not a fresh untouched test set. Kernel development used synthetic correctness and timing inputs; the model, optimizer, seeds and stopping rule were unchanged.

ARM64 training was **1.751× faster** than PyTorch, using the geometric mean of six paired training-time ratios.
This is the same 17,952-parameter causal transformer implemented with ordinary
batched PyTorch (built-in scaled-dot-product attention and autograd) and handwritten
Apple ARM64/NEON assembly. The experiment contains **12 complete fits**: three
initialization seeds, two timing repeats, and both implementations.

Measured source revision: `375af274085af1c5b55e2f9be091749a7d863bcd`. Hardware: Apple M4 Max.
Platform: `macOS-26.2-arm64-arm-64bit`. PyTorch: `2.14.1`.

| Measurement | PyTorch | ARM64 assembly |
|---|---:|---:|
| Test NLL (lower is better) | 2.4585 ± 0.0292 | 2.4621 ± 0.0211 |
| Test bits / character | 3.5469 ± 0.0422 | 3.5521 ± 0.0304 |
| Test perplexity | 11.691 ± 0.341 | 11.731 ± 0.246 |
| Next-character accuracy (%) | 35.046 ± 0.532 | 35.042 ± 0.671 |
| Training + validation (s) | 1.280 ± 0.006 | 0.731 ± 0.006 |
| Training CPU time (s) | 1.277 ± 0.006 | 0.730 ± 0.006 |
| Whole child process (s) | 2.217 ± 0.014 | 0.746 ± 0.007 |
| Forward p50, full context (ms) | 0.0537 ± 0.0002 | 0.0144 ± 0.0004 |
| Forward p95, full context (ms) | 0.0559 ± 0.0010 | 0.0161 ± 0.0005 |
| Forward p99, full context (ms) | 0.0583 ± 0.0030 | 0.0177 ± 0.0014 |
| Peak RSS through training (MiB) | 277.35 ± 0.25 | 2.33 ± 0.00 |

Entries are mean ± sample SD of three seed means. Timing repeats are averaged
within seed; they are not six independent quality seeds. The primary ratio is
Python/ARM64 training time, **1.751085**, where values above one favor assembly.
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
| 42 | 1 | PyTorch | 1.293 | 2.46377 | 34.912 | 8 |
| 42 | 1 | ARM64 assembly | 0.737 | 2.47330 | 34.790 | 8 |
| 42 | 2 | ARM64 assembly | 0.737 | 2.47330 | 34.790 | 8 |
| 42 | 2 | PyTorch | 1.266 | 2.46377 | 34.912 | 8 |
| 43 | 1 | ARM64 assembly | 0.728 | 2.47524 | 34.534 | 8 |
| 43 | 1 | PyTorch | 1.285 | 2.48474 | 34.595 | 8 |
| 43 | 2 | PyTorch | 1.287 | 2.48474 | 34.595 | 8 |
| 43 | 2 | ARM64 assembly | 0.733 | 2.47524 | 34.534 | 8 |
| 44 | 1 | PyTorch | 1.267 | 2.42697 | 35.632 | 8 |
| 44 | 1 | ARM64 assembly | 0.724 | 2.43776 | 35.803 | 8 |
| 44 | 2 | ARM64 assembly | 0.726 | 2.43776 | 35.803 | 8 |
| 44 | 2 | PyTorch | 1.282 | 2.42697 | 35.632 | 8 |

## Controls and retained evidence

Both implementations used the same float32 initial parameters, token rows,
per-epoch schedules, Adam settings, one-thread CPU budget and stopping rule.
Run order was reversed between repeats, one child ran at a time, and fixed
15-second pauses separated children. Power, thermal and load snapshots are
retained in the raw runner manifest. Operating-system scheduling, temperature
and clock frequencies were not experimentally fixed.

The independent post-run audit passed **3399 checks**.
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
