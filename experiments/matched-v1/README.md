# Matched small-transformer CPU comparison, v1

ARM64 training took **2.232× as long** as PyTorch, using the geometric mean of six paired training-time ratios.
This is the same 17,952-parameter causal transformer implemented with ordinary
batched PyTorch (built-in scaled-dot-product attention and autograd) and handwritten
Apple ARM64/NEON assembly. The experiment contains **12 complete fits**: three
initialization seeds, two timing repeats, and both implementations.

Measured source revision: `542bba1246a1def1388760ec20d2be3a944b359f`. Hardware: Apple M4 Max (verified after measurement; the original sandboxed sysctl query failed).
Platform: `macOS-26.2-arm64-arm-64bit`. PyTorch: `2.14.1`.

| Measurement | PyTorch | ARM64 assembly |
|---|---:|---:|
| Test NLL (lower is better) | 2.4585 ± 0.0292 | 2.4601 ± 0.0189 |
| Test bits / character | 3.5469 ± 0.0422 | 3.5492 ± 0.0273 |
| Test perplexity | 11.691 ± 0.341 | 11.707 ± 0.221 |
| Next-character accuracy (%) | 35.046 ± 0.532 | 34.981 ± 0.092 |
| Training + validation (s) | 1.328 ± 0.034 | 2.964 ± 0.018 |
| Training CPU time (s) | 1.323 ± 0.033 | 2.961 ± 0.019 |
| Whole child process (s) | 2.474 ± 0.062 | 3.004 ± 0.019 |
| Forward p50, full context (ms) | 0.0545 ± 0.0013 | 0.0609 ± 0.0043 |
| Forward p95, full context (ms) | 0.0581 ± 0.0024 | 0.0681 ± 0.0064 |
| Forward p99, full context (ms) | 0.0682 ± 0.0095 | 0.0731 ± 0.0071 |
| Peak RSS through training (MiB) | 276.64 ± 0.44 | 2.33 ± 0.00 |

Entries are mean ± sample SD of three seed means. Timing repeats are averaged
within seed; they are not six independent quality seeds. The primary ratio is
Python/ARM64 training time, **0.448082**, where values above one favor assembly.
All within-implementation repeat checkpoints, optimizer states, held-out logits,
and non-timing learning histories matched exactly.

## Prediction quality and baselines

The mean assembly-minus-PyTorch test NLL difference was
**+0.001617**, and the accuracy difference was
**-0.065104 percentage points**.
Mean paired next-character prediction disagreement was **8.850%**.
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
| 42 | 1 | PyTorch | 1.366 | 2.46377 | 34.912 | 8 |
| 42 | 1 | ARM64 assembly | 2.926 | 2.46357 | 35.022 | 8 |
| 42 | 2 | ARM64 assembly | 2.960 | 2.46357 | 35.022 | 8 |
| 42 | 2 | PyTorch | 1.343 | 2.46377 | 34.912 | 8 |
| 43 | 1 | ARM64 assembly | 2.952 | 2.47706 | 34.875 | 8 |
| 43 | 1 | PyTorch | 1.304 | 2.48474 | 34.595 | 8 |
| 43 | 2 | PyTorch | 1.276 | 2.48474 | 34.595 | 8 |
| 43 | 2 | ARM64 assembly | 2.994 | 2.47706 | 34.875 | 8 |
| 44 | 1 | PyTorch | 1.355 | 2.42697 | 35.632 | 8 |
| 44 | 1 | ARM64 assembly | 2.988 | 2.43971 | 35.046 | 8 |
| 44 | 2 | ARM64 assembly | 2.964 | 2.43971 | 35.046 | 8 |
| 44 | 2 | PyTorch | 1.326 | 2.42697 | 35.632 | 8 |

## Controls and retained evidence

Both implementations used the same float32 initial parameters, token rows,
per-epoch schedules, Adam settings, one-thread CPU budget and stopping rule.
Run order was reversed between repeats, one child ran at a time, and fixed
15-second pauses separated children. Power, thermal and load snapshots are
retained in the raw runner manifest. Operating-system scheduling, temperature
and clock frequencies were not experimentally fixed.

The independent post-run audit passed **3395 checks**.
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
