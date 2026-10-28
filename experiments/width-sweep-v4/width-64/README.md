# Matched transformer CPU comparison, v4, width 64

This optimization study reuses v1's fixed train/validation/test fixtures. It is an engineering regression benchmark, not a fresh untouched test set. Kernel development used synthetic correctness and timing inputs; the model, optimizer, seeds and stopping rule were unchanged.

ARM64 training was **1.792× faster** than PyTorch, using the geometric mean of six paired training-time ratios.
This is the same 52,160-parameter causal transformer implemented with ordinary
batched PyTorch (built-in scaled-dot-product attention and autograd) and handwritten
Apple ARM64 assembly. The experiment contains **12 complete fits**: three
initialization seeds, two timing repeats, and both implementations.

Measured source revision: `276c6d091a7ebdc5f7831be2169683db4715f748`. Hardware: Apple M4 Max.
Platform: `macOS-26.2-arm64-arm-64bit`. PyTorch: `2.14.1`.

Native processes requested automatic dispatch. Hardware SME capability: True; streaming vector length: 64 bytes. Actual dense calls per native fit were SME 94,040 forward / 81,920 backward, and NEON 0 forward / 0 backward. These authenticated counters cover the whole process, including evaluation and inference samples.

The auto-policy preflight contains 1,041 checks; forced-policy gates add 2,080, for 3,121 admission checks.

| Measurement | PyTorch | ARM64 assembly |
|---|---:|---:|
| Test NLL (lower is better) | 2.4145 ± 0.0291 | 2.4138 ± 0.0282 |
| Test bits / character | 3.4835 ± 0.0420 | 3.4824 ± 0.0407 |
| Test perplexity | 11.188 ± 0.326 | 11.180 ± 0.315 |
| Next-character accuracy (%) | 36.869 ± 0.494 | 36.589 ± 0.723 |
| Training + validation (s) | 1.595 ± 0.013 | 0.890 ± 0.004 |
| Training CPU time (s) | 1.590 ± 0.013 | 0.889 ± 0.005 |
| Whole child process (s) | 2.685 ± 0.044 | 0.905 ± 0.004 |
| Forward p50, full context (ms) | 0.0626 ± 0.0018 | 0.0198 ± 0.0001 |
| Forward p95, full context (ms) | 0.0653 ± 0.0016 | 0.0209 ± 0.0013 |
| Forward p99, full context (ms) | 0.0666 ± 0.0024 | 0.0216 ± 0.0011 |
| Peak RSS through training (MiB) | 278.47 ± 1.05 | 3.11 ± 0.00 |

Entries are mean ± sample SD of three seed means. Timing repeats are averaged
within seed; they are not six independent quality seeds. The primary ratio is
Python/ARM64 training time, **1.792176**, where values above one favor assembly.
All within-implementation repeat checkpoints, optimizer states, held-out logits,
and non-timing learning histories matched exactly.

## Prediction quality and baselines

The mean assembly-minus-PyTorch test NLL difference was
**-0.000695**, and the accuracy difference was
**-0.280762 percentage points**.
Mean paired next-character prediction disagreement was **20.121%**.
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
| 42 | 1 | ARM64 assembly | 0.902 | 2.38570 | 36.389 | 7 |
| 42 | 1 | PyTorch | 1.601 | 2.38665 | 37.219 | 8 |
| 43 | 1 | PyTorch | 1.598 | 2.44475 | 36.304 | 7 |
| 43 | 1 | ARM64 assembly | 0.888 | 2.44209 | 35.986 | 7 |
| 44 | 1 | ARM64 assembly | 0.895 | 2.41376 | 37.390 | 8 |
| 44 | 1 | PyTorch | 1.630 | 2.41223 | 37.085 | 8 |
| 42 | 2 | PyTorch | 1.574 | 2.38665 | 37.219 | 8 |
| 42 | 2 | ARM64 assembly | 0.886 | 2.38570 | 36.389 | 7 |
| 43 | 2 | ARM64 assembly | 0.882 | 2.44209 | 35.986 | 7 |
| 43 | 2 | PyTorch | 1.577 | 2.44475 | 36.304 | 7 |
| 44 | 2 | PyTorch | 1.589 | 2.41223 | 37.085 | 8 |
| 44 | 2 | ARM64 assembly | 0.886 | 2.41376 | 37.390 | 8 |

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
