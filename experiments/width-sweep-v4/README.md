# Transformer width sweep, v4

The complete sweep contains **48 fits** at widths 32, 64, 128 and 256:
three seeds, two timing repeats and both implementations at each width.
Each comparison uses the same initial weights, token rows and schedules within
its width/seed pair. Source revision: `276c6d091a7ebdc5f7831be2169683db4715f748`.

This is the reused engineering benchmark from v1/v2/v3, not a fresh untouched test
set. Widths and training rules were fixed before timing. Wider models change
capacity and computational cost; this report does not select a width from test quality.

All measured native processes requested automatic dispatch. The table records actual dense calls per fit, including validation, final test and inference samples; it does not imply every transformer operation used SME. Hardware capability, vector length and exact counters are authenticated against each width's numerical gate.

| Width | SME forward / backward | NEON forward / backward |
|---:|---:|---:|
| 32 | 0 / 0 | 94,040 / 81,920 |
| 64 | 94,040 / 81,920 | 0 / 0 |
| 128 | 94,040 / 81,920 | 0 / 0 |
| 256 | 94,040 / 81,920 | 0 / 0 |

## Training and forward timing

| Width | Parameters | PyTorch training (s) | ARM64 training (s) | Paired training ratio |
|---:|---:|---:|---:|---:|
| 32 | 17,952 | 1.295 ± 0.012 | 0.712 ± 0.003 | 1.819× |
| 64 | 52,160 | 1.595 ± 0.013 | 0.890 ± 0.004 | 1.792× |
| 128 | 169,728 | 2.410 ± 0.009 | 1.641 ± 0.003 | 1.468× |
| 256 | 601,472 | 5.163 ± 0.017 | 4.000 ± 0.006 | 1.291× |

The primary ratio at each width is the geometric mean of six paired
PyTorch/ARM64 training-time ratios. Above one favors assembly; below one favors
PyTorch. There is no pooled-width primary ratio or excluded unfavorable width.
Training includes validation, finite checks, checkpoint selection/writes and
history bookkeeping. Each fit performs 1,024 updates over eight epochs.

![Training and forward cost across widths](scaling.png)

| Width | PyTorch forward p50 (ms) | ARM64 forward p50 (ms) | Ratio of mean p50s |
|---:|---:|---:|---:|
| 32 | 0.0544 ± 0.0003 | 0.0150 ± 0.0003 | 3.629× |
| 64 | 0.0626 ± 0.0018 | 0.0198 ± 0.0001 | 3.154× |
| 128 | 0.0779 ± 0.0002 | 0.0367 ± 0.0005 | 2.123× |
| 256 | 0.1210 ± 0.0010 | 0.0844 ± 0.0001 | 1.433× |

Forward latency is one full 32-character context, batch one, after 20 warmups,
with 100 samples per fit. It excludes token loading, loss, softmax, sampling and
I/O. It is not cached next-token generation. The forward ratio uses the two
implementation means, a different statistic from the primary training ratio.

![Per-width implementation time ratios](speedup.png)

Entries and error bars are mean ± sample SD of three seed means, with the two
repeats averaged within each seed. Repeats are not independent quality seeds.
OS scheduling, temperatures and clock frequencies were not experimentally fixed.

## Throughput, memory and prediction quality

| Width | Implementation | Target ktoken/s | Peak RSS (MiB) | Test NLL | Accuracy (%) |
|---:|---|---:|---:|---:|---:|
| 32 | PyTorch | 405.1 ± 3.5 | 277.52 ± 0.26 | 2.4585 ± 0.0292 | 35.046 ± 0.532 |
| 32 | ARM64 assembly | 736.8 ± 3.1 | 2.34 ± 0.00 | 2.4621 ± 0.0211 | 35.042 ± 0.671 |
| 64 | PyTorch | 328.8 ± 2.6 | 278.47 ± 1.05 | 2.4145 ± 0.0291 | 36.869 ± 0.494 |
| 64 | ARM64 assembly | 589.2 ± 2.9 | 3.11 ± 0.00 | 2.4138 ± 0.0282 | 36.589 ± 0.723 |
| 128 | PyTorch | 217.6 ± 0.8 | 288.28 ± 2.35 | 2.3862 ± 0.0214 | 37.964 ± 0.254 |
| 128 | ARM64 assembly | 319.5 ± 0.7 | 5.67 ± 0.00 | 2.3859 ± 0.0171 | 37.691 ± 0.778 |
| 256 | PyTorch | 101.6 ± 0.3 | 314.72 ± 0.73 | 2.3962 ± 0.0472 | 37.203 ± 1.705 |
| 256 | ARM64 assembly | 131.1 ± 0.2 | 14.56 ± 0.00 | 2.3969 ± 0.0348 | 37.557 ± 0.702 |

Throughput divides **524,288 training target characters per fit**
by the training-plus-validation timer, then averages repetitions within seed.
Validation tokens are excluded from the numerator while validation time remains
in the denominator. RSS includes the framework, runtime and data; it does not
isolate model memory. NLL and accuracy are descriptive observations, not evidence
of statistical equivalence between implementations.

![Memory and quality across the fixed width family](memory-quality.png)

Uniform and training-only unigram baselines, all per-seed quality differences,
prediction disagreements, checkpoints and individual runs are retained in the
per-width reports. The shared data audit verifies identical corpus identity,
partitions and train/validation/test rows across widths, identical schedules for
each seed, and distinct width-specific initial parameter files. No model-quality
tuning used this test set during the sweep.

## Evidence and limits

The post-run audit contains **14,185 checks**:
549 sweep consistency checks plus all per-width
post-run records. The primary automatic-policy numerical/runtime preflights contain
**4,162 checks** in total. Forced-policy gates add
**8,318 checks**. All
**393,216 retained test prediction rows** were
independently recomputed from selected checkpoints; selected validation losses
were also checked against checkpoint-selection histories. Within-implementation
repeat equality is required before quality aggregation.

- [Width 32: full report](width-32/README.md), [audit](width-32/audit.json) (3,409 checks), [preflight](width-32/preflight.json) (1,039 checks).
- [Width 64: full report](width-64/README.md), [audit](width-64/audit.json) (3,409 checks), [preflight](width-64/preflight.json) (1,041 checks).
- [Width 128: full report](width-128/README.md), [audit](width-128/audit.json) (3,409 checks), [preflight](width-128/preflight.json) (1,041 checks).
- [Width 256: full report](width-256/README.md), [audit](width-256/audit.json) (3,409 checks), [preflight](width-256/preflight.json) (1,041 checks).

See [summary.json](summary.json) for all observations, statistics and hashes,
[audit.json](audit.json) for every post-run check, and
[report-processes.json](report-processes.json) for the separate width-selected
audit subprocesses. Full raw evidence remains in the original result/input
directories. Reporting never overwrites an existing destination.

The first repeat visited ascending widths and the second descending widths;
implementation order reversed within each seed pair. Fits ran sequentially on
AC power with one CPU compute thread and 15-second pauses. Every measured entry
matches its per-width manifest exactly, with unchanged source, binary and inputs.

The scope is this short-context float32 model family on one CPU machine against
eager PyTorch. Native training still accumulates per-sequence gradients. These
results establish neither fresh test-set generalization nor GPU, long-context,
graph-compiled PyTorch, energy, cached decoding or production-serving performance.
Historical v1/v2 comparisons remain separate campaigns with different conditions.
