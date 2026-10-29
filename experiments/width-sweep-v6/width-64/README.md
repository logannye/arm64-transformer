# Higher-precision attention study, v6, width 64

**Publication view:** this directory contains compact metrics and audit indices. Full correctness checks, numerical failure records, admission gates, and raw evidence are retained in the [release archive](https://github.com/logannye/arm64-transformer/releases/tag/transformer-attention-precision-v6). Each JSON index records the SHA-256 and archive path of its complete source. Numerical outcomes and qualification statements below are unchanged.

The implementation preserves the transformer and training budget while making
attention score precision an explicit hardware-design choice. Both engines are
replayed at every selected checkpoint, regardless of which engine trained it.
Independent NumPy FP64 equations evaluate the exact saved FP32 weights.

## Numerical result

Status: **`completed`**. Both implementations passed the prespecified FP64 accuracy requirement at all selected checkpoints.

The post-run contract is `v6-fp64-centered-attention-v1`. Each native and production
PyTorch prediction is compared with FP64 using the unchanged forward limit
`abs(error) <= 2e-5 + 2e-4*abs(FP64)`. PyTorch FP32 compatibility is separately
reported, with its actual passing or failing records preserved.

| Numerical comparison | Checks | Failed checks | Failed coordinates |
|---|---:|---:|---:|
| Native versus independent FP64 | 6,156 | 0 | 0 |
| Production PyTorch versus independent FP64 | 6,156 | 0 | 0 |
| Native versus production/functional PyTorch FP32 | 12,288 | 0 | 0 |

Counts cover every test and validation sequence at all 12 selected checkpoints,
including repeated fits, plus per-engine validation-NLL comparisons. Coordinates
are individual logits or NLL scalars. They are not independent statistical samples.
The first and last sequence of each split also cross-check NumPy FP64 against
independent functional PyTorch FP64 at fixed absolute/relative limits of `1e-10`.

The originating engine reproduced saved test logits bit-exact, using its original
evaluation grouping (batch 16 for PyTorch; individual sequences for native).
Its validation loss reproduced the selected history entry exactly. Within-engine
repeat weights, moments, saved logits and non-timing histories also matched exactly.
Evidence-integrity checks passed: **9,605**.

## Matched measurements

| Measurement | PyTorch | ARM64 assembly |
|---|---:|---:|
| Training + validation (s) | 1.565 ± 0.005 | 0.887 ± 0.004 |
| Full-context forward p50 (ms) | 0.0633 ± 0.0004 | 0.0198 ± 0.0001 |
| Peak RSS through training (MiB) | 278.35 ± 1.16 | 3.11 ± 0.00 |
| Test NLL | 2.41454 ± 0.02911 | 2.41385 ± 0.02819 |
| Test accuracy (%) | 36.869 ± 0.494 | 36.589 ± 0.723 |

Entries are mean ± sample SD across three seed means, after averaging two repeats
within each seed. The geometric mean of six paired PyTorch/native training-time
ratios is **1.764055**; values above one favor native. This descriptive statistic
does not override the qualification status above.

![Training and process time](training.png)
![Forward latency](latency.png)
![Process memory](memory.png)
![Training and validation histories](learning.png)

## Fixed workload and scope

One causal decoder block, width 64, 52,160 parameters,
four heads, context 32, feed-forward width twice the model width, ASCII-128 output,
FP32 parameters and optimizer state. The pinned Tiny Shakespeare benchmark uses
65,536 training and 8,192 validation/test targets. Seeds 42/43/44, eight epochs,
batch 16 and 1,024 Adam updates are unchanged. Minimum epoch validation NLL selects
the checkpoint; test predictions are evaluated afterward.

This is a reused engineering benchmark. Exposed v5 failures inform development
and remain failed in their original reports. V6 admission adds prospectively
declared synthetic stress tests and independent FP64 checks. It does not claim a
fresh untouched corpus assessment or equivalence of independent training paths.

Training timing includes forward/backward, Adam, finite checks, validation and
checkpoint/history work. Setup, final test and inference samples are excluded.
Forward timing is batch one over a full 32-character context, including attention
softmax, with 20 warmups and 100 samples per fit; it excludes output probability
softmax, sampling and I/O. RSS includes the framework/runtime and loaded data.
One compute thread, serial children, alternating/reversed order and 15-second
pauses remain fixed. Operating-system clocks and scheduling are not controlled.

## Every measured fit

| Seed | Repeat | Implementation | Train (s) | Test NLL | Best epoch |
|---:|---:|---|---:|---:|---:|
| 42 | 1 | ARM64 assembly | 0.8867 | 2.385702 | 7 |
| 42 | 1 | PyTorch | 1.5737 | 2.386654 | 8 |
| 43 | 1 | PyTorch | 1.5780 | 2.444746 | 7 |
| 43 | 1 | ARM64 assembly | 0.8980 | 2.442091 | 7 |
| 44 | 1 | ARM64 assembly | 0.8894 | 2.413756 | 8 |
| 44 | 1 | PyTorch | 1.5807 | 2.412234 | 8 |
| 42 | 2 | PyTorch | 1.5680 | 2.386654 | 8 |
| 42 | 2 | ARM64 assembly | 0.8892 | 2.385702 | 7 |
| 43 | 2 | ARM64 assembly | 0.8843 | 2.442091 | 7 |
| 43 | 2 | PyTorch | 1.5437 | 2.444746 | 7 |
| 44 | 2 | PyTorch | 1.5479 | 2.412234 | 8 |
| 44 | 2 | ARM64 assembly | 0.8764 | 2.413756 | 8 |

Measured revision: `2c5191047068c16b12e3312652b118d48a5f3744`. Hardware: Apple M4 Max.
Platform: `macOS-26.2-arm64-arm-64bit`. PyTorch: `2.14.1`.
Preflight contract: `strict-free-running-v1`. Auto/forced backend gates
contain 3,301 blocking checks.
All 98,304 saved test prediction rows and
their per-run metrics and numerical counts appear in [summary.json](summary.json).
[audit.json](audit.json) indexes the complete numerical evidence in the release archive. Raw checkpoints, fixtures, source, binaries and timings
are authenticated in the accompanying evidence bundle.
