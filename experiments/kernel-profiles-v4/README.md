# Matrix optimization diagnostics

These synthetic diagnostics guided the v4 implementation before the matched
training study. They use seed 777 and no benchmark checkpoints or test quality.
The preserved v3 library and final v4 library run in one process, with ten
rotated/reversed timing blocks of 200 calls per case and one CPU compute thread.
Each complete public call includes dispatch, FFI, packing, and scratch handling.
Forward outputs and all model/linear gradients are checked before timing.

![Complete native calls and sampled matrix bottleneck](kernel-diagnostics.png)

| Width | v3 forward (µs) | v4 forward (µs) | v3 backward + loss (µs) | v4 backward + loss (µs) |
|---:|---:|---:|---:|---:|
| 32 | 14.895 | 15.028 | 25.613 | 23.265 |
| 64 | 33.706 | 20.504 | 67.752 | 27.146 |
| 128 | 102.524 | 37.623 | 215.212 | 47.068 |
| 256 | 348.237 | 83.962 | 756.124 | 114.862 |

Width 256 improves **4.15× in forward and 6.58× in backward**. Width 32 keeps
NEON and has a small forward overhead in this comparison; its backward improves
1.10×. These are full native model calls, not end-to-end training or PyTorch
speedup estimates. Repeated backward calls reuse saved activations and accumulate
parameter gradients. The full training study measures actual complete updates.

Separate macOS `sample` captures of synthetic native training establish the
original bottleneck. Dense linear kernels account for approximately 57%, 71%,
86%, and 92% of post-loader sampled execution at widths 32/64/128/256. The final
implementation's shares are 54%, 26%, 36%, and 52%, including its named packing
helper. Shares are exclusive top-of-stack counts, not summed inclusive frames
or wall-clock savings. Windows contain the training/validation/runtime work
present during sampling; the final profiler deliberately stops its child after
the capture and preserves partial output. No incomplete fit is a quality result.

The development candidate initially left the packing helper unnamed, producing
unattributed addresses. Those raw captures are retained and must not be treated
as complete dense attribution. The final capture names the helper, retains
unknown-address records explicitly, and has zero unattributed leaf records.
The collapsed table omits symbols with fewer than five samples; neighboring
assembly symbols can be ambiguous, so only the combined dense family is used.

[summary.json](summary.json) records all plotted values, sample denominators,
binary identities and raw-file hashes. [figure.json](figure.json) authenticates
the image. The v4 release includes raw kernel blocks, stack samples, scripts,
preserved baseline/candidate binaries and development checks. The renderer is
[report_kernel_optimization.py](../../scripts/report_kernel_optimization.py).

To repeat a fresh kernel diagnostic with a preserved v3 library:

```sh
ARM64_TRANSFORMER_WIDTH=256 ARM64_TRANSFORMER_BACKEND=auto \
  python scripts/benchmark_kernels.py \
  --baseline-library /path/to/v3/width-256/libtransformer.dylib \
  --output results/new-width-256-diagnostic.json
```

Run profiling separately from every correctness check and measured benchmark.
No isolated instruction ablation or cache-counter measurement is implied.
