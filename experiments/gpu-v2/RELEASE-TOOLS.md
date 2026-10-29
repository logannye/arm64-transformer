# Reporting and preserving GPU v2 engineering evidence

These tools process saved evidence. They do not train models, execute GPU kernels,
open the reserved admission stream, or evaluate original validation/test arrays.
Run reporting and packaging after all experiment writers and performance
measurements have stopped.

## Baseline explanation

`scripts/summarize_gpu_v2_baselines.py` reads the completed 24-worker diagnostic,
separates exact replay, runtime completion, and FP64 accuracy, and localizes the
first retained divergent full-gradient pair for each engine/width. It reads those
saved gradient arrays without executing model arithmetic. The localization is
explicitly limited to the first divergent case; it does not characterize every
shape or imply that later training differences remain confined to embeddings.

```sh
PYTHONPATH=work/gpu-v1/python-site:. .venv/bin/python \
  scripts/summarize_gpu_v2_baselines.py \
  --baseline work/gpu-v2/repeatability/full-v3 \
  --output work/gpu-v2/repeatability/full-v3-summary
```

All outputs use new directories. Existing outputs are preserved, including
failed and superseded attempts.

## Explicitly select an eligible engineering attempt

`scripts/report_gpu_v2_engineering.py` never searches for the fastest attempt.
It reads the explicitly supplied six-width search and reference-profile
directories. The default accepted attempt is `selection-v2`. Its `attempt.json`
must identify `scope: metal_development_search_attempt`, the matching
`attempt_id`, and `eligible_for_reporting: true`. A directory containing
`invalidated.json` is always rejected. A later approved attempt requires an
explicit `--attempt-id`; that option cannot override an invalidation receipt.

The reporter requires a finalized baseline launcher receipt and all six search
receipts. A failed search remains a failed outcome. Selecting the reference
policy because no candidate improved it is also a valid outcome.

Only direct final `selected_vs_reference` pairs produce latency ratios. The
reporter recomputes block medians and ratios from the saved observations and
requires their timing controls. It does not multiply greedy candidate versus
incumbent ratios. Missing direct pairs remain unavailable. Whiskers show the
range of five paired block ratios, not a confidence interval. Training and
inference remain separate; there is no pooled-width headline or comparison
against framework/CPU performance.

The latency table also reports p50 and p95 in microseconds and milliseconds,
using linear interpolation over the 100 saved observations. Training throughput
is 51,200 next-token targets divided by the summed durations of 100 B16T32
updates at the same timing boundary. Direct comparisons require the retained
fresh-pair proof: equal initial state, matching owned allocation counts/bytes,
and exactly three batch buffers per backend. Device-global allocations and
process RSS are not substituted for owned-buffer measurements.

`metal-kernel-ablation.png` and its SVG compare the matched reference and final
selected bundle at every width, separately for training and inference. They sum
three-repetition stage medians into dense forward/dX/dW, QKV, normalization/bias
reductions, attention, embedding, Adam, and other work. QKV matrix work includes
its forward, input gradient, and weight gradient; its bias gradient remains in
reductions. Raw observations and grouped costs are retained in
`kernel-stage-costs.csv` and `kernel-category-costs.csv`. Each profile stage uses
a separate synchronized command buffer, so this is an intrusive kernel-cost
comparison, not an exact attribution of production wall time or an isolated
causal ablation. The figure uses a separate microsecond scale in each panel.
Stage command-buffer counts are retained separately from kernel dispatches;
dispatch differences remain unavailable unless actual per-stage counts exist.

```sh
PYTHONPATH=work/gpu-v1/python-site:. .venv/bin/python \
  scripts/report_gpu_v2_engineering.py \
  --baseline work/gpu-v2/repeatability/full-v3 \
  --profiles work/gpu-v2/profiles/reference-v3 \
  --searches work/gpu-v2/search/selection-v2 \
  --output results/gpu-v2-engineering
```

Use the exact profile directory authenticated by the search ledgers. The example
path is not discovered automatically. `--no-figures` emits JSON/CSV without
loading Matplotlib. The PNGs show Metal-only paired timings/candidate outcomes
and separate baseline qualification. No learning curves are generated for the
unadmitted formal campaign.

## Complete retained evidence in bounded archive parts

`scripts/package_gpu_v2_engineering.py create` requires the finalized blocked
report, its exact reporter source, all 24 baseline workers, complete declared
case inventories, explicit skips after runtime errors, first outputs, first
divergent outputs, and closing receipts for superseded attempts. Every reported
source digest must occur at its declared evidence path. Failed gates remain
failed; the packager does not require or manufacture formal admission.

The archive contains current release sources, a Git snapshot of HEAD, any
explicit `--revision` snapshots, the frozen CPU source tree and binary/layout
artifacts, the Metal binary, runtime identity records, and every retained v2
evidence file. This includes raw states/logits/gradients, failures, profiles,
search ledgers, logs, and incomplete attempts. Dependency installations and
named caches are excluded and listed. Original CPU manifests preserve all
training/validation/test hashes for all 18 input cells; only seed-42 initial
parameters and training bytes are copied. Original held-out arrays stay unread.

```sh
.venv/bin/python scripts/package_gpu_v2_engineering.py create \
  --report results/gpu-v2-engineering \
  --output ../arm64-transformer-gpu-v2-release
```

The output must be a new directory outside the repository and all input trees.
Creation checks available disk space, hashes each input, streams one tar/gzip
archive into parts of at most 1,900,000,000 bytes, and verifies every part and
embedded member without extraction. It then checks that input inventories,
bytes, and Git state stayed unchanged. A verification sidecar is written only
after those checks pass. An interrupted archive is preserved without a success
receipt. No original evidence is deleted.

Publish the `*.part-*` files, archive index, external manifest, verification
receipt, and checksum list. Generated standalone source-snapshot tar files are
already embedded in the archive and need not be uploaded separately. To verify
downloaded parts without concatenating a second full archive onto disk:

```sh
shasum -a 256 -c gpu-v2-engineering-SHA256SUMS.txt
.venv/bin/python scripts/package_gpu_v2_engineering.py verify \
  --index /path/to/downloads/gpu-v2-engineering-archive.json
```

Archive integrity establishes which bytes were preserved. It does not establish
numerical qualification, reproducibility on another machine, or permission to
open the formal campaign.
