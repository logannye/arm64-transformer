# The five-engine study: what you can check and what you can rerun

This page covers the accepted first phase,
[gpu-v5-library-phase1](../experiments/gpu-v5-library-phase1/results/README.md):
120 fits at widths 32, 64, 128 and 256 on one Apple M4 Max.

It is not a one-command rerun. From the public repository you can recompute
the reported ratios, confirm that the published inputs match their recorded
hashes, and build and audit the native engine yourself. Rerunning the timed
fits needs two frameworks built from source and the host the study was frozen
on, and the raw evidence of the accepted run is in a local archive, not here.

## What the baseline is

The headline numbers are 2.38–2.70× faster training and 3.99–7.52× faster
resident batch-one inference at widths 32–256. These speedups are over the
PyTorch CPU engine as configured for this study: a project-maintained build on
one numerical thread, with FP64 intermediates at four stages. They are not
speedups over a stock PyTorch release.

Against stock PyTorch 2.14.1, the earlier
[two-engine study](../experiments/width-sweep-v6/README.md) measured the
assembly at 1.29–1.79× faster training at widths 32–256, 0.93× at width 512 and
0.72× at width 1024. The two studies ran under different protocols and
numerical gates, so their ratios should not be combined. The study's MPS and
MLX engines use two-float FP32 arithmetic at their selected stages; how much of
their time that accounts for was not measured.

## What you can check from the public files

**The headline ratios.** The accepted
[summary.json](../experiments/gpu-v5-library-phase1/results/summary.json) holds
the three paired-seed ratios behind each headline number, and
[measurements.csv](../experiments/gpu-v5-library-phase1/results/measurements.csv)
holds the six-fit median of every engine and width. From the repository root,
with any Python 3 and no dependencies:

```sh
python - <<'EOF'
import csv, json, statistics
from pathlib import Path

results = Path("experiments/gpu-v5-library-phase1/results")
summary = json.loads((results / "summary.json").read_text())
rows = list(csv.DictReader((results / "measurements.csv").open()))
cell = {(int(r["width"]), r["engine"]): r for r in rows}

print("width  training  batch-one   (median of three paired-seed ratios)")
for entry in summary["cpu_torch_relative"]:
    if entry["engine"] != "cpu_arm64":
        continue
    seeds = entry["paired_seeds"]
    training = statistics.median(s["training_speedup"] for s in seeds)
    latency = statistics.median(s["resident_b1_speedup"] for s in seeds)
    assert training == entry["median_paired_training_speedup"]
    assert latency == entry["median_paired_resident_b1_speedup"]
    print(f"{entry['width']:5d}  {training:7.2f}x  {latency:8.2f}x")

print("width  training  batch-one   (ratio of the six-fit medians in the CSV)")
for width in summary["workload"]["widths"]:
    torch, arm64 = cell[width, "cpu_torch"], cell[width, "cpu_arm64"]
    training = float(torch["median_training_seconds"]) / float(arm64["median_training_seconds"])
    latency = float(torch["median_latency_p50_seconds"]) / float(arm64["median_latency_p50_seconds"])
    print(f"{width:5d}  {training:7.2f}x  {latency:8.2f}x")
EOF
```

```text
width  training  batch-one   (median of three paired-seed ratios)
   32     2.53x      7.52x
   64     2.70x      7.28x
  128     2.48x      5.16x
  256     2.38x      3.99x
width  training  batch-one   (ratio of the six-fit medians in the CSV)
   32     2.52x      7.53x
   64     2.70x      7.35x
  128     2.52x      5.12x
  256     2.36x      4.00x
```

The first block is the reported statistic: for each seed, the median of its two
repeats, then the median of the three per-seed ratios. The second is a different
statistic from the other public file, and lands within about 2% of the first.
The per-fit timings themselves are not public, so the per-seed ratios are the
finest grain you can check.

**The published inputs.** The comparison contract, the framework patches and
the Metal shader are in the repository, and their SHA-256 values are recorded
in the selection and summary files:

```sh
shasum -a 256 integration/library-study-v1/comparison-contract.json \
  experiments/gpu-v5-library/framework-selection.json \
  validation/owned-pytorch-numpy2-v1/changes.patch \
  integration/mlx-release-bench/inputs/embedding-fix.patch \
  integration/mlx-release-bench/inputs/attention-fix.patch \
  native/metal/precision-normalization-v1.metal
```

```text
25b5847326ef7aee4a6580423a49c2812006cbfb6aa8ad84a365930015591714  integration/library-study-v1/comparison-contract.json
f1fb641386e1a2cfd5afd49981618969c32b61f7b07a38704e7d347e42b77909  experiments/gpu-v5-library/framework-selection.json
07128b3a9bbe39f4e423ef7e038f955c52cbc562d8edb9ed28c4baac59c49b3f  validation/owned-pytorch-numpy2-v1/changes.patch
9aecf0c9b0131fe0673493035fd3c47fba3e4ffb1cfc04d00a506702e1d1067c  integration/mlx-release-bench/inputs/embedding-fix.patch
b9787a423920641034f42e611f0d8529f6dd49022d094e280fa5ce52a38b7cc3  integration/mlx-release-bench/inputs/attention-fix.patch
1df2879328a5e3493e3203f762ce8606002c397ec028754534c4f4cf4a547035  native/metal/precision-normalization-v1.metal
```

In order, these match `scope_amendment.original_contract_sha256` in the
summary, the value in the
[protocol](../experiments/PROTOCOL-gpu-v5-library.md), the PyTorch
`combined_patch_sha256` and the first two MLX `patches` in
[framework-selection.json](../experiments/gpu-v5-library/framework-selection.json),
and `selected_shader` in
[metal-selection.json](../experiments/gpu-v5-library/metal-selection.json).

**The native engine.** The [quick start](../README.md#quick-start) builds the
assembly and audits its kernels, gradients, optimizer and executable against
independent references, and
[check_native_build_identity.py](../scripts/check_native_build_identity.py)
confirms that the binaries match the measured source. That checks correctness
on your machine. It does not reproduce a timing.

**What hashes do and do not show.** The summary lists nine evidence files by
SHA-256: the accepted report, the final audit, the frozen manifest, the
campaign record and five others. None of the nine is in the repository. A hash
ties the public summary to one exact set of bytes; it does not let a reader
inspect them.

## What a rerun would need

| Requirement | What was used | Where it is |
|---|---|---|
| Host | Apple M4 Max, 128 GiB, macOS 26.2. The driver records this identity when the study is frozen and stops if it changes. | `hardware` in the summary |
| PyTorch, for the CPU and MPS engines | A source build at `71d2a08507ab47dd6b2fa9b66319d17ce2377de4` with four patches, against NumPy 2.2.6; hosted run 41158978641 | Recipe, patch and contract: [validation/owned-pytorch-numpy2-v1](../validation/owned-pytorch-numpy2-v1/README.md). Workflow: `.github/workflows/owned-pytorch-numpy2-v1.yml` |
| MLX | A source build at `a21f892a6e8e83afb117cef1d1ceae3e82f48c32` with two library patches and one test-only patch, in the `jit-on` shader mode; hosted run 40978071574 | Recipe and patches: [integration/mlx-release-bench](../integration/mlx-release-bench/CONTRACT.md) and [mlx-test-compat-v1](../integration/gpu-owned/mlx-test-compat-v1/README.md). Workflow: `.github/workflows/upstream-mlx-release-bench.yml` |
| Precision API | `torch.nn.precise_fp32` and `mlx.nn.precise_fp32`, added to the two built packages | Source: [integration/precision-library-v1](../integration/precision-library-v1/README.md). Installer: `integration/gpu-owned/derive_precision_api.py` |
| Native and Metal engines | The assembly at policy 0, built on hosted CI; the reference Metal shader path with the normalization refinement | [native/](../native/README.md), [metal-selection.json](../experiments/gpu-v5-library/metal-selection.json) |
| Driver | Mathematical source `bd2cd2823038729779a217ee5b084fa06191d3a9`, measurement control `30390c80573baf687f9287f2b1921c79d5bc67c0`, audit control `f1de5c69b77e634a10970d4a4e3dc6e6cff2bd4d` | [integration/library-study-v1](../integration/library-study-v1/README.md) |

Three things stand between these parts and a rerun:

- **The built frameworks are not a download.** The patches, recipes and build
  scripts are public, but both build workflows run only in this repository and
  keep their artifacts for 14 days. A new build has to be captured and bound
  afresh before the driver will accept it.
- **The driver is fail-closed.** It opens no reserved operand and runs no
  measured fit until it has authenticated both source roles, the complete
  runtime, the native binaries, the original fixtures, the host and the full
  set of prerequisite audits. The prerequisite audits, the runtime manifests
  and the API derivation receipts are not in the repository, which records
  hashes for the manifests and receipts.
- **Another machine makes it a new study.** The freeze binds the host, so the
  same protocol on different hardware would be a new freeze with its own
  admission and audits. It could support or contradict these results; it would
  not regenerate the accepted evidence.

Everything below the summaries is in the local archive as well: the per-fit
records, the 240 checkpoints, the 1,200 replays, the 240 oracle outputs and the
admission arrays.

## Claims, evidence and guides

| Claim | Public evidence | How to rerun |
|---|---|---|
| ARM64 trained 2.38–2.70× faster and ran batch-one inference 3.99–7.52× faster than the study's PyTorch CPU engine, widths 32–256 | Per-seed ratios in `summary.json`; medians in `measurements.csv` | Recompute with the command above. A full rerun needs everything in the previous section. |
| All 120 fits passed: 1,200 replays, 240 FP64 oracles, 240 checkpoint audits, 60 exact repeat pairs | Counts in `summary.json`; the audit itself is local, bound by hash | Not possible from public files |
| Metal had the lowest training time at width 256, 3.593 s against 3.981 s for ARM64 | `measurements.csv` | As above |
| Against stock PyTorch 2.14.1: 1.29–1.79× at widths 32–256, 0.93× at 512, 0.72× at 1024 | [width-sweep-v6](../experiments/width-sweep-v6/README.md), with raw evidence in its tagged release | [REPRODUCE.md](REPRODUCE.md), a full rerun |
| The first attempt at widths 512 and 1024 failed its checkpoint audit | [width-sweep-v5-outcome](../experiments/width-sweep-v5-outcome/README.md) | [REPRODUCE-v5.md](REPRODUCE-v5.md) |
| No QKV kernel candidate met the 3% threshold | [gpu-v3-qkv](../experiments/gpu-v3-qkv/README.md) | [Its guide](../experiments/gpu-v3-qkv/REPRODUCE.md) |
| The native kernels match independent references at every width | [Hosted CI](../.github/workflows/validate.yml) on every push | [Quick start](../README.md#quick-start) |

For an informal look at the assembly against stock PyTorch on your own machine,
[transformer/demo.py](../transformer/demo.py) trains the model once with each
and prints both times. It is a demonstration, not a measurement under either
protocol.

## Which protocol text applies

Three identifiers appear in the record, and they are one lineage:

- `gpu-v5-library-five-engine-v1` is the text in
  [PROTOCOL-gpu-v5-library.md](../experiments/PROTOCOL-gpu-v5-library.md). It
  lists three precise stages for CPU PyTorch.
- `gpu-v5-library-five-engine-v3` is the driver's contract. It adds the first
  feed-forward projection as a fourth CPU PyTorch stage and moves the
  mathematical source to `bd2cd28`; the
  [driver's notes](../integration/library-study-v1/README.md) record the change.
- `gpu-v5-library-five-engine-v4-phase1` is the
  [amendment](../experiments/gpu-v5-library-phase1/AMENDMENT.md) that staged the
  study to widths 32–256. It is the protocol the accepted summary names.

Where they differ, the summary's `model_selection` is what ran.
