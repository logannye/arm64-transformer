# Integration

The code that brings all five engines into one study: the MLX build recipes,
the tooling for project-maintained framework runtimes, the precision API, the
trained-state precision work, and the driver that ran the accepted phase. The
raw evidence these tools read and write lives under the git-ignored `work/` and
is not in the repository.

## Folders

| Folder | What it is |
|---|---|
| [library-study-v1](library-study-v1/README.md) | The driver of the accepted five-engine phase: sealed source roles, qualification, admission, the timed fits, checkpoint replay, the audits and the report. It will not open reserved operands or run a measured fit until source, runtime, fixtures and hardware all authenticate. |
| [precision-library-v1](precision-library-v1/README.md) | The precision API added to the maintained PyTorch and MLX packages: generic `linear`, last-axis `layer_norm` and causal self-attention. CPU operations use FP64 intermediates; MPS and MLX use two-float FP32. Stock operators are unchanged. |
| [gpu-owned](gpu-owned/OWNED-COMMANDS.md) | Tooling for the project-maintained runtimes: [runtime capture](gpu-owned/RUNTIME-IDENTITY.md), the PyTorch and MLX suite runners, evidence preparation and audit, the patch candidates, and `derive_precision_api.py`, which adds the precision API to a built package. |
| [trained-precision](trained-precision/README.md) | Precision development at widths 512 and 1024: replay of the 89 historical failing-sequence anchors, the arithmetic candidates for each engine, and independent raw auditors. |
| [mlx](mlx/HOSTED-VALIDATION.md) | Hosted build and validation of MLX at `a21f892a6e8e83afb117cef1d1ceae3e82f48c32` with the embedding and attention patches. |
| [mlx-official](mlx-official/CONTRACT.md) | The same patched source run through the structure of MLX's official CI matrix. |
| [mlx-release-bench](mlx-release-bench/CONTRACT.md) | Matched Release builds for comparing individual operators with and without the patches. The recipe of the selected MLX build. |

## Relation to the five-engine study

The [accepted phase](../experiments/gpu-v5-library-phase1/results/README.md) was
run by `library-study-v1`. Its framework engines are the two builds recorded in
the [framework selection](../experiments/gpu-v5-library/framework-selection.json):

- **MLX**: built from source by the `mlx-release-bench` recipe with two library
  patches, [embedding-fix.patch](mlx-release-bench/inputs/embedding-fix.patch)
  and [attention-fix.patch](mlx-release-bench/inputs/attention-fix.patch). A
  third patch, [mlx-test-compat-v1](gpu-owned/mlx-test-compat-v1/README.md),
  changes one test file in the built package.
- **PyTorch**, for both the CPU and MPS engines: built by
  [validation/owned-pytorch-numpy2-v1](../validation/owned-pytorch-numpy2-v1/README.md).
  Three of its four patches live here, in
  [gpu-owned/patches](gpu-owned/patches/mps-layout-candidate.patch),
  [half-index-candidate-v1](gpu-owned/half-index-candidate-v1/README.md) and
  [half-narrowing-candidate-v1](gpu-owned/half-narrowing-candidate-v1/PROPOSAL.md).

Both packages then receive the precision API from `precision-library-v1`. The
selected stages of each engine are listed in the accepted
[summary](../experiments/gpu-v5-library-phase1/results/summary.json). What a
reader can check or rerun is set out in the
[reproduction page](../docs/REPRODUCE-five-engine.md).

## Unit suites

`library-study-v1`, `precision-library-v1`, `trained-precision` and the MLX
helpers are tested by pytest, from [tests/](../tests/README.md). `gpu-owned`
carries its own standard-library `unittest` suites, which
[hosted CI](../.github/workflows/validate.yml) runs one file per process, so
that each import-isolation check starts from a fresh interpreter:

```sh
for test_file in integration/gpu-owned/test_*.py; do
  python -I -B -m unittest discover -s integration/gpu-owned \
    -p "$(basename "$test_file")"
done
python -I -S -B -m unittest discover \
  -s integration/gpu-owned/half-index-candidate-v1 -p 'test_*.py'
```

The array-audit tests use the installed NumPy. No test here executes a GPU
framework.
