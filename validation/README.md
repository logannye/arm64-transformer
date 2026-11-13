# Validation

Hosted-build recipes and controls for the project's PyTorch builds, plus one
hosted replay of the native code. A folder is what one GitHub Actions workflow
ran: the patch, a contract that binds source, dependencies and toolchain by
hash, the build script and the runtime verifier. PyTorch is built from source
only on hosted runners; nothing here compiles a framework locally.

## Relation to the five-engine study

The accepted phase runs its PyTorch CPU and MPS engines from one
project-maintained distribution. Its native build is the output of
[owned-pytorch-numpy2-v1](owned-pytorch-numpy2-v1/README.md): PyTorch at
`71d2a08507ab47dd6b2fa9b66319d17ce2377de4` with four patches, against NumPy
2.2.6, as recorded in the
[framework selection](../experiments/gpu-v5-library/framework-selection.json).
The other folders are the earlier builds and the controls that led to it, kept
in place because later recipes and tests refer to them by path and hash. The
MLX side is under [integration/](../integration/README.md).

## Folders

| Folder | What it is |
|---|---|
| `upstream-pytorch` | The original hosted build of patched PyTorch: the embedding patch, build script and runtime verifier. No README. |
| `upstream-pytorch-format-v2` | The same build with the formatted patch. Its `changes.patch` is the first of the four selected patches. No README. |
| [upstream-pytorch-lint-control](upstream-pytorch-lint-control/README.md), [-v2](upstream-pytorch-lint-control-v2/README.md), [-v3](upstream-pytorch-lint-control-v3/README.md), [-v4](upstream-pytorch-lint-control-v4/README.md) | Runs unchanged upstream `spin lint` on pristine and patched source, to separate findings already in the base from findings a patch introduced. Each version repairs a failure of the one before, which stays in place. |
| [upstream-pytorch-pristine-v1](upstream-pytorch-pristine-v1/README.md) | The untouched-base control: PyTorch at the pinned revision with no patch. |
| [upstream-pytorch-paired-v1](upstream-pytorch-paired-v1/README.md) | Builds pristine and patched PyTorch in one hosted job, to attribute the 27 retained failures in the MPS test module. |
| [owned-pytorch-layout-v1](owned-pytorch-layout-v1/README.md) | The embedding patch plus the matrix and convolution dispatch candidate: six files. |
| [owned-pytorch-half-v1](owned-pytorch-half-v1/README.md) | Adds the FP16 `index_put` accumulation candidate: twelve files. |
| [owned-pytorch-half-v2](owned-pytorch-half-v2/README.md) | Adds explicit Half storage: fourteen files. |
| [owned-pytorch-half-v3](owned-pytorch-half-v3/README.md) | The same fourteen-file patch, with compiler flags passed explicitly and checked early. |
| [owned-pytorch-numpy2-v1](owned-pytorch-numpy2-v1/README.md) | The same fourteen-file patch built against NumPy 2.2.6. The recipe of the selected build. |
| [trained-precision-hosted](trained-precision-hosted/README.md) | Builds native precision policy 15 on hosted macOS, forces NEON and replays all 89 historical anchors from the public v6 release. |

Each recipe has a workflow in [.github/workflows/](../.github/workflows) named
after its folder; the hosted replay's is `trained-precision-replay.yml`. Those
workflows start by manual dispatch, or on the development branches they were
written for.

## Unit suites

The helpers are tested with the standard library's `unittest`, in isolated mode,
with no framework build. [Hosted CI](../.github/workflows/validate.yml) runs
these eight on every push:

```sh
for suite in upstream-pytorch-pristine-v1 upstream-pytorch-paired-v1 \
    owned-pytorch-layout-v1 owned-pytorch-half-v1 owned-pytorch-half-v2 \
    owned-pytorch-half-v3 owned-pytorch-numpy2-v1 trained-precision-hosted; do
  python -I -S -B -m unittest discover -s validation/$suite -p 'test_*.py'
done
```

The four lint-control folders carry `test_*.py` files as well; the validation
workflow does not run those.
