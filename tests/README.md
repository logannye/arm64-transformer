# Tests

The pytest suite for the Python package, the native library, the study control
code and the evidence tools. Most of it needs no native build, GPU or corpus:
the control code is tested with small synthetic inputs and by fault injection,
where a test alters saved evidence and requires the check to reject it.

## Running

Build the native code first if the native tests should run. The tests never
invoke a compiler themselves.

```sh
make -C native widths PYTHON=python
ARM64_TRANSFORMER_WIDTH=64 python -m pytest -q
```

The width is fixed per process, so the suite runs once per width: 32 when the
variable is unset, or any of 64, 128, 256, 512 and 1024. One width takes about a
minute on an M4 Max. [Hosted CI](../.github/workflows/validate.yml) runs all six
widths twice, on Linux without a native build and on macOS with one.

The studies ran on one numerical thread, and CI pins the same setting:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

Lint is separate: `python -m ruff check .` and
`python -m ruff format --check .`.

## What is covered

| Files | Count | What they test |
|---|---:|---|
| `test_native.py`, `test_native_precision.py`, `test_backend.py` | 3 | The native library through its `ctypes` binding: its numerical contract against independent references, the FP64 precision routines, and the backend identity checks. |
| `test_reference.py`, `test_report.py`, `test_sweep*.py`, `test_runtime_replay.py`, `test_validation_contract.py` | 6 | The PyTorch reference model, the two-engine study runners and reports, and the validation gates. |
| `test_precision*.py`, `test_trained_precision*.py`, `test_compensated_*.py`, `test_framework_*.py`, `test_relu_localize.py`, `test_fresh_precision_inventory.py` | 13 | The precision work: FP64 oracles, the v6 audit, and the trained-state replay, candidates and auditors in [integration/trained-precision](../integration/trained-precision/README.md). |
| `test_gpu_*.py` | 33 | The GPU engines' adapters and policies, the Metal and QKV development tools, the five-engine study control plane, and the provenance checks for the project-maintained frameworks. |
| `test_library_*.py` | 31 | The driver of the accepted five-engine phase in [integration/library-study-v1](../integration/library-study-v1/README.md), the precision API in [integration/precision-library-v1](../integration/precision-library-v1/README.md), and the library models' common-state, optimizer and repeatability checks. |
| `test_package_*.py`, `test_publish_sweep_view.py`, `test_failed_sweep_report.py`, `test_profile_native.py`, `test_graph_evidence.py`, `test_zip_evidence.py` | 8 | The [scripts](../scripts/README.md) and storage tools that package, report and archive evidence. |
| `test_mlx_*.py`, `test_pytorch_*.py`, `test_upstream_pytorch_runtime_origins.py` | 5 | The hosted framework-build helpers under [integration/](../integration/README.md) and [validation/](../validation/README.md). |
| `test_demo.py`, `test_native_build_identity.py` | 2 | The side-by-side [demo](../transformer/demo.py), including one real run of both engines on a tiny schedule, and the bookkeeping of the [build-identity check](../scripts/check_native_build_identity.py). |

`contract_fixtures.py` holds small saved-evidence fixtures shared by the
contract tests. There is no `conftest.py`.

Some folders under `validation/` and `integration/gpu-owned/` carry their own
standard-library `unittest` suites, which pytest does not collect. Their
commands are in the [validation](../validation/README.md) and
[integration](../integration/README.md) indexes.

## Skips

A skip always means a precondition is absent, never a known failure.

| Skipped | Condition |
|---|---|
| `test_native_numerical_contract` in `test_native.py` | Not macOS on arm64, or no built library for the selected width. |
| All of `test_native_precision.py` | Not macOS, or no built library for the selected width. |
| The real run in `test_demo.py` | Not macOS on arm64, or no built executable for the selected width. |
| Two tests in `test_native_precision.py` | The compiled precision policy lacks that test's bit: 4 for the second LayerNorm, 8 for the first feed-forward projection. The policy is 0 in a default build and at every width below 512, so both skip. |
| `test_gpu_frameworks.py` and `test_gpu_training_shape.py` | `ARM64_TRANSFORMER_GPU_TEST_BACKEND` is unset. |
| Two tests in `test_gpu_metal_contract.py` | `ARM64_TRANSFORMER_GPU_TEST_BACKEND` is not `metal`. |

The device tests are opt-in: they need the GPU and the engine's framework, and
are meant to run one engine and width per fresh process. Set the variable to
`mps`, `mlx` or `metal` and name the test files as well, because
`test_gpu_frameworks.py` accepts only `mps` and `mlx` and fails on any other
value. With a default native build and no device opt-in, 17 tests skip at
width 32.
