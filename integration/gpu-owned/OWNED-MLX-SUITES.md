# Fresh MLX full suites in the owned joint runtime

`owned_mlx_suites.py` prepares and executes the complete retained MLX C++ and
Python suites using the same authenticated CPython, NumPy 1.26.4 overlay, selected
PyTorch distribution, compiled MLX build and dependencies intended for the owned
study. The earlier MLX suite results used NumPy 2.2.6. Those results remain valid
historical evidence, but cannot qualify this changed dependency environment.

The frozen inventory requires 1,053 Python test cases, the same seven skip
identities and reasons, 17,701 passing subtests, all 27 added GPU regressions, and
305 C++ cases with 4,792 assertions. All 73 upstream test source files are bound.
There are no test filters, new skips, replaced operators or source edits. Both
suites retain failures, raw output and a complete execution history.

Preparation calls `authenticate_runtime` before importing any framework. The
runtime receipt must bind the producer and every project bootstrap module. The
worker starts with `-P -S -B` and `PYTHONHASHSEED=0`. Authentication imports only
the bound project code and original isolated standard library; unverified
dependency directories cannot shadow those imports. After authentication, the
worker installs the supplemental import roots in their declared order and checks
actual package resolution against the runtime inventory. A fresh nonexistent
bytecode prefix is set before project or
framework imports; bytecode writes remain disabled. It does not execute `.pth`
files or editable site hooks. `-I` is excluded because it ignores the fixed hash
seed. The inventory must therefore be captured using the same isolated
interpreter layout.
For CPython 3.12, a site-enabled virtual-environment capture is a different layout
and is rejected rather than silently reused.

Capture must include the whole selected MLX bundle, in addition to its
`runtime/python` import root: the unchanged Python test modules, C++ executable
and native libraries must all appear in the complete runtime inventory. The
project source map is separately authenticated. The producer verifies NumPy's
version and origin, the natural GPU default, actual compiled graph replay,
loaded framework origins and imported Python dependency files. It repeats the
runtime, source, compilation and origin checks after each suite.

Each suite runs in a fresh process, serially. Python uses the ordinary full
`pytest.main` invocation and records JUnit plus its raw log. C++ executes the
existing authenticated test binary without filters or rebuilding it. Its native
proof consists of the pinned executable, selected library bytes and saved
`otool` linkage before and after execution. C++ linkage evidence is not an
in-process loaded-image trace; the supervising Python worker separately verifies
its own natural GPU default, compilation and loaded MLX images. OS shared-cache
libraries remain identified by the authenticated OS build, as in the runtime
contract.

The unchanged Python fresh-process test invokes `sys.executable` twice. During
pytest only, the producer routes that variable through a byte-bound interpreter
launcher and restores it before postflight authentication. The launcher executes
the canonical interpreter with the same `-P -S -B` controls and a bound child
bootstrap. It does not set `PYTHONPATH`, modify test source or replace subprocess
functions. Only `test_gather_grad.py --repeatability-worker` is admitted. Its
stdout, argv, numerical code and exit status are preserved through `runpy`.

Each child has a separate receipt with its actual PID and parent PID, source and
runtime identities, compiled GPU evidence, native origins and imported-file
proof before and after execution. Both receipts and the launcher are re-audited
before a parent pass. OS reuse of a PID does not invalidate distinct sequential
child launches. The inventory of all 73 test sources contains exactly two direct
subprocess call shapes: the two-invocation fresh-process test and the unchanged
`sysctl` CPU-name query. The source-bound extension tests mock their build
commands; the conformance launcher is not a measured baseline launcher.

The time limits are 1,800 seconds for C++ and 3,600 seconds for Python, plus 600
seconds of wrapper allowance. Output is bounded to 2 GiB with a 5 GiB disk
reserve. Each worker receives a private temporary directory. Nested C++ execution
remains in the worker's process group, so terminating a timed-out worker also
terminates its C++ child. Failed and timed-out attempts retain available evidence.

Before reporting completion, the parent rehashes both receipts and every retained
raw log, XML and linkage file, then independently parses the complete suite
outcomes. A worker's `passed` flag cannot override missing evidence, failed
postflight checks, an altered inventory or a failing raw result. These suite
receipts remain development eligibility evidence; they never open admission or
start a formal fit.

Run the CPU-only regression tests with the retained CPython 3.12 interpreter:

```sh
/ABSOLUTE/SELECTED/PYTHON -I -S -B -m unittest discover \
  -s integration/gpu-owned -p 'test_owned_mlx_suites.py' -v
```

The local Homebrew CPython 3.14 installation failed XML tests because its Expat
extension referenced a missing system-library symbol. No dependency was changed;
the retained CPython 3.12 interpreter passed all static tests.

After the runtime and source are frozen, preparation requires explicit authority
inputs and an external SHA256 for the runtime receipt:

```sh
env -i PATH=/usr/bin:/bin HOME="$HOME" \
  MLX_ENABLE_TF32=0 MLX_SDPA_VJP_FALLBACK=0 \
  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  METAL_DEBUG_ERROR_MODE=0 METAL_DEVICE_WRAPPER_TYPE=1 \
  TF_CPP_MIN_LOG_LEVEL=2 SKBUILD_EDITABLE_REBUILD=false \
  /ABSOLUTE/SELECTED/PYTHON -P -S -B \
  integration/gpu-owned/owned_mlx_suites.py prepare \
  --runtime-root /ABSOLUTE/RUNTIME/ROOT \
  --runtime-inventory /ABSOLUTE/RUNTIME-INVENTORY.json \
  --selection /ABSOLUTE/FRAMEWORK-SELECTION.json \
  --receipt /ABSOLUTE/RUNTIME-RECEIPT.json \
  --receipt-sha256 EXTERNALLY_RECORDED_RECEIPT_SHA256 \
  --output /ABSOLUTE/WORKSPACE/work/gpu-v4-owned/mlx-suite-preparation-v1
```

Use the same sanitized interpreter environment for `prepare` and `run`. The
producer checks the full declared control map before doing work; it does not
silently inherit unrecognized Python controls.

A separate `run --plan ... --plan-sha256 ... --output ...` command launches device
work. Its output path must be new. Execution must wait for the project's device
lease and for all other numerical project work to stop. This implementation phase
has run no GPU suite, framework installation, native build or formal experiment.
