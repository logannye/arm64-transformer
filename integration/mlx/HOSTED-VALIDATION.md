# Isolated MLX integration validation

These technical files run only in `logannye/arm64-transformer`. They do not
create an MLX commit, submission, or upstream pull request. They do not alter
the frozen transformer runtime, admit a standard framework baseline, open the
reserved admission stream, or conduct formal study fits.

The inputs are upstream commit `a21f892a6e8e83afb117cef1d1ceae3e82f48c32`
and exactly the two retained embedding/attention patches. Both patch hashes and
all eleven patched file hashes are checked. The upstream offline-kernel CMake
file and preamble generator must match the base exactly. Prior validation-only
build substitutions, prebuilt metallibs, and extracted preambles are excluded.

The workflow runs on the project's standard `macos-26` hosted runner through
explicit dispatch or a path-filtered push to `codex/upstream-integration-v2`.
It requires native arm64, Xcode, and `xcrun metal`, and does not install Xcode or
request a larger paid runner. Actual runner capabilities are checked at runtime.

## What runs

1. Check native arm64 CPython 3.12, Xcode/Metal availability, and storage.
2. Install only wheel dependencies into a dedicated virtual environment. Direct
   dependency versions are prospective integration choices in
   `hosted-requirements.txt`; they are separate from the study environment.
   Preserve the resolver report with distribution hashes and complete freeze.
3. Fetch the exact upstream commit and authenticate/apply both patches. Run the
   complete unchanged upstream pre-commit configuration in an isolated cache:
   check-yaml v6.0.0, clang-format v21.1.8, cmake-format v0.6.13, Ruff v0.16.10,
   and the local stream-argument hook. Hook failure or any source mutation stops
   before compilation. No automatic repair is accepted.
4. Build normal complete JIT ON and then OFF configurations sequentially in one
   reused CMake directory. Both compile the regular offline metallib; OFF also
   compiles the attention instantiation that the earlier runtime-JIT evidence
   could not validate. C++ tests are enabled. Debug assertions remain enabled;
   `MLX_EXPORT_ALL_SYMBOLS=ON` supports linking internal C++ tests against the
   same shared library used by Python; this is an integration build option.
   `-g0` limits debug-symbol storage. Two compiler workers are bounded to one
   native build process.
5. Package the actual extension, `libmlx.dylib`, any built `libjaccl.dylib`,
   freshly compiled metallib, and C++ executable. The normal package layout is
   `runtime/python/mlx/core*.so` plus `runtime/python/mlx/lib/`. The test binary
   uses relative RPATHs for both its CMake test-discovery location and retained
   package location. Retain complete patched sources and replay scripts.
6. Temporarily move the single build directory aside while running the package
   to expose accidental absolute build-path dependencies. Run all Python tests
   serially with CPU and GPU defaults after their real dispatch probes pass.
   Metal-enabled C++ tests contain explicit GPU operations even with a CPU
   default, so both complete C++ invocations require the GPU probe to pass.
   Required new GPU regression cases cannot be missing, failed, or skipped.
7. Preserve both archives and every result, including failures. A missing GPU
   remains `gpu_unavailable_not_validated`; the driver returns nonzero unless
   both configurations pass both CPU and GPU checks. Useful build artifacts
   still upload on failure for subsequent M4 replay.

Storage guards retain a 2 GiB free-space reserve and cap the isolated work tree
at 10 GiB, checked before/after dependency installation and build/package stages.
No unbounded compiler caches are used. The pre-commit cache is isolated inside
that tree. Existing local build trees and evidence remain untouched.

## Install the CI fragment

Copy the helper `.py` files, `hosted-requirements.txt`, and `inputs/` into
`integration/mlx/` in the project repository. Review and install
`hosted-workflow.yml.fragment` as `.github/workflows/upstream-mlx.yml`.
Dispatch that workflow, or push the approved branch and relevant paths.
This is an isolated integration study, not upstream CI approval.

## M4 replay

Verify the downloaded archive SHA256 against the retained publication inventory,
extract into a new directory, and create one isolated CPython 3.12 environment.
Install `runtime-lock.txt` with wheel-only installation. Never replace the study
environment's MLX. Run each retained mode sequentially:

```sh
/path/to/isolated/python jit-on/tools/validate_bundle.py \
  --bundle /absolute/path/jit-on --output /absolute/path/new-jit-on-replay
/path/to/isolated/python jit-off/tools/validate_bundle.py \
  --bundle /absolute/path/jit-off --output /absolute/path/new-jit-off-replay
```

The validator checks every bundled file and required runtime version before
execution, requires the loaded MLX extension to come from the bundle, records
every skipped Python case, and checks that the bundle stayed unchanged. A real
GPU arithmetic dispatch is required before GPU test results can qualify.

## Remaining scope limits

This bounded route covers one macOS deployment target (14.0), the entire normal
Python suite, and C++ suite for both JIT settings. Distributed MPI/ring scripts,
example-extension builds, the separate external array-API suite, other macOS
deployment targets, Linux/CUDA/Windows, and upstream maintainer review remain
outside this workflow. Optional TensorFlow and ml_dtypes comparisons are
included by explicit dependencies; their imports must succeed in the runtime
probe. No native framework source dependencies are installed through pip.

The initial helper bundle is prospective code. Its source-native build,
relocation behavior, dependency resolution, and test execution require the
retained CI run; authoring the scripts is not a validation pass.
