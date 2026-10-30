# Prospective official MLX integration matrix, version 1

This is a new validation contract and namespace. It does not replace, erase,
or relabel earlier attempts. The previous full CPU-default run on the
Metal-enabled hybrid build failed and then crashed; those failures remain
separate additional-coverage diagnostics under
`work/upstream-integration-v2/mlx`. Their cause still needs matched unchanged
baseline evidence. Nothing here repairs unrelated CPU code or changes a test.

At exact upstream base `a21f892a6e8e83afb117cef1d1ceae3e82f48c32`, the
official workflow builds CPU jobs with `MLX_BUILD_METAL=OFF`; Metal and JIT
jobs use the natural GPU default. It does not set `DEVICE=cpu` for the
Metal-enabled test suites. This prospective route follows that three-cell
backend/default-device structure at deployment target 14.0:

| Mode | Metal backend | Metal JIT | Test default | Required suites |
| --- | --- | --- | --- | --- |
| cpu-only | OFF | OFF | CPU | Full Python and full C++ |
| jit-on | ON | ON | GPU | Full Python and full C++ |
| jit-off | ON | OFF | GPU | Full Python and full C++ |

The same two authenticated patches are applied to the exact upstream base,
without validation-only preambles or metallibs. All eleven resulting file
hashes must match. Complete pinned upstream pre-commit hooks must pass without
mutating source. One CMake build tree is reused sequentially across all three
modes, with two compiler workers and source identity checks between stages.
The existing strict two-mode hosted workflow may finish under its original
contract; this one uses a separate workflow and output namespace. Both share
the same GitHub concurrency group to prevent overlapping MLX jobs.

Each package has the normal Python extension/shared-library layout. Only the
Metal modes contain a freshly compiled metallib. C++ tests and complete
patched sources are retained. Build-directory hiding tests relocation; all
source, runtime, patch, dependency, and command receipts remain authenticated.

`DEVICE` is removed from the test environment. The Python probe checks both
the compiled-backend flag and the actual natural default before executing on
that device. The upstream C++ test main selects CPU only for `DEVICE=cpu`,
otherwise selects GPU if a GPU backend is available. With `DEVICE` absent,
CPU-only and Metal builds therefore follow their official defaults. Actual
Metal dispatch is still required before either Metal suite can qualify; a
compiled Metal flag alone is insufficient.

**Qualification requires all three mode receipts, successful real-device
probes, complete passing Python inventories, and passing full C++ executions.**
The new GPU regression cases must be present and cannot be skipped. Missing
GPU access leaves both Metal modes unqualified, with binaries retained for
serial M4 replay. Failed attempts never become passes. CPU-on-Metal diagnostic
failures remain visible even if this prospective official matrix passes.

The CPU-only official suite skips `test_init_reduce_large_output` because its
unchanged decorator requires a Metal build. The official Metal suites execute
it on the GPU. Running that same test with CPU default on a Metal build is the
additional combination that crashed earlier; this document does not claim
that its root cause has been proved or repaired.

Copy this bundle into `integration/mlx-official/` and install the workflow
fragment as `.github/workflows/upstream-mlx-official.yml`. It uses the approved
standard `macos-26` runner, CPython 3.12.10, one dedicated environment, a
2 GiB free-space reserve, and a 10 GiB work-tree cap. Publishing remains limited
to the user's project repository. No upstream submission occurs.

This is the backend/device matrix at one deployment target, not a claim to
replicate every upstream CI job: distributed MPI/ring scripts, example
extensions, the external array-API suite, other deployment targets, and other
platforms remain outside this bounded route. Shared builds explicitly export
symbols so the retained C++ tests link the same library as Python; Debug
assertions are enabled and `-g0` bounds storage. These flags are recorded.

The transformer study remains unchanged: zero formal fits, the reserved
admission stream unopened, no baseline substitution, no tolerance change, and
no waiver of exact repeatability or any other admission requirement.
