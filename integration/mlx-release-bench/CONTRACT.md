# Prospective matched MLX Release operator comparison

This is build and measurement preparation, not an upstream PR description. No new build or numerical benchmark has executed under this contract. Historical attempts and the standard-release study gate remain unchanged.

## Why another comparison is needed

The retained MLX CONTRIBUTING.md at base `a21f892a6e8e83afb117cef1d1ceae3e82f48c32`, lines 63–64, requires benchmarks for main and the pull request when performance may change, with scripts and reproduction steps. The deterministic embedding strategy and attention dispatch repair can both change cost.

Existing 26-case embedding and 112-case attention measurements compare an exact-base upstream JIT-on runtime with the earlier locally built hybrid patch runtime. The embedding launcher requested compilation without proving that compilation was enabled; the attention benchmark explicitly enabled compilation. Those outcomes remain useful within their recorded scopes. Neither is a matched Release comparison of the newly integrated full-source runtime.

An unpatched exact-base JIT-on wheel is retained from upstream run 40864937663, artifact 12893988316. Its native package records Debug configuration. The newer validated patched JIT-on/off bundles also use Debug, specifically `-O0 -g0`. Do not compare those artifacts with newly optimized Release binaries or relabel their test durations as performance observations.

## Fixed build inventory

The helper requires `GITHUB_ACTIONS=true`, `GITHUB_REPOSITORY=logannye/arm64-transformer`, an ARM64 macOS runner, and a fresh work root inside `RUNNER_TEMP` before any mutation. Run/attempt IDs, workflow source revision, runner and available image identities are retained. One isolated hosted job, one interpreter environment, one source checkout and one reusable CMake build directory produce these four cells in order:

1. Unmodified frozen main, shader JIT on.
2. Unmodified frozen main, shader JIT off.
3. Frozen main plus the two recorded runtime patches, shader JIT on.
4. The same patched source, shader JIT off.

Every cell uses the same macOS arm64 host/toolchain, CPython 3.12.10, full recorded dependency lock, deployment target 14.0, shared-library layout, and Release `-O3 -DNDEBUG -g0` for C/C++. Native compilation uses two jobs. After configuration and again after compilation, the actual CMake cache must match the declared Release flags, architecture, backends, shader-JIT mode, layout and disabled cache/toolchain overrides; a requested flag alone is not sufficient evidence. CUDA, examples, CMake benchmark targets, stubs and compiler caches are disabled; CPU, Metal, Python bindings and C++ test executable are included. Both Metal variants build all required offline shaders using the real Xcode compiler and generate their own JIT preambles. No retained native object, metallib, or preamble is substituted.

The helper verifies the exact Git base and clean main tree. It then applies the two authenticated patches once and permits exactly their eleven changed/new paths. Build configuration files formerly changed for local validation are required to match upstream bytes. Source inventories are compared before and after every build. The CMake header component is copied from each exact source variant; base and patch header inventories can differ only through declared patch files.

Each packaged runtime, source, manifest, provenance, dependency lock, inputs and public headers is archived immediately before import or execution probes. Archive roundtrip checks stream every payload hash, byte count and executable bit. The original build directory is hidden during an isolated normal import, which authenticates the core extension and native libraries. A failed transport probe leaves its archive and failed receipt intact. A successful hosted job means build and transport only, never numerical or performance qualification.

All four planned rows exist before the first build; failure and unstarted rows remain visible. Check storage before builds and after compilation; the work-root cap is 10 GiB with at least 2 GiB free. Installation uses binary wheels only. No local compilation or frozen study-environment change is authorized by this helper. Install this directory only as the project's `integration/mlx-release-bench/`, with `workflow.yml` as `.github/workflows/upstream-mlx-release-bench.yml`, after root review.

## Separate prospective operator runner

The runner must bind all four runtime manifests, the complete case inventory, fixtures, its own sources and the unchanged numerical thresholds before scoring. Root owns execution authorization and the serial device lease. This section specifies the required adaptation; the existing historical runners must remain unchanged.

Reuse the 26 embedding shapes/patterns and two operations per case: scatter-add and embedding value/gradient. Twenty-four cases request compilation and two are eager. Reuse the 112 attention cases: lengths 1, 3, 8, 9, 31, 32, 64; head dimensions 64, 128, 192, 256; inference/training; eager/compiled; B2, H4, causal masking. This yields 138 cases per runtime. Two fresh-process repeats with paired ABBA order produce 1,104 planned case-workers across the four runtimes, before any explicitly declared additional control runs.

Use the same historical fixture seeds and bytes (embedding 817, attention 314159), 20 warmups and 100 synchronized observations per operation; no study fixtures, corpus partitions, or reserved seed 20261104. Keep cold compilation, imports and fixture generation separate from steady-state latency. Pair main and patch cases on the same device and shader mode in fixed ABBA order (main-repeat0, patched-repeat0, patched-repeat1, main-repeat1). Preserve p50/p95 and all samples; do not pool shapes into a headline speedup.

Before importing frameworks, remove `MLX_DISABLE_COMPILE` and `PYTORCH_MPS_PREFER_METAL`, clear library/path/device injection controls, require the natural GPU default, retain TF32/fallback/fast-math zero settings, and fix host numerical threading. Bind actual loaded native images to the runtime manifest. Prove generic graph compilation with the retained four-changing-input, one-trace `Compiled`-primitive probe; additionally count traces on each requested-compiled operator wrapper. Eager cases remain explicitly eager. Pure `mx.value_and_grad` on explicit arrays does not require the model-state capture workaround; any `nn.value_and_grad(model, ...)` wrapper must capture the model state in both directions.

Run independent FP64 accuracy and exact replay checks separately. Preserve the attention harness's existing `2e-5/2e-5` comparison and first/last byte replay controls. The embedding script originally only recorded FP64 error values; a prospective explicit bound must be declared in the new runner before measurements. Do not describe the old embedding timings as having passed a numerical admission gate they did not implement. Use the unchanged study gradient tolerance `3e-5/3e-3` for new embedding accuracy checks, with the same independent deterministic FP64 sum.

Known main failures are evidence: order-dependent repeated-token gradients and unsupported attention VJP cases must remain marked. Compute a main-versus-patch ratio only when both sides pass runtime provenance, accuracy, byte-exact within-process replay and byte-exact cross-process repeat equality. Preserve raw samples and descriptive costs for nondeterministic or failed cases, clearly unqualified and excluded from ratios; never turn a repeatability failure into a study pass. These isolated upstream implementation-cost diagnostics are separate from the five-engine transformer campaign and do not waive its strict exact-training requirement. Retained Debug integration results do not automatically qualify the new Release binaries; run appropriate newly bound device regressions before publishing Release results.

## Completion boundary

The build prerequisite is four authenticated normal full-source Release runtimes with passing transport checks, or complete retained failures. The submission performance prerequisite is a separately frozen, executed and audited matched operator comparison with its scripts, samples and limitations. Human upstream submission requirements, supported standard releases, fresh baseline qualification, full formal admission and the 180-fit study remain separate and unexecuted under this contract.
