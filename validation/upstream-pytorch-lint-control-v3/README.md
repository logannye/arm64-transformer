# Pristine and patched upstream lint control v3

This is a diagnostic comparison of **unchanged upstream `spin lint`**, using one hosted macOS arm64 checkout, compiler, Python environment, and staged lint tools. It separates findings already present at the pinned upstream source from findings introduced by either patch. It invokes no framework build step, numerical test, rule exclusion, upstream modification, or study admission; its official CMake preparation may compile configuration probes.

The original hosted run [40976850708](https://github.com/logannye/arm64-transformer/actions/runs/40976850708) failed lint after saving its build/import/archive receipts. Its retained lint log reports 13 narrowing-conversion errors in `torch/headeronly/util/Half.h` and `complex.h`, plus one CLANGFORMAT warning in the modified `Embedding.metal` macro. Those counts describe that failed log. They do not establish a pristine-base result. The original patch, failure, and full-artifact verification requirements remain unchanged.

## Additive repair after two incomplete controls

The first control, [run 40989945817](https://github.com/logannye/arm64-transformer/actions/runs/40989945817), failed during pristine provisioning because the upstream clang-tidy adapter unconditionally changed directory to absent `build/`. V2, [run 40993204538](https://github.com/logannye/arm64-transformer/actions/runs/40993204538), created that empty directory and retained three provisioning logs with 39 pristine, 40 original-patch, and 39 formatted-patch diagnostics. Those logs included generated-header/macro errors absent from the full-build log. V2 then stopped while sealing tool payloads because its inventory rejected a directory symlink. It completed no paired comparison and did not establish equality with the full-build findings. Both helpers, contracts, failures, and existing eligibility bindings remain unchanged.

V3 uses the upstream prerequisite itself: exactly `spin regenerate-clangtidy-files`, once on the pristine checkout before provisioning. The pinned [upstream clang CI script](https://github.com/pytorch/pytorch/blob/71d2a08507ab47dd6b2fa9b66319d17ce2377de4/.github/scripts/lintrunner.sh#L33) runs this command. Its [implementation](https://github.com/pytorch/pytorch/blob/71d2a08507ab47dd6b2fa9b66319d17ce2377de4/tools/linter/clang_tidy/generate_build_files.py) configures CMake, runs torchgen, and generates autograd/functionalization/lazy/Python support files. CMake may compile small compiler-configuration probes. **No framework build step is invoked**, and no compilation database or generated header is fabricated by this control.

The declared environment reproduces the full-build helper's requested Release/MPS/OpenMP settings and exact dependency wheels. The official generator additionally sets `USE_NCCL=0`, `USE_PRECOMPILED_HEADERS=1`, `CC=clang`, and `CXX=clang++`; these upstream overrides are recorded explicitly. The workflow provisions binary-bottle OpenMP as the full-build workflow does. The CMake cache and authoritative end-of-configure `env_forwarded.txt` values, compile commands, host/compiler identity, dependency freeze, and source configuration are retained. The comparison still uses a new hosted runner, so historical image/compiler equivalence is not assumed.

After successful official preparation, the control binds the full `build/` tree and all four output directories outside it: `torch/csrc/autograd/generated`, `torch/csrc/functionalization/generated`, `torch/csrc/lazy/generated`, and `torch/testing/_internal/generated`. These generated input bytes and their inventory receipt must remain identical through every source transition and before/after every lint command. A missing output, changed generated file, failed configure command, or unexpected tracked-source change stops the comparison. The generated-input manifest retains exact identities; the artifact includes that manifest, the CMake cache, effective-forwarded settings, and compile-command JSON.

Tool failures are detected before source-finding parsing. A traceback or general linter failure produces `classification: tool_runtime_failure`, `diagnostics_parsed: false`, explicit log-line markers, and an ineligible sidecar. An empty `findings` list in that case means not parsed, not zero findings. Parser failures are separately retained. No paired comparison is published from either class.

The later diagnostic runtime gate still needs exact agreement between the actual full-build findings and the properly controlled patched variant. This preparatory comparison cannot waive lint, qualify a framework runtime, or open the formal study. It has a new namespace, contract, workflow, schema, and hosted run; earlier failed runs cannot be rebound as passing evidence.

## Fixed sources and environments

The three variants, in fixed serial order, are:

1. Pristine PyTorch `71d2a08507ab47dd6b2fa9b66319d17ce2377de4`.
2. Original patch `3dd6bf6567284d798074e7d6102ff7f8b2a051a3a4ff2824f86184c120eb0ce9`.
3. Formatted patch `3834dca844ea3de6b40467ce214b99bce83fa06dae2bc454623516dafc0fdd71`.

The contract records every changed file's exact bytes for all three variants. Variant changes use checked patch application and reversal on a fresh private hosted checkout; no user checkout is reset. Each lint run records HEAD, all Git refs, `merge-base(HEAD, origin/main)`, tracked diff names, relevant source hashes, and the unchanged upstream lint configuration. Reference and source changes invalidate the comparison.

The workflow uses Python 3.12.10 and `macos-26`, with full upstream Git history so the default merge-base scope can be recorded. It records the actual image, OS, Xcode, SDK, clang path/hash/version, and interpreter hash. The **same 45 dependency wheel names and hashes** from the failed build are downloaded and installed with no dependency resolution or source builds. The three upstream lint binaries must reproduce their original hashes. This is a new hosted comparison; it does not claim that a fresh runner's compiler/image is identical to the previous runner.

All three variants first execute ordinary `spin lint` to provision the upstream tools and their script environments. All provisioning outcomes, including lint failures, are retained. `UV_PYTHON_PREFERENCE=only-managed` makes upstream `uvx --python 3.10` select a fully recorded managed interpreter instead of an unrecorded system installation. After this staging pass, the comparison runs the same three variants serially with `UV_OFFLINE=1`. This controls package retrieval; it does not change the upstream command, selected rules, files, or severities. Before and after each comparison, installed `.venv`, `.lintbin`, managed-Python, and UV executable/package payload inventories must be identical. Resolver timestamp indexes and Python bytecode caches are not executable-source qualification evidence. Directory aliases are expanded and authenticated rather than omitted. The bounded tree walker records each link and its resolved target, hashes every reachable non-bytecode payload, and checks link/target/file/directory identities again before returning. Aliases may resolve only within the explicitly recorded inventory roots, including the hosted Python installation and OpenMP prefix needed by interpreter/library links. Cycles, undeclared escapes, broken links, special nodes, changed payloads, and links to excluded bytecode paths fail with an identified path. Only regular `.pyc` files are excluded; `__pycache__` directories are inspected and must contain only such files. The scanner bounds expanded entries at 500,000, unique non-bytecode file bytes at 16 GiB, and traversal depth at 96. The tool inventory records the approved roots and exclusion policy; no newly discovered external root is silently admitted. Dependency freeze and clang bytes are checked again. Ambient Python/PIP/UV/linter/compiler overrides are removed; only declared environment controls are serialized.

## Exact lint semantics

Every provisioning and comparison command is exactly:

```text
<same-checkout>/.venv/bin/spin lint
```

No `--take`, `--skip`, path restriction, `--all-files` override, autofix, or local framework patch is passed. At this pinned source, bare `spin lint` runs fast linters on all files and slow linters on merge-base changes. The pristine variant therefore has no patch-specific slow-linter paths; the two patched variants have the same four declared changed paths. This difference follows the unchanged upstream command and remains explicit in the evidence.

Runtime/setup failures are distinct from source diagnostics. A failed exit without parsed findings, missing either upstream lint phase, or a detected tool traceback/runtime failure cannot qualify as a valid paired lint comparison. The raw failed command remains visible. The parser retains full diagnostic blocks, source-file hashes, and duplicate findings, including the two diagnostics sharing line 555 in the earlier log.

## Outputs and interpretation

`receipt.json` contains all commands and their raw log hashes, all three provisioning runs, all three comparison runs, environment/source/tool identities, and exact diagnostic-multiset comparisons:

- `shared`: identical full blocks and source hashes seen in pristine and patched sources, retaining counts.
- `base_only`: diagnostics only in pristine source.
- `candidate_only`: diagnostics only in the relevant patched variant.

`complete` and `evidence_valid` describe completed, controlled evidence. `all_lints_passed` is true only if **every comparison lint command exits zero**. Shared errors remain errors, and `lint_failures_waived` is always false. A complete comparison with failures is recorded as `completed_with_lint_failures`, and the CI command exits nonzero. Every original failed hosted result remains failed. A subsequent separate diagnostic runtime-eligibility receipt may cite this classification, but cannot turn failed hosted lint, standard-baseline admission, or formal-study qualification into a pass.

The artifact contains all JSON/logs, CMake cache and effective-forwarded settings, and the exact helper, contract, patch and requirements files. It excludes wheel/cache duplication and contains no framework runtime. The original wheel identities are independently reproducible from the pinned requirements and hashes.

## Running and checking

The workflow is limited to `logannye/arm64-transformer`. It can be dispatched or triggered by changes to its own workflow/validation directory on `codex/upstream-integration-v2`. Its dedicated concurrency group serializes lint-control jobs; it does not block an independent hosted framework build or execute local numerical work.

```sh
# On the declared fresh hosted runner only:
python study/validation/upstream-pytorch-lint-control-v3/run.py \
  --source pytorch-source --output lint-control-v3-result

# Small local synthetic checks; no lint tools/frameworks are executed:
python3 -I -S -B validation/upstream-pytorch-lint-control-v3/test_run.py
ruff check validation/upstream-pytorch-lint-control-v3
```

Preparation includes only synthetic orchestration tests and static checks. No pristine or formatted lint result is claimed until the hosted workflow completes and its receipts are reviewed.
