# Pristine and patched upstream lint control v2

This is a diagnostic comparison of **unchanged upstream `spin lint`**, using one hosted macOS arm64 checkout, compiler, Python environment, and staged lint tools. It separates findings already present at the pinned upstream source from findings introduced by either patch. It performs no framework build, numerical test, rule exclusion, upstream modification, or study admission.

The original hosted run [40976850708](https://github.com/logannye/arm64-transformer/actions/runs/40976850708) failed lint after saving its build/import/archive receipts. Its retained lint log reports 13 narrowing-conversion errors in `torch/headeronly/util/Half.h` and `complex.h`, plus one CLANGFORMAT warning in the modified `Embedding.metal` macro. Those counts describe that failed log. They do not establish a pristine-base result. The original patch, failure, and full-artifact verification requirements remain unchanged.

## Additive repair after the failed control

The first control, [run 40989945817](https://github.com/logannye/arm64-transformer/actions/runs/40989945817), failed during pristine-base provisioning. The upstream adapter raised `FileNotFoundError` when changing directory to the absent `source/build`. The v1 wrapper then attempted to parse the general tool failure as a source diagnostic and raised `Finding lacks a file identity`. That run produced no qualified paired comparison. Its helper, contract, logs, receipts, and exact-run development-eligibility bindings remain unchanged.

V2 creates exactly one empty, non-symlink `build/` directory before the first source check. It authenticates the upstream configuration and adapter bytes first, records the operation as filesystem setup, and verifies the directory stays empty with the same device/inode/mode before and after every lint execution and variant transition. It creates no compilation database, generated headers, framework build, or compiler result. Unexpected contents or a replaced directory fail the control; nothing is silently removed.

This narrowly satisfies an observed upstream working-directory prerequisite: the pinned [lint configuration](https://github.com/pytorch/pytorch/blob/71d2a08507ab47dd6b2fa9b66319d17ce2377de4/.lintrunner.toml#L355) passes `--build_dir=./build`, and the [adapter](https://github.com/pytorch/pytorch/blob/71d2a08507ab47dd6b2fa9b66319d17ce2377de4/tools/linter/adapters/clangtidy_linter.py#L197) calls `os.chdir(build_dir)` unconditionally. Its header-only compatibility invocation explicitly supplies C++17 flags; the adapter also documents header checks without compilation-database entries. This supports the directory setup, not an assumption that every subsequent check will pass.

The upstream [clang CI entry point](https://github.com/pytorch/pytorch/blob/71d2a08507ab47dd6b2fa9b66319d17ce2377de4/.github/scripts/lintrunner.sh#L33) separately runs `spin regenerate-clangtidy-files`, which configures CMake and generates headers. This control reproduces unchanged **bare `spin lint`** across three source variants, not that additional CI setup or a fully built tree. Missing generated inputs or any further tool failures remain blocking. Any later diagnostic-runtime eligibility gate must separately match the actual full-build lint findings and identities; this control alone cannot qualify a runtime or waive lint.

Tool-failure detection now runs before source-finding parsing. A traceback or general linter failure produces `classification: tool_runtime_failure`, `diagnostics_parsed: false`, explicit log-line markers, and an ineligible sidecar. Its empty `findings` list means **not parsed**, not zero source findings. A parser failure is separately retained as `diagnostic_parse_failure`. The raw command/log and nonzero exit remain authoritative. No paired comparison is published from either failure class.

V2 has its own namespace, contract, workflow, schema, and future hosted run. V1's failed run must not be rebound or accepted by a later gate as if it had completed.

## Fixed sources and environments

The three variants, in fixed serial order, are:

1. Pristine PyTorch `71d2a08507ab47dd6b2fa9b66319d17ce2377de4`.
2. Original patch `3dd6bf6567284d798074e7d6102ff7f8b2a051a3a4ff2824f86184c120eb0ce9`.
3. Formatted patch `3834dca844ea3de6b40467ce214b99bce83fa06dae2bc454623516dafc0fdd71`.

The contract records every changed file's exact bytes for all three variants. Variant changes use checked patch application and reversal on a fresh private hosted checkout; no user checkout is reset. Each lint run records HEAD, all Git refs, `merge-base(HEAD, origin/main)`, tracked diff names, relevant source hashes, and the unchanged upstream lint configuration. Reference and source changes invalidate the comparison.

The workflow uses Python 3.12.10 and `macos-26`, with full upstream Git history so the default merge-base scope can be recorded. It records the actual image, OS, Xcode, SDK, clang path/hash/version, and interpreter hash. The **same 45 dependency wheel names and hashes** from the failed build are downloaded and installed with no dependency resolution or source builds. The three upstream lint binaries must reproduce their original hashes. This is a new hosted comparison; it does not claim that a fresh runner's compiler/image is identical to the previous runner.

All three variants first execute ordinary `spin lint` to provision the upstream tools and their script environments. All provisioning outcomes, including lint failures, are retained. `UV_PYTHON_PREFERENCE=only-managed` makes upstream `uvx --python 3.10` select a fully recorded managed interpreter instead of an unrecorded system installation. After this staging pass, the comparison runs the same three variants serially with `UV_OFFLINE=1`. This controls package retrieval; it does not change the upstream command, selected rules, files, or severities. Before and after each comparison, installed `.venv`, `.lintbin`, managed-Python, and UV executable/package payload inventories must be identical. Resolver timestamp indexes and Python bytecode caches are not executable-source qualification evidence. Directory symlinks that could hide unhashed payloads are rejected. Dependency freeze and clang bytes are checked again. Ambient Python/PIP/UV/linter/compiler overrides are removed; only declared environment controls are serialized.

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

The artifact contains all JSON/logs and the exact helper, contract, patch and requirements files. It excludes wheel/cache duplication and contains no framework runtime. The original wheel identities are independently reproducible from the pinned requirements and hashes.

## Running and checking

The workflow is limited to `logannye/arm64-transformer`. It can be dispatched or triggered by changes to its own workflow/validation directory on `codex/upstream-integration-v2`. Its dedicated concurrency group serializes lint-control jobs; it does not block an independent hosted framework build or execute local numerical work.

```sh
# On the declared fresh hosted runner only:
python study/validation/upstream-pytorch-lint-control-v2/run.py \
  --source pytorch-source --output lint-control-v2-result

# Small local synthetic checks; no lint tools/frameworks are executed:
python3 -I -S -B validation/upstream-pytorch-lint-control-v2/test_run.py
ruff check validation/upstream-pytorch-lint-control-v2
```

Preparation includes only synthetic orchestration tests and static checks. No pristine or formatted lint result is claimed until the hosted workflow completes and its receipts are reviewed.
