# Owned PyTorch layout candidate: hosted build

This workflow builds the full framework with the retained deterministic embedding
patch and the new matrix/convolution dispatch candidate. These are three logical
repairs in two input patch files, combined into one exact artifact. The original
formatted build and its failed broad-suite qualification remain unchanged.

The combined patch modifies exactly six files. `contract.json` binds its hash,
all resulting source hashes, the original base, exact 45 dependency wheels and
the retained requirements/constraints/runtime verifier. No upstream contribution
is submitted. A successful hosted run establishes build/lint/import/transport
checks only. Device numerical tests and formal admission remain unexecuted.

The builder is an additive copy of the retained formatted-v2 helper with explicit
candidate metadata, six-file checks, exact dependency pins, source preservation
and compile-flag validation. It does not load a modified historical helper. The
new runtime cannot substitute into an old receipt or become a study baseline
without fresh device checks.

## Execution boundaries

- Only `logannye/arm64-transformer` GitHub Actions, macOS arm64, Python3.12.
- One fresh checkout, one environment, one build tree; exactly two build workers.
- Release, `DEBUG=0`, `REL_WITH_DEB_INFO=0`, C/C++ `-g0`; actual cache and changed
  translation-unit flags checked. No debug-symbol build is requested.
- At least40GiB free before preparation,8GiB reserve during commands,60GiB
  source/build/cache/output cap,8GiB source and runtime payload caps,360minutes.
- Binary-only pinned dependency wheels and binary-only Homebrew OpenMP. The
  actual runner image/toolchain/OpenMP identity is recorded as a new runtime.
- The complete root/submodule tracked source is archived before compilation,
  with member hashes/modes/symlinks, submodule revisions and the patch. Runtime
  bytes are archived before imports/lint. Failures retain available logs,
  receipts, inputs, dependencies and partial artifacts; a failed build cannot
  manufacture a complete runtime.

The workflow triggers only for changes to this directory or its workflow on
`codex/gpu-v4-owned-baselines`, and also supports manual dispatch after GitHub
registers it. Pushing such changes starts the bounded build. It shares the
existing PyTorch validation concurrency group so builds cannot overlap.

## Static checks and dispatch

Run the stdlib-only mutation checks locally without importing PyTorch or compiling:

```sh
python3 -I -B -m unittest discover \
  -s validation/owned-pytorch-layout-v1 -p 'test_hosted_build.py' -v
```

After review and source freeze, push the authorized own-project branch. If the
workflow is registered for manual use, the equivalent explicit command is:

```sh
gh workflow run owned-pytorch-layout-v1.yml \
  --repo logannye/arm64-transformer \
  --ref codex/gpu-v4-owned-baselines -f runner_label=macos-26
```

Do not run both routes for the same revision. The artifact name includes the run
ID and attempt. Retain it before its14-day expiry and verify every manifest/hash
before device testing. The candidate does not repair the separate FP16
combinations-gradient failure; the full broad suite remains a blocking gate.
