# Capture the maintained runtime before validation

`capture_runtime.py` records the committed project, canonical CPython 3.12
installation, selected complete-source frameworks, native libraries and shared
dependencies. It imports no tensor framework, executes no numerical operation,
and opens neither admission nor formal modeling data. Its result is identity
evidence, never permission to run the campaign.

Use a clean committed tree and record its full commit ID independently. Supply
canonical absolute paths: the PyTorch runtime, the authenticated NumPy 1.26.4
overlay, the complete MLX bundle and the shared dependency directory. The
interpreter must be the resolved base executable, not a virtual-environment
symlink. No installation, environment synchronization or compilation is needed.

```sh
env -i PATH="$PATH" HOME="$HOME" TMPDIR="${TMPDIR:-/tmp}" \
  PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
  "$OWNED_PYTHON" -P -S -B \
  "$OWNED_REPO/integration/gpu-owned/capture_runtime.py" \
  --project "$OWNED_REPO" --expected-head "$OWNED_COMMIT" \
  --torch-runtime "$OWNED_TORCH_RUNTIME" \
  --numpy-overlay "$OWNED_NUMPY_OVERLAY" \
  --mlx-bundle "$OWNED_MLX_BUNDLE" \
  --dependencies "$OWNED_DEPENDENCIES" \
  --output "$OWNED_WORK/runtime-capture-v1"
```

The output directory must be new and outside every input tree. The helper sets
a fresh nonexistent bytecode prefix before importing project code. `-B` alone
would still read old bytecode; `-I` would ignore the requested hash seed. The
owned launch policy therefore uses `-P -S -B`, explicit authenticated import
roots and an actual zero hash-randomization flag. Undeclared Python and dynamic
loader environment overrides are rejected.

Source validation compares the working bytes and executable modes with both the
Git commit and index, including files hidden by `assume-unchanged` or
`skip-worktree`. Runtime discovery includes the whole MLX bundle, its native
libraries and tests, the complete interpreter and resolved dependency roots.
Actual package resolution must match the declared paths. The output records
`runtime-inventory.json`, `owned-runtime.json`, `framework-selection.json` and
`capture.json`. Record the byte SHA-256 of `owned-runtime.json` externally before
passing it to subsequent commands.

Subsequent workers must recheck these files and actual loaded Python/native
origins before and after execution. A source, dependency, patch or native-binary
change requires fresh identity capture and evidence for the new identity.
System dynamic-loader shared-cache bytes are represented by the OS build; this
capture does not claim to copy or individually hash those system images.

CPU-only regression tests cover changed source and executable modes, stale
bytecode, package shadowing, unsafe output paths and undeclared import controls.
Real framework imports and numerical suites are separate checks described in
[the owned command guide](OWNED-COMMANDS.md).
