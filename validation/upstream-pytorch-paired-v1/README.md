# Same-host pristine and patched diagnostic control

This prospective control builds both variants at upstream revision `71d2a08507ab47dd6b2fa9b66319d17ce2377de4` in one hosted job. It is intended to attribute the 27 retained failures in the earlier complete MPS test module. It is not a replacement for the selected model runtime, a standard released baseline, a speed comparison, or formal-study admission. No outcome is asserted by this preparation.

The earlier pristine-only attempt stopped before any source/build command because GitHub's runner image changed from `20261013.0424.1` to `20261021.0445.8`. Its failed receipt remains unchanged. This new contract compares toolchain and environment identities **within a pair**; differences from the historical host are retained separately in `historical_drift`. This is a newly declared control, not a waiver of the earlier exact-image requirement.

The producer uses one upstream checkout, one `.venv`, one `build` tree, two native build workers, and one retained 45-wheel dependency set:

1. Initialize the exact 61 recursive submodules, download only the pinned binary wheels, verify every wheel hash and the complete installed dependency freeze, and record host/Xcode/compiler/SDK/OpenMP identities.
2. Build untouched pristine source using the official `pip install -e . -v --no-build-isolation` command. Retain its runtime archive/manifest and compact test-source archive before any import or lint failure can discard them. Verify normal editable import, transported normal import, native library portability, and an archive round trip. Retain the complete test module, Metal/conformance fixture closure, LICENSE and NOTICE separately.
3. Apply the exact reviewed formatted patch. In the same environment and build directory, run the same official editable-build command incrementally. Retain and verify an independent copied runtime and source archive. The pristine copies remain immutable.
4. Require equal compiler/SDK/OpenMP/host/build identities before and after both builds, equal effective recorded CMake settings and embedding compile-command/object paths, changed compiled embedding `.mm` and `.air` object hashes, and the unchanged 45-wheel environment. Recheck both retained archives before marking the diagnostic pair complete.

The unchanged upstream `spin lint` runs for each variant without exclusions. Its failures are retained; the overall hosted job fails if either lint invocation fails. A separately labeled `diagnostic_control_ready` means only that both native/source/import/portability controls completed and matched. It never changes the original suite failure, grants upstream acceptance or admits a standard framework baseline.

Python starts with `-B`; bytecode emission must be disabled before helper inventory capture. This avoids the unretained generated helper bytecode in the earlier failed attempt. Toolchain version identity excludes only `InstalledDir` display lines; full raw tool logs remain available. Nine canonical test/helper AST identities include empty fields, decorators and tagged constants, and were cross-checked under Python3.12/3.14. Only the known formatted test module can differ from pristine Git bytes in its compact source closure.

Artifacts use `integration-paired-v1-result/`: `pair.json`, `setup.json`, setup logs and a shared `wheels/`; separate `pristine/` and `formatted/` directories contain their receipts, logs, CMake cache, compile database, normal-import receipts, `pytorch-integrated-runtime.tar.gz`, and variant-specific test-source archives. Failed and partial archives are uploaded. No second expanded source checkout is required locally.

After build authentication, a separately reviewed device replay must run the exact failed test IDs under each normal framework with the same local hardware, dependency payloads and runtime controls. A matching pristine failure could localize a problem outside the embedding patch; it would not clear either original full-suite result. No kernel selection is forced and no tolerance or numerical requirement is changed.

Framework-free preparation checks:

```sh
python3 -I -S -B -m unittest discover \
  -s validation/upstream-pytorch-paired-v1 -p 'test_*.py'
```
