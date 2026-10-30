# Fresh validation of the v3 framework artifact

`candidate_half_v3_validation.py` selects hosted run `41125186309` at study
revision `11fd196ed3665178f389e9c25256c3561b14fe5f`. This selection identifies
the attempted build; it does not assert that the build or any device test passed.
The earlier candidate adapters and failed build records remain unchanged.

The adapter retains the original transport, source-tree, dependency, native-origin
and complete-method checks, with a new versioned identity for the v3 14-file
contract. It additionally requires:

- The exact v3 build controls and explicit `-g0` in both early and final changed
  translation-unit commands and CMake flags.
- Successful original and candidate Half compile, assembly and execution commands.
- All 65,536 Half encodings, 16 arithmetic partners, and identical retained result
  bits and floating-point exception flags. The complete raw payload is 123,826,140
  bytes per variant.
- Actual hashes of the retained probe binaries, assembly, raw outputs and original
  headers, plus command databases and CMake cache matching the receipt.

Preparation requires an independently checked hosted receipt hash. It verifies
both archives before expanding one source/runtime pair and reuses the exact
existing 45-wheel dependency installation. It performs no compilation, package
installation or framework import. Expansion requires its complete declared size
plus a 16 GiB free-space reserve. Execution uses the authenticated interpreter and
fresh serial processes on the real device.

The 34-method focused sequence retains all 27 original failures, four new FP16
regressions and three original nondeterminism methods. A focused pass cannot
qualify the full 10,357-method suite, the owned joint runtime, transformer
arithmetic or a formal fit. The complete suite uses the explicit
`owned_pytorch_suites.py prepare --suite-contract half-v3` route after fresh
runtime capture with `capture_runtime.py --half-v3-plan ...
--half-v3-plan-sha256 ...`. Its stock conformance-fixture checks and portable
archive audit remain mandatory. These adapters are implemented; no full-suite
execution is asserted. The old 10,353-method owned-suite contract is unchanged.

The 30 standard-library checks exercise stale selection, missing/changed native
evidence, both compile databases, Half coverage and payload corruption, exact
dependency payloads, source/archive traversal and retained failure handling. They
do not import a numerical framework or execute a compiler.

```sh
python3 -I -B -m unittest discover \
  -s integration/gpu-owned -p 'test_candidate_half_v3_validation.py'
```

After the hosted build succeeds and its transport and receipt are verified:

```sh
python3 -I -B integration/gpu-owned/candidate_half_v3_validation.py prepare \
  --workspace /ABSOLUTE/WORKSPACE \
  --artifact /ABSOLUTE/VERIFIED-HOSTED-ARTIFACT \
  --receipt-sha256 EXTERNALLY_VERIFIED_RECEIPT_SHA256 \
  --hosted-head 11fd196ed3665178f389e9c25256c3561b14fe5f \
  --output /ABSOLUTE/WORKSPACE/work/gpu-v4-owned/half-v3-preparation
```

The separate `run` command requires the resulting plan and its recorded hash.
Wait for all other project numerical processes to finish before launching it.
