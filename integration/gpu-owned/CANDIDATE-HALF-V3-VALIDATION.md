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
installation or framework import. Expansion requires its complete declared size,
an 8 KiB per-member metadata allowance and a 16 GiB free-space reserve by default.
The explicit `--expansion-reserve-gib 10` option records a bounded development
budget with 10 GiB reserved after expansion. Both options check the reserve during
and after expansion and retain it in the authenticated plan. This accommodates
the verified 2.84 GiB source/runtime pair while leaving room for the separately
bounded 2 GiB full-suite output, 1 GiB fixture cache and 5 GiB operating floor.
No runtime, numerical or suite gate changes. Execution uses the authenticated interpreter and
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

The 42 standard-library checks exercise stale selection, missing/changed native
evidence, both compile databases, Half coverage and payload corruption, exact
dependency payloads, source/archive traversal and retained failure handling. They
do not import a numerical framework or execute a compiler.

The verified v3 runtime archive includes empty include directories absent from
its file manifest. The first preparation rejected them before expansion. The
adapter now accepts authenticated empty runtime directory records and ignores
them during expansion. It still rejects paths outside the runtime, source
directory records, file/directory collisions, links and every missing or changed
payload file. The original failed preparation remains retained; no runtime bytes
or numerical qualification changed.

After expansion and runtime capture, the first full-suite preparation rejected
the source archive's explicit symlink records. The source verifier and portable
auditor now normalize that format alongside the earlier digest-based manifest.
They hash each literal link target, reject lexical escapes and substituted files,
and require live links to resolve within the authenticated source tree. Regular
file content and modes remain checked. All 146,435 members of the actual source
tree verify; 86 producer/archive tests pass without a framework import or build.
The original failed preparation is retained. Fresh capture must bind the corrected
verifier; this format correction provides no numerical qualification.

Two subsequent preparations stopped when initial file access changed source
metadata. A retained diagnostic caught one zero-byte placeholder changing flags
and timestamps on its first read. File binding now warms the opened descriptor
before taking its initial fingerprint, hashes that same unbuffered descriptor,
then requires its full fingerprint and the final path's fingerprint to match.
Content, timestamp, inode, mode or path changes during hashing still fail, even
when the resulting digest would match. This avoids mistaking initial file
materialization for an in-flight mutation; no numerical or source-content check
is weakened. The failed attempts remain retained.
An empty descriptor is checked for stable zero length and assigned SHA256 of
the empty byte sequence without a content read. This avoids waiting for cloud
hydration of a file with no bytes; nonempty files retain the complete read/hash.

The candidate artifact verifier uses the same descriptor-stability approach.
Its explicit `rebind-plan --plan OLD_PLAN --plan-sha256 OLD_SHA --output NEW_DIR`
route reuses an expanded pair under a new verifier snapshot. It retains the old
plan and driver as bound inputs and changes no runtime, source tree, test
inventory or execution controls. The entire artifact, dependency installation,
source/runtime trees and plan must authenticate again before `plan.json` is
published. A failed attempt retains only its pending plan. No previous device
qualification transfers, and neither source tree is duplicated or rebuilt.

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
