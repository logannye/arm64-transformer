# Trained-state precision development status

This is development evidence on an Apple M4 Max, not a completed five-engine
performance study. The historical CPU and GPU contracts, their failures and
reserved streams remain unchanged. The formal GPU campaign still has 0/180 fits.

The regression inventory contains 89 exposed sequence anchors: two at width 512
and 87 at width 1024. At batch size 16 these represent 73 distinct checkpoint/input
batches (2 + 71). Every comparison loads identical saved FP32 weights and inputs.
The new checks retain logits, mean loss and all parameter gradients, compare two
independent FP64 equations, and require exact repeated execution within an engine.
They retain the original numerical tolerances.

## Completed, independently recomputed results

| Development implementation | Historical B16 logits/loss/all gradients | Fresh development B16 checks |
|---|---|---|
| Native policy 15, both normalizations and QKV/FF1 FP64 arithmetic, M4/SME | 89/89 anchors pass | 20/20 batches pass |
| Native first-LayerNorm FP64 arithmetic, policy 1 | 89/89 anchors pass | Width 512: 9/10 pass; width 1024: 10/10 pass |
| Python FP32-saved first-LayerNorm/QKV/attention refinement, v2 | 89/89 anchors pass | 20/20 batches pass |
| Handwritten reference Metal normalization refinement, policy 1 | 89/89 anchors pass | 20/20 batches pass |
| Custom MPS first-LayerNorm/QKV/attention refinement, policy 7 | 89/89 anchors pass | 20/20 batches pass |
| Custom compiled MLX first-LayerNorm/QKV/attention refinement, policy 7 | 88/89 anchors pass; one FF1 gradient failure | 20/20 batches pass |
| Custom compiled MLX policy 15, additionally refining FF1 | 89/89 anchors pass, with separate inference and gradient audits | 20/20 batches pass in both modes |

The Python refinement is an explicit custom model using PyTorch operations, not
ordinary PyTorch and not yet a selected maintained-framework arm. Its CPU evidence
uses installed PyTorch 2.14.1. The Metal refinement applies to the reference shader
path; optimized paths cannot inherit its results. Native policy 1 also passes all
89 B16 forward anchors under forced NEON with verified dispatch counters.

The independent follow-up audit is retained locally under
`work/trained-followup-audit-v1`, with separate hash-bound historical and fresh
manifests. Earlier native and factorial audits are retained under
`work/trained-collection-audit-v1`. These local raw artifacts are not included in
the source repository; the source includes the inventory, producers, independent
raw auditor and collection auditor needed to reproduce their checks.

Native policy 15 additionally refines QKV, second LayerNorm and the first
feed-forward projection. On the M4, it passes all 89 historical B16 logits/loss/
gradient anchors and all 20 fresh logits/loss/gradient batches. The complete
historical raw audits are in `work/precision-phase6`; automatic dispatch selected
SME with 64-byte streaming vectors. The MPS and compiled-MLX
custom policy-7 models, which refine first LayerNorm, QKV and attention, each pass
all 89 historical forward anchors and all 20 fresh forward batches. These complete
collections have independent raw audits in `work/precision-phase5/prior-*-audit.json`.
MPS policy 7 now also passes the complete historical gradient inventory, with
independent raw audits for both widths. Compiled MLX passes the two width-512
gradient anchors; its complete width-1024 audit confirms one failing anchor/batch
among 87 anchors and 71 distinct batches. It is not qualified by its forward pass.
MPS and compiled MLX each also pass all seven gradient-screen anchors and all 20
fresh logits/loss/gradient batches, with independent raw audits. The screen is
the two width-512 anchors plus the five remaining width-1024 QKV-localization
cases, not the full historical gradient denominator. The machine-readable
[evidence index](evidence-index.json) binds these audits and the preceding failed
framework candidates. Referenced `work/` raw artifacts remain local.

## Failures retained and next decisions

The complete MLX policy-7 gradient replay fails 12 FF1 weight-gradient
coordinates at `w1024-37621e9d745f8bd4-validation-39`, while that batch's logits
and loss pass. The independent collection audit confirms the failure. The
[branch-localization result](mlx-branch-localization.json) reproduces the original
compiled logits, loss and every gradient byte, including after adding its trace.
One ReLU preactivation is +3.6508e-7 in MLX versus -1.2613e-7 in FP64. Substituting
only that observed gate into the FP64 backward removes all 12 gradient failures.
Every observed prefix through the second normalization retains the correct gate
when FF1 is recomputed in FP64, isolating the residual error to that projection.
The declared framework-v3 bit 8 applies the existing compensated linear operation
and explicit derivatives to FF1. Its policy-15 forward and gradient screens pass
independent raw audits, including exact repeat checks. Its complete historical
gradient replay now passes independent raw audits at both widths: all 89 anchors
and 73 distinct batches. Separate historical inference also passes both complete
widths. The existing-fresh replay passes all 20 batches in each mode, with eight
independent collection audits across the two inventories, two widths and two
operations. The first fresh launch used the wrong evidence root and stopped
before evaluating any case. A separate successor authenticated the original
checkpoint hashes under the correct root and ran those four fresh collections.
The failed zero-case launch remains retained alongside the historical passes.
This policy's bit mapping differs from native policy 15. The queued
fresh-state plans stopped at their prerequisites without launching a worker.

Ordinary maintained MPS and MLX still fail the historical width-1024 forward
inventory: 60/71 and 56/71 distinct batches, respectively. Passing short runtime
replays or fresh development states does not erase those failures. Same-QKV
compensated FP32 attention diagnostics pass forward and backward on their three
localized MPS/MLX cases. Full-model compensated-attention policies 2 and 3 pass
both operations on those three cases, including compiled MLX with graph/trace
proof. Ordinary MLX width-1024 inference fails while its gradient graph passes;
operation-matched ordinary controls reproduce exactly. The 28 operation-specific
factorial receipts and four new native screens have independent raw audits in
`work/trained-followup-audit-v3`. Complete policy-2/policy-3 forward replay then
found respectively 11/5 remaining MPS anchors and 4/5 MLX anchors at width 1024.
The five remaining MPS policy-3 cases match the CPU QKV-localization residuals.
Adding the prospectively declared compensated QKV bit removes those failures in
the full policy-7 forward inventory. These custom model results do not qualify a
framework source repair or performance comparison.

The first Python candidate (v1) passed historical logits but failed two gradient
coordinates on one width-1024 batch. V2 adds first-LayerNorm precision and passes
that complete inventory. Both the original failure and replacement identity are
retained.

The fresh batch `fresh-w512-step001-sequence16` fails native-policy-1 gradients
(469 coordinates) and ordinary PyTorch gradients (224 coordinates), while their
logits and loss pass. Native policy 3, which additionally uses FP64 QKV
accumulation, retains the 469-coordinate failure. The branch-localization
diagnostic identifies one ReLU sign disagreement: FP64 -1.65e-8 versus native
policy 3 +7.15e-8. Holding the FP64 forward fixed and substituting only the native
backward gate removes every gradient mismatch. The policy-3 prefix first changes
that gate at the second LayerNorm. The prospectively declared bit 4
refines the second normalization; policies 4–7 build successfully but all retain
the same gradient failure. Policy 7 improves the preactivation to +1.19e-8.
Its input through the second LayerNorm now gives the correct gate when the first
feed-forward dot product is recomputed in FP64. This isolates the remaining sign
error to that projection; it does not justify changing ReLU or its tolerance.
The complete bit-8 factorial retains failures for policies 8, 9, 11, 12 and 13;
policies 10, 14 and 15 pass that regression's logits, loss and gradients. The
independent raw audits retain every outcome. Policy 15 also passes both complete
fresh inventories, including the previously failing branch.

The fresh states come from the declared 64-update training-only trajectory at
steps 0, 1, 8, 32 and 64, with one training-derived and one synthetic B16 probe per
checkpoint. Two fresh native training processes match all 65 parameter and Adam
state digests at each width. These are development states, not held-out assessment
or a replacement for the formal training budget.

## Build, storage and remaining gates

Assembly and framework builds are hosted. GPU execution retains runtime shader
compilation. Native candidate builds pass their
primitive checks. Policy 1 has passed complete hosted model and executable checks;
hosted policies 1, 3 and 7 have passed full checks at both widths. Policy 7
passes 1,153 tests plus 190 subtests (14 skips), with 1,979 runtime checks each
under auto and forced NEON. Explicit policy-dependent dense-kernel accounting
uses a development-only validation contract.
Historical validators keep their original contract.
The hosted policy-15 build at `5c0899ad6f5e5f2505aa523d35f46e3ae68f0c38`
passes 1,167 tests plus 190 subtests (14 skips) at each wide width, with 1,979
executable checks in each auto/NEON run. Run
[41125590880](https://github.com/logannye/arm64-transformer/actions/runs/41125590880)
also checks all 16 primitive policies. Hosted
[41127179360](https://github.com/logannye/arm64-transformer/actions/runs/41127179360)
replays the complete historical native gradients from existing public tensors;
both widths passed their complete independent raw audits (89 anchors, 73 distinct
batches, including every gradient). Its M1/NEON platform
cannot qualify the M4/SME path or support new performance claims.
The separate M4/SME replay now passes the same complete denominator. Its width-512
and width-1024 audit hashes are respectively
`f64a61b29e5b8ffea2e970017063a48776eb56a4743278f83e98efc7dd37e5a7` and
`fb40add286b836887b3edfc6b70a6c3fcb879c0087cb82dcfb07817ac2396b3f`.
Hosted raw artifacts expire November 6; local M4 raw evidence remains in the
lossless store. The hosted raw payload has not been independently downloaded
and re-audited locally.

The later [six-width build](native-six-width-build.json) at
`44a3bca1b17bff104965848855fb8fe0954a610c` passed every hosted job in
[run 41137295729](https://github.com/logannye/arm64-transformer/actions/runs/41137295729).
All six small candidate bundles were independently hash-verified locally. Both
wide policy-15 libraries are byte-identical to the existing M4 replay libraries.
The width-1024 job passed 1,190 tests plus 190 subtests and 1,979 executable
checks in each auto/NEON run. Large hosted raw runtime artifacts were not downloaded.

The full-source PyTorch FP16/layout v2 build completed compilation, but failed
its check because four Objective-C++ commands omit explicit `-g0` despite the
CMake cache containing it. The failure is retained in run 41110294834. Recipe v3
passes CMake flag variables explicitly and checks generated commands early.
Its complete build, import and lint checks passed in
[run 41125186309](https://github.com/logannye/arm64-transformer/actions/runs/41125186309)
at `11fd196ed3665178f389e9c25256c3561b14fe5f`. Its 1,343,946,294-byte artifact
matches GitHub's SHA256 `a2b311ec1886804a9be6d348c5bbafb4276ce9b4730da13ae18a41a93b534741`.
The first preparation rejected legitimate empty runtime include directories;
the narrow verifier correction retains all file/link/path checks and leaves
those empty directory records unexpanded. The actual runtime archive's 15,707
payload files now verify. The complete source/runtime pair has been expanded;
runtime capture retains 28,909 runtime and 2,220 project file identities.
Full-suite preparation then rejected the new source manifest's explicit symlink
records. The producer and portable auditor now normalize both retained source
formats, hashing literal link targets and rejecting missing, escaped or replaced
links. The corrected producer verifies all 146,435 source members, including 21
symlinks. The failed preparation is preserved. The subsequent
[runtime capture and full-suite preparation](half-v3-preparation.json) bind the
corrected verifier at `697bc9c05e792e31571a5fbff5fa4ae0a8b22c81` and pass.
That build
addresses retained integration failures, not the MPS trained-attention arithmetic
failures. The portable MPS evidence producer and archive auditor now have 86
CPU-only checks, plus 25 eligibility checks. They verify retained evidence after
relocation without reading original paths. No actual runtime or full MPS suite is
qualified by those tests. Full framework conformance against the new runtime,
archive validation of that run and a prospectively frozen
successor comparison contract remain necessary before any formal campaign.

Two later full-suite preparations stopped on file-stability checks. The retained
diagnostic observed a zero-byte source placeholder changing flags and timestamps
on initial access. The corrected binder hashes an already-readable unbuffered
descriptor and still requires complete metadata stability through hashing and
the final path check. Mutation and identical-byte replacement tests confirm that
those checks remain enforced. Both failed preparations remain in the record.
The corrected binder also verifies all 146,435 actual source members. Empty files
are verified from their stable zero-length descriptor without a hydration read;
all nonempty contents are read and hashed completely.
The next preparation exposed one environment difference: macOS CPython adds
`__CF_USER_TEXT_ENCODING` at startup. Preserving that platform variable makes
the frozen environment match a fresh interpreter exactly. The latest capture
and full-suite plan pass; the 34-method focused replay and complete 10,357-test
MPS suite are queued after the fresh model checks. Preparation imports no
framework and runs no device test or C++ fixture build.
The unchanged MLX C++ and Python suites also prepare successfully against this
same captured runtime. Their frozen inventory retains 305 C++ cases with 4,792
assertions, 1,053 Python cases, seven unchanged skips, 17,701 passing subtests
and all 27 added GPU regressions. Their device execution is queued after MPS;
earlier MLX results under another NumPy environment cannot qualify this one.

Large raw arrays are stored losslessly in `work/trained-raw-store-v1`. Each receipt
binds the original NPZ hash; the store preserves exact compressed members and ZIP
metadata and verifies byte-identical reconstruction. New workers can write shared
members directly from memory, and auditors read one bounded archive in memory.
No arrays are rounded, downcast or discarded. The interrupted storage run and the
sandboxed Metal launch that returned `No Metal device` remain explicit records.
At the October 30 21:36 UTC checkpoint, user-authorized cleanup restored about
26.6 GiB of available disk space. Thirty local copies of published archives were
verified against their release asset hashes before removal. An obsolete stopped
CI VM was removed after preserving its source and diagnostics; a duplicate
expanded validation-cache file was removed after verifying byte identity with its
retained compressed original. Current runtimes and historical transformer raw
evidence were preserved. Cleanup receipts remain in
`work/storage-cleanup-20261030`. Logical archive size is not a measure of physical
space recovered from cloud-managed storage.

Full historical gradients for native policy 15, MPS policy 7 and compiled MLX
policy 7 resumed in a frozen serial plan in `work/precision-phase6`. Each engine
has one two-case width-512 worker and six width-1024 shards covering all 87
anchors. A collection audit follows each complete engine/width group. The plan
requires 25 GiB initially and stops at a 5 GiB operating reserve. Results remain
pending until each complete collection has been independently audited. Both
native and MPS width collections are complete and pass. MLX width 512 passes;
its width-1024 audit confirms one numerical failure. All six collection audits
are complete and evidence-valid; numerical success remains separate.

The separate [v3 framework artifact adapter](../../integration/gpu-owned/CANDIDATE-HALF-V3-VALIDATION.md)
has 42 standard-library checks. It verifies the new build's actual early/final
compiler commands and retained Half differential bytes before preparing the
34-method focused device replay. This adapter does not promote the old runtime's
evidence or qualify the new runtime's device behavior. Explicit candidate runtime capture
and the separately pinned 10,357-test full-suite route are implemented with the
same conformance-fixture and portable-evidence requirements. An explicit bounded
expansion option retains 10 GiB after the declared payload plus per-member
metadata allowance, with checks during and after expansion; the default remains
16 GiB. Full-suite output (2 GiB), fixture cache (1 GiB) and the 5 GiB operating
floor remain unchanged. The new runtime's identity capture and suite preparation
are complete; actual suite execution and device qualification remain pending.

The candidate verifier now also hashes stable readable descriptors. Its explicit
plan-rebinding command preserves the original plan and verifier, reauthenticates
every source/runtime/dependency and artifact input, and publishes a final plan
only after all checks pass. This lets the corrected verifier reuse the existing
expanded pair without another build, duplicate expansion or qualification transfer.

The policy-15 development trajectory and saved-Adam continuation route has 16
focused checks. The updated inventory reader also reconstructs the retained v1
20-case inventory exactly from its original receipts and files. A separate frozen
plan in `work/precision-phase7` required all six phase-6 collection audits to pass
before starting two 64-update policy-15 processes per width, a third-process
step-32 continuation, and all-five-engine replay of the resulting 20 common-state
probes. Actual trajectory, continuation and replay results remain pending.
The separate phase-8 inference plan required phase 7 and its complete audits;
inference cannot inherit compiled-gradient graph results. Both plans stopped
without executing workers after the retained MLX failure. A new plan can follow
the replacement candidate's complete checks; these failed prerequisites remain
in the record.

The corrected successor plans are `work/precision-phase7c` and
`work/precision-phase8c`. The former has started native policy-15 training after
all eight MLX collections and the other four implementations' historical gates
passed. It retains two independent trajectories per width, saved Adam moments,
the third-process restart and all-five-engine gradient checks. Only after the
two trajectories' complete state histories and saved bytes match may identical
checkpoint files share storage; every original path, byte and receipt remains.
The initial storage allocation is the exact 1,313,802,240-byte checkpoint size
plus 3 GiB of working space and the unchanged 5 GiB floor. Inference has a
separate 512 MiB working allocation above that floor. Both supervisors monitor
the floor throughout execution. These storage allocations do not alter test
coverage or numerical requirements. Both independent processes now match all 65
parameter and optimizer state digests at each width. Fresh-process restarts from
step 32 also reproduce every saved parameter and moment digest through step 64
at both widths. Sharing 34 byte-identical saved files then recovered 655,495,168
bytes while retaining every original path. All five implementations pass all 20
new common-state probes in both inference and gradient modes. The twenty
independent collection audits (five engines, two widths, two operations) each
verify ten distinct batches, dual-FP64 accuracy and exact within-engine repeats.
These new states were generated by native policy 15; the replay policies remain
Python v2, reference Metal policy 1, MPS policy 7 and compiled MLX policy 15.
This does not qualify the maintained-framework arithmetic or transfer results
to another runtime distribution.

The separate captured-runtime replay adapter has 16 CPU-only trust-boundary
checks. It runs the unchanged numerical producer after authenticating a selected
runtime capture and retains a separate pre/postflight origin receipt without
changing the original worker result or raw arrays. Its actual replay under the
new framework distribution remains pending; existing model passes under the old
distribution do not transfer to it.

The successor runtime capture at `29fffcc7b3f38c476fd9d73ed89448464a7d4423`
binds the captured-model adapter and both conformance producers to one project
revision. It verifies the same framework bytes as the preceding capture and
reauthenticates all source members after storage sharing. The focused MPS, full
MPS and full MLX preparations pass. The older queued plans were superseded before
any device worker ran. The first aligned launcher also stopped before a worker
when MLX-only controls were passed to the focused MPS adapter; its successor
uses that adapter's own controlled environment. The preparation receipts and
retained launcher failure are bound in `half-v3-preparation.json`. Actual suite
results remain pending. Hosted Python and native CI passed at this revision in
[run 41146051670](https://github.com/logannye/arm64-transformer/actions/runs/41146051670).

User-authorized cleanup has recorded about 22.8 GiB of cumulative free-space
gains across bounded actions; concurrent writes and APFS/cloud sharing prevent
exact physical attribution. The latest receipt is
`work/storage-cleanup-20261030/summary-v6.json`. All transformer raw evidence
remains retained. The session disk guard stops the identified conformance process
tree at 5.5 GiB free, preserving headroom above the 5 GiB operating floor.

The resumed full MPS route collected the unchanged 10,357-method inventory but
stopped before test execution when the origin checker rejected
`torch._C._dynamo.autograd_compiler`. The pinned C++ source creates that module
under this registered name and attaches the same object as
`torch._C._dynamo.compiled_autograd`. The narrow checker correction resolves
exactly that spelling difference and still requires object identity, declared
module name and authenticated native-extension origin. Six new mutation cases
and the related origin/loaded-proof tests pass (73 checks total). The failure
and C++ source hashes are retained in [origin-mapping-repair.json](origin-mapping-repair.json).
This changes provenance verification only. New capture and conformance execution
remain required; no arithmetic or framework binary changed.

The independent captured-replay provenance auditor has nine standard-library
checks. It authenticates the original numerical receipt and loaded origins and
explicitly requires a separate raw-array audit. Both hosted jobs also passed at
`6f9e8be5af92b0bd49b58eff111b7ff6532ba798` in
[run 41148065755](https://github.com/logannye/arm64-transformer/actions/runs/41148065755).
The resumed controller enforces a 5.5 GiB stop threshold on its complete worker
process group. Additional cleanup left about 12 GiB free; its local record is
`work/storage-cleanup-20261030/additional-cleanup-final.json`.

The corrected capture and both full-suite preparations pass at
`0640f2cd4912dbc22f925abcc085ca9fb3436fc9`; their exact identities are recorded
in `half-v3-preparation.json`. The framework-byte identity remains
`5698e6f6f07655d54f249533e28aabdfa75af51f768fc9ff0c49b4ca65d2d3cd`.
The serial controller in `work/resumed-framework-validation-v2` runs the full
MPS suite, full MLX suites, a matched 24-worker captured-runtime screen, then the
34 focused MPS cases. The screen compares ordinary and custom candidate models
on the same two width-512 historical cases and the width-1024 FF1 regression,
with separate inference/gradient, provenance and raw-array audits. It cannot
substitute for the complete historical/fresh inventories or all-width admission.
These actual results remain pending. Source-level framework arithmetic repairs,
optimized Metal qualification, a successor freeze and the 180-fit experiment
remain open requirements.

The next full-suite launch also stopped before numerical tests because `_VF.py`
replaces its module with a fileless forwarding instance. A single authenticated
metadata diagnostic enumerated all 46 fileless modules in the collected runtime.
The verifier now explicitly recognizes `_VF` and nine registered backend wrappers,
checking the original module/native target, actual registered object, exact
class and declared Python function/property code against authenticated source.
Unknown wrappers and altered targets, source, code or proof flags still fail.
The same records are validated in CPU, GPU, suite and portable provenance audits.
The actual 1,066-module PyTorch inventory passes this development check; no
conformance test or numerical model was executed by that diagnostic. Unit checks
pass: 134 origin/backend/CPU tests plus ten forwarding subtests, 60 suite checks
and 26 portable archive checks. The failed launches remain retained. A new
capture is required before repeating the full suites under this verifier.

The first complete MPS test execution under the corrected namespace checker ran
all 10,357 methods in 365.314 seconds. All 27 previously failing integration IDs
now pass their unchanged upstream assertions and are neither skipped nor expected
failures. The run retains one failure and one error, both in fallback-warning
subprocess tests. A standard-library reproduction shows that the shell launcher
adds `PWD` and `SHLVL`, violating the exact environment contract. An `env -S`
launcher preserves the selected interpreter, `-P -S -B`, literal arguments and
exact frozen environment; its regression check passes on the actual host.
The post-run audit separately exposed a raw symlink-manifest reader; that route
now uses the existing authenticated normalization rather than indexing every
raw record for `sha256`. The suite helper passes 63 tests and the portable
archive helper passes 26. No upstream test, tolerance or skip changed.
The [retained full-run record](mps-full-suite-harness-failure.json) remains failed
and unqualified. The independent full MLX run and matched model diagnostics
continue under the previous captured project while the new MPS retry is prepared.

The independent MLX C++ worker passes all 305 cases and 4,792 assertions with
zero skips under the captured NumPy-1.26 environment. Its outer audit fails:
`otool -L` and `otool -l` outputs were assigned filenames differing only in case,
so the latter replaced the former on this filesystem. Both original command
outputs remain embedded in the hash-bound worker receipt; the incomplete run is
retained in [mlx-linkage-artifact-failure.json](mlx-linkage-artifact-failure.json).
The corrected producer uses `libraries` and `load-commands` filenames with
exclusive creation. Its regression checks require distinct case-folded names and
byte-equal saved command output. All 31 MLX suite-helper tests pass. The Python
suite and model diagnostics did not start after this integrity failure; no pass
is transferred to another run. The corrected MPS and MLX routes need a new
capture and fresh execution, preserving all suite coverage and storage limits.

The corrected-shell full MPS retry again runs all 10,357 methods, retaining
one failure and one error in the same two child tests; all 27 original integration
regressions still pass their unchanged upstream assertions. A direct authenticated
import-only child succeeds. A separate metadata-only trace identifies the remaining
environment difference: importing the unchanged test inventory adds
`TORCHINDUCTOR_CACHE_DIR`. The producer now freezes that path before collection
inside the existing monitored 1 GiB/64-file fixture cache. Live and portable audits
reject missing or external cache paths. The 65 suite-helper and 27 portable-helper
tests pass. The [failed run and diagnostics](mps-import-environment-failure.json)
remain retained, and a fresh capture/execution is required. MLX conformance and the
matched model screen continue independently under the immutable preceding capture;
neither route may qualify the failed MPS run. The disk watchdog tracks nested
sessions by PID, creation time and command, with a 6 GiB stop reserve.

The fresh MLX C++ route passes all 305 cases and 4,792 assertions, including
its corrected linkage-file audit. Python collection then fails before any test
method: the original `test_array.py` parser cannot convert PyTorch's `0a0` patch
component to an integer. The upstream pytest teardown also clears all streams,
so a subsequent compilation probe needs its own newly verified GPU stream. The
[retained record](mlx-test-compatibility-failure.json) preserves both failures
and the old parent's misleading `complete=true` flag; the corrected producer
requires complete, integrity-checked workers for a complete parent summary.

The explicit [test-only compatibility patch](../../integration/gpu-owned/mlx-test-compat-v1/README.md)
uses `packaging.version.Version.release`, preserving all test bodies, decorators,
assertions, eligibility and the complete inventory. Its derivation tool requires
byte-identical native sources, libraries, shaders and original build provenance.
It creates a new capsule and manifest with an APFS clone and performs no native
build. Ten capsule/parser mutation tests, 22 capture tests, seven capture-review
tests and 35 MLX suite-helper tests pass. Fresh full conformance remains required.

The bounded MPS cache fix survives actual unchanged test-module import and
collection with no environment additions or changes. The matched captured-runtime
model screen now runs in the required owned-development output namespace, after
retaining a failed launcher that executed no model. Its early CPU candidate
results pass separate inference/gradient, provenance and raw-array audits at both
wide widths; ordinary CPU controls retain failures on the same selected operands.
This screen covers two historical width-512 cases and one width-1024 FF1 case,
not a complete inventory, framework repair, all-width gate or performance result.

The complete [captured-runtime screen](captured-model-screen.json) now retains
24 cells and 48 independent provenance/raw audits. All 12 custom-candidate cells
pass: CPU v2, MPS policy 7 and compiled MLX policy 15, each at both wide widths
and in separate inference/gradient modes. Ordinary controls pass only the MPS
width-512 gradient cell; the other 11 cells fail at the unchanged tolerances.
Coverage remains exactly two exposed width-512 cases and the single width-1024
FF1 regression. These observations do not transfer to another captured identity.

The new test-compatible joint capture binds source
`14bf87420e4415e48b0def3d300293450091261f` and runtime identity
`f50c56ad6f55cd383f51033a8622272ee2ba538a9fe430a4f457ab5cb1398e0f`.
The derived MLX capsule verifies unchanged native source/binary bytes and its
47,940,121-byte original logical payload. Both full-suite preparations pass. A
serial controller runs fresh complete MPS and MLX suites, then all 34 focused MPS
cases, after the model screen and every identified descendant exit. Its separate
watchdog retains a 6 GiB reserve across nested sessions. Actual suite results,
maintained-library arithmetic repairs and all formal qualifications remain pending.
Hosted Python and native CI both passed at `2c28aadc7340b3e317cba8078d7f1987986ad199`
in [run 41154376946](https://github.com/logannye/arm64-transformer/actions/runs/41154376946).

The fresh MPS worker executes all 10,357 methods successfully, with exactly the
existing 744 skips and 382 expected failures, and passes its before/after runtime
integrity checks. Its final gate still rejects eight existing empty reason
strings. The authenticated `common_mps.py` source applies bare `unittest.skip` to
exactly those eight already eligible FP16 IDs; a standard-library reproduction
confirms that spelling records an empty string. The decoder now recognizes only
those exact IDs under the two pinned test sources and rejects empty reasons
elsewhere. No skip or expected-failure ID changes. The portable reader also now
checks the launcher's source-bound generator rather than the obsolete shell form.
The saved actual test inventory passes the corrected decoder, but the old failed
producer receipt/exit remains unchanged and unqualified. The two affected helper
suites each pass 28 tests.

MLX C++ again passes all 305 cases and 4,792 assertions. Its complete Python
execution reports 1,040 passes, four failures and nine skips. Three failures
require NumPy 2-only APIs; the fourth is a retained padding-reference mismatch
under NumPy 1.26. The two additional skips require TensorFlow and ml_dtypes, which
are installed in the retained complete MLX dependency environment. The original
seven-skip inventory is preserved, not expanded. The
[complete findings](joint-runtime-conformance-findings.json) bind these outcomes.
A new hosted build retains the same fourteen-file native source patch and changes
only NumPy to 2.2.6 in its 45-wheel inventory, with actual native shared-memory
NumPy roundtrips required after build. All 30 build-helper mutation checks pass.
Fresh complete conformance and model validation remain required under that new
ABI identity. Supplemental installed dependency roots can now be explicitly
captured without shadowing the selected framework/NumPy roots; 23 capture checks
pass. The prospective generic precision API's CPU tests passed hosted Python and
native CI at `1ed35f1b10873863453d7ee640a4cd6cf1934801` in
[run 41157511140](https://github.com/logannye/arm64-transformer/actions/runs/41157511140).
These are prototype operator checks, not installed-library or GPU qualification.

The full NumPy 2 build is running at source
`a367b322eda2649309c1519017c0a2a14819b827` in
[run 41158978641](https://github.com/logannye/arm64-transformer/actions/runs/41158978641).
The candidate verifier now has an explicit, independently pinned `numpy2-v1`
selection; the preceding half-v3 selection is unchanged. It requires the same
native source and test inventories, all hosted build/source/compiler proofs and
four native NumPy dtype/alias roundtrips. Forty-five verifier mutation checks and
24 capture checks pass. A selected run ID is not a successful build or qualification.

The retained MLX Python run also used pytest 7.3.2, which missed the upstream
`[tool.pytest]` configuration, prefixed case identities with the outer workspace
and omitted the required subtest summary. The corrected producer requires
pytest 9.1.1, the selected source root and the unchanged upstream configuration.
It binds their actual files and metadata, together with the existing TensorFlow
2.20.0 and ml_dtypes 0.5.3 packages, before collection. All 40 MLX helper checks
pass, including rejection of old pytest, missing support packages and changed
configuration. The explicit NumPy 2 suite contract retains all 1,053 Python cases,
17,701 subtests, seven eligible skips, 305 C++ cases and 4,792 assertions. No old
XML, receipt, exit status or qualification is rewritten. A fresh joint capture
and complete execution remain required.

The [library API derivation](library-api-derivations.json) now authenticates an
additive full MLX package copy: three generic Python modules appear in both
source and runtime, while all old native/Python/test members remain unchanged.
Five derivation mutation checks and 26 capture checks pass. The copy is not yet
loaded or qualified. Its native build remains 40978071574 and its new manifest
is a separate identity. PyTorch's analogous copy waits for the NumPy 2 build.
An explicit model adapter calls these package APIs with fixed CPU-v2/MPS-7/MLX-15
stage selections. Its changed-token CPU forward/all-gradient reproduction check
is submitted to hosted CI at all six process-selected widths. Device, model,
full-suite and optimizer/checkpoint qualification remain pending.

The library model now has a separate captured-runtime replay flag and preserves
the existing same-byte input, dual-FP64, raw-array and byte-exact repeat gates.
Independent audits reject mixed old/custom/library labels, changed model stages,
a missing selected API, or absent/substituted loaded package modules. Seventeen
replay-helper checks, eleven provenance checks and 29 independent array-auditor
checks pass. The new numerical route has not yet executed. NumPy version claims
now come from the explicitly selected native ABI; old captures retain 1.26.4.
MLX suite preparation rejects an inconsistent Torch/NumPy suite selection.

A real non-framework pytest probe with the complete MLX dependency root first
and the retained Torch test dependencies second passes one method and all three
subtests. Its XML uses the required `python.tests.test_probe.TestProbe` namespace.
The first sandboxed probe remains failed because the installed rerun plugin
could not open its ephemeral localhost socket; the authorized unrestricted probe
passes. This is dependency/configuration evidence only, not MLX conformance.

The prospective installed-package primitive gate now covers all six widths and
CPU/MPS/compiled-MLX execution. Each of three generic operations has two changed
operand sets, every derivative, dual independent FP64 references, exact replay,
noncontiguous inputs and malformed-input rejection. CPU/MPS saved-tensor hooks
check FP32/device state. MLX requires actual compiled graphs, one trace over four
changed/repeated calls, natural GPU execution and FP32 outputs/derivatives; it
explicitly does not claim a runtime saved-tensor hook. Raw evidence is bounded
to 512 MiB per worker with the existing lossless shared store and 6 GiB operating
reserve. A separate auditor checks actual package origins, case/width layouts,
raw decisions and graph/trace bindings. Hosted prototype tests now exercise
these equations and audits at every process-selected width. The actual captured
primitive gate has not run yet.

Source preparation can now reuse an existing owned source tree only after its
complete contents match the new hosted source manifest. This copies zero source
bytes, transfers no qualification and retains full reauthentication and the
existing expansion reserve. All 47 candidate-verifier checks pass. The hosted
Python job at `e2fdb56b2efecf71b5970a93ef051601634b8205` failed in synthetic
derivation tests because that runner had less than the production 16 GiB free
reserve. A test-only free-space fixture at `915091e7b5e5cca601043d7c3dc28394505ae52a`
fixes the byte-sized unit fixture; the real reserve and its rejection test remain
unchanged. The failed run is retained; no validation is transferred.

The complete MPS suite now has an explicit `numpy2-v1` contract pinned to native
run 41158978641 and NumPy 2.2.6. Its 10,357 method IDs, 744 skip IDs, 382 expected
failure IDs, all assertion sources and existing tolerances exactly match the
half-v3 inventory. Full source checks additionally require the same precision
API derivation and all three frozen generic API modules. All 67 MPS helper and
28 portable-reader checks pass. A frozen plan may use its original verifier
from another checkout only when that verifier's actual bytes equal the current
trusted verifier; changed verifier source is rejected before loading. All six
package-derivation checks pass.

The first unused MLX API copy is superseded before any primitive/model execution
so the refreshed derivation producer can be frozen in an immutable checkout.
Its old receipt is retained as an unqualified preparation record. The waiting
NumPy 2 preparation supervisor was stopped before downloading any artifact to
bind these prospective fixes; the hosted build and serial device validation
continue. Routine hosted validation now cancels superseded jobs per event and
branch/PR, preserving separate latest push and PR checks and the dedicated
long native build's independent concurrency group.

Hosted Python and macOS arm64 native CI both pass at
`56b927a7bbda4e47a1915f4f874aec4fa5ac9fbd` in
[run 41160555792](https://github.com/logannye/arm64-transformer/actions/runs/41160555792).
This includes the new CPU primitive/dual-oracle checks and byte-equal complete
model forward/all-gradient reproduction at all six process-selected widths.
The exact installed Native/NumPy 2 runtime and real GPU checks remain separate.
The refreshed MLX API copy is authenticated from immutable source
`41d78aec3e5fa826b3048263ae3950a14cb04233`; it has not been loaded numerically.

The prospective primitive gate also now observes actual attention probabilities
through the public operation: its second value operand is an identity padded to
head width, so the first T output columns equal the probabilities. Independent
auditing requires that exact basis and preserves the existing 2e-7/2e-6 absolute/
relative probability tolerance. The first operand set retains arbitrary values;
all returned derivatives and exact repeated execution remain required. This
changes no library arithmetic or API package bytes and precedes any actual
captured primitive execution. The NumPy 2 preparation supervisor is running
under the frozen full-suite/provenance source with a 16 GiB reserve.
