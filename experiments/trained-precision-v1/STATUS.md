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
