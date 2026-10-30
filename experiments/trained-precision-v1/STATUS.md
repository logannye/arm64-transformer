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
| Custom compiled MLX policy 15, additionally refining FF1 | 89/89 gradient-graph logits/loss/all-gradient anchors pass | Complete v3 replay pending |

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
and 73 distinct batches. Separate historical inference and existing-fresh replays
are running under the frozen serial phase-9 plan; inference cannot inherit the
compiled gradient graph's result.
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
symlinks. The failed preparation is preserved. A fresh capture and preparation
must bind the corrected verifier before device qualification. That build
addresses retained integration failures, not the MPS trained-attention arithmetic
failures. The portable MPS evidence producer and archive auditor now have 85
CPU-only checks, plus 25 eligibility checks. They verify retained evidence after
relocation without reading original paths. No actual runtime or full MPS suite is
qualified by those tests. Full framework conformance against the new runtime,
archive validation of that run, new runtime capture and a prospectively frozen
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
has 36 standard-library checks. It verifies the new build's actual early/final
compiler commands and retained Half differential bytes before preparing the
34-method focused device replay. This adapter does not promote the old runtime's
evidence or qualify the new runtime's device behavior. Explicit candidate runtime capture
and the separately pinned 10,357-test full-suite route are implemented with the
same conformance-fixture and portable-evidence requirements. An explicit bounded
expansion option retains 10 GiB after the declared payload plus per-member
metadata allowance, with checks during and after expansion; the default remains
16 GiB. Full-suite output (2 GiB), fixture cache (1 GiB) and the 5 GiB operating
floor remain unchanged. Their new runtime
and actual suite execution remain pending.

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
