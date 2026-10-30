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
| Native first-LayerNorm FP64 arithmetic, policy 1 | 89/89 anchors pass | Width 512: 9/10 pass; width 1024: 10/10 pass |
| Python FP32-saved first-LayerNorm/QKV/attention refinement, v2 | 89/89 anchors pass | 20/20 batches pass |
| Handwritten reference Metal normalization refinement, policy 1 | 89/89 anchors pass | 20/20 batches pass |

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
feed-forward projection. On the M4, it passes all 89 historical B16 forward
anchors and all 20 fresh logits/loss/gradient batches. The MPS and compiled-MLX
custom policy-7 models, which refine first LayerNorm, QKV and attention, each pass
all 89 historical forward anchors and all 20 fresh forward batches. These complete
collections have independent raw audits in `work/precision-phase5/prior-*-audit.json`.
Full historical gradients on the M4 are still required for the new candidates.
MPS and compiled MLX each also pass all seven gradient-screen anchors and all 20
fresh logits/loss/gradient batches, with independent raw audits. The screen is
the two width-512 anchors plus the five remaining width-1024 QKV-localization
cases, not the full historical gradient denominator. The machine-readable
[evidence index](evidence-index.json) binds these audits and the preceding failed
framework candidates. Referenced `work/` raw artifacts remain local.

## Failures retained and next decisions

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

All native and framework compilation is hosted. Native candidate builds pass their
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

The full-source PyTorch FP16/layout v2 build completed compilation, but failed
its check because four Objective-C++ commands omit explicit `-g0` despite the
CMake cache containing it. The failure is retained in run 41110294834. Recipe v3
passes CMake flag variables explicitly and checks generated commands early;
its hosted validation remains pending. That build
addresses retained integration failures, not the MPS trained-attention arithmetic
failures. The portable MPS evidence producer and archive auditor now have 70
CPU-only checks, plus 25 eligibility checks. They verify retained evidence after
relocation without reading original paths. No actual runtime or full MPS suite is
qualified by those tests. Full framework conformance against the new runtime,
archive validation of that run, new runtime capture and a prospectively frozen
successor comparison contract remain necessary before any formal campaign.

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
pending until the complete collections have been independently audited.

The separate [v3 framework artifact adapter](../../integration/gpu-owned/CANDIDATE-HALF-V3-VALIDATION.md)
has 30 standard-library checks. It verifies the new build's actual early/final
compiler commands and retained Half differential bytes before preparing the
34-method focused device replay. This adapter neither qualifies the pending
build nor promotes the old runtime's evidence. Explicit candidate runtime capture
and the separately pinned 10,357-test full-suite route are implemented with the
same conformance-fixture and portable-evidence requirements. Their new runtime
and actual suite execution remain pending.
