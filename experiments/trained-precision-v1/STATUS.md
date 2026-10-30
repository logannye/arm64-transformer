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
| Handwritten reference Metal normalization refinement, policy 1 | 89/89 anchors pass | Width 512: 10/10 pass; width 1024 follow-up pending independent audit |

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

## Failures retained and next decisions

Ordinary maintained MPS and MLX still fail the historical width-1024 forward
inventory: 60/71 and 56/71 distinct batches, respectively. Passing short runtime
replays or fresh development states does not erase those failures. Same-QKV
compensated FP32 attention diagnostics pass forward and backward on their three
localized MPS/MLX cases. The complete-model four-policy factorial, including both
compiled and uncompiled MLX execution, is the next diagnostic, not a framework
source repair or a performance claim.

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
that gate at the second LayerNorm. A new, prospectively declared policy bit 4
refines the second normalization; policies 4–7 await hosted build and validation.

The fresh states come from the declared 64-update training-only trajectory at
steps 0, 1, 8, 32 and 64, with one training-derived and one synthetic B16 probe per
checkpoint. Two fresh native training processes match all 65 parameter and Adam
state digests at each width. These are development states, not held-out assessment
or a replacement for the formal training budget.

## Build, storage and remaining gates

All native and framework compilation is hosted. Native candidate builds pass their
primitive checks. Policy 1 has passed complete hosted model and executable checks;
new hosted checks cover both policies 1 and 3, with explicit four-versus-five
ordinary dense-kernel accounting and a development-only validation contract.
Historical validators keep their original contract.

The full-source PyTorch FP16/layout repair build is still in progress. That build
addresses retained integration failures, not the MPS trained-attention arithmetic
failures. Full framework conformance, archive-safe MPS evidence validation, new
runtime capture and a prospectively frozen successor comparison contract remain
necessary before any formal five-engine campaign.

Large raw arrays are stored losslessly in `work/trained-raw-store-v1`. Each receipt
binds the original NPZ hash; the store preserves exact compressed members and ZIP
metadata and verifies byte-identical reconstruction. New workers can write shared
members directly from memory, and auditors read one bounded archive in memory.
No arrays are rounded, downcast or discarded. The interrupted storage run and the
sandboxed Metal launch that returned `No Metal device` remain explicit records.
