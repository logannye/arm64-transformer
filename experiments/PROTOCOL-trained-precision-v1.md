# Trained-state precision development, v1

This prospective development phase follows the user's October 30 request to
repair wide-model arithmetic and enable a matched five-engine comparison.
It is separate from the immutable v6 CPU and v4 owned campaigns. Neither their
results nor their unopened admission streams are changed. No formal fit is
authorized by a development receipt.

## Fixed question and candidates

Can selective higher-precision normalization and QKV arithmetic remove the
observed trained-state failures while preserving FP32 storage, the model,
gradients, exact within-engine replay and the existing numerical tolerances?

The first native inventory is fixed before candidate execution:

| Policy | First LayerNorm | QKV projection |
|---:|---|---|
| 0 | Original | Original |
| 1 | FP64 intermediate arithmetic | Original |
| 2 | Original | FP64 accumulation |
| 3 | FP64 intermediate arithmetic | FP64 accumulation |

Both forward and backward are implemented. Policies apply only to widths 512
and 1024. Other layers and widths retain their original execution. FP32 saved
normalization intermediates remain the backward inputs; the ordinary derivative
does not differentiate rounding. These simple assembly candidates prioritize
correctness; any subsequent optimization requires a separately frozen inventory.
All candidates and failures are retained, without performance-based selection on
exposed validation/test examples. Default builds remain policy 0 until a candidate
has been qualified and a new formal contract explicitly selects it.

## Development evidence

Replay the union of historical native and PyTorch FP64 failing sequences at
both wide widths, deduplicated by checkpoint hash, split and sequence. Authenticate
the original checkpoint, split and report hashes. These exposed examples are
regressions, never fresh assessment. Each worker loads one width, implementation
and native policy in a fresh process. Record the actual binary, source, package,
hardware and input identities. Never overwrite an output or an original result.

Evaluate logits, mean cross-entropy and every parameter gradient at identical
saved weights and inputs against independent NumPy FP64 equations and PyTorch
FP64 autograd. Preserve original per-sequence versus batched execution semantics
when reproducing baseline failures; explicitly label any alternate grouping.
Keep local operation errors, propagated prefix errors, repeatability and model
quality separate. Precision failures do not disappear when aggregate loss passes.

Retain forward atol/rtol 2e-5/2e-4, gradient 3e-5/3e-3, probability
2e-7/2e-6 and independent FP64 agreement 1e-10/1e-10. Exact replay means
identical bytes within an engine, not identical independently trained models.
Candidate source, binary and policy must accompany each observation. Missing
engines, unsupported operations or failed workers remain explicit failures.

After regression localization, use the existing development-only stream 20261103
for synthetic stress cases and original training-only inputs for evolved-state
checks. No reserved stream is opened in this phase. No changes to architecture,
initialization, epsilon, learning rate, training budget, corpus or tolerances.

## Framework and formal work

Before framework arithmetic changes, a separate CPU localization probe tests the
complete eight-policy factorial of first-LayerNorm, QKV and attention FP64
substitutions (bits 1, 2, 4). Each selected module returns FP32. The attention
substitution retains FP64 internal saved state and therefore is explicitly a
counterfactual diagnostic, not a qualified FP32-state baseline. All eight outcomes
are retained. This inventory is fixed before its first execution and does not
authorize selecting or timing a replacement framework based on exposed test data.

Framework repairs need their own same-input localization and full-source binary
qualification. Existing maintained-framework short replay passes do not establish
trained-state accuracy. MPS integration failures and its unfinished archive audit
remain blocking; native precision development cannot clear them.

A successor formal protocol must explicitly bind repaired CPU sources (v4 binds
the old source), maintained framework patches, binary identities, real native
dispatch accounting, every inherited gate, and the full 180-fit denominator.
Freeze it only after development and integration pass; select a fresh assessment
stream without inspecting its operands. Audit all five engines at every selected
checkpoint. Include all precision, packing and synchronization costs in timing.

Native compilation uses hosted CI. Reuse installed Python libraries and retained
artifacts locally; do not build or replace the framework runtime locally.
