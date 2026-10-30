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

## Bounded fresh development trajectory

After the exposed-case factorial, evaluate native policy 1 on a fixed 64-update
training-only trajectory at each wide width. Start from the existing seed-42
initial checkpoint, authenticate its original training rows, and use private
stream `[20261103, width, 73]` for 64 batches of 16 training indices and fresh
development probes. Retain checkpoints 0, 1, 8, 32 and 64. Each checkpoint is
replayed on one training-derived batch and one synthetic batch containing repeated
0/127 tokens, alternating 0/127 tokens and uniform random tokens. Run two fresh
processes and require all 65 parameter/first-moment/second-moment state digests
to match. All comparisons still use identical saved weights across engines.
This is 64 development updates, not a replacement for the 1,024-update formal
training budget, a held-out assessment or evidence of convergence quality.

The expanded exposed-state check uses B16 for native policy 1 gradients, in
16-case shards bounded by the worker's 4 GiB raw-evidence limit. Retained MPS,
MLX and handwritten Metal each replay the entire 87-case width-1024 forward
inventory; width 512 has two exposed cases. Numerical failures remain completed
failures. Source/runtime changes, missing cases and provenance failures remain
integrity failures. The independent raw auditor verifies both kinds separately.

## Prospective Python candidate after localization

`python-fp32-saved-qkv-attention-v1` is a separate Python model candidate, not a
relabeling of the ordinary PyTorch baseline. It uses standard PyTorch operations
with explicit first-order autograd functions for QKV and causal attention. Only
intermediate arithmetic is FP64; parameters, module outputs, saved activations,
attention probabilities and returned gradients are FP32. The attention backward
uses the saved FP32 probabilities. The model, epsilon and optimizer remain fixed.

Before replay, require primitive forward/backward agreement, causal-prefix
isolation, repeated inputs and an autograd saved-tensor audit. Then test the
complete same-input regression inventory and the fixed fresh development states
with unchanged tolerances. The candidate remains separate from both historical
contracts and from a stock-framework performance claim. Its full cost would have
to be included in any future explicitly revised Python comparison arm.

## Prospective handwritten Metal refinement

The reference Metal path has one failing width-1024 replay batch after its existing
compensated score/QKV work. Before candidate execution, declare the complete
four-policy factorial in `integration/trained-precision/metal_candidates.py`:
bit 1 retains a two-float reciprocal-square-root refinement and normalized value
through affine normalization; bit 2 uses a compensated attention-value sum.
Both affect forward arithmetic, preserve FP32 buffers and retain the existing
backward equations using saved FP32 state. The normalization candidate recovers
the low part of the existing exact 1e-5 epsilon; it does not select a new epsilon.

All four policies are replayed on the same entire exposed inventory. Baseline
source, derived shader bytes, runtime compilation and host-library identities
remain bound. These candidates address the reference shader path only; optimized
policy paths are unchanged and cannot inherit their results. No timing or formal
selection follows from the exposed cases alone. Local and propagated prefix
errors are examined separately, with trace/ordinary-output equality reported.

The follow-up framework attention diagnostic is fixed before execution in
`compensated_attention.py`: two-float products with Dekker splitting, pairwise
two-float sums, split attention scale, compensated score centering, exponential
correction and probability division, and compensated value contraction. It uses
ordinary FP32 framework tensor operations without graph compilation or fusion.
The diagnostic is restricted to finite same-input QKV operands with absolute
values at most 10,000; this is not a generic library implementation or a new
comparison arm. It runs on the already captured MPS/MLX localization traces,
with a fixed development-only upstream derivative from `[20261103, width, 91]`.
Both forward and ordinary framework autograd are compared with independent
NumPy/PyTorch FP64 equations and repeated exactly. A passing diagnostic does not
qualify a full-source framework repair or establish its performance.

The complete v1 Python gradient replay, unlike its initial two-case screen,
exposed two failing position-embedding gradient coordinates in
`w1024-37621e9d745f8bd4-test-221`. Retain that candidate and its failure. Before
further execution, declare `python-fp32-saved-ln1-qkv-attention-v2`, which adds
first-LayerNorm FP64 intermediate arithmetic in both directions. It saves only
FP32 normalized values, inverse standard deviations and weights. All other v1
operations and every tolerance remain fixed. Screen primitive saved-state and
gradient behavior, then the entire original regression inventory and the same
fresh training/synthetic development inventory. Neither candidate is ordinary
PyTorch or an automatically admitted replacement for a historical study arm.

After the same-QKV GPU diagnostic, the next fixed model diagnostic is the complete
four-way factorial of compensated first normalization (bit 1) and compensated
attention (bit 2), with the ordinary QKV projection unchanged. It uses the three
retained MPS/MLX localization cases: MPS width 1024, MLX widths 512 and 1024. Run
all four policies through complete logits, loss and every parameter gradient.
For MLX retain both ordinary and compiled expression execution; do not assume
that passing unfused expressions survive compiler transformations. Policy 0
checks that the extracted equations reproduce the ordinary model. This is an
explicit diagnostic model implementation, not a runtime operator override,
framework source repair, admitted replacement or performance result. All input
and output identities, failures and existing tolerances remain retained.

The fresh width-512 step-1 synthetic batch fails native-policy-1 and policy-3
gradients despite passing logits. Before any further native arithmetic candidate,
trace its feed-forward preactivation signs against the FP64 equations. For each
successive native prefix, recompute the remaining prefix through the first
feed-forward projection in FP64 and retain any ReLU gate disagreements. As a
diagnostic only, hold the entire FP64 forward computation fixed and substitute
the observed native ReLU gate in its backward equation. Compare that gradient
with both the original FP64 gradient and the reproduced native gradient. This
counterfactual tests whether a branch disagreement explains the failure; it is
not a replacement gradient, a tolerance exception or an admitted model.

That diagnostic found one sign disagreement at batch/token/channel [14, 2, 248]:
the FP64 preactivation is -1.649231633270125e-8 and native policy 3 produces
7.147900760173798e-8. Holding the FP64 forward fixed and using the native backward
gate removes all 469 gradient failures. Policy 3's prefix first introduces the
gate disagreement at the second normalization. Before any new native execution,
extend the original policy bits with bit 4: FP64 second-LayerNorm forward and
backward arithmetic with the same FP32 saved buffers. Build policies 4 through 7
alongside the retained 0 through 3 controls. Replay all four new policies on the
same failing batch, then test any passing candidate on the complete historical
and existing development inventories. Policy 7 combines both normalizations and
QKV refinement; no FFN projection, ReLU rule, architecture or tolerance changes.

The first complete-model GPU factorial evaluates logits returned by the gradient
graph. Its MLX width-1024 policy-0 control passes, unlike the retained standalone
inference path. Before extending any conclusion, repeat the same four policies
in explicit forward-only mode at all three localized cases, retaining both MLX
compilation modes. Also repeat the compiled MLX gradient cases with an actual
compiled-graph proof and one-Python-trace replay check. Policy 0 additionally
records raw ordinary-backend outputs for the matching operation, using the
gradient graph's own logits for the gradient control. The independent auditor
recomputes control equality. Passing one operation never qualifies the other.

After the operation-specific four-policy controls, extend the compensated
framework candidates (policies 2 and 3) to the complete exposed 89-case inventory
and the existing 20-case fresh inventory. Keep inference and gradient executions
separate. Use a reusable fixed-width evaluator with parameters, tokens and targets
passed explicitly on every execution; no captured checkpoint is allowed. MLX
uses the already verified compiled mode and must retain one Python trace across
all changed inputs and exact repeats, with an execution count and graph proof.
This remains a named diagnostic model candidate using ordinary FP32 framework
operations, not an ordinary-framework or full-source repair claim. Begin with
complete forward replay before allocating the larger gradient evidence; retain
all failures and do not promote a candidate based only on the localized cases.
