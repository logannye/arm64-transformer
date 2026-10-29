# Preserve attention's small differences before rounding

The wider models exposed a precision problem in attention: two large query–key
scores can be close enough that their small difference determines the probability
split. Subtracting the maximum before exponentiation prevents overflow, but cannot
recover bits already lost while forming those scores.

V6 keeps the optimized width-512/1024 attention scores in FP64 (64-bit floating
point) through dot products, scaling, maximum selection and subtraction. Only the
centered scores are rounded to FP32. The existing exponential, probability
normalization and value mixing then continue in FP32. Model parameters, stored
activations and Adam state retain their FP32 layouts; the architecture and training
budget are unchanged.

![Selective precision through attention](../../docs/precision-dataflow.png)

The implementation uses ordinary NEON double instructions, independently of the
SME FP32 dense kernels. It caches 32 converted query channels, uses four FP64
partial accumulators, and retains one 32-double score row: **256 bytes of score
scratch in a 400-byte forward frame**. No optional SME double-precision feature is
required. Smaller model widths and generic attention fallbacks retain their
existing arithmetic.

## Why backward also needed attention

The initial forward-only change passed the development forward and probability
checks. A width-512 stress case with a common score near 4096 and a competing-score
gap of 0.75 still failed two gradient comparison records. Two query-gradient
coordinates differed from the FP64 reference by about `4.54e-5` and `4.47e-5`,
against allowances of `3.55e-5` and `3.23e-5`. Evaluating the backward formula with
the identical saved probabilities still gave errors near `4.10e-5`, localizing a
remaining backward arithmetic limitation.

The final wide backward path therefore retains FP64 through the dA dot products,
softmax row contraction, scaled score derivatives and complete causal query-gradient
sum. Each key-gradient contribution is added in FP64 before its FP32 store; value
gradients keep their existing FP32 accumulation order. One 32-double row is reused
for dA and score derivatives within a **384-byte backward frame**. Saved
probabilities are neither renormalized nor clipped. All 437 unchanged development
checks passed at each wide model size after this change.

The initial failure JSON records are retained as development logs. That intermediate
candidate's binary and exact source snapshot were not captured, so those logs are
not an independently replayable version of the forward-only implementation. The
final implementation and its regression replays below have explicit binary
identities; the formal campaign has its own frozen source and admission evidence.

## What changed on the 25 known failing sequences

This regression replays **all 25 unique previously exposed v5 failure sequences**:
one at width 512 and 24 at width 1024. Every comparison uses the same saved native
checkpoint and input. These are development examples, not a new untouched
assessment. No training or timing is performed in this replay.

The preserved old library reproduces every saved v5 prediction exactly. Old and
new libraries run in separate processes, and the attention-isolation comparisons
receive bit-identical QKV arrays and deterministic upstream gradients. Independent
NumPy FP64 equations evaluate the exact stored inputs and parameters.

| Comparison against FP64 | Old failing coordinates | New failing coordinates | Old pooled RMSE | New pooled RMSE |
|---|---:|---:|---:|---:|
| Whole-model logits | 1 | 0 | 1.468e-6 | 1.169e-6 |
| Attention output at identical QKV | 37 | 0 | 1.491e-6 | 4.105e-8 |
| Attention probabilities at identical QKV | 111 | 0 | 1.502e-7 | 2.611e-9 |
| Attention gradients at identical QKV/upstream gradient | 0 | 0 | 5.952e-7 | 2.798e-8 |

Whole-model logits and attention outputs use the unchanged forward limit
`2e-5 + 2e-4*abs(reference)`. Probabilities use the math limit
`2e-7 + 2e-6*abs(reference)`; gradients use `3e-5 + 3e-3*abs(reference)`.
These are coordinate counts within a selected regression inventory, not independent
statistical observations.

Attention output, probability and gradient RMSE improved in **25/25 cases**.
Whole-model RMSE improved in **16/25** and increased in **9/25**, with a maximum
relative increase of 21.2%; every new whole-model comparison nevertheless passed
the unchanged forward tolerances against FP64. FP32 rounding elsewhere remains, and changing
one stage can remove accidental cancellation between different stages' errors.
The detailed records preserve both favorable and unfavorable changes.

The original v5 audit compared native against PyTorch FP32. Its failed result
remains unchanged. This FP64 regression does not establish that the old mutual
FP32 criterion would pass, or qualify new timing measurements. The separate
[v6 protocol](../PROTOCOL-v6.md) declares the fresh synthetic admission stream,
same-checkpoint FP64 checks for both engines, compatibility reporting and matched
measurement requirements before the new campaign.

## What the fresh campaign still exposed

The formal v6 campaign retains a width-1024 `native_precision_limit`: 18 native
logit coordinates fail across 16 sequence checks including identical repeats.
Deduplication by checkpoint hash, split and sequence leaves eight sequences and
nine coordinates. A bounded post-hoc investigation replays all eight at their
exact saved weights, with no training, tuning, source edits or changed tolerance.

Every new attention output and probability check passes when FP64 receives the
identical native QKV array. Maximum local errors are `1.36e-6` and `1.01e-7`,
respectively. In six sequences, retaining the native first-LayerNorm output and
recomputing the entire remaining model in FP64 still produces a logit failure.
One further case first fails after the captured QKV projection. The final case
is a cumulative rounding effect just 0.365% beyond its allowance.

Computing the first normalization in FP64 from the exact native embedding input,
rounding once to FP32, then using an FP64 suffix passes all eight. Ideal FP64 QKV
rounded once to FP32 also passes all eight. This implicates propagated upstream
normalization and dense arithmetic rounding; these cases do not establish an
unavoidable FP32-storage floor. The substitutions use an FP64 suffix, so they do
not prove that changing LayerNorm alone would repair the full native model.

On this selected v6 sample, the old attention library has 16 same-QKV output
failures and the new library has zero. Whole-model failures remain nine
coordinates, spread across six sequences with the old library and eight with the
new one. Local improvements can remove accidental cancellation between stages;
this selected sample does not establish a global old-versus-new failure rate.

Across the complete width-1024 report, deduplicating identical checkpoint/split
repeats leaves 98,304 token positions. Native has zero argmax changes versus FP64,
a maximum probability difference of `1.057e-5`, and a largest absolute split-mean
NLL difference of `2.061e-8` nats. PyTorch has one argmax change and a largest
absolute split-mean NLL difference of `9.144e-8` nats. Both engines pass the NLL
checks; the strict logit failures and unqualified timing status remain unchanged.

Full stage traces, independent oracle agreement, per-case substitutions and all
old/new comparisons are preserved under
`development/v6-development/residual-diagnosis/` in the v6 evidence archive.
This is exposed diagnostic evidence, separate from the declared admission tests.

Compact numerical summaries are available for the [25 known cases](regression-diagnosis-summary.json)
and [eight residual cases](residual-diagnosis-summary.json). Full arrays, comparisons,
scripts and hashes remain in the evidence archive.

## Evidence identities

The final regression libraries match the wide binaries used by the v6 campaign,
whose measured source is `2c5191047068c16b12e3312652b118d48a5f3744`.

| Library | SHA-256 |
|---|---|
| Width 512 | `db1e5c6510861bf238962a19cfe47a1ded0a15da3997147723e23cd3f9d7bdd9` |
| Width 1024 | `99149b68688e4bfa58e3d862ccc96117da95cf73b2e9f7a0824948a38085cbea` |

The known-case summary, complete old/new per-case comparisons, checkpoint/input
hashes, scripts, logs and payload hashes are retained under
`development/v6-development/regression-diagnosis/` in the v6 evidence archive. Earlier width-512
candidate replay files remain under its `pre-final-build/` directory. The original
forward-only development logs are retained separately under
`development/v6-development/precision-development/` with the provenance limitation above.

This is selective precision. Key gradients still round after each query
contribution, value gradients remain FP32, and rounded probability mass can leave
small backward residuals. Neither the implementation nor these 25 examples imply
full FP64 equivalence or correctness over every possible input.
