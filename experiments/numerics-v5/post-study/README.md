# Width-512 seed-44 post-study checkpoint discrepancy

This is a diagnosis of an unchanged, failed post-study criterion, not a new admission rule or a repaired result. No training, parameter changes, kernel edits, or tolerance changes were performed. It applies to the specific saved checkpoint and row below; it does not explain the width-1024 failures or establish accuracy across every trained checkpoint.

## Exact reproduction

The measured source is `c12874c2fe6af3fd1612046950aa56d756f4d0a0`. The test uses `seed-44-repeat-1-arm64/best.f32`, width 512, test sequence 43, and the measured native library with the auto backend. Native recomputation reproduces every saved logit for this sequence bit-exact. Repeat 2 has identical best-checkpoint and saved-logit file hashes.

Exactly one of the sequence's 4,096 logits exceeds the declared forward comparison limit: zero-based token position 29, vocabulary index 109 (`m`). The largest absolute discrepancy, at index [29,76], is a different logit and passes its larger reference-relative limit.

| Same-checkpoint calculation | Logit [29,109] |
|---|---:|
| Saved/native FP32 | 0.03495997563004494 |
| Functional PyTorch FP32 audit | 0.03492925316095352 |
| Production PyTorch FP32 model | 0.03492935746908188 |
| Independent NumPy FP64 equations | 0.034962898670769176 |
| Functional PyTorch FP64 | 0.03496289867076952 |

Native versus functional FP32 error is `3.07224690914e-5`, exceeding `2e-5 + 2e-4*abs(reference) = 2.69858506322e-5` by about 13.85%. The comparison therefore remains failed. Production PyTorch also differs from native beyond the original limit at this coordinate, so the audit's expanded attention implementation alone does not explain it.

Across the sequence, native error against NumPy FP64 is max `5.0630e-6`, RMSE `6.5583e-7`, with no coordinates outside the original forward limits. Functional PyTorch FP32 error against the same FP64 equations is max `5.2953e-5`, RMSE `3.3441e-6`, with the same one failing coordinate. Production PyTorch FP32 is similar. NumPy FP64, functional PyTorch FP64, and production PyTorch FP64 agree to at most `1.69e-14` across all logits. FP64 evaluates the same real-valued equations using the exact saved FP32 parameter values cast to double; it is diagnostic, not a substitute production model.

## Localization

Full layer traces first show substantial disagreement at attention, particularly head 1 and query position 29. No engine changes a ReLU active/inactive mask on this row's full forward pass. LayerNorm and each dense kernel, when evaluated at identical incoming FP32 operands, stay within the original elementwise forward limits.

The dominant attention effect comes from FP32 score arithmetic rather than the upstream QKV projection. Holding the functional model's QKV array fixed, native attention differs from the FP64 attention equations by max `5.8524e-6`; functional and production PyTorch attention differ by about `9.60e-5`. Continuing from the functional FP32 QKV array through an otherwise FP64 suffix gives final-logit max error `4.13e-6`. Continuing instead from its already-computed FP32 attention array gives `5.30e-5`. These are diagnostic substitutions only; no model implementation was changed.

The two dominant causal keys are 29 and 27. Using exactly the same FP32 QKV inputs, a scalar `math.fsum` oracle confirms their scores as `107.19285617366322` and `107.00036893682561`. Functional PyTorch FP32 produces `107.1928939819336` and `107.00032043457031`: opposite-signed score errors of `+3.7808e-5` and `-4.8502e-5`. The score difference changes from `0.19248723684` to `0.19257354736`. This changes the approximately 0.548/0.452 probability split by `2.1379e-5`. Computing softmax itself in FP32 adds only about `5e-9` and `2.5e-8` at these two probabilities. The score arithmetic is therefore the dominant localized source.

This is not severe cancellation inside either Q·K dot product: sum-of-absolute-products divided by absolute sum is approximately 1.036 and 1.018. It is sensitivity of the difference between two scores near 107, followed by probability-weighted value mixing and downstream projections. The native blocked score reduction is closer to the FP64 equations on this captured input. The evidence identifies a finite-precision disagreement and supplies no concrete native implementation defect to repair. It does not justify changing the held-out acceptance criterion after observing this result.

## Evidence and reproduction

- `sequence43-diagnosis.json`: exact failing coordinate, provenance, whole-model comparisons, layer statistics, and primitive checks at each engine's captured inputs.
- `sequence43-traces.npz`: all native, production/functional FP32, production/functional FP64, and independent NumPy FP64 traces.
- `attention-isolation.json`: same-QKV attention comparisons and FP64 suffix substitutions.
- `score-rounding.json`: independent scalar summation and score-versus-softmax error decomposition.
- `diagnose.py`, `isolate_attention.py`, `score_rounding.py`, with matching `.log` files: replayable diagnostic code and output.

Run each script from the repository with `ARM64_TRANSFORMER_WIDTH=512 ARM64_TRANSFORMER_BACKEND=auto`, one-thread BLAS/OpenMP environment settings, and `PYTHONPATH=.`. Outputs stay in this ignored diagnosis directory. `provenance.json` authenticates the measured source/native/input identities, repeat hashes, and every diagnostic payload.

The full scripts, logs and trace arrays are retained in the release under `development/post-study-diagnosis/`. The JSON files here are byte-identical copies of the diagnostic records.
