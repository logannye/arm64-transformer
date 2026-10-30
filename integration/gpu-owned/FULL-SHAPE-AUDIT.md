# Independent saved-array audit for the full-shape development check

`audit_full_shape_probe.py` reads a closed execution of the declared B1/T32 and
B16/T32 development probe. It imports NumPy and the standard library, never a
tensor framework. The existing B3/T3 auditor and all frozen execution helpers
remain unchanged.

The accepted manifest SHA-256 is
`1617562aa8cd1e04c358aa5e4261fb72a6fad529472115f1762462c3ed470954`.
Its declared execution helper SHA-256 is
`06a772b3e28819739fc63a3f782631840dcd22cb1eb8d71b9d09b002670b6277`.
The auditor requires a separately supplied SHA-256 for the final execution
summary. A missing, incomplete or failed execution cannot produce a passing
audit, even if some individual workers passed.

Complete coverage requires:

- 72 workers: six widths, two batch/sequence shapes, three maintained framework
  paths and two fresh processes, in the exact planned order.
- Parent/worker PID agreement, protocol/manifest/cell identity and unchanged
  development-only scope flags.
- 36 exact fresh-process pairs and 12 complete common-fixture records, including
  exact initial parameter bytes across batch shapes at a given width.
- Seven NPZ files per worker, **504 total**, with exact member inventories,
  shapes, dtypes, finite values and byte hashes. These include raw checkpoint
  logits at both updates three and six.
- The declared token/target formulas; initial logits, loss and every parameter
  gradient against the saved PyTorch FP64 arrays; and agreement between the
  saved NumPy and PyTorch FP64 arrays at the unchanged tolerances.
- Ten initial replay records, warmup restoration, two six-update trajectories,
  optimizer counts, checkpoint serialization/reload and prediction replay.
- Rechecked worker receipts and all raw archives at completion, after the
  individual worker audits have finished.

The auditor recomputes comparisons from saved arrays. It does not execute the
transformer equations, recreate parameter initialization, reauthenticate live
framework libraries or repeat the GPU execution. Intermediate training states,
additional initial replays and warmup restoration retain execution-time hash
records rather than separate raw copies. Trained-state FP64 accuracy, broad
framework integration, the complete stress inventory, formal admission and
performance measurements remain separate requirements.

Thirty-one CPU-only tests cover valid synthetic B1/T32 and B16/T32 records,
malformed intermediate identities, loss shape, one-ULP raw checkpoint
substitution, mismatched PIDs, incorrect token formulas, dual-FP64 disagreement,
missing/reordered/failed parent checks, changed scope flags, late receipt/raw
mutations and the complete 504-file final recheck. The frozen manifest has also
passed static contract validation. These tests are tooling validation; they do
not establish the outcome of a running development experiment.

After the real execution closes, record its final summary SHA-256 and use a new
output path:

```sh
.venv/bin/python -m unittest discover -s integration/gpu-owned \
  -p test_audit_full_shape_probe.py -v
.venv/bin/python integration/gpu-owned/audit_full_shape_probe.py \
  --root /ABSOLUTE/WORKSPACE/work/gpu-v4-owned/model-full-shape-v1 \
  --manifest /ABSOLUTE/WORKSPACE/work/gpu-v4-owned/model-full-shape-manifest-v1.json \
  --summary-sha256 RECORDED_CLOSED_SUMMARY_SHA256 \
  --output /ABSOLUTE/FRESH/FULL-SHAPE-AUDIT.json
```

The command preserves an explicit failed audit receipt if verification fails;
it never edits the execution evidence. A passing receipt remains development
evidence and cannot open either formal admission stream.
