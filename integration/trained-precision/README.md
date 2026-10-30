# Trained-state precision development

These tools investigate the width-512 and width-1024 failures at identical saved
weights and inputs. They do not run the formal 180-fit study, open its reserved
streams, select a performance winner or revise historical results. The fixed
[development protocol](../../experiments/PROTOCOL-trained-precision-v1.md) records
the arithmetic candidates, tolerances and fresh training-only trajectory.

`transformer.trained_precision` replays the authenticated union of 89 historical
failing-sequence anchors. With batch size 16, the 87 width-1024 anchors correspond
to 71 distinct batches; the two width-512 anchors correspond to two batches.
NumPy FP64 equations and PyTorch FP64 autograd independently check logits, mean
loss and every parameter gradient. Each engine executes twice, and both raw
results are retained. Baseline and candidate failures remain in the evidence.
The same worker accepts the separately authenticated fresh-development inventory.

The native policy build is produced by the hosted
[`trained-precision.yml`](../../.github/workflows/trained-precision.yml) workflow.
It now retains binaries at all six widths from one source revision: policy 0 at
32–256 and the full 0–15 factorial at 512–1024. Full model/executable checks select
policy 0 at the four smaller widths and policy 15 at the two wide widths. These
are development candidates; the default build and formal selections are unchanged.
Policy 1 changes first-LayerNorm arithmetic; policy 2 changes QKV accumulation;
policy 3 changes both. The later bit 4 refines the second LayerNorm in both
directions, giving policies 4–7 after the fresh ReLU-branch localization.
Bit 8 refines the first feed-forward projection and its derivatives, giving the
complete 0–15 factorial. Policy 15 enables all four refinements. The ordinary
dense-kernel count is `5 - bool(policy & 2) - bool(policy & 8)` in each direction;
development workers verify observed dispatch against that declared count.
`relu_localize.py` retains that counterfactual diagnostic and its original
gradient failure. FP32 stored state remains fixed. Default builds retain
policy 0. Use externally verified artifact hashes with `--library` and
`--library-sha256`; never rebuild these candidates implicitly during replay.

`--python-candidate --python-candidate-version 1` selects the separately named
QKV/attention Python candidate. Version 2 additionally refines first LayerNorm.
Both use standard PyTorch operations with explicit first-order derivatives and
FP32 saved tensors. They are separate model implementations, not ordinary
PyTorch baselines. Installed and maintained PyTorch runtimes must also be named
separately; a pass under one distribution does not qualify another.

`metal_candidates.py` emits the declared four-way normalization/value-contraction
factorial for the reference Metal shader path. It does not change default or
optimized shaders. `localize_gpu.py` separates same-input local operation errors
from propagated prefix errors. `compensated_probe.py` tests attention in isolation;
`framework_probe.py` tests the complete model under a four-way normalization and
attention factorial, including separate compiled and uncompiled MLX controls.
`--framework-policy 2|3|6|7` extends these candidates to whole replay inventories;
bit 4 adds the tiled compensated QKV projection. Its explicit first-order
derivatives use standard device tensor operations and retain only FP32 inputs,
weights and returned gradients. This avoids saving every product/reduction
temporary for backward. It remains a custom model, not a native framework patch.
`--framework-compiled` explicitly selects compiled MLX. The reusable evaluator
takes new parameters and batches on every call and retains its graph/trace and
execution-count proof. Forward and gradient receipts remain separate.
`graph_evidence.py` retains small graphs as DOT and larger graphs as lossless gzip,
with hashes and strict bounds for both encoded and decoded bytes. The independent
auditor rejects truncated, concatenated or oversized compressed streams. Graph
transport does not change model arithmetic or relax numerical tolerances.
These probes are diagnostic Python model expressions, not full-source framework
repairs or admitted comparison arms.

`fresh_states.py` runs exactly 64 native development updates from authenticated
training-only inputs. Run it twice in fresh processes per width.
`fresh_inventory.py` requires all 65 parameter, first-moment and second-moment
digests to match, verifies saved bytes and actual dispatch counts, then emits
20 replay batches across checkpoints 0, 1, 8, 32 and 64. This short development
trajectory does not establish 1,024-update accuracy, convergence or assessment
performance.

The explicit `fresh_states.py --precision-policy 15` successor retains a separate
v2 identity and also saves both Adam moments. A v2 pair manifest must select
policy 15; the inventory checks its three-call ordinary dense accounting and
every saved moment. `resume_native.py` authenticates a completed v2 receipt and
resumes its saved step-32 state in a fresh process, checking every parameter and
moment digest through step 64 against the uninterrupted run. These checks are
development-only and must run serially after other numerical workers finish.

## Independent checks and bounded storage

`audit_replay.py` recomputes decisions from externally hash-bound raw arrays,
checks dual-oracle agreement and exact repeatability, and verifies available
native dispatch counters. Historical receipts without after-counters remain
explicitly unverified for dispatch. `audit_collection.py` additionally rejects
missing, reordered, duplicated or mixed-implementation shards and reports both
sequence-anchor and distinct-batch counts. Evidence validity and numerical
success are separate results. Neither auditor returns formal eligibility.

Large replay archives repeat the same oracle arrays. `zip_evidence.py` can share
identical **compressed member bytes**, retaining each original ZIP header,
directory and trailer. It does not decode, cast, recompress or approximate an
array. Reconstruction must reproduce the original archive SHA256 exactly.
`pack-worker --replace` removes a materialized NPZ only after its complete byte
reconstruction has been verified; the original receipt and raw hashes remain
unchanged. Storage operates only on completed task-owned workers.

Use the original receipt's `raw_sha256` to restore one archive:

```sh
python -I -S -B integration/trained-precision/zip_evidence.py restore \
  --store /ABSOLUTE/EVIDENCE/STORE --sha256 ORIGINAL_RAW_SHA256 \
  --destination /ABSOLUTE/RESTORED/ARRAYS.npz
```

Alternatively pass `--raw-store /ABSOLUTE/EVIDENCE/STORE` to either auditor.
It authenticates and checks one archive in memory at a time, with a 512 MiB
compressed-archive bound. Auditing does not write temporary evidence copies.
New replay workers and `framework_probe.py` also accept `--raw-store`: each
bounded NPZ is assembled in memory and only previously unseen compressed members
are written. The receipt still binds the original complete NPZ hash and size.
This preserves the 4 GiB logical evidence limit per replay worker even when
several archives share most of their stored bytes.
Materialized raw files with changed bytes are rejected; the store never masks a
corrupted existing file. This permits a complete re-audit without materializing
the entire campaign. Keep the store's `objects` and `recipes` directories with
the original worker receipts, inventory and authenticated input checkpoints.

The hosted [`trained-precision-replay.yml`](../../.github/workflows/trained-precision-replay.yml)
workflow restores only hash-authenticated tensor inputs from the existing public
CPU-v6 release, then replays all historical native-policy-15 gradients and audits
the raw arrays. Its Apple M1/NEON result is separate from Apple M4/SME evidence.
Large hosted raw artifacts need not be downloaded onto the local development disk.

The complete maintained-framework builds, full upstream suites, archive-safe MPS
qualification, fresh runtime capture, prospective formal contract and all-engine
checkpoint audit remain separate requirements. A development pass clears none
of those gates by itself.
