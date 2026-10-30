# Full-shape development evidence publication

This additive package retains the closed B1/T32 and B16/T32 development run. It does not modify earlier archives or promote the run to formal admission. The retained full-source PyTorch CPU/MPS and compiled MLX builds passed 72 workers, 36 fresh-process pairs and 12 common fixtures. The independent saved-array auditor passed all 504 NPZ files. Native CPU, custom Metal, corpus fits, performance measurements and post-training FP64 accuracy remain outside this check.

The [compact report](../../experiments/gpu-v4-owned/full-shape-report-v1/README.md) contains 36 per-width/engine/shape rows, static figures and 75 byte-preserved receipts. The original failed model attempt remains in the earlier evidence package. Broad framework qualification remains blocking; neither reserved admission stream was opened and formal fits remain 0/180.

The trusted closed identities are:

| Input | SHA-256 |
|---|---|
| Full-shape summary | `2995a572d97799a48ca436634ac7959590fff5e2d5904feac2ea8d45790d3df8` |
| Prospective manifest | `1617562aa8cd1e04c358aa5e4261fb72a6fad529472115f1762462c3ed470954` |
| Model execution helper | `06a772b3e28819739fc63a3f782631840dcd22cb1eb8d71b9d09b002670b6277` |
| Independent array auditor | `8ff4565f77e7f535ec72a77eb03fe3b210b9ba60d55f28408810319887c344e8` |
| Passing array audit | `e7b6498e3180d7bfd038c5adffc4b3f947c54a35a0e9b78bce4c76daeecb038f` |
| Frozen archive utility | `2a30d78a30cfe75e56bf3a133f7611e0544706c4b1171a40732eb113b0b4ba51` |
| Full-shape archive plan | `0c29c26e47308b45dab62b81b854e1e97837352441bf3c8204d6683668921064` |

[The plan](full-shape-evidence-plan-v1.json) enumerates every original workspace-relative member exactly once: 651 files, including 504 named raw-array archives, totaling 6,882,696,502 uncompressed bytes. Six raw-evidence groups split by engine and batch shape, each containing 12 workers and their logs; a seventh metadata group contains the summary, manifest and passing array audit. Every input group and resulting gzip archive must remain strictly below 2,000,000,000 bytes. Source and runtime distributions are referenced by their authenticated identities and are not duplicated in these groups.

`prepare_full_shape_evidence.py` reuses the unchanged, hash-pinned `prepare_evidence.py` inventory, bounded gzip/tar writer and streaming archive verifier. It requires externally recorded summary and audit hashes, the fixed manifest and auditor identities, a closed passing execution, the complete ordered worker inventory, matching cross-process and cross-engine receipts, and exactly seven expected NPZ filenames per worker. Symlinks, missing or duplicate members, native payloads, unknown raw-array roles and changed sources fail. All source inventories are rechecked after packaging; incomplete attempts retain their failure receipt and partial evidence.

From the repository root, regenerate to fresh output paths:

```sh
.venv/bin/python integration/gpu-owned/prepare_full_shape_evidence.py plan \
  --workspace /ABSOLUTE/WORKSPACE \
  --summary-sha256 2995a572d97799a48ca436634ac7959590fff5e2d5904feac2ea8d45790d3df8 \
  --audit-sha256 e7b6498e3180d7bfd038c5adffc4b3f947c54a35a0e9b78bce4c76daeecb038f \
  --output /FRESH/full-shape-evidence-plan.json

.venv/bin/python integration/gpu-owned/prepare_full_shape_evidence.py build \
  --workspace /ABSOLUTE/WORKSPACE \
  --plan integration/gpu-owned/full-shape-evidence-plan-v1.json \
  --plan-sha256 0c29c26e47308b45dab62b81b854e1e97837352441bf3c8204d6683668921064 \
  --output /FRESH/full-shape-evidence-archives

.venv/bin/python integration/gpu-owned/report_full_shape_probe.py \
  --workspace /ABSOLUTE/WORKSPACE \
  --summary-sha256 2995a572d97799a48ca436634ac7959590fff5e2d5904feac2ea8d45790d3df8 \
  --audit-sha256 e7b6498e3180d7bfd038c5adffc4b3f947c54a35a0e9b78bce4c76daeecb038f \
  --output /FRESH/full-shape-report
```

The archive plan binds the producer source hash, all source bytes and file modes. A source, contract or evidence change requires a new plan and new output directory. Report copies bind source and destination hashes both immediately and after generation. The report includes only independently rechecked initial FP64 errors; unsaved replay/intermediate arrays are described as execution-time hash evidence.

Run each CPU-only integration test file in a fresh process, as CI does:

```sh
.venv/bin/python -m unittest discover -s integration/gpu-owned -p test_prepare_full_shape_evidence.py -q
.venv/bin/python -m unittest discover -s integration/gpu-owned -p test_report_full_shape_probe.py -q
.venv/bin/python -m unittest discover -s integration/gpu-owned -p test_audit_full_shape_probe.py -q
```

These suites cover 14 archive-contract mutations, 12 reporting/copy mutations and 31 raw-auditor checks. A successful local archive receipt means byte-preserved packaging only. Release upload and independent remote asset verification remain separate steps; use the [read-only release verifier](REMOTE-VERIFICATION.md) against the actual release and asset identifiers.

The [local packaging receipt](full-shape-evidence-archives-v1.json) records all seven archives as verified, with every member hash, size and inventory checked and all source files rechecked after packaging. The files are under `work/gpu-v4-owned/full-shape-evidence-archives-v1/`; publication status remains separate.

| Archive | Compressed bytes | SHA-256 |
|---|---:|---|
| `model-full-shape-v1-cpu-torch-b1-t32.tar.gz` | 1,014,374,610 | `4ddfbbe8150ba56949785468335e48b2c89d5301f428204b3471f6ee1b46c2dd` |
| `model-full-shape-v1-cpu-torch-b16-t32.tar.gz` | 1,049,773,364 | `c344013b963c9d9a1ec2ed9adbfa49457f707b9dc1f1212766e4e19772d37011` |
| `model-full-shape-v1-mps-b1-t32.tar.gz` | 1,014,286,695 | `08cd57ca91dec4331416f2b263a7bf4c0973c21a98a0874296d13ab6e4fd4b9d` |
| `model-full-shape-v1-mps-b16-t32.tar.gz` | 1,049,651,099 | `c2c4c08ea7c4ee314359039e2e44aabc57fa37e42bba87ce5389fc3ad17c77c4` |
| `model-full-shape-v1-mlx-b1-t32.tar.gz` | 1,014,330,520 | `2e62d4d875f819f78b94a1bbc5599e5b33d2ac137d9f9e7847dbd22e1dfba8b8` |
| `model-full-shape-v1-mlx-b16-t32.tar.gz` | 1,049,685,250 | `d779cc951dd214dd6aac7e770a4f1357890b502638b87329c146fb6ad3da194f` |
| `model-full-shape-v1-metadata.tar.gz` | 109,091 | `1681af8280cda9c100d635ea018fe006a673ad16d4c9b9211744bb79518b7217` |
