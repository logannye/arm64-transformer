# Closed model-v2 evidence archives

This additive plan packages the now-complete B3/T3 model development check.
The original four-group archive plan and `prepare_evidence.py` remain unchanged.
`prepare_model_evidence.py` authenticates that frozen utility before reusing its
bounded archive creation and complete member-verification routines.

The [model plan](model-evidence-plan-v1.json) has SHA-256
`eebc219327aa6be013d504c6bd77e3783dd7df93e145f1487b4d0e50163dcee7`.
Its producer SHA-256 is
`ea95ae4dd113e78e09f4037e7b52a0e29ef9b8acd2f1cda1422f5967bb67bbf5`.
It requires the exact completed execution summary, original model manifest and
passing independent array audit. It retains the broad-suite blocking state and
the explicit absence of formal fits/admission.

| Archive | Original files | Uncompressed bytes | Compressed bytes |
|---|---:|---:|---:|
| `model-compatibility-v2-cpu-torch.tar.gz` | 96 | 1,135,384,926 | 1,002,761,498 |
| `model-compatibility-v2-mps.tar.gz` | 96 | 1,135,393,466 | 1,002,621,927 |
| `model-compatibility-v2-mlx.tar.gz` | 96 | 1,135,573,866 | 1,002,671,723 |
| `model-compatibility-v2-metadata.tar.gz` | 3 | 748,156 | 55,769 |

Each engine archive contains its 12 worker directories in full and the 12
matching parent logs. The metadata archive contains the final execution
summary, frozen manifest and passing independent audit. The planner checks
that their combined inventory covers the entire closed model tree plus those
two external metadata files **exactly once**. All members retain their original
`work/gpu-v4-owned/...` relative paths, including failures referenced in the
separate preserved model-v1 archive.

The result covers 291 files, including all 216 NPZ archives, and 3,407,100,414
uncompressed bytes. That total includes the final parent summary, model
manifest and audit in addition to the worker evidence. Each archive and each
group's input remain strictly below 2,000,000,000 bytes. Source/runtime binaries
and native trees are not duplicated. The final repository source archive binds
the producer/auditor code separately.

All four local archives passed per-member SHA-256, size, mode, order, inventory
and padding verification. Source closures, contents and complete inventory
were checked again afterward. The
[packaging receipt](model-evidence-archives-v1.json) records the exact outer
archive hashes. Nine additional CPU-only tests check the engine split,
missing/failed workers, frozen utility identity, omitted/duplicate logs and
files, late extra evidence and excluded runtime paths. The frozen utility's
21 archive mutation controls remain applicable.

```sh
.venv/bin/python -m unittest discover -s integration/gpu-owned \
  -p test_prepare_model_evidence.py -v
.venv/bin/python integration/gpu-owned/prepare_model_evidence.py build \
  --workspace /ABSOLUTE/RECONSTRUCTED/WORKSPACE \
  --plan integration/gpu-owned/model-evidence-plan-v1.json \
  --plan-sha256 eebc219327aa6be013d504c6bd77e3783dd7df93e145f1487b4d0e50163dcee7 \
  --output /ABSOLUTE/FRESH/MODEL-ARCHIVE/DIRECTORY
```

This command verifies every archive before returning success. It imports no
framework, performs no numerical experiment and uploads nothing. Release
publication and streamed remote verification are separate steps governed by
[the owned-release verification contract](REMOTE-VERIFICATION.md). The
[compact model report](../../experiments/gpu-v4-owned/model-report-v2/README.md)
explains the development results and their limits.
