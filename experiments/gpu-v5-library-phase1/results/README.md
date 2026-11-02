# Maintained-library five-engine transformer comparison

Project-maintained five-engine FP32-storage transformer; original ASCII, T32, B16 workload; resident B1 forward latency; phase 1 widths 32–256 only. Widths 512/1024 and 60 fits are deferred, not accepted. Original study incomplete.

First-phase scientific acceptance: **True**. Audited: 120 fits, 240 checkpoints, 60 repeat pairs and 983,040 test predictions.

All model state uses FP32 storage. The selected CPU and GPU precision operators use the internal arithmetic declared in the frozen model selection. Framework builds are project-maintained. Timing applies to this hardware and workload.

Each row is the median of six fits across three paired seeds and two repeats. Training throughput includes the frozen training boundary; latency measures resident B1 forward and completion, with 20 warmups and 100 samples.

| Width | Engine | Training targets/s | B1 p50 ms | B1 p95 ms | Test NLL |
|---:|---|---:|---:|---:|---:|
| 32 | cpu_arm64 | 692120.01 | 0.016 | 0.016 | 2.473305 |
| 32 | cpu_torch | 274339.40 | 0.122 | 0.125 | 2.453750 |
| 32 | metal | 372577.25 | 0.265 | 0.295 | 2.452584 |
| 32 | mps | 8383.85 | 7.948 | 8.374 | 2.472005 |
| 32 | mlx | 24135.91 | 2.080 | 2.228 | 2.458924 |
| 64 | cpu_arm64 | 585686.01 | 0.020 | 0.020 | 2.413756 |
| 64 | cpu_torch | 216955.20 | 0.147 | 0.152 | 2.411997 |
| 64 | metal | 332720.88 | 0.315 | 0.346 | 2.412701 |
| 64 | mps | 6145.63 | 9.249 | 9.693 | 2.411570 |
| 64 | mlx | 16023.05 | 2.469 | 2.708 | 2.403599 |
| 128 | cpu_arm64 | 319719.90 | 0.036 | 0.038 | 2.377682 |
| 128 | cpu_torch | 127066.69 | 0.187 | 0.203 | 2.399026 |
| 128 | metal | 256331.72 | 0.411 | 0.440 | 2.379816 |
| 128 | mps | 3146.69 | 12.468 | 12.952 | 2.381126 |
| 128 | mlx | 7554.71 | 3.354 | 3.741 | 2.390828 |
| 256 | cpu_arm64 | 131697.27 | 0.082 | 0.087 | 2.386181 |
| 256 | cpu_torch | 55739.25 | 0.329 | 0.354 | 2.395190 |
| 256 | metal | 145910.20 | 0.626 | 0.663 | 2.383403 |
| 256 | mps | 1530.92 | 19.694 | 20.404 | 2.391122 |
| 256 | mlx | 3115.36 | 5.901 | 6.649 | 2.384476 |

## Matched CPU comparison

Speedups below are the median of the three paired-seed ratios after taking the
median of each seed's two repeats. The denominator is the selected maintained
PyTorch CPU arm. Values greater than 1 mean ARM64 CPU is faster.

| Width | ARM64 CPU training speedup | ARM64 CPU resident B1 speedup |
|---:|---:|---:|
| 32 | 2.53× | 7.52× |
| 64 | 2.70× | 7.28× |
| 128 | 2.48× | 5.16× |
| 256 | 2.38× | 3.99× |

ARM64 CPU has the lowest median training time at widths 32, 64 and 128. Metal
has the lowest median training time at width 256 (3.593 s versus 3.981 s for
ARM64 CPU). ARM64 CPU has the lowest resident B1 latency at every included
width. The selected MPS and MLX arms are slower than the CPU arms on this
workload. These measurements use project-maintained frameworks and the frozen
precision operators; standard framework releases remain outside this acceptance.

Test NLL and accuracy differ across trained engines. The audit checks every
saved checkpoint on all five engines against independent FP64 equations and
cross-engine tolerances. It does not require identical training trajectories
across engines. All 60 same-engine repeat pairs are exact.

## Scope and validation

The accepted phase contains 120 fits: four widths, five engines, three seeds,
and two repeats. All 1,200 selected-engine checkpoint replays, 240 independent
FP64 oracles, 240 checkpoint audits and 60 final repeat checks passed, covering
983,040 audited best-model test-token predictions. Required qualification
and admission preceded fitting. Each checkpoint retained its own comparison
against both independent FP64 equations and the other selected engines.

The hardware is an Apple M4 Max with 128 GiB memory, running macOS 26.2.
Training uses B16, context 32, eight epochs and 1,024 updates. The frozen timer
includes forward/backward/Adam, finite checks, loss reads, validation, checkpoint
writes, packing, synchronization and bookkeeping. B1 latency uses resident
inputs and outputs, forward execution and completion wait, 20 warmups and
100 samples. Scope, precision stages, per-seed ratios and complete numeric
values are in [summary.json](summary.json); all 20 arm/width medians are also
available as [measurements.csv](measurements.csv).

Widths **512 and 1024 and their 60 fits remain deferred**. The original 180-fit
study is incomplete. The [scope amendment](../AMENDMENT.md) was approved after
development and partial admission, with zero formal fits; this is a staged
result, not retrospective qualification of the deferred widths.

## Failures retained

There were zero failed formal fits and zero failed final checkpoint or repeat
audits in this phase. Two execution failures required recovery:

- The first replay attempt repeated PyTorch process-wide interop-thread setup.
  The narrow audit-only correction was validated at its exact hosted revision,
  followed by a real two-checkpoint preflight. The original measurements stayed
  bound to their original source revision.
- After all 1,200 selected-engine replays completed, the coordinator supplied
  engine maps in execution order while the unchanged checkpoint auditor required
  lexical order. Fresh job files reordered the same values. All completed
  engine replays and the first independent oracle were retained and reused;
  the frozen source, numerical calculations and tolerances stayed unchanged.

Both failures occurred before the affected numerical comparisons. Original
failure records, source-bound checkpoints and raw evidence are preserved.
The final disk watchdog observed at least 47.28 GiB free, above the 16 GiB floor;
no cleanup was necessary during this audit continuation.

Earlier development negatives remain part of the record. Ordinary, unselected
MLX attention-backward diagnostics retain a Metal buffer-offset assertion,
including width 256. Those diagnostics are distinct from the selected maintained
precision arm accepted here. Wider-width evidence and standard-release study
failures retain their existing status; this result does not qualify them.

## Evidence identity

The measured mathematical source is `bd2cd2823038729779a217ee5b084fa06191d3a9`;
measurement control is `30390c80573baf687f9287f2b1921c79d5bc67c0`; audit control is
`f1de5c69b77e634a10970d4a4e3dc6e6cff2bd4d`. Native builds were performed on hosted
CI. The final operational continuation changed job ordering only and used the
unchanged auditors and original report implementation.

The public files are bounded summaries. Full source-bound evidence remains in
the retained local campaign archive. SHA-256 identities in `summary.json` link
the summary to that evidence; hashes establish byte identity, not independent
public reproduction of the entire archive.

Accepted report SHA-256: `8c6b3d842b9108195a4d6f6354c96aee0d73b70e59a88697c2f338ea813ac592`.

Final campaign audit SHA-256: `d14de612258434d05def0b9e927f835cb7eac75b82dd079bae00f6d990b7844e`.
