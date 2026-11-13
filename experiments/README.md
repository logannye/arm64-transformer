# Experiments

Every study this project ran, in the order it was added: 19 study folders, the
14 protocol and plan files that govern them, and one cross-campaign comparison.
Each folder is the public record of one study: a report, compact summaries, and
the hashes that tie them to the raw evidence.

Two folders carry the results quoted in the [root README](../README.md):

- [gpu-v5-library-phase1](gpu-v5-library-phase1/results/README.md) is the
  accepted five-engine result: 120 fits at widths 32–256, where the assembly
  trained 2.38–2.70× faster than the study's PyTorch CPU engine. That engine is
  a project-maintained build with FP64 intermediates at four stages, not a
  stock release.
- [width-sweep-v6](width-sweep-v6/README.md) is the comparison against stock
  PyTorch 2.14.1: the assembly trained 1.29–1.79× faster at widths 32–256 and
  was slower at widths 512 (0.93×) and 1024 (0.72×).

The rest is the path that led there, including the studies that stopped at
their own gates. They stay where they are because the results are bound to
these paths: summaries record file locations and SHA-256 identities, so nothing
here is moved, renamed or rewritten.

## Timeline

Dates are when each folder first entered the repository. Outcomes use each
study's own words and status values. A training ratio above 1 favors the
assembly.

| Added | Study | Protocol | Question | Outcome |
|---|---|---|---|---|
| 2026-10-28 | [matched-v1](matched-v1/README.md) | [v1](PROTOCOL.md) | Is a handwritten ARM64/NEON implementation of the width-32 model faster than ordinary PyTorch on one thread? | No. 12 fits: "ARM64 training took 2.232× as long as PyTorch". |
| 2026-10-28 | [matched-v2](matched-v2/README.md) | [v2](PROTOCOL-v2.md) | Do optimized native kernels change that, from the same initial weights and schedules? | Yes. 12 fits: "ARM64 training was 1.751× faster than PyTorch". |
| 2026-10-28 | [width-sweep-v3](width-sweep-v3/README.md) | [v3](PROTOCOL-v3.md) | How does the comparison change at widths 32, 64, 128 and 256? | 48 fits. Training ratio 1.719× at width 32, then 0.893×, 0.435× and 0.265×: PyTorch won above width 32. |
| 2026-10-28 | [kernel-profiles-v4](kernel-profiles-v4/README.md) | [v4](PROTOCOL-v4.md) | Where does the native time go, and does a matrix (SME) backend recover it? | Synthetic diagnostics, no fits. Dense linear kernels took about 57–92% of sampled execution; at width 256 the new library "improves 4.15× in forward and 6.58× in backward". |
| 2026-10-28 | [width-sweep-v4](width-sweep-v4/README.md) | [v4](PROTOCOL-v4.md) | Does the SME backend change the four-width comparison? | 48 fits. Training ratio 1.819×, 1.792×, 1.468× and 1.291× at widths 32–256. |
| 2026-10-28 | [numerics-v5](numerics-v5/README.md) | [v5](PROTOCOL-v5.md) | Do widths 512 and 1024 pass the numerical admission checks? | The initial wider implementation "failed the original admission contract". The diagnosis and a revised, prospective contract are recorded. |
| 2026-10-28 | [width-sweep-v5](width-sweep-v5/README.md) | [v5](PROTOCOL-v5.md) | Does the assembly stay competitive at widths 512 and 1024? | The report is titled "audit failed": "No validated aggregate result is claimed." |
| 2026-10-28 | [width-sweep-v5-outcome](width-sweep-v5-outcome/README.md) | [v5](PROTOCOL-v5.md) | What exactly failed in v5? | Reporting only. The 24-fit campaign "failed its frozen post-run checkpoint audit": 50 of 6,822 checks, 2 at width 512 and 48 at width 1024. |
| 2026-10-28 | [numerics-v6](numerics-v6/README.md) | [v6](PROTOCOL-v6.md) | Does keeping the wide attention scores in FP64 until they are centered remove those failures? | Replay of all 25 known failing sequences: failing whole-model logit coordinates went from 1 to 0 and attention-output coordinates from 37 to 0. |
| 2026-10-28 | [width-sweep-v6](width-sweep-v6/README.md) | [v6](PROTOCOL-v6.md) | With that change, how do all six widths compare against stock PyTorch? | 72 fits. Widths 32–256 `completed` at 1.794×–1.293×; width 512 `baseline_precision_limit` (0.928×); width 1024 `native_precision_limit` (0.719×). |
| 2026-10-28 | [gpu-development-v1](gpu-development-v1/README.md) | [gpu-v1](PROTOCOL-gpu-v1.md), [plan](GPU-PLAN-v1.md), [precision candidates](GPU-PRECISION-CANDIDATES-v1.md) | Can the same model run on the Apple GPU in handwritten Metal, matched against MPS and MLX? | "Phase closed at the strict repeatability gate on 2026-10-29." All 108 planned formal fits remain blocked and unrun; no GPU timing is claimed. |
| 2026-10-29 | [gpu-v2](gpu-v2/README.md) | [gpu-v2](PROTOCOL-gpu-v2.md) | How far can Metal kernels designed around the hardware go? | Development result: the selected kernels cut a training step by 45.6–58.3% and forward latency by 18.1–63.8% against the original Metal implementation. Five-engine study at "0 of 180 formal fits". |
| 2026-10-29 | [gpu-v3-qkv](gpu-v3-qkv/README.md) | [gpu-v3-qkv](PROTOCOL-gpu-v3-qkv.md) | Can re-tiled, precision-protected QKV kernels beat the GPU v2 choices? | All 144 candidate configurations pass their numerical checks; none meets the 3% adoption threshold, so "all six widths retain the GPU v2 kernel choices". |
| 2026-10-29 | [gpu-compile-control-v1](gpu-compile-control-v1/CORRECTION.md) | none | Were the MLX and MPS baselines running in the intended mode? | Correction. Two variables set to `"0"` switched modes by being present, so earlier baselines ran graph-disabled MLX and forced-Metal MPS. Fresh qualification required. |
| 2026-10-30 | [upstream-integration-v2](upstream-integration-v2/README.md) | none | Can deterministic reductions and attention dispatch be repaired inside PyTorch and MLX? | Patched MLX passes its declared integration checks. Patched PyTorch passes targeted checks; "its broad MPS suite remains failed". The five-engine study stayed at "0 of 180 formal fits". |
| 2026-10-30 | [gpu-v4-owned](gpu-v4-owned/README.md) | [gpu-v4-owned](PROTOCOL-gpu-v4-owned.md) | Can the study continue on project-maintained PyTorch and MLX builds, without upstream changes? | "Status: preparation, not admission. Zero owned-study formal fits have run." The library protocol is its successor. |
| 2026-10-30 | [trained-precision-v1](trained-precision-v1/STATUS.md) | [trained-precision-v1](PROTOCOL-trained-precision-v1.md) | Can selective higher precision remove the trained-state failures at widths 512 and 1024? | Development evidence, no formal fits. Native policy 15 passes all 89 historical anchors and 20 of 20 fresh batches; the framework candidates are listed with the failures they retain. |
| 2026-10-31 | [gpu-v5-library](gpu-v5-library/framework-selection.json) | [gpu-v5-library](PROTOCOL-gpu-v5-library.md) | Which framework builds, precision API and Metal shader does the five-engine comparison use? | Selection records only: the PyTorch and MLX source builds with their patches, the precision API and the reference Metal shader. A selection is not an admission (`formal_eligible: false`). |
| 2026-11-01 | [gpu-v5-library-phase1](gpu-v5-library-phase1/results/README.md) | [gpu-v5-library](PROTOCOL-gpu-v5-library.md), [amendment](gpu-v5-library-phase1/AMENDMENT.md) | Staged to widths 32–256, do all five engines pass every gate, and how do they compare? | Accepted ("First-phase scientific acceptance: True"). 120 fits; all 1,200 replays, 240 FP64 oracles, 240 checkpoint audits and 60 repeat pairs passed. Widths 512 and 1024 and their 60 fits are deferred. |

## Protocols and other files

A protocol records a study's design, numerical checks and limits. Later
protocols leave the earlier ones and their results unchanged.

| File | Scope |
|---|---|
| [PROTOCOL.md](PROTOCOL.md) | v1: the first matched comparison, width 32, PyTorch against ARM64/NEON. |
| [PROTOCOL-v2.md](PROTOCOL-v2.md) | v2: the same twelve fits with optimized native kernels. |
| [PROTOCOL-v3.md](PROTOCOL-v3.md) | v3: width sweep at 32, 64, 128 and 256. |
| [PROTOCOL-v4.md](PROTOCOL-v4.md) | v4: the SME matrix backend, same four widths. |
| [PROTOCOL-v5.md](PROTOCOL-v5.md) | v5: widths 512 and 1024. |
| [PROTOCOL-v6.md](PROTOCOL-v6.md) | v6: FP64 attention scores at the wide widths, all six widths measured. |
| [GPU-PLAN-v1.md](GPU-PLAN-v1.md) | The eight-stage plan for mapping the model to the Apple GPU. |
| [GPU-PRECISION-CANDIDATES-v1.md](GPU-PRECISION-CANDIDATES-v1.md) | The finite list of Metal precision candidates, recorded before they were built. |
| [PROTOCOL-gpu-v1.md](PROTOCOL-gpu-v1.md) | GPU v1: the prospective three-engine GPU comparison, closed at the repeatability gate. |
| [PROTOCOL-gpu-v2.md](PROTOCOL-gpu-v2.md) | GPU v2: Metal engineering and the first five-engine design, 180 planned fits. |
| [PROTOCOL-gpu-v3-qkv.md](PROTOCOL-gpu-v3-qkv.md) | GPU v3: the bounded QKV kernel search. |
| [PROTOCOL-gpu-v4-owned.md](PROTOCOL-gpu-v4-owned.md) | GPU v4: the move to project-maintained framework builds. |
| [PROTOCOL-trained-precision-v1.md](PROTOCOL-trained-precision-v1.md) | Trained-state precision development at widths 512 and 1024. |
| [PROTOCOL-gpu-v5-library.md](PROTOCOL-gpu-v5-library.md) | GPU v5: the maintained precision-library comparison that the accepted phase follows. |
| [v1-v2.json](v1-v2.json), [v1-v2.png](v1-v2.png) | The v1 and v2 campaigns side by side, drawn by [compare_versions.py](../scripts/compare_versions.py). Its recorded scope: "historical across separate campaigns; not a new paired experiment". |

## Reading a study folder

Most folders open with a `README.md`; the four that do not are linked above to
the file that states their result. In the measured studies, `summary.json`
holds the reported numbers with the source revision and the SHA-256 of the raw
evidence manifest. Raw evidence is not in this repository. Some reports link a
tagged GitHub release that holds theirs; the raw archive for the accepted
five-engine phase is local and is linked by hash only. The
[reproduction page](../docs/REPRODUCE-five-engine.md) says what that leaves a
reader able to check.
