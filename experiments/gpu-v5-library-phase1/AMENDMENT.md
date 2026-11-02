# Maintained-library comparison: staged completion amendment

Protocol: `gpu-v5-library-five-engine-v4-phase1`. Approved November 1, 2026,
after development qualification and partial reserved admission, with **zero
formal fits completed**. This scope decision is informed by that existing
evidence; it is not an untouched, prospectively selected six-width study.

The first accepted report covers widths **32, 64, 128, and 256**. Widths **512
and 1024** and their 60 formal fits are deferred to a separately qualified second
phase. First-phase acceptance does not imply that the original study, either
deferred width, or a standard framework release has passed.

## Unchanged scientific contract

The [original comparison contract](../../integration/library-study-v1/comparison-contract.json)
is preserved byte for byte (SHA-256
`25b5847326ef7aee4a6580423a49c2812006cbfb6aa8ad84a365930015591714`).
Its original 180-cell rotation is filtered by width, preserving the order and
identity of every retained cell and admission group. The original frozen
six-width manifests and all positive and negative campaign evidence remain
retained. The last completed six-width qualification used control `143c58c`;
its frozen manifest SHA-256 is
`baa0ff69dd31e928ad5d604e131ff280ce9eb3c3dfab91347d24b163861919c3`.

The equations, five selected engines (`cpu_arm64`, `cpu_torch`, `metal`, `mps`,
`mlx`), maintained framework builds, and approved CPU FF1 correction are
unchanged. Mathematical source remains
`bd2cd2823038729779a217ee5b084fa06191d3a9`. Fixtures, seeds 42/43/44, two repeats,
eight epochs, 1,024 updates, optimizer, checkpoint selection, tolerances, timing
boundaries, warmups, latency samples, cooldowns, and power checks are unchanged.
Reserved admission seed 20261112 remains the same already-exposed stream.

## First-phase denominators and acceptance

| Required evidence | First phase | Derivation |
|---|---:|---|
| Formal fits | 120 | 4 widths × 5 engines × 3 seeds × 2 repeats |
| Admission groups | 1,392 | All 348 original groups at each retained width |
| Blocking admission groups | 1,032 | All 258 blocking groups at each retained width |
| Admission members | 6,572 | Exact retained inventory, including diagnostic negatives |
| Best/last checkpoints | 240 | Both checkpoints from every fit |
| Selected-engine checkpoint replays | 1,200 | Every checkpoint through all five engines |
| Existing untimed replay batches | 20 | 4 widths × 5 engines, 60 replays each |
| Independent FP64 oracle workers | 240 | One per checkpoint, original Torch FP64 anchor rows |
| Independent checkpoint auditors | 240 | Original NumPy FP64 full-split checks and all comparisons |
| Exact formal repeat pairs | 60 | 4 widths × 5 engines × 3 seeds |
| Audited best-checkpoint test predictions | 983,040 | 120 × 8,192 tokens |

All 56 prerequisite raw audit jobs belonging to retained widths must pass,
alongside the unchanged runtime/source authentication, hosted Python/native
checks and full-suite audit. The 232 already-completed wide prerequisite jobs
remain retained development evidence, outside first-phase qualification. They
are not converted into first-phase or second-phase formal results.

Widths 32/64/128 have completed admission workers. Reuse is restricted to their
recorded bindings and requires fresh independent audits using their original
source, with identical numerical source, runtime, fixtures, native assets and
hardware. Only this documented scope amendment and the diagnosed diagnostic
isolation control changes are permitted across that reuse boundary. Required
fresh-process repeat and restart evidence is retained and re-audited.

Width 256 must finish admission before any first-phase fit starts. Its ordinary
MLX single-token SDPA backward diagnostic produced a native buffer-offset abort.
An isolated probe reproduced the abort; the adjacent three-token probe passed.
The affected nonblocking diagnostic runs in a separate authenticated process,
with the crash retained as a negative. It is not removed from the denominator,
and no selected model implementation or tolerance changes. Unrecognized process,
storage, source or provenance failures remain blocking.

Timed fits and required isolated repeat/restart checks remain serial. Existing
untimed replay batching is retained. Acceptance requires all 120 fits and every
listed final audit to pass; a required negative result stops acceptance. The
first report must name this four-width scope and retain the deferred study status.

## Deferred work and preserved negatives

The original wide-width investigation remains open. Existing ordinary-framework
failures, historical unqualified timings, the four retained nonblocking native
and Metal trajectory negatives at widths 512/1024, and the MLX diagnostic abort
remain negative evidence. Deferral does not repair or qualify them. Completing
the original study requires the remaining admission, 60 formal fits, and their
full checkpoint, oracle, repeat and final audits under an explicit second-phase
authority. No second-phase acceptance is claimed here.

At this amendment, first-phase formal progress is **0/120**, with **60 fits
deferred**. The controller maintains a 16 GiB free-space floor; cleanup preserves
bound evidence and useful failures and targets only identified inactive,
regenerable artifacts.
