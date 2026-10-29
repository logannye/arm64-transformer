# Protected QKV tiling: complete development results

**The new kernels preserve precision, but none improves complete-operation
latency enough to replace the existing policy.** The bounded search is complete
at widths 32, 64, 128, 256, 512 and 1024. All 144 candidate configurations pass
their numerical checks; all six widths retain the GPU v2 kernel choices.

## What changed

The new Metal forward and input-gradient kernels cooperatively load padded,
transposed tiles and assign four or eight outputs to each thread. This exposes
independent accumulation chains and reuses operands across outputs. Each
individual output retains the original ascending `ff_product`/`ff_add` order,
bias operation and final FP32 rounding. No atomics, split-K reduction, storage
precision change or tolerance relaxation is introduced. QKV weight gradients
keep their existing separately qualified dense kernel.

The fixed grid has M={16,32}, N={32,64}, K={16,32}: eight forward and eight dX
schedules, with dX excluded from inference. Every candidate is qualified before
timing. Selection uses complete B16/T32 training steps and B1/T32 forward passes,
20 warmups and 100 observations in five alternating paired blocks. A candidate
must reduce median paired latency by at least 3%. No candidate grid expansion
followed the measurements.

## Results

Positive values below mean lower latency; negative values mean a slowdown.
These are the best observed candidates within each mode, including candidates
that do not meet the adoption threshold. They are not selected speedups.

| Width | Best training latency reduction | Best forward latency reduction | Decision |
|---:|---:|---:|---|
| 32 | −0.21% | +2.83% | Retain incumbent |
| 64 | +2.10% | −5.04% | Retain incumbent |
| 128 | +0.91% | −4.69% | Retain incumbent |
| 256 | +0.14% | −7.00% | Retain incumbent |
| 512 | +1.63% | −8.90% | Retain incumbent |
| 1024 | +2.41% | −1.03% | Retain incumbent |

![Every measured QKV candidate](results/qkv-candidate-grid.png)

At width 512, the best training candidate changes dX with a 16/32/32 tile.
At width 1024, it changes dX with a 32/32/32 tile. Forward tiling contributes no
qualifying improvement. Some wider batch-one tiles are substantially slower:
the 16/64/32 schedule raises width-1024 forward latency by 37.3%.

The arithmetic workload is unchanged. More reuse also means more live
accumulators, fewer independent output owners and larger coordinated groups.
For B1/T32 at width 1024, a 32×64 output tile launches 48 QKV forward groups;
the retained 16×16 tile launches 384. This is a structural reason to suspect
insufficient parallel work or resource pressure, especially for small batches.
We did not collect hardware occupancy, register or bandwidth counters, so those
mechanisms are hypotheses rather than measured diagnoses.

The final selected-versus-incumbent comparison consequently measures the same
kernel choices twice. Small differences around 1× describe timing variation,
not a newly achieved speedup. The [full report](results/README.md) provides
p50/p95, throughput, paired-block ranges, intrusive stage profiles, and all
candidate outcomes. Instrumented stage costs are separate from normal timings.

## Qualification and provenance

The unchanged forward, gradient, probability/math and independent-FP64 gates
pass. Direct tests additionally require exact output, dX, dW and bias-gradient
bytes against the original protected kernels, input immutability, finite state
and intact guards. The selected bundles match original schema-2 full-model
forward, loss, every gradient and first-Adam-state bytes. The schema-3 null-QKV
incumbent gives both timing instances matching pipeline and buffer inventories.

Final checks cover 53 retained stress cases, including all 23 attention and 22
LayerNorm/QKV families, plus 16 direct QKV cases at every width. Fresh-process
Metal replay passes in 12 workers: two processes per width, 12 cases per worker,
and ten identical executions per case. Full gradients, six-step parameters and
moments, and checkpoint predictions repeat exactly. This is development
qualification, not formal admission or a full checkpoint study.

The numerical source freeze is
[`aac359f`](https://github.com/logannye/arm64-transformer/commit/aac359fb54b19334a532ca3f0a1bd084545836fe).
The six searches share that source, binary and runtime identity. The
[protocol](../PROTOCOL-gpu-v3-qkv.md), [candidate ledger](candidates.json),
[selected policies](policies/selected.json), [kernel design](../../native/metal/QKV-KERNELS.md),
and [reproduction guide](REPRODUCE.md) describe the experiment. Local regression
validation passed 705 tests with 14 tests skipped; real Metal
checks run separately. The native cleanup ownership probe also passes.

## What remains blocked

The requested complete standard-Python/native CPU/GPU comparison is **not
complete: 0/180 new formal fits have run**. Stable MPS/MLX baseline requirements
are unchanged. Official tagged-source inspection found no qualifying repair;
the previously prepared local patches cannot substitute for released standard
baselines. The [feasibility receipt](baseline-feasibility.json) distinguishes
that inspection from runtime validation.

Unblocking requires eligible supported framework releases, passing baseline
checks, a fresh source/runtime/policy freeze and admission, then 72 fresh CPU
and 108 GPU fits with all-checkpoint five-engine and FP64 audits. Historical CPU
wide-model accuracy failures remain recorded. Neither the reserved admission
stream nor original validation/test arrays were opened by this QKV phase.

The [evidence release](https://github.com/logannye/arm64-transformer/releases/tag/transformer-gpu-qkv-v3)
retains raw states and logits, timings, candidate and failure evidence, runtime
identities, measured source snapshots and archive verification. The historical
[GPU v2 release](https://github.com/logannye/arm64-transformer/releases/tag/transformer-gpu-engineering-v2)
remains unchanged.
