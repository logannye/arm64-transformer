# Optimized Metal kernel integration

`optimized.metal` is appended to the reference `kernels.metal` source before
runtime compilation. It reuses `Args` and the existing two-float arithmetic.
The reference kernels remain independently selectable. These are optimization
candidates; source implementation alone does not establish correctness or speed.

## Dispatch contract

All kernels retain `Args` at buffer 7. `M = B*T`, `H = 4`, and `D = W/H`.
Widths are the existing supported multiples of 32, context is at most 32, and
batch is at most 16. Use complete threadgroups for every cooperative kernel.

| Kernel | Buffers starting at 0 | Threadgroups | Threads per group |
|---|---|---|---|
| `v2_linear_forward` | X, weight, bias, Y | ceil(N/tileN), ceil(M/tileM) | 32*SG |
| `v2_linear_dx` | dY, weight, dX | ceil(K/tileN), ceil(M/tileM) | 32*SG |
| `v2_linear_dw` | X, dY, dW | ceil(K/tileN), ceil(N/tileM) | 32*SG |
| `v2_norm_forward` | X, gamma, beta, Y, saved norm, saved inverse | M | 32*normSG |
| `v2_norm_dx` | saved norm, saved inverse, gamma, dY, dX | M | 32*normSG |
| `v2_norm_dparam` | saved norm, dY, dGamma, dBeta | W | 32*normSG |
| `v2_linear_db` | dY, dBias | N | 32*normSG |
| `v2_attention_forward` | QKV, scores, score low, P, output | ceil(T/rows), H, B | 32*rows*attSG |
| `v2_attention_da` | QKV, dOutput, dA, dA low | ceil(B*H*T*T/rows) | 32*rows*attSG |
| `v2_attention_ds` | dA, dA low, P, dScore, dScore low | ceil(B*H*T/rows) | 32*rows*attSG |
| `v2_attention_dads` | QKV, dOutput, P, dA, dA low, dScore, dScore low | ceil(B*H*T/rows) | 32*rows*attSG |
| `v2_embedding_index` | token IDs, row indices, counts | 128 | 32 |

`v2_linear_forward_epilogue` uses the forward matrix tile dispatch and buffers
X, weight, bias, raw output, residual, fused output. `Args.step` selects ReLU (1)
or residual addition (2). `v2_reference_forward_epilogue` has the same bindings
but uses the reference 16x16 threadgroup layout and ceil(N/16), ceil(M/16)
groups. Both retain the raw output needed for tracing/backward and produce the
fused output directly from the completed accumulator. The reference variant
allows fusion to be tested independently of matrix arithmetic changes.

`v2_adam_finite` uses ordinary element dispatch over the parameter count and
buffers parameter, gradient, first moment, second moment, integer failure flag.
Reset the flag before the update. The update expressions match reference Adam;
the kernel ORs nonfinite updated parameters/moments or incoming gradients into
the flag. Logit/loss checks remain a separate host-dispatched responsibility.

`v2_embedding_grad` uses ordinary element dispatch over `128*W`, with buffers
row indices, counts, dX, dToken, dPosition. Row indices need `128*M` uint32
elements and counts need 128; allocate for maximum capacity but use the current
`M` as the row-index stride. Rebuild these indices for the current batch before
consuming them.

## Candidate parameters and numerical boundaries

Dense macros `V2_FWD_{M,N,K,SG}`, `V2_DX_*`, and `V2_DW_*` independently choose
output tile M in {16,32}, output tile N in {32,64}, inner tile K in {16,32}, and
SIMD groups in {2,4}. Defaults inherit `V2_MAT_*`. Inputs and accumulation use
FP32 SIMD-group 8x8 matrices. Loads are cooperatively guarded and zero padded;
each output tile has one owner and no floating atomics. The most demanding
listed shared allocation is 20 KiB. These routines do not replace compensated
QKV forward or dX unless independently qualified.

`V2_NORM_SG` is 1, 2, or 4 and `V2_NORM_VEC` is 1 or 4. A group owns a row or
parameter-gradient channel. All reduction stages preserve FP32 residual pairs
and have a fixed shuffle/tree order. Vector mode performs aligned `float4`
loads and stores; the supported widths guarantee complete vectors. The reduction
tree differs from the reference, so repeatability does not imply bitwise equality
to the reference. Compare against the unchanged numerical tolerances and FP64
oracle.

The same six reduction-policy choices also cover dense bias gradients. A group
owns an output channel and reduces its input rows with the fixed compensated
tree. Here `V2_NORM_VEC` controls one or four consecutive rows per lane; these
loads are row-strided, rather than contiguous `float4` loads. Bias-gradient
stages contribute to the reduction family's measured reference cost.

`V2_ATT_ROWS` and `V2_ATT_SG` are each 1 or 2. `V2_ATT_STAGE` is 0 or 1 and
controls Q-vector staging only. It does not stage full K/V matrices. Scores,
score residuals, maximum selection, centering, denominator summation and quotient
correction retain the two-float design. The fused forward retains all diagnostic
score/P writes while consuming its own intermediates locally. This removes
separate launches and global rereads, but does not eliminate trace allocations
or claim a reduction in peak workspace memory.

`v2_attention_dads` fuses the dA dot products with the softmax backward contraction.
It retains dA/dScore diagnostic writes and uses register dA values for dScore.
It does not recompute forward probabilities. If the host evaluates a policy that
recomputes forward P before backward, the repeated forward cost must be counted;
retaining the trace allocations still prevents a workspace-memory saving claim.
The compensated reference `attention_dqkv` completes the backward pass.

Embedding indexing uses only integer prefix sums. Each token's rows are emitted
in ascending order. The subsequent floating sums use precisely that order, as in
the reference, while sharing token lookup work across channels. `V2_EMBED_BLOCK`
chooses 64 or 128 input rows per iteration: each lane counts matches in its 2 or
4 consecutive rows, and one SIMD prefix sum assigns their stable output offsets.
The two candidates change the amount of local work per prefix operation.

## Required device checks

The policy search compiles the reference and all 75 isolated candidate
configurations before its first performance score. It records shader compilation,
SIMD width, required threadgroup size and static threadgroup-memory limits in a
separate eligibility receipt. Unsupported configurations remain unranked. Greedy
composition depends on earlier winners, so each subsequent family's complete
composed inventory also receives a fresh compile/resource preflight before that
family's timing. An unused unsupported optional kernel does not disqualify a
policy that does not dispatch it.

Qualification contexts are destroyed before performance measurement. Primitive
audits retain diagnostic buffers, so merely restoring parameters and optimizer
state does not restore the allocation state. Each paired comparison creates two
fresh contexts and uploads exactly the same B16T32 training and B1T32 inference
batch shapes. Before warmup, the harness records and requires equality of owned
buffer counts, workspace/input/batch counts, total owned bytes including guards,
and initial numerical state. Device-global allocation counters remain descriptive.
The final selected-bundle comparison uses the same clean-context rule, followed
by intrusive profiling in another fresh context. This harness repair leaves the
75-candidate inventory unchanged; earlier attempts remain separate evidence.

Before timing a policy, compile it on the actual device and check all three dense
directions against FP64, including a final batch with a row count below or between
tile sizes. Check all six widths and short contexts including T=1 and T=3 for
normalization and attention, causal zeros, sequence isolation, repeated-token and
absent-token gradients, and buffer sentinels. Exercise both staged-Q modes, both
row counts, each SIMD-group count and both norm vector modes. Run each candidate
twice from identical inputs/state, and retain any numerical rejection before
performance selection. Full-model forward/backward/Adam and saved-state replay
must then pass the existing contract. No compile or device validation is implied
by these integration notes.
