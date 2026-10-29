# Protected QKV contractions

`qkv.metal` changes the ownership and memory layout of the protected QKV
forward and input-gradient contractions. The original `qkv_forward` and
`qkv_dx` kernels in `kernels.metal` remain independently selectable. QKV
weight gradients still use the existing, separately qualified FP32 dense
policy; this change does not alter their arithmetic.

The forward product is `Y[M,N] = X[M,K] W[N,K]^T + bias[N]`. Its backward
input product is `dX[M,K] = dY[M,N] W[N,K]`. Both protected kernels retain
the original recurrence:

```text
accumulator = (0, 0)
for k in ascending inner-dimension order:
    accumulator = ff_add(accumulator, ff_product(left[k], right[k]))
```

Forward adds bias with the original `ff_add` and rounds the expansion to FP32
once at output. Input gradients use the original final `hi + lo` rounding.
Each output has one owner, with no cross-thread floating-point reduction,
atomics, split-K sum, partial-sum reassociation, or lower-precision matrix
accumulator. Safe Metal math and precise floating-point functions remain
required. Supported QKV inner dimensions are `W` and `3W`, both multiples of
32, so K16 and K32 tiles traverse the same terms as the reference without
introducing additional padding additions.

## Register microtiles and shared operands

The frozen schedules use `M ∈ {16,32}`, `N ∈ {32,64}`, and `K ∈ {16,32}`:
eight schedules for forward and eight for dX. A thread owns two output rows
and `N/16` adjacent columns. It therefore maintains four or eight independent
two-float accumulators. Two left operands and one right operand can update
multiple outputs before the next shared-memory access. The independent
accumulation chains expose instruction-level parallelism without changing
the order inside a chain.

There are `8*M` threads per group: 128 or 256. For thread index `i`, the
microtile starts at local row `2*(i/16)` and column `(N/16)*(i%16)`. These
microtiles partition the complete output tile without overlap.

Cooperative global loads follow each operand's contiguous physical dimension.
Forward weights are output-major, so the load transposes them into shared
rows indexed by inner position and output channel. dX weights already have
that orientation. The shared right operand has row stride `N+1`; the padding
avoids the power-of-two stride during the forward transpose. Repeated
channel reads then use adjacent shared-memory locations. The left operand
uses ordinary row-major `M*K` storage.

The exact static shared allocation is `4*K*(M+N+1)` bytes. The largest
candidate uses 12,416 bytes. Four or eight two-float accumulators require
eight or sixteen scalar accumulator components per thread, before compiler
temporaries. That is a source-level count, not a measured register allocation
or occupancy claim.

## Barriers, tails, and dispatch

Every thread participates in the same inner-tile loop. Global loads are
guarded and write zero for inactive rows or columns. A threadgroup barrier
follows loading; another follows all consumption before the tile is reused.
There is no early return before either barrier. Only final stores branch on
the actual output shape. The padded shared column is never read or written.
Groups cover `ceil(output_columns/N)` by `ceil(M_actual/M)` tiles.

The new entry points are `v3_qkv_forward` and `v3_qkv_dx`. They retain the
reference buffer bindings and `Args` at buffer 7. Compile-time controls are
`V3_QKV_FWD_{M,N,K}` and `V3_QKV_DX_{M,N,K}`. Pipeline resource limits are
checked independently of numerical validation; a rejected resource schedule
cannot receive a timing rank.

## Versioned policy ABI

`tfm_create` and `tfm_create_v2` keep their existing interfaces. The new
`tfm_create_v3` receives the v2 arguments plus a final `uint32_t controlCount`,
which must be exactly 54. Each mode has 27 words: the unchanged 21 v2 words,
followed by forward `(M,N,K)` and dX `(M,N,K)`. Training precedes inference.
An all-zero triple selects the original protected kernel; partial-zero
triples and values outside the frozen schedule grid are rejected.

The v3 capability receipt records both selected QKV triples, compiled pipeline
resource limits, and the output-ownership description. The Python interface
also binds the complete policy and shader/host identities. Schema-2 policy
values and their identities are not rewritten.

## Required evidence

Each candidate must match the original protected output bytes and preserve
the existing FP64 numerical gates, input immutability, buffer guards, causal
behavior, and full-model checks. Partial rows and N64 column tails are
included. Full-step and B1-forward comparisons must use fresh contexts with
identical state and allocation ownership, and must compare against the
published selected v2 policy at that width. Microkernel timing alone cannot
select a policy. Source-level reuse and parallelism describe the design;
only retained matched measurements establish a performance effect.
