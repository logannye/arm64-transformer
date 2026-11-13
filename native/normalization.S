// Four-wide NEON LayerNorm, forward and backward.
//
// Exported routines (C prototypes in ABI.md):
//   tf_ln_forward   y = norm * gamma + beta; keeps norm and inv for backward
//   tf_ln_backward  dx from dy; ADDs to the dgamma and dbeta accumulators
//
// Both handle widths that are a multiple of four. Any other width tail-calls
// the scalar tf_ln_forward_generic or tf_ln_backward_generic in kernels.S
// with the argument registers untouched.
// Both are leaf routines: Darwin caller-saved registers only, no stack frame.
// Row reductions keep four lane sums (lane j takes the elements whose index
// is j mod 4) and combine them as (lane0 + lane1) + (lane2 + lane3).
.text
.p2align 2
// tf_ln_forward(x,gamma,beta,y,norm,inv,rows,width)
//   x0 = x      float32 [rows, width]  input
//   x1 = gamma  float32 [width]        scale
//   x2 = beta   float32 [width]        bias
//   x3 = y      float32 [rows, width]  output
//   x4 = norm   float32 [rows, width]  normalized input, kept for backward
//   x5 = inv    float32 [rows]         1 / sqrt(variance + eps), per row
//   x6 = rows, x7 = width              64-bit integers
// For each row r, with eps = 1e-5:
//   mean       = (1/width) * sum_c x[r][c]
//   var        = (1/width) * sum_c (x[r][c] - mean)^2    biased, second pass
//   inv[r]     = 1 / sqrt(var + eps)
//   norm[r][c] = (x[r][c] - mean) * inv[r]
//   y[r][c]    = norm[r][c] * gamma[c] + beta[c]
// Returns nothing. Leaf, no frame; scratch x9-x14, v0-v2, v6, v7, v16, v17.
// x7 is also overwritten (width becomes the row length in bytes).
.globl _tf_ln_forward
_tf_ln_forward:
    // Widths that are not a multiple of four go to the scalar routine.
    tst x7, #3
    b.eq Lnvf_fast
    b _tf_ln_forward_generic
Lnvf_fast:
    // s7 = 1 / width. s6 = eps = 1e-5 (bit pattern 0x3727c5ac).
    // x7 = row length in bytes. x9 = row index r.
    scvtf s7, x7
    fmov s0, #1.0
    fdiv s7, s0, s7
    mov w9, #0xc5ac
    movk w9, #0x3727, lsl #16
    fmov s6, w9
    lsl x7, x7, #2
    mov x9, #0
// Row loop: r = 0 .. rows-1.
Lnvf_row:
    // x12 = x row, x13 = y row, x14 = norm row; x10 = byte offset in the row.
    mul x11, x9, x7
    add x12, x0, x11
    add x13, x3, x11
    add x14, x4, x11
    mov x10, #0
    movi v0.4s, #0
// Pass 1: v0 = four lane sums of the row, four elements per step.
Lnvf_mean:
    ldr q1, [x12, x10]
    fadd v0.4s, v0.4s, v1.4s
    add x10, x10, #16
    cmp x10, x7
    b.lo Lnvf_mean
    // s16 = mean = ((lane0 + lane1) + (lane2 + lane3)) * (1/width), then
    // broadcast to v16. v0 is cleared for the second pass.
    faddp v0.4s, v0.4s, v0.4s
    faddp s0, v0.2s
    fmul s16, s0, s7
    dup v16.4s, v16.s[0]
    movi v0.4s, #0
    mov x10, #0
// Pass 2: v0 = four lane sums of (x - mean)^2.
Lnvf_var:
    ldr q1, [x12, x10]
    fsub v1.4s, v1.4s, v16.4s
    fmul v1.4s, v1.4s, v1.4s
    fadd v0.4s, v0.4s, v1.4s
    add x10, x10, #16
    cmp x10, x7
    b.lo Lnvf_var
    // s17 = inv[r] = 1 / sqrt(sum * (1/width) + eps), computed with fsqrt and
    // fdiv. It is stored to inv[r] and broadcast to v17.
    faddp v0.4s, v0.4s, v0.4s
    faddp s0, v0.2s
    fmul s0, s0, s7
    fadd s0, s0, s6
    fsqrt s0, s0
    fmov s1, #1.0
    fdiv s17, s1, s0
    str s17, [x5, x9, lsl #2]
    dup v17.4s, v17.s[0]
    mov x10, #0
// Pass 3: norm = (x - mean) * inv, stored; y = norm * gamma + beta, stored.
// The multiply and the add for y are separate instructions, not fused.
Lnvf_output:
    ldr q0, [x12, x10]
    fsub v0.4s, v0.4s, v16.4s
    fmul v0.4s, v0.4s, v17.4s
    str q0, [x14, x10]
    ldr q1, [x1, x10]
    ldr q2, [x2, x10]
    fmul v0.4s, v0.4s, v1.4s
    fadd v0.4s, v0.4s, v2.4s
    str q0, [x13, x10]
    add x10, x10, #16
    cmp x10, x7
    b.lo Lnvf_output
    add x9, x9, #1
    cmp x9, x6
    b.lo Lnvf_row
    ret

// tf_ln_backward(dy,norm,inv,gamma,dx,dgamma,dbeta,shape)
//   x0 = dy      float32 [rows, width]  gradient of y
//   x1 = norm    float32 [rows, width]  from tf_ln_forward
//   x2 = inv     float32 [rows]         from tf_ln_forward
//   x3 = gamma   float32 [width]
//   x4 = dx      float32 [rows, width]  overwritten
//   x5 = dgamma  float32 [width]        accumulator, added to
//   x6 = dbeta   float32 [width]        accumulator, added to
//   x7 = shape   int32 [2] = [rows, width]
// For each row r, with g[c] = dy[r][c] * gamma[c]:
//   m1 = (1/width) * sum_c g[c]
//   m2 = (1/width) * sum_c g[c] * norm[r][c]
//   dx[r][c]   = inv[r] * (g[c] - m1 - norm[r][c] * m2)
//   dgamma[c] += dy[r][c] * norm[r][c]
//   dbeta[c]  += dy[r][c]
// Returns nothing. Leaf, no frame; scratch x9-x15, v0-v3, v7, v16-v18.
.globl _tf_ln_backward
_tf_ln_backward:
    // w10 = width. Widths that are not a multiple of four go to the scalar
    // routine.
    ldr w10, [x7, #4]
    tst x10, #3
    b.eq Lnvb_fast
    b _tf_ln_backward_generic
Lnvb_fast:
    // w9 = rows. s7 = 1 / width. x10 = row length in bytes. x11 = row index r.
    ldr w9, [x7]
    scvtf s7, x10
    fmov s0, #1.0
    fdiv s7, s0, s7
    lsl x10, x10, #2
    mov x11, #0
// Row loop: r = 0 .. rows-1.
Lnvb_row:
    // x13 = dy row, x14 = norm row, x15 = dx row. v16 and v17 collect the lane
    // sums for m1 and m2. x12 is then reused as the byte offset in the row.
    mul x12, x11, x10
    add x13, x0, x12
    add x14, x1, x12
    add x15, x4, x12
    movi v16.4s, #0
    movi v17.4s, #0
    mov x12, #0
// Pass 1: v2 = g = dy * gamma; v16 += g; v17 += g * norm (fused).
// In the same step dgamma += dy * norm (fused) and dbeta += dy, in place.
Lnvb_sums:
    ldr q0, [x13, x12]
    ldr q1, [x14, x12]
    ldr q2, [x3, x12]
    fmul v2.4s, v0.4s, v2.4s
    fadd v16.4s, v16.4s, v2.4s
    fmla v17.4s, v2.4s, v1.4s
    ldr q3, [x5, x12]
    fmla v3.4s, v0.4s, v1.4s
    str q3, [x5, x12]
    ldr q3, [x6, x12]
    fadd v3.4s, v3.4s, v0.4s
    str q3, [x6, x12]
    add x12, x12, #16
    cmp x12, x10
    b.lo Lnvb_sums
    // v16 = m1 and v17 = m2, each reduced pairwise, scaled by 1/width and
    // broadcast. v18 = inv[r] broadcast.
    faddp v16.4s, v16.4s, v16.4s
    faddp s16, v16.2s
    fmul s16, s16, s7
    dup v16.4s, v16.s[0]
    faddp v17.4s, v17.4s, v17.4s
    faddp s17, v17.2s
    fmul s17, s17, s7
    dup v17.4s, v17.s[0]
    ldr s18, [x2, x11, lsl #2]
    dup v18.4s, v18.s[0]
    mov x12, #0
// Pass 2: dx = (g - m1 - norm * m2) * inv[r], with g recomputed from dy.
Lnvb_dx:
    ldr q0, [x13, x12]
    ldr q1, [x14, x12]
    ldr q2, [x3, x12]
    fmul v0.4s, v0.4s, v2.4s
    fsub v0.4s, v0.4s, v16.4s
    fmul v1.4s, v1.4s, v17.4s
    fsub v0.4s, v0.4s, v1.4s
    fmul v0.4s, v0.4s, v18.4s
    str q0, [x15, x12]
    add x12, x12, #16
    cmp x12, x10
    b.lo Lnvb_dx
    add x11, x11, #1
    cmp x11, x9
    b.lo Lnvb_row
    ret
