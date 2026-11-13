// Handwritten Darwin AArch64 numerical primitives. See ABI.md/kernel_notes.md.
// No numerical library calls. x18 and callee-saved SIMD v8-v15 are unused.
//
// This file holds the scalar and plain four-lane NEON kernels. The seven
// `_generic` entries are the paths that linear.S, normalization.S,
// attention.S and softmax.S branch to for shapes they do not specialize;
// ReLU, Adam and the scalar tf_exp and tf_log are defined only here.
//
//   tf_linear_forward_generic      y = x @ w^T + b
//   tf_linear_backward_generic     dx, dw and db of the same layer
//   tf_ln_forward_generic          LayerNorm over the last dimension
//   tf_ln_backward_generic         LayerNorm derivative
//   tf_attention_forward_generic   causal scaled dot-product attention
//   tf_attention_backward_generic  its derivative with respect to Q, K, V
//   tf_relu_forward                y = max(x, 0)
//   tf_relu_backward               dx = dy where x > 0, otherwise 0
//   tf_cross_entropy_generic       mean token loss and its logit gradient
//   tf_adam                        one Adam step over a flat array
//   tf_exp, tf_log                 float32 exponential and natural logarithm
//
// Conventions: arrays are contiguous row-major float32, so element [r][c]
// of an [R,C] array is at byte offset 4*(r*C + c). Shape arrays and targets
// are int32; a 32-bit load into wN zero-extends, so xN is then used as a
// 64-bit count. Except where a count is tested first (ReLU, Adam, and the
// vector/tail splits), loops test at the bottom, so row and feature
// dimensions must be at least one, as kernel_notes.md requires of callers.
// The leaf routines use only argument and scratch registers (x0-x17, v0-v7,
// v16-v21). The two attention kernels and cross-entropy build a frame and
// save x19-x28.
.text
.p2align 2
// CONST dst, label: dst = address of a table in __TEXT,__const (ADRP page
// address plus page offset).
.macro CONST dst, label
    adrp \dst, \label@PAGE
    add \dst, \dst, \label@PAGEOFF
.endm

// tf_linear_forward_generic: generic kernel behind
// tf_linear_forward(x,w,b,y,shape). tf_linear_forward_neon (linear.S)
// tail-calls here when R, I or O is not a multiple of four.
//   x0 = x      float32 [R,I]
//   x1 = w      float32 [O,I]
//   x2 = b      float32 [O]
//   x3 = y      float32 [R,O]   overwritten
//   x4 = shape  int32 [3] = {R, I, O}
// Computes y[r][o] = b[o] + sum over i of x[r][i] * w[o][i].
// Each dot product keeps four partial sums in the lanes of v0: over the
// first I - I % 4 inputs, lane l sums the i with i % 4 == l. They are
// combined as (lane0 + lane1) + (lane2 + lane3); the remaining I % 4
// products are then added one at a time with FMADD, and the bias last.
// Leaf, no frame. w9 = R, w10 = I, w11 = O, x12 = r, x13 = o.
.globl _tf_linear_forward_generic
_tf_linear_forward_generic:
    ldr w9, [x4]
    ldr w10, [x4, #4]
    ldr w11, [x4, #8]
    mov x12, #0
// Row loop: r = x12 runs over 0..R-1.
Llf_row:
    mov x13, #0
// Output loop: o = x13 runs over 0..O-1. x14 = &x[r][0], x15 = &w[o][0].
Llf_out:
    mul x14, x12, x10
    add x14, x0, x14, lsl #2
    mul x15, x13, x10
    add x15, x1, x15, lsl #2
    movi v0.4s, #0
    lsr x17, x10, #2
    cbz x17, Llf_reduce
// I / 4 iterations: v0 += x[r][i..i+3] * w[o][i..i+3], lane by lane.
Llf_vec:
    ldr q1, [x14], #16
    ldr q2, [x15], #16
    fmla v0.4s, v1.4s, v2.4s
    subs x17, x17, #1
    b.ne Llf_vec
// s0 = (lane0 + lane1) + (lane2 + lane3), then the I % 4 scalar tail.
Llf_reduce:
    faddp v0.4s, v0.4s, v0.4s
    faddp s0, v0.2s
    and x17, x10, #3
    cbz x17, Llf_bias
Llf_tail:
    ldr s1, [x14], #4
    ldr s2, [x15], #4
    fmadd s0, s1, s2, s0
    subs x17, x17, #1
    b.ne Llf_tail
// y[r][o] = s0 + b[o]; x16 = r*O + o indexes y.
Llf_bias:
    ldr s1, [x2, x13, lsl #2]
    fadd s0, s0, s1
    madd x16, x12, x11, x13
    str s0, [x3, x16, lsl #2]
    add x13, x13, #1
    cmp x13, x11
    b.lo Llf_out
    add x12, x12, #1
    cmp x12, x9
    b.lo Llf_row
    ret

// tf_linear_backward_generic: generic kernel behind
// tf_linear_backward(x,w,dy,dx,dw,db,shape). tf_linear_backward_neon
// (linear.S) tail-calls here when R or O is not a multiple of four or I is
// not a multiple of sixteen.
//   x0 = x      float32 [R,I]   forward input
//   x1 = w      float32 [O,I]
//   x2 = dy     float32 [R,O]
//   x3 = dx     float32 [R,I]   overwritten
//   x4 = dw     float32 [O,I]   added to
//   x5 = db     float32 [O]     added to
//   x6 = shape  int32 [3] = {R, I, O}
// For each row r in order, after zeroing dx[r], and each output o in order:
//   db[o]    += dy[r][o]
//   dx[r][i] += dy[r][o] * w[o][i]   for every i
//   dw[o][i] += dy[r][o] * x[r][i]   for every i
// So each dx element accumulates over o in increasing order, and each dw
// and db element over r in increasing order on top of the caller's value.
// Overwrite dx; ADD dw and db. All dimensions support scalar tails: four i
// per FMLA, then I % 4 scalar FMADDs.
// Leaf, no frame. w9 = R, w10 = I, w11 = O, x12 = r, x13 = o. x6 is reused
// as the dw row pointer once the shape has been read; x7 = &dy[r][0].
.globl _tf_linear_backward_generic
_tf_linear_backward_generic:
    ldr w9, [x6]
    ldr w10, [x6, #4]
    ldr w11, [x6, #8]
    mov x12, #0
// Row loop: r = x12 runs over 0..R-1.
Llb_row:
    mul x8, x12, x10
    add x15, x3, x8, lsl #2
    mov x17, x10
// I iterations: zero dx[r] so the output loop can accumulate into it.
Llb_zero:
    str wzr, [x15], #4
    subs x17, x17, #1
    b.ne Llb_zero
    mul x8, x12, x11
    add x7, x2, x8, lsl #2
    mov x13, #0
// Output loop: o = x13 runs over 0..O-1. db[o] += dy[r][o], then
// v0 = dy[r][o] in all four lanes. x14 = &x[r][0], x15 = &dx[r][0],
// x16 = &w[o][0], x6 = &dw[o][0].
Llb_out:
    ldr s0, [x7, x13, lsl #2]
    ldr s1, [x5, x13, lsl #2]
    fadd s1, s1, s0
    str s1, [x5, x13, lsl #2]
    dup v0.4s, v0.s[0]
    mul x8, x12, x10
    add x14, x0, x8, lsl #2
    add x15, x3, x8, lsl #2
    mul x8, x13, x10
    add x16, x1, x8, lsl #2
    add x6, x4, x8, lsl #2
    lsr x17, x10, #2
    cbz x17, Llb_tail_start
// I / 4 iterations: dx[r][i..i+3] += dy * w[o][i..i+3] and
// dw[o][i..i+3] += dy * x[r][i..i+3].
Llb_vec:
    ldr q1, [x14], #16
    ldr q2, [x16], #16
    ldr q3, [x15]
    ldr q4, [x6]
    fmla v3.4s, v0.4s, v2.4s
    fmla v4.4s, v0.4s, v1.4s
    str q3, [x15], #16
    str q4, [x6], #16
    subs x17, x17, #1
    b.ne Llb_vec
// I % 4 remaining inputs, one at a time; s0 is still dy[r][o].
Llb_tail_start:
    and x17, x10, #3
    cbz x17, Llb_next_out
Llb_tail:
    ldr s1, [x14], #4
    ldr s2, [x16], #4
    ldr s3, [x15]
    ldr s4, [x6]
    fmadd s3, s0, s2, s3
    fmadd s4, s0, s1, s4
    str s3, [x15], #4
    str s4, [x6], #4
    subs x17, x17, #1
    b.ne Llb_tail
Llb_next_out:
    add x13, x13, #1
    cmp x13, x11
    b.lo Llb_out
    add x12, x12, #1
    cmp x12, x9
    b.lo Llb_row
    ret

// tf_ln_forward_generic: scalar kernel behind
// tf_ln_forward(x,gamma,beta,y,norm,inv,rows,width). tf_ln_forward
// (normalization.S) branches here when width is not a multiple of four.
//   x0 = x      float32 [rows,width]
//   x1 = gamma  float32 [width]
//   x2 = beta   float32 [width]
//   x3 = y      float32 [rows,width]   overwritten
//   x4 = norm   float32 [rows,width]   overwritten
//   x5 = inv    float32 [rows]         overwritten
//   x6 = rows, x7 = width (64-bit integers)
// For each row r:
//   mean       = (sum over j of x[r][j]) / width
//   var        = (sum over j of (x[r][j] - mean)^2) / width
//   inv[r]     = 1 / sqrt(var + 1e-5)
//   norm[r][j] = (x[r][j] - mean) * inv[r]
//   y[r][j]    = norm[r][j] * gamma[j] + beta[j]
// Two-pass biased variance, as kernel_notes.md describes. Each sum is one
// scalar float32 chain in j order.
// Leaf, no frame. s7 = float(width), s6 = 1e-5, x9 = r, x10 = j,
// s16 = mean, s17 = inv[r].
.globl _tf_ln_forward_generic
_tf_ln_forward_generic:
    scvtf s7, x7
    mov w9, #0xc5ac
    movk w9, #0x3727, lsl #16 // float32 1e-5
    fmov s6, w9
    mov x9, #0
// Row loop: r = x9 runs over 0..rows-1. x12 = &x[r][0], x13 = &y[r][0],
// x14 = &norm[r][0].
Llnf_row:
    mul x11, x9, x7
    add x12, x0, x11, lsl #2
    add x13, x3, x11, lsl #2
    add x14, x4, x11, lsl #2
    mov x10, #0
    fmov s0, wzr
// width iterations: s0 = sum of x[r][j].
Llnf_mean:
    ldr s1, [x12, x10, lsl #2]
    fadd s0, s0, s1
    add x10, x10, #1
    cmp x10, x7
    b.lo Llnf_mean
    fdiv s16, s0, s7
    fmov s0, wzr
    mov x10, #0
// width iterations: s0 = sum of (x[r][j] - mean)^2.
Llnf_var:
    ldr s1, [x12, x10, lsl #2]
    fsub s1, s1, s16
    fmul s1, s1, s1
    fadd s0, s0, s1
    add x10, x10, #1
    cmp x10, x7
    b.lo Llnf_var
    // inv[r] = 1 / sqrt(var + 1e-5), with FSQRT and FDIV.
    fdiv s0, s0, s7
    fadd s0, s0, s6
    fsqrt s0, s0
    fmov s1, #1.0
    fdiv s17, s1, s0
    str s17, [x5, x9, lsl #2]
    mov x10, #0
// width iterations: write norm[r][j] and y[r][j].
Llnf_normalize:
    ldr s0, [x12, x10, lsl #2]
    fsub s0, s0, s16
    fmul s0, s0, s17
    str s0, [x14, x10, lsl #2]
    ldr s1, [x1, x10, lsl #2]
    ldr s2, [x2, x10, lsl #2]
    fmul s0, s0, s1
    fadd s0, s0, s2
    str s0, [x13, x10, lsl #2]
    add x10, x10, #1
    cmp x10, x7
    b.lo Llnf_normalize
    add x9, x9, #1
    cmp x9, x6
    b.lo Llnf_row
    ret

// tf_ln_backward_generic: scalar kernel behind
// tf_ln_backward(dy,norm,inv,gamma,dx,dgamma,dbeta,shape). tf_ln_backward
// (normalization.S) branches here when the width is not a multiple of four.
//   x0 = dy      float32 [R,D]
//   x1 = norm    float32 [R,D]   from tf_ln_forward
//   x2 = inv     float32 [R]     from tf_ln_forward
//   x3 = gamma   float32 [D]
//   x4 = dx      float32 [R,D]   overwritten
//   x5 = dgamma  float32 [D]     added to
//   x6 = dbeta   float32 [D]     added to
//   x7 = shape   int32 [2] = {R, D}
// For each row r, with g[j] = dy[r][j] * gamma[j]:
//   dgamma[j] += dy[r][j] * norm[r][j]
//   dbeta[j]  += dy[r][j]
//   m1 = (sum over j of g[j]) / D
//   m2 = (sum over j of g[j] * norm[r][j]) / D
//   dx[r][j] = inv[r] * (g[j] - m1 - norm[r][j] * m2)
// Leaf, no frame. w9 = R, w10 = D, s7 = float(D), x11 = r, s16 = m1,
// s17 = m2, s18 = inv[r]. x12 = j inside the two D loops; at the top of
// each row it first holds the row offset r*D.
.globl _tf_ln_backward_generic
_tf_ln_backward_generic:
    ldr w9, [x7]
    ldr w10, [x7, #4]
    scvtf s7, x10
    mov x11, #0
// Row loop: r = x11 runs over 0..R-1. x13 = &dy[r][0], x14 = &norm[r][0],
// x15 = &dx[r][0].
Llnb_row:
    mul x12, x11, x10
    add x13, x0, x12, lsl #2
    add x14, x1, x12, lsl #2
    add x15, x4, x12, lsl #2
    fmov s16, wzr
    fmov s17, wzr
    mov x12, #0
// D iterations: the two sums in s16 and s17, and the dgamma and dbeta
// accumulation.
Llnb_sums:
    ldr s0, [x13, x12, lsl #2]
    ldr s1, [x14, x12, lsl #2]
    ldr s2, [x3, x12, lsl #2]
    fmul s2, s0, s2
    fadd s16, s16, s2
    fmadd s17, s2, s1, s17
    ldr s3, [x5, x12, lsl #2]
    fmadd s3, s0, s1, s3
    str s3, [x5, x12, lsl #2]
    ldr s3, [x6, x12, lsl #2]
    fadd s3, s3, s0
    str s3, [x6, x12, lsl #2]
    add x12, x12, #1
    cmp x12, x10
    b.lo Llnb_sums
    fdiv s16, s16, s7
    fdiv s17, s17, s7
    ldr s18, [x2, x11, lsl #2]
    mov x12, #0
// D iterations: dx[r][j].
Llnb_dx:
    ldr s0, [x13, x12, lsl #2]
    ldr s1, [x14, x12, lsl #2]
    ldr s2, [x3, x12, lsl #2]
    fmul s0, s0, s2
    fsub s0, s0, s16
    fmul s1, s1, s17
    fsub s0, s0, s1
    fmul s0, s0, s18
    str s0, [x15, x12, lsl #2]
    add x12, x12, #1
    cmp x12, x10
    b.lo Llnb_dx
    add x11, x11, #1
    cmp x11, x9
    b.lo Llnb_row
    ret

// tf_attention_forward_generic: scalar kernel behind
// tf_attention_forward(qkv,out,probs,shape). tf_attention_forward
// (attention.S) branches here for shapes it does not specialize.
//   x0 = qkv    float32 [T,3D]    each row is Q, then K, then V, D floats each
//   x1 = out    float32 [T,D]     overwritten
//   x2 = probs  float32 [H,T,T]   overwritten; entries with j > t are zero
//   x3 = shape  int32 [3] = {T, D, H}
// With d = D / H and scale = 1 / sqrt(d), head h owns channels
// h*d .. h*d+d-1 of each segment. For every head h and query row t:
//   score[j]       = scale * sum over k of Q[t][h*d+k] * K[j][h*d+k], j <= t
//   probs[h][t][j] = exp(score[j] - max) / sum over j' of exp(score[j'] - max)
//   out[t][h*d+k]  = sum over j <= t of probs[h][t][j] * V[j][h*d+k]
// where max is the largest score of the row. Row [h][t] of probs is the
// working storage: it holds the scores, then the exponentials, then the
// normalized probabilities.
// Builds a 176-byte frame and calls tf_exp once per unmasked score. All
// ten callee-saved GPRs are in use, so the remaining state lives in the
// frame, where it survives the calls.
//   x19 = qkv, x20 = out, x21 = probs, x22 = T, x23 = D, x24 = H, x25 = d,
//   x26 = 12*D, the byte length of one qkv row, x27 = 4*T, the byte length
//   of one probs row, x28 = h.
// Stack locals: t112,j120,probrow128,qptr136,scale144,max148,sum152.
//   [sp,#112] t                   [sp,#120] j, in the exponential loop
//   [sp,#128] &probs[h][t][0]     [sp,#136] &Q[t][h*d]
//   [sp,#144] float32 scale       [sp,#148] float32 max
//   [sp,#152] float32 sum of the exponentials
.globl _tf_attention_forward_generic
_tf_attention_forward_generic:
    stp x29, x30, [sp, #-176]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    ldr w22, [x3]
    ldr w23, [x3, #4]
    ldr w24, [x3, #8]
    udiv x25, x23, x24
    add x26, x23, x23, lsl #1
    lsl x26, x26, #2
    lsl x27, x22, #2
    scvtf s0, x25
    fsqrt s0, s0
    fmov s1, #1.0
    fdiv s0, s1, s0
    str s0, [sp, #144]
    mov x28, #0
// Head loop: h = x28 runs over 0..H-1.
Laf_head:
    str xzr, [sp, #112]
// Query loop: t runs over 0..T-1. First zero all T entries of probs[h][t],
// which leaves the masked entries j > t at zero.
Laf_row:
    ldr x9, [sp, #112]
    madd x10, x28, x22, x9
    mul x10, x10, x27
    add x10, x21, x10
    str x10, [sp, #128]
    mov x11, x22
    mov x12, x10
Laf_zero_probs:
    str wzr, [x12], #4
    subs x11, x11, #1
    b.ne Laf_zero_probs
    // x11 = &Q[t][h*d]: row t of qkv, head h of the Q segment.
    mul x11, x9, x26
    madd x12, x28, x25, xzr
    add x11, x11, x12, lsl #2
    add x11, x19, x11
    str x11, [sp, #136]
    // s16 = running maximum, started at -infinity.
    mov w12, #0xff800000
    fmov s16, w12
    mov x9, #0
// Key loop: j = x9 runs over 0..t (b.ls keeps the diagonal).
// x12 = &K[j][h*d]; the K segment starts D floats into row j.
Laf_scores:
    mul x12, x9, x26
    add x12, x19, x12
    add x12, x12, x23, lsl #2
    mul x13, x28, x25
    add x12, x12, x13, lsl #2
    ldr x11, [sp, #136]
    mov x13, x25
    fmov s0, wzr
// d iterations: s0 += Q[t][h*d+k] * K[j][h*d+k], one FMADD per k.
Laf_dot:
    ldr s1, [x11], #4
    ldr s2, [x12], #4
    fmadd s0, s1, s2, s0
    subs x13, x13, #1
    b.ne Laf_dot
    // Scale the score, park it in probs[h][t][j] and fold it into the maximum.
    ldr s1, [sp, #144]
    fmul s0, s0, s1
    ldr x10, [sp, #128]
    str s0, [x10, x9, lsl #2]
    fmax s16, s16, s0
    add x9, x9, #1
    ldr x11, [sp, #112]
    cmp x9, x11
    b.ls Laf_scores
    str s16, [sp, #148]
    str wzr, [sp, #152]
    str xzr, [sp, #120]
// Exponential loop: j runs over 0..t. Replace each score by
// exp(score - max) and add it to the sum; j and the row pointer are
// reloaded from the frame after each call.
Laf_exp:
    ldr x9, [sp, #120]
    ldr x10, [sp, #128]
    ldr s0, [x10, x9, lsl #2]
    ldr s1, [sp, #148]
    fsub s0, s0, s1
    bl _tf_exp
    ldr x9, [sp, #120]
    ldr x10, [sp, #128]
    str s0, [x10, x9, lsl #2]
    ldr s1, [sp, #152]
    fadd s1, s1, s0
    str s1, [sp, #152]
    add x9, x9, #1
    str x9, [sp, #120]
    ldr x11, [sp, #112]
    cmp x9, x11
    b.ls Laf_exp
    // Zero this query/head output, then accumulate normalized A times V.
    // x14 = &out[t][h*d].
    ldr x9, [sp, #112]
    mul x10, x9, x23
    madd x10, x28, x25, x10
    add x14, x20, x10, lsl #2
    mov x11, x25
    mov x12, x14
Laf_zero_out:
    str wzr, [x12], #4
    subs x11, x11, #1
    b.ne Laf_zero_out
    mov x9, #0
// Value loop: j = x9 runs over 0..t. Normalize probs[h][t][j] in place with
// one FDIV, then add it times V[j] to the output. x11 = &V[j][h*d]; the V
// segment starts 2*D floats into row j.
Laf_value:
    ldr x10, [sp, #128]
    ldr s0, [x10, x9, lsl #2]
    ldr s1, [sp, #152]
    fdiv s0, s0, s1
    str s0, [x10, x9, lsl #2]
    mul x11, x9, x26
    add x11, x19, x11
    add x11, x11, x23, lsl #3
    mul x12, x28, x25
    add x11, x11, x12, lsl #2
    mov x12, x14
    mov x13, x25
// d iterations: out[t][h*d+k] += probs[h][t][j] * V[j][h*d+k].
Laf_value_dot:
    ldr s1, [x11], #4
    ldr s2, [x12]
    fmadd s2, s0, s1, s2
    str s2, [x12], #4
    subs x13, x13, #1
    b.ne Laf_value_dot
    add x9, x9, #1
    ldr x10, [sp, #112]
    cmp x9, x10
    b.ls Laf_value
    add x10, x10, #1
    str x10, [sp, #112]
    cmp x10, x22
    b.lo Laf_row
    add x28, x28, #1
    cmp x28, x24
    b.lo Laf_head
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #176
    ret

// tf_attention_backward_generic: scalar kernel behind
// tf_attention_backward(qkv,probs,dout,dqkv,shape). tf_attention_backward
// (attention.S) branches here for shapes it does not specialize.
//   x0 = qkv    float32 [T,3D]    forward input; Q, K, V segments per row
//   x1 = probs  float32 [H,T,T]   forward probabilities, read only
//   x2 = dout   float32 [T,D]     gradient with respect to out
//   x3 = dqkv   float32 [T,3D]    overwritten: zeroed, then accumulated
//   x4 = shape  int32 [3] = {T, D, H}
// With d = D / H, scale = 1 / sqrt(d), A[j] = probs[h][t][j], and every
// vector below restricted to head h's d channels, for each head h and
// query row t:
//   first pass, j = 0..t:    dA[j]   = dout[t] . V[j]
//                            dV[j]  += A[j] * dout[t]
//                            rowDot += A[j] * dA[j]
//   second pass, j = 0..t:   g       = A[j] * (dA[j] - rowDot) * scale
//                            dQ[t]  += g * K[j]
//                            dK[j]  += g * Q[t]
// g is the gradient with respect to the unscaled dot product Q[t] . K[j].
// Recompute dA in the second pass, without a temporary T*T tensor.
// Builds a 176-byte frame; calls nothing.
//   x19 = qkv, x20 = probs, x21 = dout, x22 = dqkv, x23 = T, x24 = D,
//   x25 = H, x26 = d, x27 = 12*D, the byte length of one qkv row, x28 = h.
// Stack: t112, scale120, rowDot124, pointers128..152.
//   [sp,#112] t                   [sp,#120] float32 scale
//   [sp,#124] float32 rowDot      [sp,#128] &dout[t][h*d]
//   [sp,#136] &probs[h][t][0]     [sp,#144] &Q[t][h*d]
//   [sp,#152] &dQ[t][h*d]
.globl _tf_attention_backward_generic
_tf_attention_backward_generic:
    stp x29, x30, [sp, #-176]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    ldr w23, [x4]
    ldr w24, [x4, #4]
    ldr w25, [x4, #8]
    udiv x26, x24, x25
    // x27 = 3*D for now. Zero all T*3*D floats of dqkv; every later write to it
    // accumulates.
    add x27, x24, x24, lsl #1
    mul x9, x23, x27
    mov x10, x22
Lab_zero:
    str wzr, [x10], #4
    subs x9, x9, #1
    b.ne Lab_zero
    // x27 = 12*D bytes from here on.
    lsl x27, x27, #2
    scvtf s0, x26
    fsqrt s0, s0
    fmov s1, #1.0
    fdiv s0, s1, s0
    str s0, [sp, #120]
    mov x28, #0
// Head loop: h = x28 runs over 0..H-1.
Lab_head:
    str xzr, [sp, #112]
// Query loop: t runs over 0..T-1. Compute the four row pointers once.
Lab_row:
    ldr x9, [sp, #112]
    mul x10, x9, x24
    madd x10, x28, x26, x10
    add x10, x21, x10, lsl #2
    str x10, [sp, #128] // dout row/head
    madd x10, x28, x23, x9
    mul x10, x10, x23
    add x10, x20, x10, lsl #2
    str x10, [sp, #136] // probabilities row
    mul x10, x9, x27
    mul x11, x28, x26
    add x10, x10, x11, lsl #2
    add x11, x19, x10
    str x11, [sp, #144] // Q
    add x11, x22, x10
    str x11, [sp, #152] // dQ
    // First pass. s16 accumulates rowDot.
    fmov s16, wzr
    mov x9, #0
// j = x9 runs over 0..t. x11 = &V[j][h*d] and x12 = &dV[j][h*d]; the V
// segment starts 2*D floats into row j.
Lab_first_j:
    mul x10, x9, x27
    add x10, x10, x24, lsl #3
    mul x11, x28, x26
    add x10, x10, x11, lsl #2
    add x11, x19, x10 // V
    add x12, x22, x10 // dV
    ldr x13, [sp, #128]
    ldr x14, [sp, #136]
    ldr s0, [x14, x9, lsl #2] // A
    fmov s1, wzr // dA
    mov x15, x26
// d iterations: dA += dout[t][k] * V[j][k] and dV[j][k] += A * dout[t][k].
Lab_first_k:
    ldr s2, [x13], #4 // dout
    ldr s3, [x11], #4 // V
    fmadd s1, s2, s3, s1
    ldr s4, [x12]
    fmadd s4, s0, s2, s4
    str s4, [x12], #4
    subs x15, x15, #1
    b.ne Lab_first_k
    fmadd s16, s0, s1, s16
    add x9, x9, #1
    ldr x10, [sp, #112]
    cmp x9, x10
    b.ls Lab_first_j
    str s16, [sp, #124]
    mov x9, #0
// Second pass: j = x9 runs over 0..t. dA[j] is recomputed with the same
// FMADD chain as in the first pass.
Lab_second_j:
    mul x10, x9, x27
    add x10, x10, x24, lsl #3
    mul x11, x28, x26
    add x10, x10, x11, lsl #2
    add x11, x19, x10 // V
    ldr x13, [sp, #128]
    fmov s1, wzr
    mov x15, x26
// d iterations: s1 += dout[t][k] * V[j][k].
Lab_second_dot:
    ldr s2, [x13], #4
    ldr s3, [x11], #4
    fmadd s1, s2, s3, s1
    subs x15, x15, #1
    b.ne Lab_second_dot
    // s1 = g = A[j] * (dA[j] - rowDot) * scale.
    ldr s0, [sp, #124]
    fsub s1, s1, s0
    ldr x14, [sp, #136]
    ldr s0, [x14, x9, lsl #2]
    fmul s1, s1, s0
    ldr s0, [sp, #120]
    fmul s1, s1, s0 // d score before scaling: d(QK)
    // x11 = &K[j][h*d] and x12 = &dK[j][h*d]; the K segment starts D floats
    // into row j.
    mul x10, x9, x27
    add x10, x10, x24, lsl #2
    mul x11, x28, x26
    add x10, x10, x11, lsl #2
    add x11, x19, x10 // K
    add x12, x22, x10 // dK
    ldr x13, [sp, #144] // Q
    ldr x14, [sp, #152] // dQ
    mov x15, x26
// d iterations: dQ[t][k] += g * K[j][k] and dK[j][k] += g * Q[t][k].
Lab_second_k:
    ldr s2, [x11], #4
    ldr s3, [x13], #4
    ldr s4, [x14]
    fmadd s4, s1, s2, s4
    str s4, [x14], #4
    ldr s4, [x12]
    fmadd s4, s1, s3, s4
    str s4, [x12], #4
    subs x15, x15, #1
    b.ne Lab_second_k
    add x9, x9, #1
    ldr x10, [sp, #112]
    cmp x9, x10
    b.ls Lab_second_j
    add x10, x10, #1
    str x10, [sp, #112]
    cmp x10, x23
    b.lo Lab_row
    add x28, x28, #1
    cmp x28, x25
    b.lo Lab_head
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #176
    ret

// tf_relu_forward(x,y,count)
//   x0 = x      float32 [count]
//   x1 = y      float32 [count]   overwritten; may be the same buffer as x
//   x2 = count  may be zero
// y[i] = max(x[i], 0), with FMAX against a zero vector: count / 4 vector
// iterations, then count % 4 scalar ones. Leaf, no frame; advances x0 and
// x1.
.globl _tf_relu_forward
_tf_relu_forward:
    movi v0.4s, #0
    lsr x9, x2, #2
    cbz x9, Lrf_tail_start
Lrf_vec:
    ldr q1, [x0], #16
    fmax v1.4s, v1.4s, v0.4s
    str q1, [x1], #16
    subs x9, x9, #1
    b.ne Lrf_vec
Lrf_tail_start:
    and x9, x2, #3
    cbz x9, Lrf_done
Lrf_tail:
    ldr s1, [x0], #4
    fmax s1, s1, s0
    str s1, [x1], #4
    subs x9, x9, #1
    b.ne Lrf_tail
Lrf_done:
    ret

// tf_relu_backward(x,dy,dx,count)
//   x0 = x      float32 [count]   forward input
//   x1 = dy     float32 [count]
//   x2 = dx     float32 [count]   overwritten
//   x3 = count  may be zero
// dx[i] = dy[i] where x[i] > 0, otherwise 0. The vector loop ANDs dy with
// the FCMGT mask; the count % 4 tail selects with FCSEL. The comparison is
// strict, so the derivative at x == 0 is zero. Leaf, no frame.
.globl _tf_relu_backward
_tf_relu_backward:
    movi v0.4s, #0
    lsr x9, x3, #2
    cbz x9, Lrb_tail_start
Lrb_vec:
    ldr q1, [x0], #16
    ldr q2, [x1], #16
    fcmgt v1.4s, v1.4s, v0.4s
    and v2.16b, v2.16b, v1.16b
    str q2, [x2], #16
    subs x9, x9, #1
    b.ne Lrb_vec
Lrb_tail_start:
    and x9, x3, #3
    cbz x9, Lrb_done
Lrb_tail:
    ldr s1, [x0], #4
    ldr s2, [x1], #4
    fcmp s1, #0.0
    fcsel s2, s2, s0, gt
    str s2, [x2], #4
    subs x9, x9, #1
    b.ne Lrb_tail
Lrb_done:
    ret

// tf_cross_entropy_generic: scalar kernel behind
// tf_cross_entropy(logits,targets,dlogits,rows,vocab). tf_cross_entropy
// (softmax.S) branches here when vocab is not 128.
//   x0 = logits   float32 [rows,vocab]
//   x1 = targets  int32 [rows]
//   x2 = dlogits  float32 [rows,vocab]   overwritten
//   x3 = rows, x4 = vocab (64-bit integers)
//   returns the float32 mean loss in s0
// Stable mean cross entropy, targets int32, dlogits = (softmax-one_hot)/rows.
// For each row r, with m = max over c of logits[r][c],
// e[c] = exp(logits[r][c] - m), S = sum over c of e[c] and y = targets[r]:
//   loss_r        = log(S) - (logits[r][y] - m)
//   dlogits[r][c] = (e[c] / S - (1 if c == y else 0)) / rows
// The return value is (sum over r of loss_r) / rows. Each loss_r is
// float32; the sum and the division are float64, rounded to float32 last.
// Builds a 144-byte frame; calls tf_exp vocab times and tf_log once per
// row.
//   x19 = logits, x20 = targets, x21 = dlogits, x22 = rows, x23 = vocab,
//   x24 = r, x25 = c, x26 = &logits[r][0], x27 = &dlogits[r][0]. x28 is
//   saved but not used.
// Stack locals max112,sum116,loss_sum double120.
//   [sp,#112] float32 m   [sp,#116] float32 S   [sp,#120] float64 loss sum
.globl _tf_cross_entropy_generic
_tf_cross_entropy_generic:
    stp x29, x30, [sp, #-144]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, x4
    mov x24, #0
    str xzr, [sp, #120]
// Row loop: r = x24 runs over 0..rows-1.
Lce_row:
    mul x9, x24, x23
    add x26, x19, x9, lsl #2
    add x27, x21, x9, lsl #2
    mov w9, #0xff800000
    fmov s0, w9
    mov x25, #0
// vocab iterations: m = maximum of the row, starting from -infinity.
Lce_max:
    ldr s1, [x26, x25, lsl #2]
    fmax s0, s0, s1
    add x25, x25, #1
    cmp x25, x23
    b.lo Lce_max
    str s0, [sp, #112]
    str wzr, [sp, #116]
    mov x25, #0
// vocab iterations: dlogits[r][c] = e[c] for now; S accumulates in the
// frame across the calls.
Lce_exp:
    ldr s0, [x26, x25, lsl #2]
    ldr s1, [sp, #112]
    fsub s0, s0, s1
    bl _tf_exp
    str s0, [x27, x25, lsl #2]
    ldr s1, [sp, #116]
    fadd s1, s1, s0
    str s1, [sp, #116]
    add x25, x25, #1
    cmp x25, x23
    b.lo Lce_exp
    // loss_r = log(S) - (logits[r][y] - m), widened and added to the float64
    // sum. w9 = y stays live through the gradient loop; no call follows.
    ldr s0, [sp, #116]
    bl _tf_log
    ldr w9, [x20, x24, lsl #2]
    ldr s1, [x26, x9, lsl #2]
    ldr s2, [sp, #112]
    fsub s1, s1, s2
    fsub s0, s0, s1
    fcvt d0, s0
    ldr d1, [sp, #120]
    fadd d1, d1, d0
    str d1, [sp, #120]
    ldr s1, [sp, #116]
    scvtf s2, x22
    mov x25, #0
// vocab iterations: turn e[c] into the gradient. s1 = S, s2 = float(rows).
Lce_gradient:
    ldr s0, [x27, x25, lsl #2]
    fdiv s0, s0, s1
    cmp x25, x9
    b.ne Lce_not_target
    fmov s3, #1.0
    fsub s0, s0, s3
Lce_not_target:
    fdiv s0, s0, s2
    str s0, [x27, x25, lsl #2]
    add x25, x25, #1
    cmp x25, x23
    b.lo Lce_gradient
    add x24, x24, #1
    cmp x24, x22
    b.lo Lce_row
    // Mean loss: float64 sum / rows, rounded to float32 in s0.
    ldr d0, [sp, #120]
    scvtf d1, x22
    fdiv d0, d0, d1
    fcvt s0, d0
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #144
    ret

// tf_adam(params,grads,m,v,step,count)
//   x0 = params  float32 [count]   updated in place
//   x1 = grads   float32 [count]
//   x2 = m       float32 [count]   first moment, updated in place
//   x3 = v       float32 [count]   second moment, updated in place
//   x4 = step    t >= 1
//   x5 = count   may be zero
// With g the gradient and p the parameter, one element is updated as
//   m = m + (g - m) * 0.1
//   v = v * 0.999 + (g * g) * 0.001
//   p = p + (m * a) / (sqrt(v) / c + 1e-8)
// where a = -0.001 / (1 - 0.9^t) and c = sqrt(1 - 0.999^t). 0.9^t and
// 0.999^t are float64, computed by square-and-multiply over the bits of t;
// a and c are computed in float64 and then rounded to float32.
// Leaf, no frame. v2 = a, v3 = c, v4 = 0.1, v5 = 0.999, v6 = 0.001,
// v7 = 1e-8, each in all four lanes. count / 4 vector iterations, then
// count % 4 scalar ones with the same sequence of operations.
.globl _tf_adam
_tf_adam:
    cbz x5, Ladam_done
    CONST x9, Ladam_constants
    ldp d0, d1, [x9]
    fmov d2, #1.0
    fmov d3, #1.0
    mov x10, x4
// Square-and-multiply: one iteration per significant bit of t, lowest
// first. d2 and d3 collect 0.9^t and 0.999^t; d0 and d1 hold the repeated
// squares.
Ladam_power:
    tbz x10, #0, Ladam_square
    fmul d2, d2, d0
    fmul d3, d3, d1
Ladam_square:
    lsr x10, x10, #1
    cbz x10, Ladam_bias
    fmul d0, d0, d0
    fmul d1, d1, d1
    b Ladam_power
// d2 = a = -0.001 / (1 - 0.9^t), d3 = c = sqrt(1 - 0.999^t), then both to
// float32 and broadcast.
Ladam_bias:
    fmov d0, #1.0
    fsub d2, d0, d2
    fsub d3, d0, d3
    fsqrt d3, d3
    ldr d0, [x9, #16]
    fdiv d2, d0, d2
    fneg d2, d2
    fcvt s2, d2
    fcvt s3, d3
    dup v2.4s, v2.s[0]
    dup v3.4s, v3.s[0]
    // The four float32 constants of Ladam_constants: 0.1, 0.999, 0.001, 1e-8.
    ldr s4, [x9, #24]
    ldr s5, [x9, #28]
    ldr s6, [x9, #32]
    ldr s7, [x9, #36]
    dup v4.4s, v4.s[0]
    dup v5.4s, v5.s[0]
    dup v6.4s, v6.s[0]
    dup v7.4s, v7.s[0]
    lsr x10, x5, #2
    cbz x10, Ladam_tail_start
// count / 4 iterations. v16 = p, v17 = g, v18 = m, v19 = v.
Ladam_vector:
    ldr q16, [x0]
    ldr q17, [x1], #16
    ldr q18, [x2]
    ldr q19, [x3]
    fsub v20.4s, v17.4s, v18.4s
    fmla v18.4s, v20.4s, v4.4s
    fmul v19.4s, v19.4s, v5.4s
    fmul v20.4s, v17.4s, v17.4s
    fmla v19.4s, v20.4s, v6.4s
    fsqrt v20.4s, v19.4s
    fdiv v20.4s, v20.4s, v3.4s
    fadd v20.4s, v20.4s, v7.4s
    fmul v21.4s, v18.4s, v2.4s
    fdiv v21.4s, v21.4s, v20.4s
    fadd v16.4s, v16.4s, v21.4s
    str q18, [x2], #16
    str q19, [x3], #16
    str q16, [x0], #16
    subs x10, x10, #1
    b.ne Ladam_vector
Ladam_tail_start:
    and x10, x5, #3
    cbz x10, Ladam_done
// count % 4 iterations, the same sequence on scalar registers.
Ladam_scalar:
    ldr s16, [x0]
    ldr s17, [x1], #4
    ldr s18, [x2]
    ldr s19, [x3]
    fsub s20, s17, s18
    fmadd s18, s20, s4, s18
    fmul s19, s19, s5
    fmul s20, s17, s17
    fmadd s19, s20, s6, s19
    fsqrt s20, s19
    fdiv s20, s20, s3
    fadd s20, s20, s7
    fmul s21, s18, s2
    fdiv s21, s21, s20
    fadd s16, s16, s21
    str s18, [x2], #4
    str s19, [x3], #4
    str s16, [x0], #4
    subs x10, x10, #1
    b.ne Ladam_scalar
Ladam_done:
    ret

// tf_exp(float): float32 exponential, s0 in and s0 out.
// NaN is returned unchanged, x < -104 returns +0 and x > 88.72283905206835
// returns +infinity. Otherwise, in float64:
//   n = x * log2(e) rounded to the nearest integer (FCVTAS, ties away)
//   r = x - n * ln(2)
//   p = sum over k = 0..12 of r^k / k!   (Horner, from Lexp_constants)
//   result = p * 2^n, rounded to float32
// Range-reduced exponential: float32 input, float64 degree-12 polynomial.
// Leaf, no frame. Uses x9-x12 and d0-d2 only.
.globl _tf_exp
_tf_exp:
    // A NaN compares unordered with itself; return it as it is.
    fcmp s0, s0
    b.vs Lexp_return
    CONST x9, Lexp_constants
    fcvt d0, s0
    ldr d1, [x9]
    fcmp d0, d1
    b.lt Lexp_zero
    ldr d1, [x9, #8]
    fcmp d0, d1
    b.gt Lexp_inf
    // Range reduction: x10 = n, d0 = r.
    ldr d1, [x9, #16]
    fmul d1, d0, d1
    fcvtas x10, d1
    scvtf d1, x10
    ldr d2, [x9, #24]
    fmul d1, d1, d2
    fsub d0, d0, d1
    add x11, x9, #32
    ldr d1, [x11], #8
    mov w12, #12
// 12 iterations: d1 = d1 * r + the next coefficient, 1/11! down to 1/0!.
Lexp_horner:
    ldr d2, [x11], #8
    fmadd d1, d1, d0, d2
    subs w12, w12, #1
    b.ne Lexp_horner
    // 2^n as a float64 bit pattern: biased exponent n + 1023, zero fraction.
    add x10, x10, #1023
    lsl x10, x10, #52
    fmov d2, x10
    fmul d0, d1, d2
    fcvt s0, d0
Lexp_return:
    ret
Lexp_zero:
    fmov s0, wzr
    ret
Lexp_inf:
    mov w9, #0x7f800000
    fmov s0, w9
    ret

// tf_log(float): float32 natural logarithm, s0 in and s0 out.
// x == 0 returns -infinity; x < 0 or NaN returns the quiet NaN 0x7fc00000;
// +infinity is returned unchanged. Otherwise the input is widened to
// float64, which is exact, and split as x = m * 2^e with 1 <= m < 2 by
// rewriting the exponent field. Then, in float64:
//   z = (m - 1) / (m + 1)
//   ln(m) = 2 * z * sum over k = 0..11 of z^(2k) / (2k + 1)
//   result = e * ln(2) + ln(m), rounded to float32
// float32 log through exact double normalization and an atanh series.
// Leaf, no frame. Uses x9-x11 and d0-d3 only.
.globl _tf_log
_tf_log:
    fcmp s0, #0.0
    b.eq Llog_minus_inf
    b.lt Llog_nan
    fcvt d0, s0
    // x10 = biased exponent field. 2047 here can only be +infinity, which is
    // returned through the final FCVT.
    fmov x9, d0
    ubfx x10, x9, #52, #11
    cmp x10, #2047
    b.eq Llog_return
    // x10 = e. Force the exponent field to 1023 so that d1 = m in [1, 2).
    sub x10, x10, #1023
    and x9, x9, #0x000fffffffffffff
    mov x11, #0x3ff0000000000000
    orr x9, x9, x11
    fmov d1, x9
    // d0 = z = (m - 1) / (m + 1), d1 = z^2.
    fmov d2, #1.0
    fsub d0, d1, d2
    fadd d1, d1, d2
    fdiv d0, d0, d1
    fmul d1, d0, d0
    CONST x9, Llog_coefficients
    ldr d2, [x9], #8
    mov w11, #11
// 11 iterations: d2 = d2 * z^2 + the next coefficient, 1/21 down to 1.
Llog_horner:
    ldr d3, [x9], #8
    fmadd d2, d2, d1, d3
    subs w11, w11, #1
    b.ne Llog_horner
    // ln(m) = 2 * z * d2, then e * ln(2) is added with one FMADD.
    fmul d0, d0, d2
    fadd d0, d0, d0
    scvtf d1, x10
    ldr d2, [x9]
    fmadd d0, d1, d2, d0
Llog_return:
    fcvt s0, d0
    ret
Llog_minus_inf:
    mov w9, #0xff800000
    fmov s0, w9
    ret
Llog_nan:
    mov w9, #0x7fc00000
    fmov s0, w9
    ret

// Constant tables, 8-byte aligned.
// Lexp_constants: lower bound, upper bound, log2(e), ln(2), then the 13
//   coefficients 1/12!, 1/11!, .. 1/0! in the order Horner's rule uses them.
// Llog_coefficients: 1/23, 1/21, .. 1/3, 1 in Horner order, then ln(2).
// Ladam_constants: float64 0.9, 0.999, 0.001 (beta1, beta2, learning rate),
//   then float32 0.1, 0.999, 0.001, 1e-8 (1 - beta1, beta2, 1 - beta2,
//   epsilon).
.section __TEXT,__const
.p2align 3
Lexp_constants:
    .double -104.0, 88.72283905206835, 1.4426950408889634074, 0.69314718055994530942
    .double 0.0000000020876756987868098979
    .double 0.000000025052108385441718775
    .double 0.00000027557319223985890653
    .double 0.0000027557319223985890653
    .double 0.000024801587301587301587
    .double 0.00019841269841269841270
    .double 0.0013888888888888888889
    .double 0.0083333333333333333333
    .double 0.041666666666666666667
    .double 0.16666666666666666667
    .double 0.5, 1.0, 1.0
Llog_coefficients:
    .double 0.043478260869565217391
    .double 0.047619047619047619048
    .double 0.052631578947368421053
    .double 0.058823529411764705882
    .double 0.066666666666666666667
    .double 0.076923076923076923077
    .double 0.090909090909090909091
    .double 0.11111111111111111111
    .double 0.14285714285714285714
    .double 0.2, 0.33333333333333333333, 1.0
    .double 0.69314718055994530942
Ladam_constants:
    .double 0.9, 0.999, 0.001
    .float 0.1, 0.999, 0.001, 0.00000001
