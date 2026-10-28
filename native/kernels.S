// Handwritten Darwin AArch64 numerical primitives. See ABI.md/kernel_notes.md.
// No numerical library calls. x18 and callee-saved SIMD v8-v15 are unused.
.text
.p2align 2
.macro CONST dst, label
    adrp \dst, \label@PAGE
    add \dst, \dst, \label@PAGEOFF
.endm

// x:[R,I], w:[O,I], b:[O], y:[R,O]; shape int32[R,I,O].
.globl _tf_linear_forward
_tf_linear_forward:
    ldr w9, [x4]
    ldr w10, [x4, #4]
    ldr w11, [x4, #8]
    mov x12, #0
Llf_row:
    mov x13, #0
Llf_out:
    mul x14, x12, x10
    add x14, x0, x14, lsl #2
    mul x15, x13, x10
    add x15, x1, x15, lsl #2
    movi v0.4s, #0
    lsr x17, x10, #2
    cbz x17, Llf_reduce
Llf_vec:
    ldr q1, [x14], #16
    ldr q2, [x15], #16
    fmla v0.4s, v1.4s, v2.4s
    subs x17, x17, #1
    b.ne Llf_vec
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

// Overwrite dx; ADD dw and db. All dimensions support scalar tails.
.globl _tf_linear_backward
_tf_linear_backward:
    ldr w9, [x6]
    ldr w10, [x6, #4]
    ldr w11, [x6, #8]
    mov x12, #0
Llb_row:
    mul x8, x12, x10
    add x15, x3, x8, lsl #2
    mov x17, x10
Llb_zero:
    str wzr, [x15], #4
    subs x17, x17, #1
    b.ne Llb_zero
    mul x8, x12, x11
    add x7, x2, x8, lsl #2
    mov x13, #0
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

// x0 x, x1 gamma, x2 beta, x3 y, x4 norm, x5 inv, x6 rows, x7 width.
// Two-pass biased variance; scalar accumulation is deliberate and documented.
.globl _tf_ln_forward
_tf_ln_forward:
    scvtf s7, x7
    mov w9, #0xc5ac
    movk w9, #0x3727, lsl #16 // float32 1e-5
    fmov s6, w9
    mov x9, #0
Llnf_row:
    mul x11, x9, x7
    add x12, x0, x11, lsl #2
    add x13, x3, x11, lsl #2
    add x14, x4, x11, lsl #2
    mov x10, #0
    fmov s0, wzr
Llnf_mean:
    ldr s1, [x12, x10, lsl #2]
    fadd s0, s0, s1
    add x10, x10, #1
    cmp x10, x7
    b.lo Llnf_mean
    fdiv s16, s0, s7
    fmov s0, wzr
    mov x10, #0
Llnf_var:
    ldr s1, [x12, x10, lsl #2]
    fsub s1, s1, s16
    fmul s1, s1, s1
    fadd s0, s0, s1
    add x10, x10, #1
    cmp x10, x7
    b.lo Llnf_var
    fdiv s0, s0, s7
    fadd s0, s0, s6
    fsqrt s0, s0
    fmov s1, #1.0
    fdiv s17, s1, s0
    str s17, [x5, x9, lsl #2]
    mov x10, #0
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

// x0 dy, x1 norm, x2 inv, x3 gamma, x4 dx, x5 dg, x6 db, x7 shape[R,D].
.globl _tf_ln_backward
_tf_ln_backward:
    ldr w9, [x7]
    ldr w10, [x7, #4]
    scvtf s7, x10
    mov x11, #0
Llnb_row:
    mul x12, x11, x10
    add x13, x0, x12, lsl #2
    add x14, x1, x12, lsl #2
    add x15, x4, x12, lsl #2
    fmov s16, wzr
    fmov s17, wzr
    mov x12, #0
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

// qkv:[T,3D], out:[T,D], probs:[H,T,T], shape int32[T,D,H].
// Stack locals: t112,j120,probrow128,qptr136,scale144,max148,sum152.
.globl _tf_attention_forward
_tf_attention_forward:
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
Laf_head:
    str xzr, [sp, #112]
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
    mul x11, x9, x26
    madd x12, x28, x25, xzr
    add x11, x11, x12, lsl #2
    add x11, x19, x11
    str x11, [sp, #136]
    mov w12, #0xff800000
    fmov s16, w12
    mov x9, #0
Laf_scores:
    mul x12, x9, x26
    add x12, x19, x12
    add x12, x12, x23, lsl #2
    mul x13, x28, x25
    add x12, x12, x13, lsl #2
    ldr x11, [sp, #136]
    mov x13, x25
    fmov s0, wzr
Laf_dot:
    ldr s1, [x11], #4
    ldr s2, [x12], #4
    fmadd s0, s1, s2, s0
    subs x13, x13, #1
    b.ne Laf_dot
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

// qkv,probs,dout,dqkv,shape. Recompute dA in the second pass, without a
// temporary T*T tensor. Stack: t112, scale120, rowDot124, pointers128..152.
.globl _tf_attention_backward
_tf_attention_backward:
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
    add x27, x24, x24, lsl #1
    mul x9, x23, x27
    mov x10, x22
Lab_zero:
    str wzr, [x10], #4
    subs x9, x9, #1
    b.ne Lab_zero
    lsl x27, x27, #2
    scvtf s0, x26
    fsqrt s0, s0
    fmov s1, #1.0
    fdiv s0, s1, s0
    str s0, [sp, #120]
    mov x28, #0
Lab_head:
    str xzr, [sp, #112]
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
    fmov s16, wzr
    mov x9, #0
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
Lab_second_j:
    mul x10, x9, x27
    add x10, x10, x24, lsl #3
    mul x11, x28, x26
    add x10, x10, x11, lsl #2
    add x11, x19, x10 // V
    ldr x13, [sp, #128]
    fmov s1, wzr
    mov x15, x26
Lab_second_dot:
    ldr s2, [x13], #4
    ldr s3, [x11], #4
    fmadd s1, s2, s3, s1
    subs x15, x15, #1
    b.ne Lab_second_dot
    ldr s0, [sp, #124]
    fsub s1, s1, s0
    ldr x14, [sp, #136]
    ldr s0, [x14, x9, lsl #2]
    fmul s1, s1, s0
    ldr s0, [sp, #120]
    fmul s1, s1, s0 // d score before scaling: d(QK)
    mul x10, x9, x27
    add x10, x10, x24, lsl #2
    mul x11, x28, x26
    add x10, x10, x11, lsl #2
    add x11, x19, x10 // K
    add x12, x22, x10 // dK
    ldr x13, [sp, #144] // Q
    ldr x14, [sp, #152] // dQ
    mov x15, x26
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

// Stable mean cross entropy, targets int32, dlogits = (softmax-one_hot)/rows.
// Stack locals max112,sum116,loss_sum double120.
.globl _tf_cross_entropy
_tf_cross_entropy:
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
Lce_row:
    mul x9, x24, x23
    add x26, x19, x9, lsl #2
    add x27, x21, x9, lsl #2
    mov w9, #0xff800000
    fmov s0, w9
    mov x25, #0
Lce_max:
    ldr s1, [x26, x25, lsl #2]
    fmax s0, s0, s1
    add x25, x25, #1
    cmp x25, x23
    b.lo Lce_max
    str s0, [sp, #112]
    str wzr, [sp, #116]
    mov x25, #0
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

// x0 params x1 grads x2 m x3 v x4 step x5 count.
.globl _tf_adam
_tf_adam:
    cbz x5, Ladam_done
    CONST x9, Ladam_constants
    ldp d0, d1, [x9]
    fmov d2, #1.0
    fmov d3, #1.0
    mov x10, x4
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

// Range-reduced exponential: float32 input, float64 degree-12 polynomial.
.globl _tf_exp
_tf_exp:
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
Lexp_horner:
    ldr d2, [x11], #8
    fmadd d1, d1, d0, d2
    subs w12, w12, #1
    b.ne Lexp_horner
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

// float32 log through exact double normalization and an atanh series.
.globl _tf_log
_tf_log:
    fcmp s0, #0.0
    b.eq Llog_minus_inf
    b.lt Llog_nan
    fcvt d0, s0
    fmov x9, d0
    ubfx x10, x9, #52, #11
    cmp x10, #2047
    b.eq Llog_return
    sub x10, x10, #1023
    and x9, x9, #0x000fffffffffffff
    mov x11, #0x3ff0000000000000
    orr x9, x9, x11
    fmov d1, x9
    fmov d2, #1.0
    fsub d0, d1, d2
    fadd d1, d1, d2
    fdiv d0, d0, d1
    fmul d1, d0, d0
    CONST x9, Llog_coefficients
    ldr d2, [x9], #8
    mov w11, #11
Llog_horner:
    ldr d3, [x9], #8
    fmadd d2, d2, d1, d3
    subs w11, w11, #1
    b.ne Llog_horner
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
