// Four-wide LayerNorm. Nonmultiple-of-four widths use the audited scalar path.
// Darwin caller-saved registers only; gamma/beta gradients ADD to accumulators.
.text
.p2align 2
.globl _tf_ln_forward
_tf_ln_forward:
    tst x7, #3
    b.eq Lnvf_fast
    b _tf_ln_forward_generic
Lnvf_fast:
    scvtf s7, x7
    fmov s0, #1.0
    fdiv s7, s0, s7
    mov w9, #0xc5ac
    movk w9, #0x3727, lsl #16
    fmov s6, w9
    lsl x7, x7, #2
    mov x9, #0
Lnvf_row:
    mul x11, x9, x7
    add x12, x0, x11
    add x13, x3, x11
    add x14, x4, x11
    mov x10, #0
    movi v0.4s, #0
Lnvf_mean:
    ldr q1, [x12, x10]
    fadd v0.4s, v0.4s, v1.4s
    add x10, x10, #16
    cmp x10, x7
    b.lo Lnvf_mean
    faddp v0.4s, v0.4s, v0.4s
    faddp s0, v0.2s
    fmul s16, s0, s7
    dup v16.4s, v16.s[0]
    movi v0.4s, #0
    mov x10, #0
Lnvf_var:
    ldr q1, [x12, x10]
    fsub v1.4s, v1.4s, v16.4s
    fmul v1.4s, v1.4s, v1.4s
    fadd v0.4s, v0.4s, v1.4s
    add x10, x10, #16
    cmp x10, x7
    b.lo Lnvf_var
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

.globl _tf_ln_backward
_tf_ln_backward:
    ldr w10, [x7, #4]
    tst x10, #3
    b.eq Lnvb_fast
    b _tf_ln_backward_generic
Lnvb_fast:
    ldr w9, [x7]
    scvtf s7, x10
    fmov s0, #1.0
    fdiv s7, s0, s7
    lsl x10, x10, #2
    mov x11, #0
Lnvb_row:
    mul x12, x11, x10
    add x13, x0, x12
    add x14, x1, x12
    add x15, x4, x12
    movi v16.4s, #0
    movi v17.4s, #0
    mov x12, #0
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
