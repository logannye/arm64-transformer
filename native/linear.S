// Handwritten NEON matrix tiles for the fixed transformer; public ABI unchanged.
// Non-tiled shapes tail-call the preserved v1 generic kernels.
.text
.p2align 4

// Four independent rows, sixteen contiguous columns. v0-v15 are the tile;
// v16-v19 are four adjacent vectors. Coefficients use their source lanes
// directly, avoiding scalar replicate loads and separate DUP instructions.
.macro OUTER4X16 c0,l0,c1,l1,c2,l2,c3,l3
    fmla v0.4s, v16.4s, \c0\().s[\l0]
    fmla v1.4s, v17.4s, \c0\().s[\l0]
    fmla v2.4s, v18.4s, \c0\().s[\l0]
    fmla v3.4s, v19.4s, \c0\().s[\l0]
    fmla v4.4s, v16.4s, \c1\().s[\l1]
    fmla v5.4s, v17.4s, \c1\().s[\l1]
    fmla v6.4s, v18.4s, \c1\().s[\l1]
    fmla v7.4s, v19.4s, \c1\().s[\l1]
    fmla v8.4s, v16.4s, \c2\().s[\l2]
    fmla v9.4s, v17.4s, \c2\().s[\l2]
    fmla v10.4s, v18.4s, \c2\().s[\l2]
    fmla v11.4s, v19.4s, \c2\().s[\l2]
    fmla v12.4s, v16.4s, \c3\().s[\l3]
    fmla v13.4s, v17.4s, \c3\().s[\l3]
    fmla v14.4s, v18.4s, \c3\().s[\l3]
    fmla v15.4s, v19.4s, \c3\().s[\l3]
.endm

.macro ZERO_TILE
    movi v0.4s, #0
    movi v1.4s, #0
    movi v2.4s, #0
    movi v3.4s, #0
    movi v4.4s, #0
    movi v5.4s, #0
    movi v6.4s, #0
    movi v7.4s, #0
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
.endm

.macro SAVE_LINEAR frame
    stp x29, x30, [sp, #-\frame]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    stp d8, d9, [sp, #96]
    stp d10, d11, [sp, #112]
    stp d12, d13, [sp, #128]
    stp d14, d15, [sp, #144]
.endm

.macro RESTORE_LINEAR frame
    ldp d14, d15, [sp, #144]
    ldp d12, d13, [sp, #128]
    ldp d10, d11, [sp, #112]
    ldp d8, d9, [sp, #96]
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #\frame
    ret
.endm

// x0 X[R,I], x1 W[O,I], x2 bias[O], x3 Y[R,O], x4 shape[R,I,O].
// Four token rows x four output columns, sixteen independent dot accumulators.
.globl _tf_linear_forward_neon
_tf_linear_forward_neon:
    ldr w9, [x4]
    ldr w10, [x4, #4]
    ldr w11, [x4, #8]
    orr w12, w9, w10
    orr w12, w12, w11
    tst w12, #3
    b.eq Ltile_forward
    b _tf_linear_forward_generic
Ltile_forward:
    SAVE_LINEAR 160
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, x9
    mov x24, x10
    mov x25, x11
    lsl x28, x24, #2
    mov x26, #0
Ltf_row:
    mov x27, #0
Ltf_col:
    madd x9, x26, x28, x19
    add x10, x9, x28
    add x11, x10, x28
    add x12, x11, x28
    madd x13, x27, x28, x20
    add x14, x13, x28
    add x15, x14, x28
    add x16, x15, x28
    lsr x17, x24, #2
    ZERO_TILE
Ltf_reduce:
    ldr q16, [x9], #16
    ldr q17, [x10], #16
    ldr q18, [x11], #16
    ldr q19, [x12], #16
    ldr q20, [x13], #16
    ldr q21, [x14], #16
    ldr q22, [x15], #16
    ldr q23, [x16], #16
    fmla v0.4s, v16.4s, v20.4s
    fmla v1.4s, v16.4s, v21.4s
    fmla v2.4s, v16.4s, v22.4s
    fmla v3.4s, v16.4s, v23.4s
    fmla v4.4s, v17.4s, v20.4s
    fmla v5.4s, v17.4s, v21.4s
    fmla v6.4s, v17.4s, v22.4s
    fmla v7.4s, v17.4s, v23.4s
    fmla v8.4s, v18.4s, v20.4s
    fmla v9.4s, v18.4s, v21.4s
    fmla v10.4s, v18.4s, v22.4s
    fmla v11.4s, v18.4s, v23.4s
    fmla v12.4s, v19.4s, v20.4s
    fmla v13.4s, v19.4s, v21.4s
    fmla v14.4s, v19.4s, v22.4s
    fmla v15.4s, v19.4s, v23.4s
    subs x17, x17, #1
    b.ne Ltf_reduce
    // Pairwise reductions combine four scalar outputs into one vector per row.
    add x9, x21, x27, lsl #2
    ldr q26, [x9]
    madd x9, x26, x25, x27
    add x9, x22, x9, lsl #2
    lsl x10, x25, #2
    faddp v24.4s, v0.4s, v1.4s
    faddp v25.4s, v2.4s, v3.4s
    faddp v24.4s, v24.4s, v25.4s
    fadd v24.4s, v24.4s, v26.4s
    str q24, [x9]
    add x9, x9, x10
    faddp v24.4s, v4.4s, v5.4s
    faddp v25.4s, v6.4s, v7.4s
    faddp v24.4s, v24.4s, v25.4s
    fadd v24.4s, v24.4s, v26.4s
    str q24, [x9]
    add x9, x9, x10
    faddp v24.4s, v8.4s, v9.4s
    faddp v25.4s, v10.4s, v11.4s
    faddp v24.4s, v24.4s, v25.4s
    fadd v24.4s, v24.4s, v26.4s
    str q24, [x9]
    add x9, x9, x10
    faddp v24.4s, v12.4s, v13.4s
    faddp v25.4s, v14.4s, v15.4s
    faddp v24.4s, v24.4s, v25.4s
    fadd v24.4s, v24.4s, v26.4s
    str q24, [x9]
    add x27, x27, #4
    cmp x27, x25
    b.lo Ltf_col
    add x26, x26, #4
    cmp x26, x23
    b.lo Ltf_row
    RESTORE_LINEAR 160

// x0 X,x1 W,x2 dY,x3 dX,x4 dW,x5 db,x6 shape[R,I,O].
// dX overwrite, dW/db ADD. Fast tiles require R,O %4 ==0 and I %16 ==0.
.p2align 4
.globl _tf_linear_backward_neon
_tf_linear_backward_neon:
    ldr w9, [x6]
    ldr w10, [x6, #4]
    ldr w11, [x6, #8]
    orr w12, w9, w11
    tst w12, #3
    b.ne Ltb_generic
    tst w10, #15
    b.eq Ltile_backward
Ltb_generic:
    b _tf_linear_backward_generic
Ltile_backward:
    SAVE_LINEAR 208
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, x4
    mov x24, x5
    mov x25, x9 // R
    mov x26, x10 // I
    mov x27, x11 // O
    lsl x28, x26, #2 // input/weight row bytes
    lsl x9, x27, #2
    str x9, [sp, #160] // output/dY row bytes
    str xzr, [sp, #168] // row block

    // dX = dY @ W. Keep a 4-row x16-input-column tile in registers over O.
Ltb_dx_rows:
    str xzr, [sp, #176] // input column block
Ltb_dx_columns:
    ldr x8, [sp, #168]
    ldr x14, [sp, #160]
    madd x9, x8, x14, x21
    add x10, x9, x14
    add x11, x10, x14
    add x12, x11, x14
    ldr x8, [sp, #176]
    add x13, x20, x8, lsl #2
    lsr x15, x27, #2
    ZERO_TILE
Ltb_dx_reduce:
    // Four consecutive coefficients per row; process lanes in the original
    // output-channel order so every accumulator has the same FMA sequence.
    ldr q20, [x9], #16
    ldr q21, [x10], #16
    ldr q22, [x11], #16
    ldr q23, [x12], #16
    ldp q16, q17, [x13]
    ldp q18, q19, [x13, #32]
    add x13, x13, x28
    OUTER4X16 v20,0,v21,0,v22,0,v23,0
    ldp q16, q17, [x13]
    ldp q18, q19, [x13, #32]
    add x13, x13, x28
    OUTER4X16 v20,1,v21,1,v22,1,v23,1
    ldp q16, q17, [x13]
    ldp q18, q19, [x13, #32]
    add x13, x13, x28
    OUTER4X16 v20,2,v21,2,v22,2,v23,2
    ldp q16, q17, [x13]
    ldp q18, q19, [x13, #32]
    add x13, x13, x28
    OUTER4X16 v20,3,v21,3,v22,3,v23,3
    subs x15, x15, #1
    b.ne Ltb_dx_reduce
    ldr x8, [sp, #168]
    ldr x9, [sp, #176]
    madd x8, x8, x26, x9
    add x8, x22, x8, lsl #2
    stp q0, q1, [x8]
    stp q2, q3, [x8, #32]
    add x8, x8, x28
    stp q4, q5, [x8]
    stp q6, q7, [x8, #32]
    add x8, x8, x28
    stp q8, q9, [x8]
    stp q10, q11, [x8, #32]
    add x8, x8, x28
    stp q12, q13, [x8]
    stp q14, q15, [x8, #32]
    add x9, x9, #16
    str x9, [sp, #176]
    cmp x9, x26
    b.lo Ltb_dx_columns
    ldr x8, [sp, #168]
    add x8, x8, #4
    str x8, [sp, #168]
    cmp x8, x25
    b.lo Ltb_dx_rows

    // dW += dY^T @ X. Keep four output channels x16 inputs in registers over R.
    str xzr, [sp, #184] // output channel block
Ltb_dw_outputs:
    str xzr, [sp, #176]
Ltb_dw_columns:
    ldr x8, [sp, #184]
    ldr x9, [sp, #176]
    madd x8, x8, x26, x9
    add x8, x23, x8, lsl #2
    ldp q0, q1, [x8]
    ldp q2, q3, [x8, #32]
    add x8, x8, x28
    ldp q4, q5, [x8]
    ldp q6, q7, [x8, #32]
    add x8, x8, x28
    ldp q8, q9, [x8]
    ldp q10, q11, [x8, #32]
    add x8, x8, x28
    ldp q12, q13, [x8]
    ldp q14, q15, [x8, #32]
    ldr x8, [sp, #176]
    add x9, x19, x8, lsl #2
    ldr x8, [sp, #184]
    add x10, x21, x8, lsl #2
    ldr x12, [sp, #160]
    mov x15, x25
Ltb_dw_reduce:
    ldp q16, q17, [x9]
    ldp q18, q19, [x9, #32]
    add x9, x9, x28
    ldr q24, [x10]
    add x10, x10, x12
    OUTER4X16 v24,0,v24,1,v24,2,v24,3
    subs x15, x15, #1
    b.ne Ltb_dw_reduce
    ldr x8, [sp, #184]
    ldr x9, [sp, #176]
    madd x8, x8, x26, x9
    add x8, x23, x8, lsl #2
    stp q0, q1, [x8]
    stp q2, q3, [x8, #32]
    add x8, x8, x28
    stp q4, q5, [x8]
    stp q6, q7, [x8, #32]
    add x8, x8, x28
    stp q8, q9, [x8]
    stp q10, q11, [x8, #32]
    add x8, x8, x28
    stp q12, q13, [x8]
    stp q14, q15, [x8, #32]
    add x9, x9, #16
    str x9, [sp, #176]
    cmp x9, x26
    b.lo Ltb_dw_columns

    // Bias reduction uses the same row ordering as the generic kernel.
    ldr x8, [sp, #184]
    add x9, x21, x8, lsl #2
    add x10, x24, x8, lsl #2
    ldr q0, [x10]
    ldr x12, [sp, #160]
    mov x15, x25
Ltb_db_reduce:
    ldr q1, [x9]
    fadd v0.4s, v0.4s, v1.4s
    add x9, x9, x12
    subs x15, x15, #1
    b.ne Ltb_db_reduce
    str q0, [x10]
    add x8, x8, #4
    str x8, [sp, #184]
    cmp x8, x27
    b.lo Ltb_dw_outputs
    RESTORE_LINEAR 208
