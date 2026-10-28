// Handwritten head-width-eight attention for [T,96] QKV, width32/head4.
// T in 1..32; other shapes retain the original general implementation.
// Forward uses NEON QK/value operations and four-lane stable exponentiation.
// Backward keeps each dA row on stack, avoiding recomputation of dot(dout,V).
.text
.p2align 2
.macro AADDR reg, name
    adrp \reg,\name@PAGE
    add \reg,\reg,\name@PAGEOFF
.endm
.macro AENTER size
    stp x29,x30,[sp,#-\size]!
    mov x29,sp
    stp x19,x20,[sp,#16]
    stp x21,x22,[sp,#32]
    stp x23,x24,[sp,#48]
    stp x25,x26,[sp,#64]
    stp x27,x28,[sp,#80]
.endm
.macro ALEAVE size
    ldp x27,x28,[sp,#80]
    ldp x25,x26,[sp,#64]
    ldp x23,x24,[sp,#48]
    ldp x21,x22,[sp,#32]
    ldp x19,x20,[sp,#16]
    ldp x29,x30,[sp],#\size
    ret
.endm

.globl _tf_attention_forward
_tf_attention_forward:
    ldr w9,[x3,#4]
    cmp w9,#32
    b.ne Laf8_fallback
    ldr w9,[x3,#8]
    cmp w9,#4
    b.ne Laf8_fallback
    ldr w9,[x3]
    sub w10,w9,#1
    cmp w10,#32
    b.lo Laf8_start
Laf8_fallback:
    b _tf_attention_forward_generic
Laf8_start:
    AENTER 96
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x9
    mov x23,#0
    AADDR x9,Lattention_scale
    ldr s31,[x9]
Laf8_head:
    mov x24,#0
Laf8_row:
    mov x9,#384
    mul x26,x24,x9
    add x26,x19,x26
    add x26,x26,x23,lsl #5
    ldp q20,q21,[x26]
    madd x9,x23,x22,x24
    mul x9,x9,x22
    add x25,x21,x9,lsl #2
    add x27,x19,#128
    add x27,x27,x23,lsl #5
    mov x28,#0
    mov w9,#0xff800000
    fmov s22,w9
Laf8_score:
    ldp q0,q1,[x27]
    add x27,x27,#384
    fmul v2.4s,v20.4s,v0.4s
    fmla v2.4s,v21.4s,v1.4s
    faddp v2.4s,v2.4s,v2.4s
    faddp s2,v2.2s
    fmul s2,s2,s31
    str s2,[x25,x28,lsl #2]
    fmax s22,s22,s2
    add x28,x28,#1
    cmp x28,x24
    b.ls Laf8_score
    // All masked entries are explicitly written, including short sequence tails.
    mov x9,x28
Laf8_mask:
    cmp x9,x22
    b.hs Laf8_exp_start
    str wzr,[x25,x9,lsl #2]
    add x9,x9,#1
    b Laf8_mask
Laf8_exp_start:
    dup v22.4s,v22.s[0]
    movi v23.4s,#0
    mov x27,x25
    lsr x28,x28,#2
    cbz x28,Laf8_exp_tail
Laf8_exp_vector:
    ldr q0,[x27]
    fsub v0.4s,v0.4s,v22.4s
    bl _tf_exp4
    str q0,[x27],#16
    fadd v23.4s,v23.4s,v0.4s
    subs x28,x28,#1
    b.ne Laf8_exp_vector
Laf8_exp_tail:
    add x28,x24,#1
    and x28,x28,#3
    cbz x28,Laf8_normalize
    // Inactive lanes start at the row maximum, then are masked after exp.
    // This avoids out-of-bounds loads and exceptional exp fallback for tails.
    mov v0.16b,v22.16b
    mov x9,x27
    ld1 {v0.s}[0],[x9],#4
    cmp x28,#1
    b.eq Laf8_tail_exp
    ld1 {v0.s}[1],[x9],#4
    cmp x28,#2
    b.eq Laf8_tail_exp
    ld1 {v0.s}[2],[x9]
Laf8_tail_exp:
    fsub v0.4s,v0.4s,v22.4s
    bl _tf_exp4
    AADDR x9,Lattention_indices
    ldr q1,[x9]
    dup v2.4s,w28
    cmgt v1.4s,v2.4s,v1.4s
    and v0.16b,v0.16b,v1.16b
    fadd v23.4s,v23.4s,v0.4s
    st1 {v0.s}[0],[x27],#4
    cmp x28,#1
    b.eq Laf8_normalize
    st1 {v0.s}[1],[x27],#4
    cmp x28,#2
    b.eq Laf8_normalize
    st1 {v0.s}[2],[x27]
Laf8_normalize:
    faddp v23.4s,v23.4s,v23.4s
    faddp s23,v23.2s
    dup v24.4s,v23.s[0]
    mov x27,x25
    add x28,x24,#1
    lsr x28,x28,#2
    cbz x28,Laf8_norm_tail_start
Laf8_norm_vector:
    ldr q0,[x27]
    fdiv v0.4s,v0.4s,v24.4s
    str q0,[x27],#16
    subs x28,x28,#1
    b.ne Laf8_norm_vector
Laf8_norm_tail_start:
    add x28,x24,#1
    and x28,x28,#3
    cbz x28,Laf8_value_start
Laf8_norm_tail:
    ldr s0,[x27]
    fdiv s0,s0,s24
    str s0,[x27],#4
    subs x28,x28,#1
    b.ne Laf8_norm_tail
Laf8_value_start:
    movi v20.4s,#0
    movi v21.4s,#0
    add x27,x19,#256
    add x27,x27,x23,lsl #5
    mov x28,#0
Laf8_value:
    ldr s0,[x25,x28,lsl #2]
    dup v0.4s,v0.s[0]
    ldp q1,q2,[x27]
    add x27,x27,#384
    fmla v20.4s,v1.4s,v0.4s
    fmla v21.4s,v2.4s,v0.4s
    add x28,x28,#1
    cmp x28,x24
    b.ls Laf8_value
    add x9,x20,x24,lsl #7
    add x9,x9,x23,lsl #5
    stp q20,q21,[x9]
    add x24,x24,#1
    cmp x24,x22
    b.lo Laf8_row
    add x23,x23,#1
    cmp x23,#4
    b.lo Laf8_head
    ALEAVE 96

.globl _tf_attention_backward
_tf_attention_backward:
    ldr w9,[x4,#4]
    cmp w9,#32
    b.ne Lab8_fallback
    ldr w9,[x4,#8]
    cmp w9,#4
    b.ne Lab8_fallback
    ldr w9,[x4]
    sub w10,w9,#1
    cmp w10,#32
    b.lo Lab8_start
Lab8_fallback:
    b _tf_attention_backward_generic
Lab8_start:
    AENTER 224
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x3
    mov x23,x9
    mov x10,#24
    mul x10,x10,x23
    mov x11,x22
    movi v0.4s,#0
Lab8_zero:
    str q0,[x11],#16
    subs x10,x10,#1
    b.ne Lab8_zero
    mov x24,#0
    AADDR x9,Lattention_scale
    ldr s31,[x9]
Lab8_head:
    mov x25,#0
Lab8_row:
    add x9,x21,x25,lsl #7
    add x9,x9,x24,lsl #5
    ldp q20,q21,[x9] // dout, eight channels
    madd x9,x24,x23,x25
    mul x9,x9,x23
    add x26,x20,x9,lsl #2
    mov x9,#384
    mul x9,x25,x9
    add x9,x9,x24,lsl #5
    add x27,x19,x9
    add x28,x22,x9
    ldp q22,q23,[x27] // Q
    movi v24.4s,#0
    movi v25.4s,#0
    fmov s26,wzr
    add x9,x19,#256
    add x9,x9,x24,lsl #5
    add x10,x22,#256
    add x10,x10,x24,lsl #5
    add x11,sp,#96
    mov x12,#0
Lab8_first:
    ldp q0,q1,[x9]
    fmul v2.4s,v20.4s,v0.4s
    fmla v2.4s,v21.4s,v1.4s
    faddp v2.4s,v2.4s,v2.4s
    faddp s2,v2.2s
    str s2,[x11,x12,lsl #2] // dA, reused in second pass
    ldr s0,[x26,x12,lsl #2]
    fmadd s26,s0,s2,s26
    dup v0.4s,v0.s[0]
    ldp q3,q4,[x10]
    fmla v3.4s,v20.4s,v0.4s
    fmla v4.4s,v21.4s,v0.4s
    stp q3,q4,[x10]
    add x9,x9,#384
    add x10,x10,#384
    add x12,x12,#1
    cmp x12,x25
    b.ls Lab8_first
    add x9,x19,#128
    add x9,x9,x24,lsl #5
    add x10,x22,#128
    add x10,x10,x24,lsl #5
    mov x12,#0
Lab8_second:
    ldr s0,[x11,x12,lsl #2]
    fsub s0,s0,s26
    ldr s1,[x26,x12,lsl #2]
    fmul s0,s0,s1
    fmul s0,s0,s31
    dup v0.4s,v0.s[0]
    ldp q1,q2,[x9]
    ldp q3,q4,[x10]
    fmla v24.4s,v1.4s,v0.4s
    fmla v25.4s,v2.4s,v0.4s
    fmla v3.4s,v22.4s,v0.4s
    fmla v4.4s,v23.4s,v0.4s
    stp q3,q4,[x10]
    add x9,x9,#384
    add x10,x10,#384
    add x12,x12,#1
    cmp x12,x25
    b.ls Lab8_second
    stp q24,q25,[x28]
    add x25,x25,#1
    cmp x25,x23
    b.lo Lab8_row
    add x24,x24,#1
    cmp x24,#4
    b.lo Lab8_head
    ALEAVE 224

.section __TEXT,__const
.p2align 4
Lattention_scale:
    .float 0.3535533905932737622
Lattention_indices:
    .long 0,1,2,3
