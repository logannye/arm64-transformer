// Handwritten attention for widths32/64/128/256/512/1024, four heads, T in1..32.
// Width32 retains its existing head-width-eight path. Wider heads use unrolled
// vector-register blocks; other shapes retain the original general implementation.
// Wide (512/1024) forward keeps QK scores in FP64 through max subtraction;
// centered scores, probabilities, value mixing and stored tensors remain FP32.
// Its backward retains dA, the softmax contraction, dScore and dQ sums in FP64.
// Other forward paths use NEON FP32 QK/value operations and stable exponentiation.
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
    b.ne Laf_wide_dispatch
    ldr w9,[x3,#8]
    cmp w9,#4
    b.ne Laf8_fallback
    ldr w9,[x3]
    sub w10,w9,#1
    cmp w10,#32
    b.lo Laf8_start
Laf8_fallback:
    b _tf_attention_forward_generic
Laf_wide_dispatch:
    ldr w10,[x3,#8]
    cmp w10,#4
    b.ne Laf8_fallback
    ldr w10,[x3]
    sub w11,w10,#1
    cmp w11,#32
    b.hs Laf8_fallback
    cmp w9,#64
    b.eq Law64_start
    cmp w9,#128
    b.eq Law128_start
    cmp w9,#256
    b.eq Law256_start
    cmp w9,#512
    b.eq Labf512_start
    cmp w9,#1024
    b.eq Labf1024_start
    b Laf8_fallback
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
    b.ne Lab_wide_dispatch
    ldr w9,[x4,#8]
    cmp w9,#4
    b.ne Lab8_fallback
    ldr w9,[x4]
    sub w10,w9,#1
    cmp w10,#32
    b.lo Lab8_start
Lab8_fallback:
    b _tf_attention_backward_generic
Lab_wide_dispatch:
    ldr w10,[x4,#8]
    cmp w10,#4
    b.ne Lab8_fallback
    ldr w10,[x4]
    sub w11,w10,#1
    cmp w11,#32
    b.hs Lab8_fallback
    cmp w9,#64
    b.eq Labw64_start
    cmp w9,#128
    b.eq Labw128_start
    cmp w9,#256
    b.eq Labw256_start
    cmp w9,#512
    b.eq Labb512_start
    cmp w9,#1024
    b.eq Labb1024_start
    b Lab8_fallback
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

// Apply a register-block operation to a head of16,32 or64 channels.
// Each operation takes register index, byte offset, and dot accumulator index.
.macro AHEAD op, h
    \op 16,0,0
    \op 17,16,1
    \op 18,32,2
    \op 19,48,3
    .if \h >= 32
    \op 20,64,0
    \op 21,80,1
    \op 22,96,2
    \op 23,112,3
    .endif
    .if \h >= 64
    \op 24,128,0
    \op 25,144,1
    \op 26,160,2
    \op 27,176,3
    \op 28,192,0
    \op 29,208,1
    \op 30,224,2
    \op 31,240,3
    .endif
.endm
.macro AHLOAD r,off,acc
    ldr q\r,[x26,#\off]
.endm
.macro AHDOT r,off,acc
    ldr q4,[x27,#\off]
    fmla v\acc\().4s,v\r\().4s,v4.4s
.endm
.macro AHZERO r,off,acc
    movi v\r\().4s,#0
.endm
.macro AHVALUE r,off,acc
    ldr q1,[x27,#\off]
    fmla v\r\().4s,v1.4s,v0.4s
.endm
.macro AHSTORE r,off,acc
    str q\r,[x9,#\off]
.endm
.macro BHLOAD r,off,acc
    ldr q\r,[x9,#\off]
.endm
.macro BHDV r,off,acc
    ldr q5,[x9,#\off]
    fmla v\acc\().4s,v\r\().4s,v5.4s
    ldr q6,[x10,#\off]
    fmla v6.4s,v\r\().4s,v4.4s
    str q6,[x10,#\off]
.endm
.macro BHDQ r,off,acc
    ldr q1,[x27,#\off]
    ldr q2,[x9,#\off]
    ldr q3,[x10,#\off]
    fmla v\r\().4s,v2.4s,v0.4s
    fmla v3.4s,v1.4s,v0.4s
    str q3,[x10,#\off]
.endm
.macro BHSTORE r,off,acc
    str q\r,[x28,#\off]
.endm

// Macro expansion specializes strides and register blocks in handwritten code.
// w=model width, h=head width, lw=log2(model width), lh=log2(head width).
.macro AWIDE_FORWARD tag,w,h,lw,lh,scale
\tag\()_start:
    AENTER 112
    mov x19,x0
    mov x20,x1
    mov x21,x2
    ldr w22,[x3]
    mov x23,#0
    AADDR x9,\scale
    ldr s0,[x9]
    str s0,[sp,#100]
\tag\()_head:
    mov x24,#0
\tag\()_row:
    mov x9,#(\w*12)
    mul x26,x24,x9
    add x26,x19,x26
    add x26,x26,x23,lsl #(\lh+2)
    AHEAD AHLOAD,\h
    madd x9,x23,x22,x24
    mul x9,x9,x22
    add x25,x21,x9,lsl #2
    add x27,x19,#(\w*4)
    add x27,x27,x23,lsl #(\lh+2)
    mov x28,#0
    mov w9,#0xff800000
    str w9,[sp,#96]
\tag\()_score:
    movi v0.4s,#0
    movi v1.4s,#0
    movi v2.4s,#0
    movi v3.4s,#0
    AHEAD AHDOT,\h
    fadd v0.4s,v0.4s,v1.4s
    fadd v2.4s,v2.4s,v3.4s
    fadd v0.4s,v0.4s,v2.4s
    faddp v0.4s,v0.4s,v0.4s
    faddp s0,v0.2s
    ldr s7,[sp,#100]
    fmul s0,s0,s7
    str s0,[x25,x28,lsl #2]
    ldr s6,[sp,#96]
    fmax s6,s6,s0
    str s6,[sp,#96]
    add x27,x27,#(\w*12)
    add x28,x28,#1
    cmp x28,x24
    b.ls \tag\()_score
    mov x9,x28
\tag\()_mask:
    cmp x9,x22
    b.hs \tag\()_exp_start
    str wzr,[x25,x9,lsl #2]
    add x9,x9,#1
    b \tag\()_mask
\tag\()_exp_start:
    ldr s22,[sp,#96]
    dup v22.4s,v22.s[0]
    movi v23.4s,#0
    mov x27,x25
    lsr x28,x28,#2
    cbz x28,\tag\()_exp_tail
\tag\()_exp_vector:
    ldr q0,[x27]
    fsub v0.4s,v0.4s,v22.4s
    bl _tf_exp4
    str q0,[x27],#16
    fadd v23.4s,v23.4s,v0.4s
    subs x28,x28,#1
    b.ne \tag\()_exp_vector
\tag\()_exp_tail:
    add x28,x24,#1
    and x28,x28,#3
    cbz x28,\tag\()_normalize
    mov v0.16b,v22.16b
    mov x9,x27
    ld1 {v0.s}[0],[x9],#4
    cmp x28,#1
    b.eq \tag\()_tail_exp
    ld1 {v0.s}[1],[x9],#4
    cmp x28,#2
    b.eq \tag\()_tail_exp
    ld1 {v0.s}[2],[x9]
\tag\()_tail_exp:
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
    b.eq \tag\()_normalize
    st1 {v0.s}[1],[x27],#4
    cmp x28,#2
    b.eq \tag\()_normalize
    st1 {v0.s}[2],[x27]
\tag\()_normalize:
    faddp v23.4s,v23.4s,v23.4s
    faddp s23,v23.2s
    dup v24.4s,v23.s[0]
    mov x27,x25
    add x28,x24,#1
    lsr x28,x28,#2
    cbz x28,\tag\()_norm_tail_start
\tag\()_norm_vector:
    ldr q0,[x27]
    fdiv v0.4s,v0.4s,v24.4s
    str q0,[x27],#16
    subs x28,x28,#1
    b.ne \tag\()_norm_vector
\tag\()_norm_tail_start:
    add x28,x24,#1
    and x28,x28,#3
    cbz x28,\tag\()_value_start
\tag\()_norm_tail:
    ldr s0,[x27]
    fdiv s0,s0,s24
    str s0,[x27],#4
    subs x28,x28,#1
    b.ne \tag\()_norm_tail
\tag\()_value_start:
    AHEAD AHZERO,\h
    add x27,x19,#(\w*8)
    add x27,x27,x23,lsl #(\lh+2)
    mov x28,#0
\tag\()_value:
    ldr s0,[x25,x28,lsl #2]
    dup v0.4s,v0.s[0]
    AHEAD AHVALUE,\h
    add x27,x27,#(\w*12)
    add x28,x28,#1
    cmp x28,x24
    b.ls \tag\()_value
    add x9,x20,x24,lsl #(\lw+2)
    add x9,x9,x23,lsl #(\lh+2)
    AHEAD AHSTORE,\h
    add x24,x24,#1
    cmp x24,x22
    b.lo \tag\()_row
    add x23,x23,#1
    cmp x23,#4
    b.lo \tag\()_head
    ALEAVE 112
.endm

.macro AWIDE_BACKWARD tag,w,h,lw,lh,scale
\tag\()_start:
    AENTER 240
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x3
    ldr w23,[x4]
    mov x10,#(\w*3/4)
    mul x10,x10,x23
    mov x11,x22
    movi v0.4s,#0
\tag\()_zero:
    str q0,[x11],#16
    subs x10,x10,#1
    b.ne \tag\()_zero
    AADDR x9,\scale
    ldr s0,[x9]
    str s0,[sp,#224]
    mov x24,#0
\tag\()_head:
    mov x25,#0
\tag\()_row:
    add x9,x21,x25,lsl #(\lw+2)
    add x9,x9,x24,lsl #(\lh+2)
    AHEAD BHLOAD,\h
    madd x9,x24,x23,x25
    mul x9,x9,x23
    add x26,x20,x9,lsl #2
    mov x9,#(\w*12)
    mul x9,x25,x9
    add x9,x9,x24,lsl #(\lh+2)
    add x27,x19,x9
    add x28,x22,x9
    str wzr,[sp,#228]
    add x9,x19,#(\w*8)
    add x9,x9,x24,lsl #(\lh+2)
    add x10,x22,#(\w*8)
    add x10,x10,x24,lsl #(\lh+2)
    add x11,sp,#96
    mov x12,#0
\tag\()_first:
    movi v0.4s,#0
    movi v1.4s,#0
    movi v2.4s,#0
    movi v3.4s,#0
    ldr s4,[x26,x12,lsl #2]
    dup v4.4s,v4.s[0]
    AHEAD BHDV,\h
    fadd v0.4s,v0.4s,v1.4s
    fadd v2.4s,v2.4s,v3.4s
    fadd v0.4s,v0.4s,v2.4s
    faddp v0.4s,v0.4s,v0.4s
    faddp s0,v0.2s
    str s0,[x11,x12,lsl #2]
    ldr s6,[sp,#228]
    fmadd s6,s4,s0,s6
    str s6,[sp,#228]
    add x9,x9,#(\w*12)
    add x10,x10,#(\w*12)
    add x12,x12,#1
    cmp x12,x25
    b.ls \tag\()_first
    AHEAD AHZERO,\h
    add x9,x19,#(\w*4)
    add x9,x9,x24,lsl #(\lh+2)
    add x10,x22,#(\w*4)
    add x10,x10,x24,lsl #(\lh+2)
    mov x12,#0
\tag\()_second:
    ldr s0,[x11,x12,lsl #2]
    ldr s1,[sp,#228]
    fsub s0,s0,s1
    ldr s1,[x26,x12,lsl #2]
    fmul s0,s0,s1
    ldr s1,[sp,#224]
    fmul s0,s0,s1
    dup v0.4s,v0.s[0]
    AHEAD BHDQ,\h
    add x9,x9,#(\w*12)
    add x10,x10,#(\w*12)
    add x12,x12,#1
    cmp x12,x25
    b.ls \tag\()_second
    AHEAD BHSTORE,\h
    add x25,x25,#1
    cmp x25,x23
    b.lo \tag\()_row
    add x24,x24,#1
    cmp x24,#4
    b.lo \tag\()_head
    ALEAVE 240
.endm

// Heads larger than64 channels use bounded feature blocks. A Q or dout block
// occupies v16-v31; its four dot accumulators stay in caller-saved v0-v3.
// The existing small-width paths above are intentionally unchanged.
.macro ABLOCK_SOFTMAX tag
\tag\()_exp_start:
    ldr s22,[sp,#96]
    dup v22.4s,v22.s[0]
    movi v23.4s,#0
    mov x27,x25
    lsr x28,x28,#2
    cbz x28,\tag\()_exp_tail
\tag\()_exp_vector:
    ldr q0,[x27]
    fsub v0.4s,v0.4s,v22.4s
    bl _tf_exp4
    str q0,[x27],#16
    fadd v23.4s,v23.4s,v0.4s
    subs x28,x28,#1
    b.ne \tag\()_exp_vector
\tag\()_exp_tail:
    add x28,x24,#1
    and x28,x28,#3
    cbz x28,\tag\()_normalize
    mov v0.16b,v22.16b
    mov x9,x27
    ld1 {v0.s}[0],[x9],#4
    cmp x28,#1
    b.eq \tag\()_tail_exp
    ld1 {v0.s}[1],[x9],#4
    cmp x28,#2
    b.eq \tag\()_tail_exp
    ld1 {v0.s}[2],[x9]
\tag\()_tail_exp:
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
    b.eq \tag\()_normalize
    st1 {v0.s}[1],[x27],#4
    cmp x28,#2
    b.eq \tag\()_normalize
    st1 {v0.s}[2],[x27]
\tag\()_normalize:
    faddp v23.4s,v23.4s,v23.4s
    faddp s23,v23.2s
    dup v24.4s,v23.s[0]
    mov x27,x25
    add x28,x24,#1
    lsr x28,x28,#2
    cbz x28,\tag\()_norm_tail_start
\tag\()_norm_vector:
    ldr q0,[x27]
    fdiv v0.4s,v0.4s,v24.4s
    str q0,[x27],#16
    subs x28,x28,#1
    b.ne \tag\()_norm_vector
\tag\()_norm_tail_start:
    add x28,x24,#1
    and x28,x28,#3
    cbz x28,\tag\()_value_start
\tag\()_norm_tail:
    ldr s0,[x27]
    fdiv s0,s0,s24
    str s0,[x27],#4
    subs x28,x28,#1
    b.ne \tag\()_norm_tail
.endm

// Cache32 converted query channels in caller-saved v16-v31. Conversion before
// multiplication preserves the exact stored FP32 operands, including cancellation
// residuals; four FP64 chains bound the depth of each 32-channel block reduction.
.macro AQ64BLOCK op
    \op 16,17,0,0,1
    \op 18,19,16,2,3
    \op 20,21,32,0,1
    \op 22,23,48,2,3
    \op 24,25,64,0,1
    \op 26,27,80,2,3
    \op 28,29,96,0,1
    \op 30,31,112,2,3
.endm
.macro AQ64LOAD lo,hi,off,acc0,acc1
    ldr q4,[x26,#\off]
    fcvtl v\lo\().2d,v4.2s
    fcvtl2 v\hi\().2d,v4.4s
.endm
.macro AQ64DOT lo,hi,off,acc0,acc1
    ldr q4,[x27,#\off]
    fcvtl v5.2d,v4.2s
    fcvtl2 v6.2d,v4.4s
    fmla v\acc0\().2d,v\lo\().2d,v5.2d
    fmla v\acc1\().2d,v\hi\().2d,v6.2d
.endm

// Stack after saved registers: FP32 centered maximum96 (=0), feature-byte
// offset104, FP64 scale112, and32 FP64 scores128..383. One 400-byte frame serves
// every supported T and head width. Future Q/K/V rows never enter a causal sum.
.macro ABLOCK_FORWARD tag,w,h,lw,lh,scale
\tag\()_start:
    AENTER 400
    mov x19,x0
    mov x20,x1
    mov x21,x2
    ldr w22,[x3]
    AADDR x9,\scale
    ldr d0,[x9]
    str d0,[sp,#112]
    mov x23,#0
\tag\()_head:
    mov x24,#0
\tag\()_row:
    madd x9,x23,x22,x24
    mul x9,x9,x22
    add x25,x21,x9,lsl #2
    add x15,sp,#128
    mov x9,#0
\tag\()_clear_scores:
    str wzr,[x25,x9,lsl #2]
    str xzr,[x15,x9,lsl #3]
    add x9,x9,#1
    cmp x9,x22
    b.lo \tag\()_clear_scores
    str xzr,[sp,#104]
\tag\()_query_block:
    mov x9,#(\w*12)
    mul x26,x24,x9
    add x26,x19,x26
    add x26,x26,x23,lsl #(\lh+2)
    ldr x10,[sp,#104]
    add x26,x26,x10
    AQ64BLOCK AQ64LOAD
    add x27,x19,#(\w*4)
    add x27,x27,x23,lsl #(\lh+2)
    add x27,x27,x10
    mov x10,#(\w*12)
    mov x28,#0
\tag\()_score_block:
    movi v0.4s,#0
    movi v1.4s,#0
    movi v2.4s,#0
    movi v3.4s,#0
    AQ64BLOCK AQ64DOT
    fadd v0.2d,v0.2d,v1.2d
    fadd v2.2d,v2.2d,v3.2d
    fadd v0.2d,v0.2d,v2.2d
    faddp d0,v0.2d
    ldr d7,[x15,x28,lsl #3]
    fadd d0,d0,d7
    str d0,[x15,x28,lsl #3]
    add x27,x27,x10
    add x28,x28,#1
    cmp x28,x24
    b.ls \tag\()_score_block
    ldr x9,[sp,#104]
    add x9,x9,#128
    str x9,[sp,#104]
    cmp x9,#(\h*4)
    b.lo \tag\()_query_block
    mov x9,#0xfff0000000000000
    fmov d6,x9
    ldr d7,[sp,#112]
    mov x28,#0
\tag\()_scale_scores:
    ldr d0,[x15,x28,lsl #3]
    fmul d0,d0,d7
    str d0,[x15,x28,lsl #3]
    fmax d6,d6,d0
    add x28,x28,#1
    cmp x28,x24
    b.ls \tag\()_scale_scores
    // Keep both scores and their maximum in FP64 until their small differences
    // are formed. Converting uncentered scores first loses the precision gain.
    mov x28,#0
\tag\()_center_scores:
    ldr d0,[x15,x28,lsl #3]
    fsub d0,d0,d6
    fcvt s0,d0
    str s0,[x25,x28,lsl #2]
    add x28,x28,#1
    cmp x28,x24
    b.ls \tag\()_center_scores
    str wzr,[sp,#96]
    ABLOCK_SOFTMAX \tag
\tag\()_value_start:
    str xzr,[sp,#104]
\tag\()_value_block:
    AHEAD AHZERO,64
    add x27,x19,#(\w*8)
    add x27,x27,x23,lsl #(\lh+2)
    ldr x9,[sp,#104]
    add x27,x27,x9
    mov x10,#(\w*12)
    mov x28,#0
\tag\()_value:
    ldr s0,[x25,x28,lsl #2]
    dup v0.4s,v0.s[0]
    AHEAD AHVALUE,64
    add x27,x27,x10
    add x28,x28,#1
    cmp x28,x24
    b.ls \tag\()_value
    add x9,x20,x24,lsl #(\lw+2)
    add x9,x9,x23,lsl #(\lh+2)
    ldr x11,[sp,#104]
    add x9,x9,x11
    AHEAD AHSTORE,64
    add x11,x11,#256
    str x11,[sp,#104]
    cmp x11,#(\h*4)
    b.lo \tag\()_value_block
    add x24,x24,#1
    cmp x24,x22
    b.lo \tag\()_row
    add x23,x23,#1
    cmp x23,#4
    b.lo \tag\()_head
    ALEAVE 400
.endm

// dA is a cancellation-sensitive reduction when K has a large common component.
// Retain it in FP64 through the row contraction, scaled dScore and dQ sum. dV
// retains its existing FP32 multiply-add order and caller-owned FP32 destination.
.macro BHDV64 r,off,acc
    ldr q5,[x9,#\off]
    fcvtl v6.2d,v5.2s
    fcvtl v7.2d,v\r\().2s
    fmla v\acc\().2d,v6.2d,v7.2d
    fcvtl2 v6.2d,v5.4s
    fcvtl2 v7.2d,v\r\().4s
    fmla v\acc\().2d,v6.2d,v7.2d
    ldr q6,[x10,#\off]
    fmla v6.4s,v\r\().4s,v4.4s
    str q6,[x10,#\off]
.endm

// Cache16 query channels and their16 dQ accumulators in FP64. Each dQ value is
// rounded only after all causal keys. dK adds each query contribution in FP64
// before rounding to its caller-owned FP32 accumulator; no quadratic cache.
.macro BQ64BLOCK op
    \op 16,17,24,25,0
    \op 18,19,26,27,16
    \op 20,21,28,29,32
    \op 22,23,30,31,48
.endm
.macro BQ64LOAD qlo,qhi,dlo,dhi,off
    ldr q1,[x27,#\off]
    fcvtl v\qlo\().2d,v1.2s
    fcvtl2 v\qhi\().2d,v1.4s
    movi v\dlo\().4s,#0
    movi v\dhi\().4s,#0
.endm
.macro BQ64UPDATE qlo,qhi,dlo,dhi,off
    ldr q1,[x9,#\off]
    fcvtl v2.2d,v1.2s
    fcvtl2 v3.2d,v1.4s
    fmla v\dlo\().2d,v2.2d,v0.2d
    fmla v\dhi\().2d,v3.2d,v0.2d
    ldr q4,[x10,#\off]
    fcvtl v5.2d,v4.2s
    fcvtl2 v6.2d,v4.4s
    fmla v5.2d,v\qlo\().2d,v0.2d
    fmla v6.2d,v\qhi\().2d,v0.2d
    fcvtn v4.2s,v5.2d
    fcvtn2 v4.4s,v6.2d
    str q4,[x10,#\off]
.endm
.macro BQ64STORE qlo,qhi,dlo,dhi,off
    fcvtn v1.2s,v\dlo\().2d
    fcvtn2 v1.4s,v\dhi\().2d
    str q1,[x28,#\off]
.endm

// FP64 dA[32] occupies96..351 and is replaced by FP64 dScore after contraction.
// FP64 scale352, row contraction360, feature-byte offset368; one384-byte frame.
.macro ABLOCK_BACKWARD tag,w,h,lw,lh,scale
\tag\()_start:
    AENTER 384
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x3
    ldr w23,[x4]
    mov x10,#(\w*3/4)
    mul x10,x10,x23
    mov x11,x22
    movi v0.4s,#0
\tag\()_zero:
    str q0,[x11],#16
    subs x10,x10,#1
    b.ne \tag\()_zero
    AADDR x9,\scale
    ldr d0,[x9]
    str d0,[sp,#352]
    mov x24,#0
\tag\()_head:
    mov x25,#0
\tag\()_row:
    madd x9,x24,x23,x25
    mul x9,x9,x23
    add x26,x20,x9,lsl #2
    movi v0.4s,#0
    stp q0,q0,[sp,#96]
    stp q0,q0,[sp,#128]
    stp q0,q0,[sp,#160]
    stp q0,q0,[sp,#192]
    stp q0,q0,[sp,#224]
    stp q0,q0,[sp,#256]
    stp q0,q0,[sp,#288]
    stp q0,q0,[sp,#320]
    str xzr,[sp,#368]
\tag\()_first_block:
    ldr x14,[sp,#368]
    add x9,x21,x25,lsl #(\lw+2)
    add x9,x9,x24,lsl #(\lh+2)
    add x9,x9,x14
    AHEAD BHLOAD,64
    add x9,x19,#(\w*8)
    add x9,x9,x24,lsl #(\lh+2)
    add x9,x9,x14
    add x10,x22,#(\w*8)
    add x10,x10,x24,lsl #(\lh+2)
    add x10,x10,x14
    add x11,sp,#96
    mov x12,#0
    mov x13,#(\w*12)
\tag\()_first:
    movi v0.4s,#0
    movi v1.4s,#0
    movi v2.4s,#0
    movi v3.4s,#0
    ldr s4,[x26,x12,lsl #2]
    dup v4.4s,v4.s[0]
    AHEAD BHDV64,64
    fadd v0.2d,v0.2d,v1.2d
    fadd v2.2d,v2.2d,v3.2d
    fadd v0.2d,v0.2d,v2.2d
    faddp d0,v0.2d
    ldr d7,[x11,x12,lsl #3]
    fadd d0,d0,d7
    str d0,[x11,x12,lsl #3]
    add x9,x9,x13
    add x10,x10,x13
    add x12,x12,#1
    cmp x12,x25
    b.ls \tag\()_first
    ldr x14,[sp,#368]
    add x14,x14,#256
    str x14,[sp,#368]
    cmp x14,#(\h*4)
    b.lo \tag\()_first_block
    mov x12,#0
    fmov d6,xzr
\tag\()_row_dot:
    ldr d0,[x11,x12,lsl #3]
    ldr s1,[x26,x12,lsl #2]
    fcvt d1,s1
    fmadd d6,d0,d1,d6
    add x12,x12,#1
    cmp x12,x25
    b.ls \tag\()_row_dot
    str d6,[sp,#360]
    ldr d7,[sp,#352]
    mov x12,#0
\tag\()_dscore:
    ldr d0,[x11,x12,lsl #3]
    fsub d0,d0,d6
    ldr s1,[x26,x12,lsl #2]
    fcvt d1,s1
    fmul d0,d0,d1
    fmul d0,d0,d7
    str d0,[x11,x12,lsl #3]
    add x12,x12,#1
    cmp x12,x25
    b.ls \tag\()_dscore
    str xzr,[sp,#368]
\tag\()_second_block:
    ldr x14,[sp,#368]
    mov x9,#(\w*12)
    mul x9,x25,x9
    add x9,x9,x24,lsl #(\lh+2)
    add x9,x9,x14
    add x27,x19,x9
    add x28,x22,x9
    BQ64BLOCK BQ64LOAD
    add x9,x19,#(\w*4)
    add x9,x9,x24,lsl #(\lh+2)
    add x9,x9,x14
    add x10,x22,#(\w*4)
    add x10,x10,x24,lsl #(\lh+2)
    add x10,x10,x14
    add x11,sp,#96
    mov x12,#0
    mov x13,#(\w*12)
\tag\()_second:
    ldr d0,[x11,x12,lsl #3]
    dup v0.2d,v0.d[0]
    BQ64BLOCK BQ64UPDATE
    add x9,x9,x13
    add x10,x10,x13
    add x12,x12,#1
    cmp x12,x25
    b.ls \tag\()_second
    BQ64BLOCK BQ64STORE
    ldr x14,[sp,#368]
    add x14,x14,#64
    str x14,[sp,#368]
    cmp x14,#(\h*4)
    b.lo \tag\()_second_block
    add x25,x25,#1
    cmp x25,x23
    b.lo \tag\()_row
    add x24,x24,#1
    cmp x24,#4
    b.lo \tag\()_head
    ALEAVE 384
.endm

AWIDE_FORWARD Law64,64,16,6,4,Lattention_scale16
AWIDE_FORWARD Law128,128,32,7,5,Lattention_scale32
AWIDE_FORWARD Law256,256,64,8,6,Lattention_scale64
AWIDE_BACKWARD Labw64,64,16,6,4,Lattention_scale16
AWIDE_BACKWARD Labw128,128,32,7,5,Lattention_scale32
AWIDE_BACKWARD Labw256,256,64,8,6,Lattention_scale64
ABLOCK_FORWARD Labf512,512,128,9,7,Lattention_scale128_f64
ABLOCK_FORWARD Labf1024,1024,256,10,8,Lattention_scale256_f64
ABLOCK_BACKWARD Labb512,512,128,9,7,Lattention_scale128_f64
ABLOCK_BACKWARD Labb1024,1024,256,10,8,Lattention_scale256_f64

.section __TEXT,__const
.p2align 4
Lattention_scale:
    .float 0.3535533905932737622
Lattention_indices:
    .long 0,1,2,3
Lattention_scale16:
    .float 0.25
Lattention_scale32:
    .float 0.1767766952966368811
Lattention_scale64:
    .float 0.125
.p2align 3
Lattention_scale128_f64:
    .double 0.08838834764831844055
Lattention_scale256_f64:
    .double 0.0625
