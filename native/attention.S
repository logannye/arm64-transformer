// Handwritten attention for widths32/64/128/256/512/1024, four heads, T in1..32.
// Width32 retains its existing head-width-eight path. Wider heads use unrolled
// vector-register blocks; other shapes retain the original general implementation.
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

// Stack after saved registers: maximum96, scale100, feature-byte offset104.
.macro ABLOCK_FORWARD tag,w,h,lw,lh,scale
\tag\()_start:
    AENTER 128
    mov x19,x0
    mov x20,x1
    mov x21,x2
    ldr w22,[x3]
    AADDR x9,\scale
    ldr s0,[x9]
    str s0,[sp,#100]
    mov x23,#0
\tag\()_head:
    mov x24,#0
\tag\()_row:
    madd x9,x23,x22,x24
    mul x9,x9,x22
    add x25,x21,x9,lsl #2
    mov x9,#0
\tag\()_clear_scores:
    str wzr,[x25,x9,lsl #2]
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
    AHEAD AHLOAD,64
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
    AHEAD AHDOT,64
    fadd v0.4s,v0.4s,v1.4s
    fadd v2.4s,v2.4s,v3.4s
    fadd v0.4s,v0.4s,v2.4s
    faddp v0.4s,v0.4s,v0.4s
    faddp s0,v0.2s
    ldr s7,[x25,x28,lsl #2]
    fadd s0,s0,s7
    str s0,[x25,x28,lsl #2]
    add x27,x27,x10
    add x28,x28,#1
    cmp x28,x24
    b.ls \tag\()_score_block
    ldr x9,[sp,#104]
    add x9,x9,#256
    str x9,[sp,#104]
    cmp x9,#(\h*4)
    b.lo \tag\()_query_block
    mov w9,#0xff800000
    fmov s6,w9
    ldr s7,[sp,#100]
    mov x28,#0
\tag\()_scale_scores:
    ldr s0,[x25,x28,lsl #2]
    fmul s0,s0,s7
    str s0,[x25,x28,lsl #2]
    fmax s6,s6,s0
    add x28,x28,#1
    cmp x28,x24
    b.ls \tag\()_scale_scores
    str s6,[sp,#96]
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
    ALEAVE 128
.endm

// Cache32 query channels and their32 dQ accumulators together. This keeps the
// query block in registers while visiting every causal key for that block.
.macro BQBLOCK32 op
    \op 16,24,0
    \op 17,25,16
    \op 18,26,32
    \op 19,27,48
    \op 20,28,64
    \op 21,29,80
    \op 22,30,96
    \op 23,31,112
.endm
.macro BQLOAD q,d,off
    ldr q\q,[x27,#\off]
    movi v\d\().4s,#0
.endm
.macro BQUPDATE q,d,off
    ldr q1,[x9,#\off]
    ldr q2,[x10,#\off]
    fmla v\d\().4s,v1.4s,v0.4s
    fmla v2.4s,v\q\().4s,v0.4s
    str q2,[x10,#\off]
.endm
.macro BQSTORE q,d,off
    str q\d,[x28,#\off]
.endm

// dA[32] occupies96..223 and is replaced by dScore after one row contraction.
// Scale224, row contraction228, feature-byte offset232; frame remains bounded.
.macro ABLOCK_BACKWARD tag,w,h,lw,lh,scale
\tag\()_start:
    AENTER 256
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
    madd x9,x24,x23,x25
    mul x9,x9,x23
    add x26,x20,x9,lsl #2
    movi v0.4s,#0
    stp q0,q0,[sp,#96]
    stp q0,q0,[sp,#128]
    stp q0,q0,[sp,#160]
    stp q0,q0,[sp,#192]
    str xzr,[sp,#232]
\tag\()_first_block:
    ldr x14,[sp,#232]
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
    AHEAD BHDV,64
    fadd v0.4s,v0.4s,v1.4s
    fadd v2.4s,v2.4s,v3.4s
    fadd v0.4s,v0.4s,v2.4s
    faddp v0.4s,v0.4s,v0.4s
    faddp s0,v0.2s
    ldr s7,[x11,x12,lsl #2]
    fadd s0,s0,s7
    str s0,[x11,x12,lsl #2]
    add x9,x9,x13
    add x10,x10,x13
    add x12,x12,#1
    cmp x12,x25
    b.ls \tag\()_first
    ldr x14,[sp,#232]
    add x14,x14,#256
    str x14,[sp,#232]
    cmp x14,#(\h*4)
    b.lo \tag\()_first_block
    mov x12,#0
    fmov s6,wzr
\tag\()_row_dot:
    ldr s0,[x11,x12,lsl #2]
    ldr s1,[x26,x12,lsl #2]
    fmadd s6,s0,s1,s6
    add x12,x12,#1
    cmp x12,x25
    b.ls \tag\()_row_dot
    str s6,[sp,#228]
    ldr s7,[sp,#224]
    mov x12,#0
\tag\()_dscore:
    ldr s0,[x11,x12,lsl #2]
    fsub s0,s0,s6
    ldr s1,[x26,x12,lsl #2]
    fmul s0,s0,s1
    fmul s0,s0,s7
    str s0,[x11,x12,lsl #2]
    add x12,x12,#1
    cmp x12,x25
    b.ls \tag\()_dscore
    str xzr,[sp,#232]
\tag\()_second_block:
    ldr x14,[sp,#232]
    mov x9,#(\w*12)
    mul x9,x25,x9
    add x9,x9,x24,lsl #(\lh+2)
    add x9,x9,x14
    add x27,x19,x9
    add x28,x22,x9
    BQBLOCK32 BQLOAD
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
    ldr s0,[x11,x12,lsl #2]
    dup v0.4s,v0.s[0]
    BQBLOCK32 BQUPDATE
    add x9,x9,x13
    add x10,x10,x13
    add x12,x12,#1
    cmp x12,x25
    b.ls \tag\()_second
    BQBLOCK32 BQSTORE
    ldr x14,[sp,#232]
    add x14,x14,#128
    str x14,[sp,#232]
    cmp x14,#(\h*4)
    b.lo \tag\()_second_block
    add x25,x25,#1
    cmp x25,x23
    b.lo \tag\()_row
    add x24,x24,#1
    cmp x24,#4
    b.lo \tag\()_head
    ALEAVE 256
.endm

AWIDE_FORWARD Law64,64,16,6,4,Lattention_scale16
AWIDE_FORWARD Law128,128,32,7,5,Lattention_scale32
AWIDE_FORWARD Law256,256,64,8,6,Lattention_scale64
AWIDE_BACKWARD Labw64,64,16,6,4,Lattention_scale16
AWIDE_BACKWARD Labw128,128,32,7,5,Lattention_scale32
AWIDE_BACKWARD Labw256,256,64,8,6,Lattention_scale64
ABLOCK_FORWARD Labf512,512,128,9,7,Lattention_scale128
ABLOCK_FORWARD Labf1024,1024,256,10,8,Lattention_scale256
ABLOCK_BACKWARD Labb512,512,128,9,7,Lattention_scale128
ABLOCK_BACKWARD Labb1024,1024,256,10,8,Lattention_scale256

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
Lattention_scale128:
    .float 0.08838834764831844055
Lattention_scale256:
    .float 0.0625
