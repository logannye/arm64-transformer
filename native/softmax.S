// Handwritten four-lane exponential and ASCII-128 cross entropy.
// tf_exp4 is an internal vector ABI: v0.4s -> v0.4s. It clobbers only
// x9-x12, v0-v7, v16-v19, condition flags. v20-v31 are preserved for callers.
// Range reduction/polynomial use two pairs of float64 lanes, then round to f32.
// Exceptional/out-of-range lanes use the original scalar function's behavior.
.text
.p2align 2
.macro SADDR reg, name
    adrp \reg, \name@PAGE
    add \reg, \reg, \name@PAGEOFF
.endm

.globl _tf_exp4
_tf_exp4:
    SADDR x9, Lexp4_bounds
    ldp q1,q2,[x9]
    fcmge v3.4s,v0.4s,v1.4s
    fcmge v4.4s,v2.4s,v0.4s
    and v3.16b,v3.16b,v4.16b
    uminv s3,v3.4s
    fmov w10,s3
    cmn w10,#1
    b.ne Lexp4_scalar
    fcvtl v1.2d,v0.2s
    fcvtl2 v2.2d,v0.4s
    ldr q5,[x9,#32] // log2(e)
    fmul v3.2d,v1.2d,v5.2d
    fmul v4.2d,v2.2d,v5.2d
    fcvtas v3.2d,v3.2d
    fcvtas v4.2d,v4.2d
    scvtf v5.2d,v3.2d
    scvtf v6.2d,v4.2d
    ldr q7,[x9,#48] // ln(2)
    fmls v1.2d,v5.2d,v7.2d
    fmls v2.2d,v6.2d,v7.2d
    mov x10,#1023
    dup v5.2d,x10
    add v3.2d,v3.2d,v5.2d
    add v4.2d,v4.2d,v5.2d
    shl v3.2d,v3.2d,#52
    shl v4.2d,v4.2d,#52
    // Degree nine Taylor on [-ln(2)/2,ln(2)/2], with double evaluation.
    // Truncation is below 1e-11 relative; final error is dominated by f32 rounding.
    add x9,x9,#64
    ldr q16,[x9],#16
    mov v17.16b,v16.16b
    mov w10,#9
Lexp4_poly:
    ldr q6,[x9],#16
    mov v7.16b,v6.16b
    fmla v6.2d,v16.2d,v1.2d
    fmla v7.2d,v17.2d,v2.2d
    mov v16.16b,v6.16b
    mov v17.16b,v7.16b
    subs w10,w10,#1
    b.ne Lexp4_poly
    fmul v16.2d,v16.2d,v3.2d
    fmul v17.2d,v17.2d,v4.2d
    fcvtn v0.2s,v16.2d
    fcvtn2 v0.4s,v17.2d
    ret
Lexp4_scalar:
    stp x29,x30,[sp,#-32]!
    mov x29,sp
    str q0,[sp,#16]
    ldr s0,[sp,#16]
    bl _tf_exp
    str s0,[sp,#16]
    ldr s0,[sp,#20]
    bl _tf_exp
    str s0,[sp,#20]
    ldr s0,[sp,#24]
    bl _tf_exp
    str s0,[sp,#24]
    ldr s0,[sp,#28]
    bl _tf_exp
    str s0,[sp,#28]
    ldr q0,[sp,#16]
    ldp x29,x30,[sp],#32
    ret

// logits,targets,dlogits,rows,vocab -> mean loss in s0.
.globl _tf_cross_entropy
_tf_cross_entropy:
    cmp x4,#128
    b.eq Lce128_start
    b _tf_cross_entropy_generic
Lce128_start:
    stp x29,x30,[sp,#-96]!
    mov x29,sp
    stp x19,x20,[sp,#16]
    stp x21,x22,[sp,#32]
    stp x23,x24,[sp,#48]
    stp x25,x26,[sp,#64]
    stp x27,x28,[sp,#80]
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x3
    mov x23,#0
    fmov d28,xzr
Lce128_row:
    mov x25,x19
    ldr q20,[x25]
    mov x24,#16
Lce128_max:
    ldp q0,q1,[x25],#32
    fmax v20.4s,v20.4s,v0.4s
    fmax v20.4s,v20.4s,v1.4s
    subs x24,x24,#1
    b.ne Lce128_max
    fmaxv s20,v20.4s
    dup v20.4s,v20.s[0]
    movi v21.4s,#0
    mov x25,x19
    mov x26,x21
    mov x24,#32
Lce128_exp:
    ldr q0,[x25],#16
    fsub v0.4s,v0.4s,v20.4s
    bl _tf_exp4
    str q0,[x26],#16
    fadd v21.4s,v21.4s,v0.4s
    subs x24,x24,#1
    b.ne Lce128_exp
    faddp v21.4s,v21.4s,v21.4s
    faddp s21,v21.2s
    fmov s0,s21
    bl _tf_log
    ldr w27,[x20,x23,lsl #2]
    ldr s1,[x19,x27,lsl #2]
    fsub s1,s1,s20
    fsub s0,s0,s1
    fcvt d0,s0
    fadd d28,d28,d0
    dup v21.4s,v21.s[0]
    scvtf s22,x22
    dup v22.4s,v22.s[0]
    SADDR x9,Lindices
    ldr q23,[x9]
    dup v24.4s,w27
    fmov v25.4s,#1.0
    movi v26.4s,#4
    mov x26,x21
    mov x24,#32
Lce128_grad:
    ldr q0,[x26]
    fdiv v0.4s,v0.4s,v21.4s
    cmeq v1.4s,v23.4s,v24.4s
    and v1.16b,v1.16b,v25.16b
    fsub v0.4s,v0.4s,v1.4s
    fdiv v0.4s,v0.4s,v22.4s
    str q0,[x26],#16
    add v23.4s,v23.4s,v26.4s
    subs x24,x24,#1
    b.ne Lce128_grad
    add x19,x19,#512
    add x21,x21,#512
    add x23,x23,#1
    cmp x23,x22
    b.lo Lce128_row
    scvtf d0,x22
    fdiv d0,d28,d0
    fcvt s0,d0
    ldp x27,x28,[sp,#80]
    ldp x25,x26,[sp,#64]
    ldp x23,x24,[sp,#48]
    ldp x21,x22,[sp,#32]
    ldp x19,x20,[sp,#16]
    ldp x29,x30,[sp],#96
    ret

.section __TEXT,__const
.p2align 4
Lexp4_bounds:
    .float -104.0,-104.0,-104.0,-104.0
    .float 88.0,88.0,88.0,88.0
    .double 1.4426950408889634074,1.4426950408889634074
    .double 0.69314718055994530942,0.69314718055994530942
    .double 0.0000027557319223985890653,0.0000027557319223985890653
    .double 0.000024801587301587301587,0.000024801587301587301587
    .double 0.00019841269841269841270,0.00019841269841269841270
    .double 0.0013888888888888888889,0.0013888888888888888889
    .double 0.0083333333333333333333,0.0083333333333333333333
    .double 0.041666666666666666667,0.041666666666666666667
    .double 0.16666666666666666667,0.16666666666666666667
    .double 0.5,0.5
    .double 1.0,1.0
    .double 1.0,1.0
Lindices:
    .long 0,1,2,3
