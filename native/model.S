// Complete handwritten one-block transformer forward/backward and embeddings.
#include "layout.inc"
.text
.p2align 2
.macro ADDR reg, symbol
 adrp \reg, \symbol@PAGE
 add \reg, \reg, \symbol@PAGEOFF
.endm
.macro PTR reg, base, offset
 add \reg, \base, #(\offset >> 12), lsl #12
 add \reg, \reg, #(\offset & 4095)
.endm
.macro ENTER
 stp x29,x30,[sp,#-96]!
 mov x29,sp
 stp x19,x20,[sp,#16]
 stp x21,x22,[sp,#32]
 stp x23,x24,[sp,#48]
 stp x25,x26,[sp,#64]
 stp d8,d9,[sp,#80]
.endm
.macro LEAVE
 ldp d8,d9,[sp,#80]
 ldp x25,x26,[sp,#64]
 ldp x23,x24,[sp,#48]
 ldp x21,x22,[sp,#32]
 ldp x19,x20,[sp,#16]
 ldp x29,x30,[sp],#96
 ret
.endm

.globl _tf_forward
_tf_forward:
 ENTER
 mov x19,x0
 mov x20,x1
 mov x21,x2
 PTR x9,x19,P_POSITION
 PTR x10,x21,W_X
 mov w11,#CONTEXT
 mov x12,x20
Lembed:
 ldr w13,[x12],#4
 add x13,x19,x13,lsl #7
 mov w14,#8
Lembed_vec:
 ldr q0,[x13],#16
 ldr q1,[x9],#16
 fadd v0.4s,v0.4s,v1.4s
 str q0,[x10],#16
 subs w14,w14,#1
 b.ne Lembed_vec
 subs w11,w11,#1
 b.ne Lembed
 PTR x0,x21,W_X
 PTR x1,x19,P_LN1_WEIGHT
 PTR x2,x19,P_LN1_BIAS
 PTR x3,x21,W_N1
 PTR x4,x21,W_NORM1
 PTR x5,x21,W_INV1
 mov x6,#CONTEXT
 mov x7,#WIDTH
 bl _tf_ln_forward
 PTR x0,x21,W_N1
 PTR x1,x19,P_QKV_WEIGHT
 PTR x2,x19,P_QKV_BIAS
 PTR x3,x21,W_QKV
 ADDR x4,Lqkvshape
 bl _tf_linear_forward
 PTR x0,x21,W_QKV
 PTR x1,x21,W_ATTENTION
 PTR x2,x21,W_PROBS
 ADDR x3,Lattshape
 bl _tf_attention_forward
 PTR x0,x21,W_ATTENTION
 PTR x1,x19,P_ATTENTION_WEIGHT
 PTR x2,x19,P_ATTENTION_BIAS
 PTR x3,x21,W_PROJECTED
 ADDR x4,Lprojshape
 bl _tf_linear_forward
 PTR x0,x21,W_X
 PTR x1,x21,W_PROJECTED
 PTR x2,x21,W_R1
 mov x3,#1024
 bl Ladd
 PTR x0,x21,W_R1
 PTR x1,x19,P_LN2_WEIGHT
 PTR x2,x19,P_LN2_BIAS
 PTR x3,x21,W_N2
 PTR x4,x21,W_NORM2
 PTR x5,x21,W_INV2
 mov x6,#CONTEXT
 mov x7,#WIDTH
 bl _tf_ln_forward
 PTR x0,x21,W_N2
 PTR x1,x19,P_FF1_WEIGHT
 PTR x2,x19,P_FF1_BIAS
 PTR x3,x21,W_FF1
 ADDR x4,Lff1shape
 bl _tf_linear_forward
 PTR x0,x21,W_FF1
 PTR x1,x21,W_RELU
 mov x2,#2048
 bl _tf_relu_forward
 PTR x0,x21,W_RELU
 PTR x1,x19,P_FF2_WEIGHT
 PTR x2,x19,P_FF2_BIAS
 PTR x3,x21,W_FF2
 ADDR x4,Lff2shape
 bl _tf_linear_forward
 PTR x0,x21,W_R1
 PTR x1,x21,W_FF2
 PTR x2,x21,W_R2
 mov x3,#1024
 bl Ladd
 PTR x0,x21,W_R2
 PTR x1,x19,P_LN3_WEIGHT
 PTR x2,x19,P_LN3_BIAS
 PTR x3,x21,W_N3
 PTR x4,x21,W_NORM3
 PTR x5,x21,W_INV3
 mov x6,#CONTEXT
 mov x7,#WIDTH
 bl _tf_ln_forward
 PTR x0,x21,W_N3
 PTR x1,x19,P_HEAD_WEIGHT
 PTR x2,x19,P_HEAD_BIAS
 PTR x3,x21,W_LOGITS
 ADDR x4,Lheadshape
 bl _tf_linear_forward
 LEAVE

.globl _tf_backward
_tf_backward:
 ENTER
 mov x19,x0
 mov x20,x1
 mov x21,x3
 mov x22,x4
 mov x1,x2
 PTR x0,x21,W_LOGITS
 PTR x2,x21,W_DLOGITS
 mov x3,#CONTEXT
 mov x4,#VOCAB
 bl _tf_cross_entropy
 fmov s8,s0
 PTR x0,x21,W_N3
 PTR x1,x19,P_HEAD_WEIGHT
 PTR x2,x21,W_DLOGITS
 PTR x3,x21,W_DN3
 PTR x4,x22,P_HEAD_WEIGHT
 PTR x5,x22,P_HEAD_BIAS
 ADDR x6,Lheadshape
 bl _tf_linear_backward
 PTR x0,x21,W_DN3
 PTR x1,x21,W_NORM3
 PTR x2,x21,W_INV3
 PTR x3,x19,P_LN3_WEIGHT
 PTR x4,x21,W_DR2
 PTR x5,x22,P_LN3_WEIGHT
 PTR x6,x22,P_LN3_BIAS
 ADDR x7,Lnormshape
 bl _tf_ln_backward
 PTR x0,x21,W_RELU
 PTR x1,x19,P_FF2_WEIGHT
 PTR x2,x21,W_DR2
 PTR x3,x21,W_DRELU
 PTR x4,x22,P_FF2_WEIGHT
 PTR x5,x22,P_FF2_BIAS
 ADDR x6,Lff2shape
 bl _tf_linear_backward
 PTR x0,x21,W_FF1
 PTR x1,x21,W_DRELU
 PTR x2,x21,W_DFF1
 mov x3,#2048
 bl _tf_relu_backward
 PTR x0,x21,W_N2
 PTR x1,x19,P_FF1_WEIGHT
 PTR x2,x21,W_DFF1
 PTR x3,x21,W_DN2
 PTR x4,x22,P_FF1_WEIGHT
 PTR x5,x22,P_FF1_BIAS
 ADDR x6,Lff1shape
 bl _tf_linear_backward
 PTR x0,x21,W_DN2
 PTR x1,x21,W_NORM2
 PTR x2,x21,W_INV2
 PTR x3,x19,P_LN2_WEIGHT
 PTR x4,x21,W_TEMP
 PTR x5,x22,P_LN2_WEIGHT
 PTR x6,x22,P_LN2_BIAS
 ADDR x7,Lnormshape
 bl _tf_ln_backward
 PTR x0,x21,W_DR2
 PTR x1,x21,W_TEMP
 PTR x2,x21,W_DR1
 mov x3,#1024
 bl Ladd
 PTR x0,x21,W_ATTENTION
 PTR x1,x19,P_ATTENTION_WEIGHT
 PTR x2,x21,W_DR1
 PTR x3,x21,W_DATTENTION
 PTR x4,x22,P_ATTENTION_WEIGHT
 PTR x5,x22,P_ATTENTION_BIAS
 ADDR x6,Lprojshape
 bl _tf_linear_backward
 PTR x0,x21,W_QKV
 PTR x1,x21,W_PROBS
 PTR x2,x21,W_DATTENTION
 PTR x3,x21,W_DQKV
 ADDR x4,Lattshape
 bl _tf_attention_backward
 PTR x0,x21,W_N1
 PTR x1,x19,P_QKV_WEIGHT
 PTR x2,x21,W_DQKV
 PTR x3,x21,W_DN1
 PTR x4,x22,P_QKV_WEIGHT
 PTR x5,x22,P_QKV_BIAS
 ADDR x6,Lqkvshape
 bl _tf_linear_backward
 PTR x0,x21,W_DN1
 PTR x1,x21,W_NORM1
 PTR x2,x21,W_INV1
 PTR x3,x19,P_LN1_WEIGHT
 PTR x4,x21,W_DX
 PTR x5,x22,P_LN1_WEIGHT
 PTR x6,x22,P_LN1_BIAS
 ADDR x7,Lnormshape
 bl _tf_ln_backward
 PTR x0,x21,W_DX
 PTR x1,x21,W_DR1
 PTR x2,x21,W_DX
 mov x3,#1024
 bl Ladd
 PTR x9,x22,P_POSITION
 PTR x10,x21,W_DX
 mov x12,x20
 mov w11,#CONTEXT
Lscatter:
 ldr w13,[x12],#4
 add x13,x22,x13,lsl #7
 mov w14,#8
Lscatter_vec:
 ldr q0,[x10],#16
 ldr q1,[x13]
 ldr q2,[x9]
 fadd v1.4s,v1.4s,v0.4s
 fadd v2.4s,v2.4s,v0.4s
 str q1,[x13],#16
 str q2,[x9],#16
 subs w14,w14,#1
 b.ne Lscatter_vec
 subs w11,w11,#1
 b.ne Lscatter
 fmov s0,s8
 LEAVE

// x0 a,x1 b,x2 output,x3 count divisible by four; overlap is supported.
Ladd:
 ldr q0,[x0],#16
 ldr q1,[x1],#16
 fadd v0.4s,v0.4s,v1.4s
 str q0,[x2],#16
 subs x3,x3,#4
 b.ne Ladd
 ret

.globl _tf_scale
_tf_scale:
 dup v0.4s,v0.s[0]
 cbz x1,Lscale_done
Lscale_loop:
 ldr q1,[x0]
 fmul v1.4s,v1.4s,v0.4s
 str q1,[x0],#16
 subs x1,x1,#4
 b.ne Lscale_loop
Lscale_done:
 ret

.globl _tf_finite
_tf_finite:
 cbz x1,Lfinite_yes
Lfinite_loop:
 ldr w9,[x0],#4
 and w9,w9,#0x7f800000
 mov w10,#0x7f800000
 cmp w9,w10
 b.eq Lfinite_no
 subs x1,x1,#1
 b.ne Lfinite_loop
Lfinite_yes:
 mov w0,#1
 ret
Lfinite_no:
 mov w0,#0
 ret

.section __TEXT,__const
.p2align 2
Lqkvshape: .long CONTEXT,WIDTH,3*WIDTH
Lprojshape: .long CONTEXT,WIDTH,WIDTH
Lff1shape: .long CONTEXT,WIDTH,HIDDEN
Lff2shape: .long CONTEXT,HIDDEN,WIDTH
Lheadshape: .long CONTEXT,WIDTH,VOCAB
Lattshape: .long CONTEXT,WIDTH,HEADS
Lnormshape: .long CONTEXT,WIDTH
