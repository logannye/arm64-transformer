// Development precision candidates. Scalar FP64 arithmetic, FP32 interfaces.
// Caller-saved registers only; no heap, libm, BLAS or shared numerical scratch.
#include <layout.inc>
#include <precision.h>
.text
.p2align 2
.globl _tf_precision_policy
_tf_precision_policy:
 .if WIDTH >= 512
 mov x0,#TF_PRECISION_POLICY
 .else
 mov x0,#0
 .endif
 ret

// Same ABI as tf_ln_forward. Keep all arithmetic through affine output in FP64.
.globl _tf_ln_forward_precise
_tf_ln_forward_precise:
 scvtf d7,x7
 fmov d0,#1.0
 fdiv d7,d0,d7
 adrp x8,Lprecision_epsilon@PAGE
 ldr d6,[x8,Lprecision_epsilon@PAGEOFF]
 mov x9,#0
Lpn_row:
 mul x8,x9,x7
 lsl x8,x8,#2
 add x11,x0,x8
 add x12,x3,x8
 add x13,x4,x8
 fmov d16,xzr
 mov x10,#0
Lpn_sum:
 ldr s0,[x11,x10,lsl #2]
 fcvt d0,s0
 fadd d16,d16,d0
 add x10,x10,#1
 cmp x10,x7
 b.lo Lpn_sum
 fmul d16,d16,d7
 fmov d17,xzr
 mov x10,#0
Lpn_variance:
 ldr s0,[x11,x10,lsl #2]
 fcvt d0,s0
 fsub d0,d0,d16
 fmadd d17,d0,d0,d17
 add x10,x10,#1
 cmp x10,x7
 b.lo Lpn_variance
 fmadd d17,d17,d7,d6
 fsqrt d17,d17
 fmov d0,#1.0
 fdiv d17,d0,d17
 fcvt s0,d17
 str s0,[x5,x9,lsl #2]
 mov x10,#0
Lpn_output:
 ldr s0,[x11,x10,lsl #2]
 fcvt d0,s0
 fsub d0,d0,d16
 fmul d0,d0,d17
 fcvt s1,d0
 str s1,[x13,x10,lsl #2]
 ldr s1,[x1,x10,lsl #2]
 ldr s2,[x2,x10,lsl #2]
 fcvt d1,s1
 fcvt d2,s2
 fmadd d0,d0,d1,d2
 fcvt s0,d0
 str s0,[x12,x10,lsl #2]
 add x10,x10,#1
 cmp x10,x7
 b.lo Lpn_output
 add x9,x9,#1
 cmp x9,x6
 b.lo Lpn_row
 ret

// Same ABI as tf_ln_backward; dx overwritten, dgamma/dbeta accumulated.
.globl _tf_ln_backward_precise
_tf_ln_backward_precise:
 ldr w9,[x7]
 ldr w10,[x7,#4]
 scvtf d7,x10
 fmov d0,#1.0
 fdiv d7,d0,d7
 mov x11,#0
Lpnb_row:
 mul x12,x11,x10
 fmov d16,xzr
 fmov d17,xzr
 mov x13,#0
Lpnb_sums:
 add x14,x12,x13
 ldr s0,[x0,x14,lsl #2]
 ldr s1,[x1,x14,lsl #2]
 ldr s2,[x3,x13,lsl #2]
 fcvt d0,s0
 fcvt d1,s1
 fcvt d2,s2
 fmul d2,d0,d2
 fadd d16,d16,d2
 fmadd d17,d2,d1,d17
 ldr s3,[x5,x13,lsl #2]
 fcvt d3,s3
 fmadd d3,d0,d1,d3
 fcvt s3,d3
 str s3,[x5,x13,lsl #2]
 ldr s3,[x6,x13,lsl #2]
 fcvt d3,s3
 fadd d3,d3,d0
 fcvt s3,d3
 str s3,[x6,x13,lsl #2]
 add x13,x13,#1
 cmp x13,x10
 b.lo Lpnb_sums
 fmul d16,d16,d7
 fmul d17,d17,d7
 ldr s18,[x2,x11,lsl #2]
 fcvt d18,s18
 mov x13,#0
Lpnb_dx:
 add x14,x12,x13
 ldr s0,[x0,x14,lsl #2]
 ldr s1,[x1,x14,lsl #2]
 ldr s2,[x3,x13,lsl #2]
 fcvt d0,s0
 fcvt d1,s1
 fcvt d2,s2
 fmul d0,d0,d2
 fsub d0,d0,d16
 fmul d1,d1,d17
 fsub d0,d0,d1
 fmul d0,d0,d18
 fcvt s0,d0
 str s0,[x4,x14,lsl #2]
 add x13,x13,#1
 cmp x13,x10
 b.lo Lpnb_dx
 add x11,x11,#1
 cmp x11,x9
 b.lo Lpnb_row
 ret

// Same ABI as tf_linear_forward; weights are [out,in].
.globl _tf_linear_forward_precise
_tf_linear_forward_precise:
 ldr w9,[x4]
 ldr w10,[x4,#4]
 ldr w11,[x4,#8]
 mov x12,#0
Lpl_row:
 mul x8,x12,x10
 add x6,x0,x8,lsl #2
 mov x13,#0
Lpl_column:
 mul x8,x13,x10
 add x7,x1,x8,lsl #2
 fmov d16,xzr
 mov x14,#0
Lpl_dot:
 ldr s0,[x6,x14,lsl #2]
 ldr s1,[x7,x14,lsl #2]
 fcvt d0,s0
 fcvt d1,s1
 fmadd d16,d0,d1,d16
 add x14,x14,#1
 cmp x14,x10
 b.lo Lpl_dot
 ldr s0,[x2,x13,lsl #2]
 fcvt d0,s0
 fadd d16,d16,d0
 fcvt s0,d16
 madd x8,x12,x11,x13
 str s0,[x3,x8,lsl #2]
 add x13,x13,#1
 cmp x13,x11
 b.lo Lpl_column
 add x12,x12,#1
 cmp x12,x9
 b.lo Lpl_row
 ret

// Same ABI as tf_linear_backward; dx overwritten, dw/db accumulated.
.globl _tf_linear_backward_precise
_tf_linear_backward_precise:
 ldr w9,[x6]
 ldr w10,[x6,#4]
 ldr w11,[x6,#8]
 mov x12,#0
Lplb_dx_row:
 mov x13,#0
Lplb_dx_column:
 fmov d16,xzr
 mov x14,#0
Lplb_dx_sum:
 madd x8,x12,x11,x14
 ldr s0,[x2,x8,lsl #2]
 madd x8,x14,x10,x13
 ldr s1,[x1,x8,lsl #2]
 fcvt d0,s0
 fcvt d1,s1
 fmadd d16,d0,d1,d16
 add x14,x14,#1
 cmp x14,x11
 b.lo Lplb_dx_sum
 madd x8,x12,x10,x13
 fcvt s0,d16
 str s0,[x3,x8,lsl #2]
 add x13,x13,#1
 cmp x13,x10
 b.lo Lplb_dx_column
 add x12,x12,#1
 cmp x12,x9
 b.lo Lplb_dx_row
 mov x12,#0
Lplb_dw_output:
 mov x13,#0
Lplb_dw_input:
 madd x7,x12,x10,x13
 ldr s16,[x4,x7,lsl #2]
 fcvt d16,s16
 mov x14,#0
Lplb_dw_sum:
 madd x8,x14,x11,x12
 ldr s0,[x2,x8,lsl #2]
 madd x8,x14,x10,x13
 ldr s1,[x0,x8,lsl #2]
 fcvt d0,s0
 fcvt d1,s1
 fmadd d16,d0,d1,d16
 add x14,x14,#1
 cmp x14,x9
 b.lo Lplb_dw_sum
 fcvt s0,d16
 str s0,[x4,x7,lsl #2]
 add x13,x13,#1
 cmp x13,x10
 b.lo Lplb_dw_input
 ldr s16,[x5,x12,lsl #2]
 fcvt d16,s16
 mov x14,#0
Lplb_db_sum:
 madd x8,x14,x11,x12
 ldr s0,[x2,x8,lsl #2]
 fcvt d0,s0
 fadd d16,d16,d0
 add x14,x14,#1
 cmp x14,x9
 b.lo Lplb_db_sum
 fcvt s0,d16
 str s0,[x5,x12,lsl #2]
 add x12,x12,#1
 cmp x12,x11
 b.lo Lplb_dw_output
 ret

.section __TEXT,__const
.p2align 3
Lprecision_epsilon:
 .quad 0x3ee4f8b588e368f1 // nearest FP64 representation of 1e-5
