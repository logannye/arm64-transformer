// Development precision candidates. Scalar FP64 arithmetic, FP32 interfaces.
// Caller-saved registers only; no heap, libm, BLAS or shared numerical scratch.
//
// These routines replace four kernels call for call: same arguments, same
// float32 inputs and outputs, but the arithmetic is scalar FP64 and a value
// is rounded to float32 only when it is stored. Each output element is
// stored once per call, except dgamma and dbeta in tf_ln_backward_precise,
// which are reloaded, updated and rounded again for every row. model.S
// calls them only in builds with WIDTH >= 512, and only for the stages
// whose bit is set in TF_PRECISION_POLICY (precision.h, written by the
// Makefile from PRECISION=0..15):
//   1  LayerNorm 1   tf_ln_forward_precise, tf_ln_backward_precise
//   2  QKV linear    tf_linear_forward_precise, tf_linear_backward_precise
//   4  LayerNorm 2   tf_ln_forward_precise, tf_ln_backward_precise
//   8  FF1 linear    tf_linear_forward_precise, tf_linear_backward_precise
// With the default PRECISION=0, or at any width below 512, model.S does not
// call them; they are still assembled into every build.
//
// Exported routines:
//   tf_precision_policy         the policy in effect for this binary
//   tf_ln_forward_precise       LayerNorm forward
//   tf_ln_backward_precise      LayerNorm backward
//   tf_linear_forward_precise   dense layer forward
//   tf_linear_backward_precise  dense layer backward
//
// All five are leaf routines without a stack frame. model.S calls the
// linear variants directly, not through the dispatcher in backend.S, so
// they do not advance its NEON/SME call counters. Row and feature counts
// must be positive: every loop tests its bound after the first pass.
#include <layout.inc>
#include <precision.h>
.text
.p2align 2
// tf_precision_policy: returns in x0 the precision policy in effect:
// TF_PRECISION_POLICY in builds with WIDTH >= 512, and 0 in narrower builds,
// where model.S ignores the policy. Leaf.
.globl _tf_precision_policy
_tf_precision_policy:
 .if WIDTH >= 512
 mov x0,#TF_PRECISION_POLICY
 .else
 mov x0,#0
 .endif
 ret

// tf_ln_forward_precise: same ABI as
// tf_ln_forward(x, gamma, beta, y, norm, inv, rows, width). Keeps all
// arithmetic through the affine output in FP64.
//   x0 = x      float32[rows][width]
//   x1 = gamma  float32[width]
//   x2 = beta   float32[width]
//   x3 = y      float32[rows][width], written
//   x4 = norm   float32[rows][width], written
//   x5 = inv    float32[rows], written
//   x6 = rows, x7 = width (64-bit counts)
// For each row r, in FP64, with every float32 input widened exactly:
//   rw      = 1 / width
//   mean    = rw * sum_c x[r][c]
//   var_eps = rw * sum_c (x[r][c] - mean)^2 + 1e-5    (biased variance)
//   s       = 1 / sqrt(var_eps)
//   inv[r]     = float32(s)
//   norm[r][c] = float32((x[r][c] - mean) * s)
//   y[r][c]    = float32((x[r][c] - mean) * s * gamma[c] + beta[c])
// y is formed from the unrounded FP64 normalized value, not from the float32
// stored in norm. Each sum is one chain over c = 0..width-1. Leaf; uses
// x8-x13, d0-d2, d6, d7, d16 and d17.
.globl _tf_ln_forward_precise
_tf_ln_forward_precise:
 // d7 = 1/width, d6 = epsilon, x9 = row index r.
 scvtf d7,x7
 fmov d0,#1.0
 fdiv d7,d0,d7
 adrp x8,Lprecision_epsilon@PAGE
 ldr d6,[x8,Lprecision_epsilon@PAGEOFF]
 mov x9,#0
// One pass per row r (rows passes).
Lpn_row:
 // x11, x12, x13 = start of row r in x, y and norm.
 mul x8,x9,x7
 lsl x8,x8,#2
 add x11,x0,x8
 add x12,x3,x8
 add x13,x4,x8
 fmov d16,xzr
 mov x10,#0
// First pass over the row: d16 = sum of x[r][c].
Lpn_sum:
 ldr s0,[x11,x10,lsl #2]
 fcvt d0,s0
 fadd d16,d16,d0
 add x10,x10,#1
 cmp x10,x7
 b.lo Lpn_sum
 // d16 = mean.
 fmul d16,d16,d7
 fmov d17,xzr
 mov x10,#0
// Second pass: d17 = sum of squared deviations.
Lpn_variance:
 ldr s0,[x11,x10,lsl #2]
 fcvt d0,s0
 fsub d0,d0,d16
 fmadd d17,d0,d0,d17
 add x10,x10,#1
 cmp x10,x7
 b.lo Lpn_variance
 // d17 = 1/sqrt(variance + epsilon). Scaling the sum by 1/width and adding
 // epsilon is one fused operation. inv[r] receives the float32 rounding.
 fmadd d17,d17,d7,d6
 fsqrt d17,d17
 fmov d0,#1.0
 fdiv d17,d0,d17
 fcvt s0,d17
 str s0,[x5,x9,lsl #2]
 mov x10,#0
// Third pass: write norm[r][c], then y[r][c].
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

// tf_ln_backward_precise: same ABI as
// tf_ln_backward(dy, norm, inv, gamma, dx, dgamma, dbeta, shape).
//   x0 = dy      float32[rows][width]
//   x1 = norm    float32[rows][width], saved by the forward pass
//   x2 = inv     float32[rows], saved by the forward pass
//   x3 = gamma   float32[width]
//   x4 = dx      float32[rows][width], overwritten
//   x5 = dgamma  float32[width], accumulated into
//   x6 = dbeta   float32[width], accumulated into
//   x7 = shape   int32[2] = [rows, width]
// For each row r, in FP64, with g[c] = dy[r][c] * gamma[c] and rw = 1/width:
//   m1 = rw * sum_c g[c]
//   m2 = rw * sum_c g[c] * norm[r][c]
//   dx[r][c]   = float32(inv[r] * (g[c] - m1 - norm[r][c] * m2))
//   dgamma[c] += dy[r][c] * norm[r][c]
//   dbeta[c]  += dy[r][c]
// Each dgamma and dbeta update widens the stored float32, adds in FP64 and
// rounds back to float32, once per row. Leaf; uses x9-x14, d0-d3, d7 and
// d16-d18.
.globl _tf_ln_backward_precise
_tf_ln_backward_precise:
 // w9 = rows, w10 = width, d7 = 1/width, x11 = row index r.
 ldr w9,[x7]
 ldr w10,[x7,#4]
 scvtf d7,x10
 fmov d0,#1.0
 fdiv d7,d0,d7
 mov x11,#0
// One pass per row r (rows passes).
Lpnb_row:
 // x12 = element index of the start of row r; d16 and d17 are the row sums.
 mul x12,x11,x10
 fmov d16,xzr
 fmov d17,xzr
 mov x13,#0
// First pass over the row: accumulate both sums and add this row's
// contribution to dgamma[c] and dbeta[c].
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
 // d16 = m1, d17 = m2, d18 = inv[r].
 fmul d16,d16,d7
 fmul d17,d17,d7
 ldr s18,[x2,x11,lsl #2]
 fcvt d18,s18
 mov x13,#0
// Second pass: write dx[r][c].
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

// tf_linear_forward_precise: same ABI as
// tf_linear_forward(x, w, b, y, shape); weights are [out,in].
//   x0 = x      float32[rows][input]
//   x1 = w      float32[output][input]
//   x2 = b      float32[output]
//   x3 = y      float32[rows][output], written
//   x4 = shape  int32[3] = [rows, input, output]
//   y[r][o] = float32(sum_k x[r][k] * w[o][k] + b[o])
// The sum is one FP64 fused multiply-add chain over k = 0..input-1; the bias
// is added last, still in FP64. Leaf; uses x6-x14, d0, d1 and d16.
.globl _tf_linear_forward_precise
_tf_linear_forward_precise:
 // w9 = rows, w10 = input, w11 = output, x12 = row index r.
 ldr w9,[x4]
 ldr w10,[x4,#4]
 ldr w11,[x4,#8]
 mov x12,#0
// One pass per row r; x6 = start of x[r].
Lpl_row:
 mul x8,x12,x10
 add x6,x0,x8,lsl #2
 mov x13,#0
// One pass per output o (x13); x7 = start of weight row w[o].
Lpl_column:
 mul x8,x13,x10
 add x7,x1,x8,lsl #2
 fmov d16,xzr
 mov x14,#0
// One pass per input k (x14): d16 += x[r][k] * w[o][k].
Lpl_dot:
 ldr s0,[x6,x14,lsl #2]
 ldr s1,[x7,x14,lsl #2]
 fcvt d0,s0
 fcvt d1,s1
 fmadd d16,d0,d1,d16
 add x14,x14,#1
 cmp x14,x10
 b.lo Lpl_dot
 // Add b[o], round once and store y[r][o] at element r*output + o.
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

// tf_linear_backward_precise: same ABI as
// tf_linear_backward(x, w, dy, dx, dw, db, shape).
//   x0 = x      float32[rows][input], the forward input
//   x1 = w      float32[output][input]
//   x2 = dy     float32[rows][output]
//   x3 = dx     float32[rows][input], overwritten
//   x4 = dw     float32[output][input], accumulated into
//   x5 = db     float32[output], accumulated into
//   x6 = shape  int32[3] = [rows, input, output]
//   dx[r][i] = float32(sum_o dy[r][o] * w[o][i])
//   dw[o][i] = float32(dw[o][i] + sum_r dy[r][o] * x[r][i])
//   db[o]    = float32(db[o] + sum_r dy[r][o])
// Every sum is one FP64 chain in increasing index order; the dw and db
// chains start from the caller's stored float32 value. Leaf; uses x7-x14,
// d0, d1 and d16.
.globl _tf_linear_backward_precise
_tf_linear_backward_precise:
 // w9 = rows, w10 = input, w11 = output.
 ldr w9,[x6]
 ldr w10,[x6,#4]
 ldr w11,[x6,#8]
 mov x12,#0
// dx: one pass per row r (x12).
Lplb_dx_row:
 mov x13,#0
// One pass per input i (x13).
Lplb_dx_column:
 fmov d16,xzr
 mov x14,#0
// One pass per output o (x14): d16 += dy[r][o] * w[o][i].
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
 // dw and db: the outer index x12 is now the output o.
 mov x12,#0
// One pass per output o: row o of dw, then db[o].
Lplb_dw_output:
 mov x13,#0
// One pass per input i (x13); d16 starts from the stored dw[o][i].
Lplb_dw_input:
 madd x7,x12,x10,x13
 ldr s16,[x4,x7,lsl #2]
 fcvt d16,s16
 mov x14,#0
// One pass per row r (x14): d16 += dy[r][o] * x[r][i].
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
 // db[o]: d16 starts from the stored db[o].
 ldr s16,[x5,x12,lsl #2]
 fcvt d16,s16
 mov x14,#0
// One pass per row r (x14): d16 += dy[r][o].
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

// LayerNorm epsilon as a double, loaded by tf_ln_forward_precise.
.section __TEXT,__const
.p2align 3
Lprecision_epsilon:
 .quad 0x3ee4f8b588e368f1 // nearest FP64 representation of 1e-5
