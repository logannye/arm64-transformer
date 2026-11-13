// Handwritten four-lane exponential and ASCII-128 cross entropy.
//
// Exported routines:
//   tf_exp4           exp of four float32 lanes; private vector ABI
//   tf_cross_entropy  mean token loss and its gradient; C ABI (ABI.md)
//
// tf_exp4 is an internal vector ABI: v0.4s -> v0.4s. It clobbers only
// x9-x12, v0-v7, v16-v19 and the condition flags. v20-v31 are preserved,
// and the callers here and in attention.S keep live values there.
// Range reduction and the polynomial use two pairs of float64 lanes; the
// result is rounded to float32 once at the end.
// If any lane is NaN or outside [-104, 88], all four lanes are evaluated
// one at a time by the scalar tf_exp in kernels.S.
.text
.p2align 2
// SADDR: address of a local symbol (page, then page offset).
.macro SADDR reg, name
    adrp \reg, \name@PAGE
    add \reg, \reg, \name@PAGEOFF
.endm

// tf_exp4: v0.4s = exp(v0.4s). Private helper, not part of the public C
// interface; ABI.md gives no prototype.
// Fast path, per lane x, in float64:
//   n = round(x * log2(e))         to nearest, ties away from zero
//   r = x - n * ln(2)              one fused multiply-subtract
//   p = sum_{k=0..9} r^k / k!      Horner form, fused multiply-adds
//   result = float32(p * 2^n)
// Lanes 0-1 travel in v1, v3 and v16; lanes 2-3 in v2, v4 and v17.
// The fast path is a leaf; the scalar path builds a 32-byte frame.
.globl _tf_exp4
_tf_exp4:
    // v1 = lower bounds (-104), v2 = upper bounds (88). A lane passes when
    // both compares are true; a NaN lane fails both. uminv gives 0xffffffff
    // only if every lane passed, which cmn detects by adding 1.
    SADDR x9, Lexp4_bounds
    ldp q1,q2,[x9]
    fcmge v3.4s,v0.4s,v1.4s
    fcmge v4.4s,v2.4s,v0.4s
    and v3.16b,v3.16b,v4.16b
    uminv s3,v3.4s
    fmov w10,s3
    cmn w10,#1
    b.ne Lexp4_scalar
    // Widen to float64: v1 = lanes 0-1, v2 = lanes 2-3.
    fcvtl v1.2d,v0.2s
    fcvtl2 v2.2d,v0.4s
    // v3, v4 = n as int64 (fcvtas rounds to nearest, ties away);
    // v5, v6 = n as float64.
    ldr q5,[x9,#32] // log2(e)
    fmul v3.2d,v1.2d,v5.2d
    fmul v4.2d,v2.2d,v5.2d
    fcvtas v3.2d,v3.2d
    fcvtas v4.2d,v4.2d
    scvtf v5.2d,v3.2d
    scvtf v6.2d,v4.2d
    // v1, v2 = r = x - n * ln(2).
    ldr q7,[x9,#48] // ln(2)
    fmls v1.2d,v5.2d,v7.2d
    fmls v2.2d,v6.2d,v7.2d
    // v3, v4 = 2^n as float64 bit patterns: (n + 1023) << 52.
    mov x10,#1023
    dup v5.2d,x10
    add v3.2d,v3.2d,v5.2d
    add v4.2d,v4.2d,v5.2d
    shl v3.2d,v3.2d,#52
    shl v4.2d,v4.2d,#52
    // Degree nine Taylor on [-ln(2)/2,ln(2)/2], with double evaluation.
    // Truncation is below 1e-11 relative; final error is dominated by f32
    // rounding.
    // x9 steps through the ten coefficients 1/9! .. 1/0! stored after the
    // bounds, log2(e) and ln(2). v16, v17 = running value, starting at 1/9!.
    add x9,x9,#64
    ldr q16,[x9],#16
    mov v17.16b,v16.16b
    mov w10,#9
// Nine Horner steps: value = value * r + next coefficient.
Lexp4_poly:
    ldr q6,[x9],#16
    mov v7.16b,v6.16b
    fmla v6.2d,v16.2d,v1.2d
    fmla v7.2d,v17.2d,v2.2d
    mov v16.16b,v6.16b
    mov v17.16b,v7.16b
    subs w10,w10,#1
    b.ne Lexp4_poly
    // Scale by 2^n, then narrow the four float64 results to float32 in v0.
    fmul v16.2d,v16.2d,v3.2d
    fmul v17.2d,v17.2d,v4.2d
    fcvtn v0.2s,v16.2d
    fcvtn2 v0.4s,v17.2d
    ret
// Scalar path. Frame: x29, x30 at [sp,#0], the four input lanes at
// [sp,#16..31]. Each lane is replaced by tf_exp of itself, in lane order,
// and the vector is reloaded.
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

// tf_cross_entropy(logits,targets,dlogits,rows,vocab) -> mean loss in s0
//   x0 = logits   float32 [rows, vocab]
//   x1 = targets  int32 [rows]            class index of each row
//   x2 = dlogits  float32 [rows, vocab]   overwritten
//   x3 = rows, x4 = vocab                 64-bit integers
// Only vocab = 128 is handled here; any other value tail-calls
// tf_cross_entropy_generic in kernels.S with the arguments untouched.
// For each row r, with m = max_c logits[r][c] and
// S = sum_c exp(logits[r][c] - m):
//   loss[r]       = log(S) - (logits[r][target] - m)
//   dlogits[r][c] = (exp(logits[r][c] - m) / S - [c == target]) / rows
// Returns s0 = float32((sum_r loss[r]) / rows). Each loss[r] is float32;
// their sum and the division are float64.
// Frame: 96 bytes holding x29, x30 and x19-x28; no locals.
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
    // x19 = logits row     x20 = targets        x21 = dlogits row
    // x22 = rows           x23 = row index r    x24 = loop count
    // x25, x26 = cursors within the row         w27 = target of row r
    // d28 = float64 sum of the row losses
    // v20 = row maximum m, v21 = sum S. These and d28 stay live across the
    // calls to tf_exp4 and tf_log; neither callee touches v20-v31.
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x3
    mov x23,#0
    fmov d28,xzr
// Row loop: r = 0 .. rows-1.
Lce128_row:
    // v20 starts as the first four logits and is folded with all 128 below.
    mov x25,x19
    ldr q20,[x25]
    mov x24,#16
// Maximum loop: 16 steps of eight logits.
Lce128_max:
    ldp q0,q1,[x25],#32
    fmax v20.4s,v20.4s,v0.4s
    fmax v20.4s,v20.4s,v1.4s
    subs x24,x24,#1
    b.ne Lce128_max
    // v20 = m in every lane. v21 = 0. x25 / x26 = logits / dlogits cursors.
    fmaxv s20,v20.4s
    dup v20.4s,v20.s[0]
    movi v21.4s,#0
    mov x25,x19
    mov x26,x21
    mov x24,#32
// Exponential loop: 32 groups of four. exp(logits - m) is parked in the
// dlogits row and summed in the four lanes of v21.
Lce128_exp:
    ldr q0,[x25],#16
    fsub v0.4s,v0.4s,v20.4s
    bl _tf_exp4
    str q0,[x26],#16
    fadd v21.4s,v21.4s,v0.4s
    subs x24,x24,#1
    b.ne Lce128_exp
    // s21 = S = (lane0 + lane1) + (lane2 + lane3). s0 = log(S).
    faddp v21.4s,v21.4s,v21.4s
    faddp s21,v21.2s
    fmov s0,s21
    bl _tf_log
    // loss[r] = log(S) - (logits[r][target] - m), added to d28 as float64.
    ldr w27,[x20,x23,lsl #2]
    ldr s1,[x19,x27,lsl #2]
    fsub s1,s1,s20
    fsub s0,s0,s1
    fcvt d0,s0
    fadd d28,d28,d0
    // Constants for the gradient loop: v21 = S, v22 = rows as float32,
    // v23 = column indices of the current group (0,1,2,3 at first),
    // v24 = target, v25 = 1.0, v26 = 4 (the index step).
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
// Gradient loop: 32 groups of four, in place over the parked exponentials.
// The compare mask ANDed with the bits of 1.0 is 1.0 in the target column
// and 0.0 elsewhere.
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
    // Advance both row pointers by 128 float32 = 512 bytes.
    add x19,x19,#512
    add x21,x21,#512
    add x23,x23,#1
    cmp x23,x22
    b.lo Lce128_row
    // s0 = float32(d28 / rows).
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
// tf_exp4 constants by byte offset: 0 lower bounds, 16 upper bounds (four
// float32 each); 32 log2(e), 48 ln(2); from 64 the coefficients 1/9! down
// to 1/0!. Every float64 is stored twice, once per lane.
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
// Column indices of the first group in the gradient loop.
Lindices:
    .long 0,1,2,3
