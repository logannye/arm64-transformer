// Handwritten NEON matrix tiles for the fixed transformer; public ABI
// unchanged. Non-tiled shapes tail-call the preserved v1 generic kernels.
//
//   tf_linear_forward_neon   y = x @ w^T + b, four rows by four outputs
//                            per tile
//   tf_linear_backward_neon  dx, dw and db, four rows (or four outputs) by
//                            sixteen inputs per tile
//
// Both take the arguments of the public tf_linear_forward and
// tf_linear_backward (ABI.md). They are reached from those dispatchers in
// backend.S and from the fallback branches in linear_sme.S. Shapes the tiles
// do not cover tail-call tf_linear_forward_generic or
// tf_linear_backward_generic in kernels.S with the arguments untouched.
//
// Conventions: arrays are contiguous row-major float32. shape is int32
// {R, I, O}: R token rows, I inputs, O outputs. x is [R,I], w is [O,I] and
// y is [R,O]. The tiled paths use v8-v15 as accumulators, so they save and
// restore d8-d15 as well as x19-x28 (SAVE_LINEAR and RESTORE_LINEAR).
.text
.p2align 4

// OUTER4X16 c0,l0,c1,l1,c2,l2,c3,l3: rank-one update of a 4-by-16 tile.
// The tile is v0-v15: row a (0..3) is v(4a)..v(4a+3), sixteen contiguous
// float32 columns. v16-v19 hold one 16-float vector u. Row a receives
//   tile[a][0..15] += u[0..15] * ca.s[la]
// where ca is a vector register and la the lane that holds row a's scalar
// coefficient. Coefficients use their source lanes directly, avoiding
// scalar replicate loads and separate DUP instructions.
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

// ZERO_TILE: clear the sixteen accumulators v0-v15.
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

// SAVE_LINEAR frame: push a frame of `frame` bytes and set x29 = sp.
//   [sp,#0]         x29, x30
//   [sp,#16..#95]   x19-x28
//   [sp,#96..#159]  d8-d15
//   [sp,#160..]     the routine's locals, when frame is larger than 160
// RESTORE_LINEAR frame: reload the same registers, pop the frame and
// return; the macro ends in RET.
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

// tf_linear_forward_neon: NEON kernel behind
// tf_linear_forward(x,w,b,y,shape).
//   x0 = x      float32 [R,I]
//   x1 = w      float32 [O,I]
//   x2 = b      float32 [O]
//   x3 = y      float32 [R,O]   overwritten
//   x4 = shape  int32 [3] = {R, I, O}
// Computes y[r][o] = b[o] + sum over i of x[r][i] * w[o][i].
// If R, I and O are all multiples of four it runs the tiled path below;
// otherwise it tail-calls tf_linear_forward_generic before building a frame.
// Tiled path: four token rows x four output columns, sixteen independent
// dot accumulators. For the block of rows r0..r0+3 and outputs o0..o0+3,
// accumulator v(4a+b) belongs to y[r0+a][o0+b] and holds four partial sums,
// lane l summing the input indices i with i % 4 == l. They are combined as
// (lane0 + lane1) + (lane2 + lane3) and the bias is added last, the order
// tf_linear_forward_generic uses when I is a multiple of four.
// Frame: 160 bytes (SAVE_LINEAR), no locals.
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
    // x19 = x, x20 = w, x21 = b, x22 = y, x23 = R, x24 = I, x25 = O,
    // x26 = r0 (row block), x27 = o0 (output block), x28 = 4*I, the byte
    // length of one row of x or w.
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, x9
    mov x24, x10
    mov x25, x11
    lsl x28, x24, #2
    mov x26, #0
// Row-block loop: r0 = x26 steps by four while r0 < R.
Ltf_row:
    mov x27, #0
// Output-block loop: o0 = x27 steps by four while o0 < O. x9-x12 point at
// rows r0..r0+3 of x, x13-x16 at rows o0..o0+3 of w, and x17 = I / 4.
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
// I / 4 iterations, four input indices each. v16-v19 = the next four floats
// of x rows r0..r0+3; v20-v23 = the same four input positions of w rows
// o0..o0+3. Accumulator v(4a+b) gains v(16+a) * v(20+b), lane by lane.
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
    // x9 = &b[o0] first, so v26 = b[o0..o0+3]; then x9 = &y[r0][o0] and
    // x10 = 4*O, the byte length of one y row. For each row a, the three
    // FADDPs leave lane b of v24 = (lane0 + lane1) + (lane2 + lane3) of
    // accumulator v(4a+b); the bias is added and the vector is stored at
    // y[r0+a][o0..o0+3].
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

// tf_linear_backward_neon: NEON kernel behind
// tf_linear_backward(x,w,dy,dx,dw,db,shape).
//   x0 = x      float32 [R,I]   forward input
//   x1 = w      float32 [O,I]
//   x2 = dy     float32 [R,O]
//   x3 = dx     float32 [R,I]   overwritten
//   x4 = dw     float32 [O,I]   added to
//   x5 = db     float32 [O]     added to
//   x6 = shape  int32 [3] = {R, I, O}
// Computes
//   dx[r][i] = sum over o of dy[r][o] * w[o][i]
//   dw[o][i] = dw[o][i] + sum over r of dy[r][o] * x[r][i]
//   db[o]    = db[o]    + sum over r of dy[r][o]
// Each dx element is accumulated from zero in increasing o; each dw and db
// element is accumulated onto its existing value in increasing r.
// dX overwrite, dW/db ADD. Fast tiles require R and O to be multiples of
// four and I a multiple of sixteen; any other shape tail-calls
// tf_linear_backward_generic before a frame is built.
// Frame: 208 bytes (SAVE_LINEAR plus 48 bytes for locals). All ten
// callee-saved GPRs hold arguments and sizes, so the block indices live in
// the frame:
//   [sp,#160]  4*O, the byte length of one dy row
//   [sp,#168]  r0, row block (dx phase)
//   [sp,#176]  i0, input-column block (dx phase, then dw phase)
//   [sp,#184]  o0, output-channel block (dw and db phase)
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
    // x19 = x, x20 = w, x21 = dy, x22 = dx, x23 = dw, x24 = db, x25 = R,
    // x26 = I, x27 = O, x28 = 4*I, the byte length of one row of x, w, dx
    // or dw.
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

    // dX = dY @ W. Keep a 4-row x 16-input-column tile in registers over O:
    // tile row a, column c accumulates dx[r0+a][i0+c] from zero.
    // Outer loop: r0 = [sp,#168] steps by four while r0 < R.
Ltb_dx_rows:
    str xzr, [sp, #176] // input column block
// Input-block loop: i0 = [sp,#176] steps by sixteen while i0 < I. x9-x12
// point at rows r0..r0+3 of dy; x13 = &w[0][i0] and moves down one w row
// (x28 bytes) per coefficient; x15 = O / 4.
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
    // O / 4 iterations. v20-v23 = dy[r0+a][o..o+3] for a = 0..3. For lane
    // l = 0..3 in turn, v16-v19 = w[o+l][i0..i0+15] and tile row a gains
    // that vector times dy[r0+a][o+l].
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
    // Store the tile over dx[r0..r0+3][i0..i0+15]; x8 = &dx[r0][i0].
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

    // dW += dY^T @ X. Keep four output channels x 16 inputs in registers over
    // R: tile row b, column c accumulates dw[o0+b][i0+c].
    str xzr, [sp, #184] // output channel block
// Output-block loop: o0 = [sp,#184] steps by four while o0 < O.
Ltb_dw_outputs:
    str xzr, [sp, #176]
// Input-block loop: i0 = [sp,#176] steps by sixteen while i0 < I. Load the
// caller's dw[o0..o0+3][i0..i0+15] into the tile so the products add to it.
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
    // x9 = &x[0][i0] and x10 = &dy[0][o0]; each moves down one row per
    // iteration (x28 = 4*I and x12 = 4*O bytes). x15 = R.
    ldr x8, [sp, #176]
    add x9, x19, x8, lsl #2
    ldr x8, [sp, #184]
    add x10, x21, x8, lsl #2
    ldr x12, [sp, #160]
    mov x15, x25
// R iterations, r in increasing order. v16-v19 = x[r][i0..i0+15] and
// v24 = dy[r][o0..o0+3]; tile row b gains x[r][i0..i0+15] * dy[r][o0+b].
Ltb_dw_reduce:
    ldp q16, q17, [x9]
    ldp q18, q19, [x9, #32]
    add x9, x9, x28
    ldr q24, [x10]
    add x10, x10, x12
    OUTER4X16 v24,0,v24,1,v24,2,v24,3
    subs x15, x15, #1
    b.ne Ltb_dw_reduce
    // Store the tile back to dw[o0..o0+3][i0..i0+15].
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

    // Bias reduction uses the same row ordering as the generic kernel:
    // v0 = db[o0..o0+3], then one FADD of dy[r][o0..o0+3] per row r = 0..R-1.
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
