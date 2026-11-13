// Handwritten FP32 SME matrix kernels for 32-token transformer projections.
// Entry is non-streaming, with no live caller ZA state. The outer dispatcher
// checks hardware/OS support; these entries check shapes and the 512-bit
// streaming vector length. No numerical library, allocation routine, or
// shared scratch. In forward, four ZA tiles hold a 32x32 output panel at
// smaller compiled widths, or four interleaved partial sums of a 16x16
// subtile at compiled widths 512/1024. In backward they hold a 32x32 dx
// panel or dw block at every width. Scratch is bounded and page-touched.
//
//   tf_linear_forward_sme   y = x @ w^T + b for 32 rows
//   tf_linear_backward_sme  dx, dw and db for 32 rows
//   tf_linear_pack32_sme    internal transpose helper, streaming ABI
//
// The two kernels take the arguments of the public tf_linear_forward and
// tf_linear_backward (ABI.md) and are reached from those dispatchers in
// backend.S.
//
// SME terms used below. SMSTART enters streaming mode and enables the ZA
// array; SMSTOP leaves both. The code requires a streaming vector length of
// 64 bytes (CNTW == 16), so a Z register holds 16 float32 lanes and each of
// the four float32 tiles ZA0-ZA3 is 16 by 16. FMOPA zaN, zA, zB adds an
// outer product to a tile, zaN[i][j] += zA[i] * zB[j], as one fused
// multiply-add per element. zaNh[k] is horizontal slice (row) k of a tile,
// zaNv[k] vertical slice (column) k.
//
// Conventions: arrays are contiguous row-major float32; shape is int32
// {R, I, O} with R token rows, I inputs and O outputs. A change of streaming
// mode does not preserve the vector registers, so both kernels save d8-d15
// before SMSTART and restore them after SMSTOP. Each kernel issues one
// SMSTART/SMSTOP pair per call and takes its packing scratch from the
// stack, below its frame.
#include <layout.inc>
.text
.p2align 4

// SME_SAVE frame / SME_RESTORE frame: push or pop a frame of `frame` bytes.
//   [sp,#0]         x29, x30 (x29 = sp after SME_SAVE)
//   [sp,#16..#95]   x19-x28
//   [sp,#96..#159]  d8-d15
//   [sp,#160..]     the routine's locals, when frame is larger than 160
// SME_RESTORE does not return; the code after it adds RET or a branch.
.macro SME_SAVE frame
    stp x29,x30,[sp,#-\frame]!
    mov x29,sp
    stp x19,x20,[sp,#16]
    stp x21,x22,[sp,#32]
    stp x23,x24,[sp,#48]
    stp x25,x26,[sp,#64]
    stp x27,x28,[sp,#80]
    stp d8,d9,[sp,#96]
    stp d10,d11,[sp,#112]
    stp d12,d13,[sp,#128]
    stp d14,d15,[sp,#144]
.endm

.macro SME_RESTORE frame
    ldp d14,d15,[sp,#144]
    ldp d12,d13,[sp,#128]
    ldp d10,d11,[sp,#112]
    ldp d8,d9,[sp,#96]
    ldp x27,x28,[sp,#80]
    ldp x25,x26,[sp,#64]
    ldp x23,x24,[sp,#48]
    ldp x21,x22,[sp,#32]
    ldp x19,x20,[sp,#16]
    ldp x29,x30,[sp],#\frame
.endm

// SME_ALLOC bytes: lower sp by `bytes`, 4096 at a time, storing a zero at
// each new sp. Every supported scratch size is a positive multiple of 4096.
// Touch each page as the stack grows, rather than jumping over Darwin's
// stack guard interval. Clobbers x9.
.macro SME_ALLOC bytes
    mov x9,\bytes
991:
    sub sp,sp,#1,lsl #12
    str xzr,[sp]
    subs x9,x9,#1,lsl #12
    b.ne 991b
.endm

// SME_SHAPE shape,fallback: read the int32 shape {R, I, O} and branch to
// `fallback` unless R == 32, I is a power of two in [64, 2048], and O is a
// multiple of 32 in [64, 3072]. On fall-through w9 = R, w10 = I, w11 = O;
// w12 is clobbered. Existing ABI shapes [R,I,O]. Only positive, bounded
// study-sized tiles enter.
.macro SME_SHAPE shape,fallback
    ldr w9,[\shape]
    cmp w9,#32
    b.ne \fallback
    ldr w10,[\shape,#4]
    cmp w10,#64
    b.lo \fallback
    cmp w10,#2048
    b.hi \fallback
    sub w12,w10,#1
    tst w10,w12
    b.ne \fallback
    ldr w11,[\shape,#8]
    cmp w11,#64
    b.lo \fallback
    cmp w11,#3072
    b.hi \fallback
    tst w11,#31
    b.ne \fallback
.endm

// tf_linear_pack32_sme (local name Lsme_pack32): transpose 32 source rows
// into [columns,32] packed rows, so that dst[c][r] = src[r][c], r = 0..31.
// Already streaming, ZA enabled, p0.s all true, SVL = 16 float32.
//   x0 = src      float32, 32 rows, `columns` floats used from each
//   x1 = dst      float32 [columns,32], 128 bytes per packed row
//   x2 = source stride, the bytes from one source row to the next
//   x3 = columns, a positive multiple of 16
// Works on 16 columns at a time. Sixteen horizontal loads per tile put
// source rows 0..15 of the column block in ZA0 and rows 16..31 in ZA1; then
// vertical slice k is stored as dst[c0+k][0..15] (from ZA0) followed by
// dst[c0+k][16..31] (from ZA1).
// Leaf. Leaves x0-x3 unchanged; uses x9, x10 and x12-x16 as scratch and
// overwrites ZA0/ZA1; no vector/GPR callee state.
// Named for sampled profiling only: internal streaming ABI, not a public API.
.globl _tf_linear_pack32_sme
_tf_linear_pack32_sme:
Lsme_pack32:
    mov x16,x0
    mov x13,x1
    lsr x14,x3,#4
    lsl x15,x2,#4
// Column-block loop: x14 = columns / 16 blocks, c0 = 0, 16, 32, ...
// x16 = &src[0][c0]; x9 walks source rows 0..15 and x10 rows 16..31
// (x15 = 16 * stride).
Lsme_pack_columns:
    mov x9,x16
    add x10,x9,x15
    mov w12,#0
// 16 iterations: source rows k and 16+k of the block into horizontal slice
// k of ZA0 and ZA1.
Lsme_pack_load:
    ld1w {za0h.s[w12,0]},p0/z,[x9]
    ld1w {za1h.s[w12,0]},p0/z,[x10]
    add x9,x9,x2
    add x10,x10,x2
    add w12,w12,#1
    cmp w12,#16
    b.lo Lsme_pack_load
    mov w12,#0
// 16 iterations: vertical slice k of ZA0, then of ZA1, to the next 128
// bytes of dst.
Lsme_pack_store:
    st1w {za0v.s[w12,0]},p0,[x13]
    add x13,x13,#64
    st1w {za1v.s[w12,0]},p0,[x13]
    add x13,x13,#64
    add w12,w12,#1
    cmp w12,#16
    b.lo Lsme_pack_store
    add x16,x16,#64
    subs x14,x14,#1
    b.ne Lsme_pack_columns
    ret

// tf_linear_forward_sme: SME kernel behind tf_linear_forward(x,w,b,y,shape).
//   x0 = x      float32 [32,I]
//   x1 = w      float32 [O,I]
//   x2 = b      float32 [O]
//   x3 = y      float32 [32,O]   overwritten
//   x4 = shape  int32 [3] = {R, I, O}
// Computes y[r][o] = b[o] + sum over i of x[r][i] * w[o][i].
// Shapes that SME_SHAPE rejects branch to tf_linear_forward_neon before a
// frame is built; a streaming vector length other than 64 bytes does the
// same after undoing the frame. The arguments are untouched in both cases.
//
// Method: pack x once as px[i][r] = x[r][i]. Then, for each panel of 32
// outputs o0..o0+31, pack pw[i][b] = w[o0+b][i]. Row i of either pack is 32
// contiguous floats, so a reduction step over i loads 16-lane vectors from
// px and pw and adds their outer product to a tile whose rows are token
// rows and whose columns are outputs.
//
// Frame: 160 bytes (SME_SAVE), no locals. Below it, 256*I bytes of scratch:
// px at sp and pw 128*I bytes above it.
//   x19 = x, x20 = w, x21 = b, x22 = y, x23 = I, x24 = O, x25 = px,
//   x26 = pw, x27 = o0, x28 = 4*I, the byte length of one row of x or w.
//
// One of two bodies is assembled, chosen by `.if WIDTH >= 512` on the
// compiled WIDTH from layout.inc. They are identical up to and including
// the panel pack, and again from the panel-loop increment (add x27) on.
// The tile clear, the reduction, the bias add and the store to y differ,
// and only the 512/1024 body wraps them in a four-pass subtile loop.
.globl _tf_linear_forward_sme
_tf_linear_forward_sme:
.if WIDTH >= 512
    // Compiled widths 512 and 1024. Each 32-by-32 output panel is computed as
    // four 16-by-16 subtiles. For one subtile ZA0-ZA3 hold four partial sums
    // of the same 16 rows by 16 outputs: ZAk accumulates the input indices i
    // with i % 4 == k.
    SME_SHAPE x4,Lsme_forward_fallback
    SME_SAVE 160
    // Streaming mode and ZA are on from here to the SMSTOP on each exit path.
    // CNTW counts float32 lanes per streaming vector; any value but 16 takes
    // the wrong-SVL exit. p0.s is set all-true and never changed.
    smstart
    cntw x9
    cmp x9,#16
    b.ne Lsme_forward_wrong_svl
    ptrue p0.s
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x3
    mov x23,x10 // I
    mov x24,x11 // O
    lsl x28,x23,#2
    lsl x10,x23,#8 // two 32-by-I float32 packs, at most 512 KiB
    SME_ALLOC x10
    // x25 = px, x26 = pw. Pack x: x0 still holds x from entry, the source
    // stride is one x row and there are I columns.
    mov x25,sp
    add x26,x25,x23,lsl #7
    mov x1,x25
    mov x2,x28
    mov x3,x23
    bl Lsme_pack32
    mov x27,#0
// Panel loop: o0 = x27 steps by 32 while o0 < O. Pack w rows o0..o0+31
// into pw, replacing the previous panel.
Lsme_forward_panel:
    madd x0,x27,x28,x20
    mov x1,x26
    mov x2,x28
    mov x3,x23
    bl Lsme_pack32
    // Four subtiles cover the 32-by-32 panel. Within each 16-by-16 subtile,
    // ZA0..ZA3 retain the partial sums over input index i modulo 4, matching
    // NEON's lane chains.
    mov x0,#0
// Subtile loop: s = x0 runs over 0..3. Bit 0 of s selects the output half
// (x1 = 0 or 64 bytes into each pw row) and bit 1 the token-row half
// (x2 = 0 or 64 bytes into each px row). x9 and x10 then step through px
// and pw one packed row (128 bytes) per input index; x11 = I / 4.
Lsme_forward_subtile:
    and x1,x0,#1
    lsl x1,x1,#6 // packed output-half byte offset 0/64
    lsr x2,x0,#1
    lsl x2,x2,#6 // packed token-half byte offset 0/64
    zero {za}
    add x9,x25,x2
    add x10,x26,x1
    lsr x11,x23,#2
// I / 4 iterations, four input indices i..i+3 each. For each index,
// z0 = x[row][i] for the 16 rows of this half and z1 = w[output][i] for the
// 16 outputs of this half; FMOPA adds z0[row] * z1[output] to ZA0 for i,
// ZA1 for i+1, ZA2 for i+2 and ZA3 for i+3.
Lsme_forward_reduce:
    ld1w {z0.s},p0/z,[x9]
    ld1w {z1.s},p0/z,[x10]
    fmopa za0.s,p0/m,p0/m,z0.s,z1.s
    add x9,x9,#128
    add x10,x10,#128
    ld1w {z0.s},p0/z,[x9]
    ld1w {z1.s},p0/z,[x10]
    fmopa za1.s,p0/m,p0/m,z0.s,z1.s
    add x9,x9,#128
    add x10,x10,#128
    ld1w {z0.s},p0/z,[x9]
    ld1w {z1.s},p0/z,[x10]
    fmopa za2.s,p0/m,p0/m,z0.s,z1.s
    add x9,x9,#128
    add x10,x10,#128
    ld1w {z0.s},p0/z,[x9]
    ld1w {z1.s},p0/z,[x10]
    fmopa za3.s,p0/m,p0/m,z0.s,z1.s
    add x9,x9,#128
    add x10,x10,#128
    subs x11,x11,#1
    b.ne Lsme_forward_reduce
    // z4 = the 16 bias values of this output half. Then x9 = the address in y
    // of the first row and first output of the subtile; x11 = 4*O.
    add x9,x21,x27,lsl #2
    add x9,x9,x1
    ld1w {z4.s},p0/z,[x9]
    add x9,x22,x27,lsl #2
    add x9,x9,x1
    // Packed token-half offset is row index * 4, so multiply by O to address Y.
    madd x9,x2,x24,x9
    lsl x11,x24,#2
    mov w12,#0
// 16 iterations: horizontal slice k of each tile is the four partial sums
// for one token row and 16 outputs.
Lsme_forward_store:
    mov z0.s,p0/m,za0h.s[w12,0]
    mov z1.s,p0/m,za1h.s[w12,0]
    mov z2.s,p0/m,za2h.s[w12,0]
    mov z3.s,p0/m,za3h.s[w12,0]
    // NEON FADDP order: (lane0+lane1)+(lane2+lane3), followed by bias. Here
    // ZA0..ZA3 take the place of lanes 0..3.
    fadd z0.s,z0.s,z1.s
    fadd z2.s,z2.s,z3.s
    fadd z0.s,z0.s,z2.s
    fadd z0.s,z0.s,z4.s
    st1w {z0.s},p0,[x9]
    add x9,x9,x11
    add w12,w12,#1
    cmp w12,#16
    b.lo Lsme_forward_store
    add x0,x0,#1
    cmp x0,#4
    b.lo Lsme_forward_subtile
    add x27,x27,#32
    cmp x27,x24
    b.lo Lsme_forward_panel
    // Release the scratch, leave streaming mode, restore the frame and return.
    lsl x9,x23,#8
    add sp,sp,x9
    smstop
    SME_RESTORE 160
    ret
// Wrong vector length: no scratch has been allocated and no output written.
// Leave streaming mode, pop the frame and fall into the NEON branch.
Lsme_forward_wrong_svl:
    smstop
    SME_RESTORE 160
Lsme_forward_fallback:
    b _tf_linear_forward_neon

.else
    // Compiled widths 32 through 256. The four tiles hold one whole 32-by-32
    // output panel: ZA0 = rows 0..15 by outputs o0..o0+15, ZA1 = rows 0..15 by
    // outputs o0+16..o0+31, ZA2 and ZA3 the same outputs for rows 16..31. Each
    // element is one chain of FMOPA updates over i in increasing order.
    SME_SHAPE x4,Lsme_forward_fallback
    SME_SAVE 160
    // Streaming mode and ZA are on from here to the SMSTOP on each exit path.
    // CNTW counts float32 lanes per streaming vector; any value but 16 takes
    // the wrong-SVL exit. p0.s is set all-true and never changed.
    smstart
    cntw x9
    cmp x9,#16
    b.ne Lsme_forward_wrong_svl
    ptrue p0.s
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x3
    mov x23,x10 // I
    mov x24,x11 // O
    lsl x28,x23,#2
    lsl x10,x23,#8 // two 32-by-I float32 packs, at most 512 KiB
    SME_ALLOC x10
    // x25 = px, x26 = pw. Pack x: x0 still holds x from entry, the source
    // stride is one x row and there are I columns.
    mov x25,sp
    add x26,x25,x23,lsl #7
    mov x1,x25
    mov x2,x28
    mov x3,x23
    bl Lsme_pack32
    mov x27,#0
// Panel loop: o0 = x27 steps by 32 while o0 < O. Pack w rows o0..o0+31
// into pw, replacing the previous panel, and clear the tiles.
Lsme_forward_panel:
    madd x0,x27,x28,x20
    mov x1,x26
    mov x2,x28
    mov x3,x23
    bl Lsme_pack32
    zero {za}
    mov x9,x25
    mov x10,x26
    mov x11,x23
// I iterations, one input index i each. z0, z1 = x[0..15][i], x[16..31][i]
// and z2, z3 = w[o0..o0+15][i], w[o0+16..o0+31][i]; one outer product per
// tile.
Lsme_forward_reduce:
    ld1w {z0.s},p0/z,[x9]
    ld1w {z1.s},p0/z,[x9,#1,mul vl]
    ld1w {z2.s},p0/z,[x10]
    ld1w {z3.s},p0/z,[x10,#1,mul vl]
    fmopa za0.s,p0/m,p0/m,z0.s,z2.s
    fmopa za1.s,p0/m,p0/m,z0.s,z3.s
    fmopa za2.s,p0/m,p0/m,z1.s,z2.s
    fmopa za3.s,p0/m,p0/m,z1.s,z3.s
    add x9,x9,#128
    add x10,x10,#128
    subs x11,x11,#1
    b.ne Lsme_forward_reduce
    // z4, z5 = b[o0..o0+15], b[o0+16..o0+31]. x9 = &y[0][o0],
    // x10 = &y[16][o0], x11 = 4*O.
    add x9,x21,x27,lsl #2
    ld1w {z4.s},p0/z,[x9]
    ld1w {z5.s},p0/z,[x9,#1,mul vl]
    add x9,x22,x27,lsl #2
    lsl x11,x24,#2
    add x10,x9,x11,lsl #4
    mov w12,#0
// 16 iterations: horizontal slice k of ZA0/ZA1 is row k and of ZA2/ZA3 row
// 16+k. Add the bias and store 32 outputs for each of the two rows.
Lsme_forward_store:
    mov z0.s,p0/m,za0h.s[w12,0]
    mov z1.s,p0/m,za1h.s[w12,0]
    mov z2.s,p0/m,za2h.s[w12,0]
    mov z3.s,p0/m,za3h.s[w12,0]
    fadd z0.s,z0.s,z4.s
    fadd z1.s,z1.s,z5.s
    fadd z2.s,z2.s,z4.s
    fadd z3.s,z3.s,z5.s
    st1w {z0.s},p0,[x9]
    st1w {z1.s},p0,[x9,#1,mul vl]
    st1w {z2.s},p0,[x10]
    st1w {z3.s},p0,[x10,#1,mul vl]
    add x9,x9,x11
    add x10,x10,x11
    add w12,w12,#1
    cmp w12,#16
    b.lo Lsme_forward_store
    add x27,x27,#32
    cmp x27,x24
    b.lo Lsme_forward_panel
    // Release the scratch, leave streaming mode, restore the frame and return.
    lsl x9,x23,#8
    add sp,sp,x9
    smstop
    SME_RESTORE 160
    ret
// Wrong vector length: no scratch has been allocated and no output written.
// Leave streaming mode, pop the frame and fall into the NEON branch.
Lsme_forward_wrong_svl:
    smstop
    SME_RESTORE 160
Lsme_forward_fallback:
    b _tf_linear_forward_neon

.endif

// tf_linear_backward_sme: SME kernel behind
// tf_linear_backward(x,w,dy,dx,dw,db,shape). One body for every width.
//   x0 = x      float32 [32,I]   forward input
//   x1 = w      float32 [O,I]
//   x2 = dy     float32 [32,O]
//   x3 = dx     float32 [32,I]   overwritten
//   x4 = dw     float32 [O,I]    added to
//   x5 = db     float32 [O]      added to
//   x6 = shape  int32 [3] = {R, I, O}
// Computes
//   dx[r][i] = sum over o of dy[r][o] * w[o][i]
//   dw[o][i] = dw[o][i] + sum over r of dy[r][o] * x[r][i]
//   db[o]    = db[o]    + sum over r of dy[r][o]
// dX overwrite, dW/db ADD. Outer products preserve the O/R accumulation
// order: each dx element accumulates from zero in increasing o, each dw and
// db element onto its existing value in increasing r.
// Shapes that SME_SHAPE rejects branch to tf_linear_backward_neon before a
// frame is built; a streaming vector length other than 64 bytes does the
// same after undoing the frame. The arguments are untouched in both cases.
//
// Frame: 192 bytes (SME_SAVE plus 32 bytes for locals). The locals are
// addressed through x29 because sp moves down for the scratch. Below the
// frame, 128*O bytes hold the pack pdy[o][r] = dy[r][o].
//   [x29,#160]  4*I, the byte length of one row of x, w, dx or dw
//   [x29,#168]  4*O, the byte length of one dy row
//   [x29,#176]  i0, the input-channel block of the dw phase
//   x19 = x, x20 = w, x21 = dy, x22 = dx, x23 = dw, x24 = db, x25 = I,
//   x26 = O, x27 = pdy, x28 = i0 in the dx phase, then o0 in the dw phase.
.p2align 4
.globl _tf_linear_backward_sme
_tf_linear_backward_sme:
    SME_SHAPE x6,Lsme_backward_fallback
    SME_SAVE 192
    // Streaming mode and ZA are on from here to the SMSTOP on each exit path.
    // CNTW counts float32 lanes per streaming vector; any value but 16 takes
    // the wrong-SVL exit. p0.s is set all-true and never changed.
    smstart
    cntw x9
    cmp x9,#16
    b.ne Lsme_backward_wrong_svl
    ptrue p0.s
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x3
    mov x23,x4
    mov x24,x5
    mov x25,x10 // I
    mov x26,x11 // O
    lsl x9,x25,#2
    str x9,[x29,#160]
    lsl x9,x26,#2
    str x9,[x29,#168]
    lsl x10,x26,#7 // 32-by-O float32 pack, at most 384 KiB
    SME_ALLOC x10
    // x27 = pdy. Pack dy: the source stride is one dy row and there are O
    // columns.
    mov x27,sp
    mov x0,x21
    mov x1,x27
    ldr x2,[x29,#168]
    mov x3,x26
    bl Lsme_pack32
    mov x28,#0
// dx phase. Panel loop: i0 = x28 steps by 32 while i0 < I. The four tiles
// hold dx[0..31][i0..i0+31]: ZA0 = rows 0..15 by inputs i0..i0+15, ZA1 =
// rows 0..15 by inputs i0+16..i0+31, ZA2 and ZA3 the same inputs for rows
// 16..31.
Lsme_dx_panel:
    zero {za}
    mov x9,x27
    add x10,x20,x28,lsl #2
    ldr x13,[x29,#160]
    mov x11,x26
// O iterations, o in increasing order. z0, z1 = dy[0..15][o], dy[16..31][o]
// from the pack; z2, z3 = w[o][i0..i0+15], w[o][i0+16..i0+31] read straight
// from weight row o. x9 steps one packed row, x10 one w row.
Lsme_dx_reduce:
    ld1w {z0.s},p0/z,[x9]
    ld1w {z1.s},p0/z,[x9,#1,mul vl]
    ld1w {z2.s},p0/z,[x10]
    ld1w {z3.s},p0/z,[x10,#1,mul vl]
    fmopa za0.s,p0/m,p0/m,z0.s,z2.s
    fmopa za1.s,p0/m,p0/m,z0.s,z3.s
    fmopa za2.s,p0/m,p0/m,z1.s,z2.s
    fmopa za3.s,p0/m,p0/m,z1.s,z3.s
    add x9,x9,#128
    add x10,x10,x13
    subs x11,x11,#1
    b.ne Lsme_dx_reduce
    // Store the panel over dx: x9 = &dx[0][i0], x10 = &dx[16][i0]. Horizontal
    // slice k of ZA0/ZA1 is row k and of ZA2/ZA3 row 16+k.
    add x9,x22,x28,lsl #2
    add x10,x9,x13,lsl #4
    mov w12,#0
Lsme_dx_store:
    st1w {za0h.s[w12,0]},p0,[x9]
    add x11,x9,#64
    st1w {za1h.s[w12,0]},p0,[x11]
    st1w {za2h.s[w12,0]},p0,[x10]
    add x11,x10,#64
    st1w {za3h.s[w12,0]},p0,[x11]
    add x9,x9,x13
    add x10,x10,x13
    add w12,w12,#1
    cmp w12,#16
    b.lo Lsme_dx_store
    add x28,x28,#32
    cmp x28,x25
    b.lo Lsme_dx_panel

    // dw phase. For each 32-by-32 block dw[o0..o0+31][i0..i0+31]: load the
    // caller's values into the tiles, add the outer product of
    // dy[r][o0..o0+31] and x[r][i0..i0+31] for the 32 rows r in order, and
    // store the block back. Tile rows are outputs and tile columns inputs.
    mov x28,#0 // output-channel block
// Output-block loop: o0 = x28 steps by 32 while o0 < O.
Lsme_dw_outputs:
    str xzr,[x29,#176] // input-channel block
// Input-block loop: i0 = [x29,#176] steps by 32 while i0 < I.
// x9 = &dw[o0][i0], x10 = &dw[o0+16][i0], x13 = 4*I.
Lsme_dw_inputs:
    ldr x13,[x29,#160]
    ldr x14,[x29,#176]
    madd x9,x28,x13,x23
    add x9,x9,x14,lsl #2
    add x10,x9,x13,lsl #4
    mov w12,#0
// 16 iterations: horizontal slice k of ZA0/ZA1 = dw[o0+k][i0..i0+31] and of
// ZA2/ZA3 = dw[o0+16+k][i0..i0+31].
Lsme_dw_load:
    ld1w {za0h.s[w12,0]},p0/z,[x9]
    add x11,x9,#64
    ld1w {za1h.s[w12,0]},p0/z,[x11]
    ld1w {za2h.s[w12,0]},p0/z,[x10]
    add x11,x10,#64
    ld1w {za3h.s[w12,0]},p0/z,[x11]
    add x9,x9,x13
    add x10,x10,x13
    add w12,w12,#1
    cmp w12,#16
    b.lo Lsme_dw_load
    // x9 = &dy[0][o0] and x10 = &x[0][i0]; each steps one row per iteration.
    // x14 is still i0 for the second ADD; the LDR then replaces it with 4*O,
    // the dy row step, which it holds until i0 is reloaded after the loop.
    add x9,x21,x28,lsl #2
    add x10,x19,x14,lsl #2
    ldr x14,[x29,#168]
    mov x11,#32
// 32 iterations, r in increasing order. z0, z1 = dy[r][o0..o0+15],
// dy[r][o0+16..o0+31] and z2, z3 = x[r][i0..i0+15], x[r][i0+16..i0+31].
Lsme_dw_reduce:
    ld1w {z0.s},p0/z,[x9]
    ld1w {z1.s},p0/z,[x9,#1,mul vl]
    ld1w {z2.s},p0/z,[x10]
    ld1w {z3.s},p0/z,[x10,#1,mul vl]
    fmopa za0.s,p0/m,p0/m,z0.s,z2.s
    fmopa za1.s,p0/m,p0/m,z0.s,z3.s
    fmopa za2.s,p0/m,p0/m,z1.s,z2.s
    fmopa za3.s,p0/m,p0/m,z1.s,z3.s
    add x9,x9,x14
    add x10,x10,x13
    subs x11,x11,#1
    b.ne Lsme_dw_reduce
    // Reload i0 and store the four tiles back to the same dw block.
    ldr x14,[x29,#176]
    madd x9,x28,x13,x23
    add x9,x9,x14,lsl #2
    add x10,x9,x13,lsl #4
    mov w12,#0
Lsme_dw_store:
    st1w {za0h.s[w12,0]},p0,[x9]
    add x11,x9,#64
    st1w {za1h.s[w12,0]},p0,[x11]
    st1w {za2h.s[w12,0]},p0,[x10]
    add x11,x10,#64
    st1w {za3h.s[w12,0]},p0,[x11]
    add x9,x9,x13
    add x10,x10,x13
    add w12,w12,#1
    cmp w12,#16
    b.lo Lsme_dw_store
    add x14,x14,#32
    str x14,[x29,#176]
    cmp x14,x25
    b.lo Lsme_dw_inputs

    // db phase for this output block: z0, z1 = db[o0..o0+15],
    // db[o0+16..o0+31]; then, per row r, z0 += dy[r][o0..o0+15] and
    // z1 += dy[r][o0+16..o0+31], one FADD each.
    add x9,x24,x28,lsl #2
    ld1w {z0.s},p0/z,[x9]
    ld1w {z1.s},p0/z,[x9,#1,mul vl]
    add x10,x21,x28,lsl #2
    ldr x13,[x29,#168]
    mov x11,#32
// 32 iterations, r in increasing order.
Lsme_db_reduce:
    ld1w {z2.s},p0/z,[x10]
    ld1w {z3.s},p0/z,[x10,#1,mul vl]
    fadd z0.s,z0.s,z2.s
    fadd z1.s,z1.s,z3.s
    add x10,x10,x13
    subs x11,x11,#1
    b.ne Lsme_db_reduce
    st1w {z0.s},p0,[x9]
    st1w {z1.s},p0,[x9,#1,mul vl]
    add x28,x28,#32
    cmp x28,x26
    b.lo Lsme_dw_outputs
    // Release the scratch, leave streaming mode, restore the frame and return.
    lsl x9,x26,#7
    add sp,sp,x9
    smstop
    SME_RESTORE 192
    ret
// Wrong vector length: no scratch has been allocated and no output written.
// Leave streaming mode, pop the frame and fall into the NEON branch.
Lsme_backward_wrong_svl:
    smstop
    SME_RESTORE 192
Lsme_backward_fallback:
    b _tf_linear_backward_neon
