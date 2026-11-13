// Handwritten attention for widths 32/64/128/256/512/1024, four heads,
// T in 1..32. Other shapes tail-call the generic routines in kernels.S.
//
// Exported routines (C prototypes in ABI.md):
//   tf_attention_forward   causal scaled dot-product attention; writes the
//                          mixed values and the attention probabilities
//   tf_attention_backward  gradients of Q, K and V from the stored
//                          probabilities; overwrites dqkv
//
// Paths, selected by model width:
//   32            head width 8, two NEON vectors per head (Laf8, Lab8)
//   64/128/256    head width 16/32/64 held in a v16.. register block
//                 (AWIDE_FORWARD, AWIDE_BACKWARD)
//   512/1024      head width 128/256 processed in feature blocks
//                 (ABLOCK_FORWARD, ABLOCK_BACKWARD)
// The 512/1024 forward (ABLOCK_FORWARD) keeps QK scores in FP64 through
// max subtraction; centered scores, probabilities, value mixing and stored
// tensors remain FP32. Its backward (ABLOCK_BACKWARD) retains dA, the
// softmax contraction, dScore and dQ sums in FP64. Other forward paths use
// NEON FP32 QK/value operations and stable exponentiation. Backward keeps
// each dA row on stack, avoiding recomputation of dot(dout,V).
//
// Layout (w = model width, h = w/4 = head width, T = tokens, head = 0..3):
//   qkv    float32 [T, 3w]. Row t holds Q at byte 0, K at byte 4w and V at
//          byte 8w; one head is the h channels at byte 4h*head of a
//          segment. The row stride is 12w bytes.
//   probs  float32 [4, T, T]. Row (head, t) starts at element
//          (head*T + t)*T.
//   out    float32 [T, w], head slices in the same order as Q.
// Query row t uses keys s = 0 .. t only. probs entries with s > t are
// written as zero.
//
// Conventions in every specialized path: AENTER saves x29, x30 and
// x19-x28; v8-v15 are never used; x9-x15 and v0-v7 are scratch; values in
// v20-v31 survive the calls to tf_exp4 (softmax.S), which preserves them
// by its private ABI.
.text
.p2align 2
// AADDR: address of a local symbol. AENTER: prologue for a frame of size
// bytes, with x29, x30 and x19-x28 at [sp,#0..95] and any locals above
// them. ALEAVE: the matching epilogue, which also returns.
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

// tf_attention_forward(qkv,out,probs,shape)
//   x0 = qkv    float32 [T, 3w]        input, Q/K/V segments per row
//   x1 = out    float32 [T, w]         written
//   x2 = probs  float32 [heads, T, T]  written
//   x3 = shape  int32 [3] = [tokens, width, heads], T = tokens
// For each head and query row t, over the head slices q, k, v and with
// scale = 1/sqrt(h):
//   score[s]          = scale * sum_c q[t][c] * k[s][c]            s <= t
//   probs[head][t][s] = exp(score[s] - max) / sum_u exp(score[u] - max)
//   probs[head][t][s] = 0                                          s > t
//   out[t][c]         = sum_{s<=t} probs[head][t][s] * v[s][c]
// where max and the sum over u run over the same keys 0 .. t.
// Returns nothing. This entry only dispatches; each specialized path builds
// its own frame.
.globl _tf_attention_forward
_tf_attention_forward:
    // w9 = width. Width 32 is checked here, the other widths further down.
    ldr w9,[x3,#4]
    cmp w9,#32
    b.ne Laf_wide_dispatch
    // Width 32 needs heads = 4 and 1 <= tokens <= 32, tested as the unsigned
    // compare tokens - 1 < 32. x9 = tokens on entry to Laf8_start.
    ldr w9,[x3,#8]
    cmp w9,#4
    b.ne Laf8_fallback
    ldr w9,[x3]
    sub w10,w9,#1
    cmp w10,#32
    b.lo Laf8_start
// Unsupported shapes tail-call the generic routine with x0-x3 unchanged.
Laf8_fallback:
    b _tf_attention_forward_generic
// Other widths: the same heads and tokens tests, then one entry per
// supported width (w9 still holds the width).
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
// Width 32 forward: head width 8, so a head slice is two vectors. Byte
// constants: row stride 384, K at 128 and V at 256 within a row, head
// stride 32, out row 128. Frame: 96 bytes, saved registers only.
Laf8_start:
    AENTER 96
    // x19 = qkv   x20 = out   x21 = probs   x22 = T   x23 = head
    // x24 = query row t   x25 = probs row (head, t)   x26 = Q slice of t
    // x27 = K, then V, cursor over keys; probs cursor during the softmax
    // x28 = key index s, or a loop count
    // s31 = scale = 1/sqrt(8), live across the whole routine
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x9
    mov x23,#0
    AADDR x9,Lattention_scale
    ldr s31,[x9]
// Head loop: head = 0 .. 3.
Laf8_head:
    mov x24,#0
// Query-row loop: t = 0 .. T-1.
Laf8_row:
    // x26 = qkv + t*384 + head*32; v20, v21 = the eight Q channels.
    // x25 = probs + 4*((head*T + t)*T). x27 = K slice of token 0.
    // s22 = running maximum, starting at -infinity (0xff800000).
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
// Score loop: s = 0 .. t. score = scale * dot(Q[t], K[s]) is written to
// probs[head][t][s] for now. Lane j of v2 holds q[j]*k[j] + q[j+4]*k[j+4];
// the lanes are combined as (lane0 + lane1) + (lane2 + lane3).
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
    // Zero probs[head][t][s] for s = t+1 .. T-1. All masked entries are
    // explicitly written, including short sequence tails.
    mov x9,x28
// Mask loop: s = t+1 .. T-1, with s in x9.
Laf8_mask:
    cmp x9,x22
    b.hs Laf8_exp_start
    str wzr,[x25,x9,lsl #2]
    add x9,x9,#1
    b Laf8_mask
// Softmax numerators: exp(score - max) in place, four scores per call to
// tf_exp4, with lane sums in v23. x28 = (t+1)/4 full groups.
Laf8_exp_start:
    dup v22.4s,v22.s[0]
    movi v23.4s,#0
    mov x27,x25
    lsr x28,x28,#2
    cbz x28,Laf8_exp_tail
// Group loop: (t+1)/4 groups of four scores.
Laf8_exp_vector:
    ldr q0,[x27]
    fsub v0.4s,v0.4s,v22.4s
    bl _tf_exp4
    str q0,[x27],#16
    fadd v23.4s,v23.4s,v0.4s
    subs x28,x28,#1
    b.ne Laf8_exp_vector
// The remaining (t+1) mod 4 scores, loaded and stored one lane at a time.
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
    // Zero the lanes whose index is not below the tail count, so that they
    // add nothing to the sum.
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
// Divide the t+1 numerators by their sum (s23, broadcast to v24): full
// groups of four, then a scalar tail.
Laf8_normalize:
    faddp v23.4s,v23.4s,v23.4s
    faddp s23,v23.2s
    dup v24.4s,v23.s[0]
    mov x27,x25
    add x28,x24,#1
    lsr x28,x28,#2
    cbz x28,Laf8_norm_tail_start
// Group loop: (t+1)/4 groups of four numerators.
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
// Tail loop: the last (t+1) mod 4 numerators, one at a time.
Laf8_norm_tail:
    ldr s0,[x27]
    fdiv s0,s0,s24
    str s0,[x27],#4
    subs x28,x28,#1
    b.ne Laf8_norm_tail
// Value mixing: v20, v21 = sum over s <= t of probs[head][t][s] * V[s],
// eight channels, one fused multiply-add per key and vector.
Laf8_value_start:
    movi v20.4s,#0
    movi v21.4s,#0
    add x27,x19,#256
    add x27,x27,x23,lsl #5
    mov x28,#0
// Key loop: s = 0 .. t, with v0 = probs[head][t][s] in every lane.
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
    // Store the head slice to out + t*128 + head*32.
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

// tf_attention_backward(qkv,probs,dout,dqkv,shape)
//   x0 = qkv    float32 [T, 3w]        forward input
//   x1 = probs  float32 [heads, T, T]  forward probabilities, only read
//   x2 = dout   float32 [T, w]         gradient of out
//   x3 = dqkv   float32 [T, 3w]        overwritten; dQ, dK, dV in qkv layout
//   x4 = shape  int32 [3] = [tokens, width, heads], T = tokens
// For each head and query row t, with p[s] = probs[head][t][s], s <= t:
//   dA[s]     = sum_c dout[t][c] * v[s][c]
//   dV[s][c] += p[s] * dout[t][c]
//   dScore[s] = scale * p[s] * (dA[s] - sum_u p[u] * dA[u])
//   dQ[t][c]  = sum_s dScore[s] * k[s][c]
//   dK[s][c] += dScore[s] * q[t][c]
// dqkv is zeroed first. dK and dV then accumulate over the query rows;
// each dQ head slice is written once.
// Returns nothing. This entry only dispatches; each specialized path builds
// its own frame.
.globl _tf_attention_backward
_tf_attention_backward:
    // Same dispatch as tf_attention_forward, with the shape pointer in x4.
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
// Width 32 backward: head width 8. Frame: 224 bytes = 96 saved + a
// 32-float dA row at [sp,#96..223].
Lab8_start:
    AENTER 224
    // x19 = qkv   x20 = probs   x21 = dout   x22 = dqkv   x23 = T
    // x24 = head   x25 = query row t   x26 = probs row (head, t)
    // x27 = Q slice of t   x28 = dQ slice of t
    // x9 / x10 = V / dV cursors in the first pass, K / dK in the second
    // x11 = dA row   x12 = key index s
    // v20, v21 = dout slice of t   v22, v23 = Q slice   v24, v25 = dQ sums
    // s26 = sum_u p[u] * dA[u]   s31 = scale = 1/sqrt(8)
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x3
    mov x23,x9
    // Zero dqkv: T rows of 96 floats = 24*T vectors of 16 bytes.
    mov x10,#24
    mul x10,x10,x23
    mov x11,x22
    movi v0.4s,#0
// Zero loop: 24*T steps, one vector each.
Lab8_zero:
    str q0,[x11],#16
    subs x10,x10,#1
    b.ne Lab8_zero
    mov x24,#0
    AADDR x9,Lattention_scale
    ldr s31,[x9]
// Head loop: head = 0 .. 3.
Lab8_head:
    mov x25,#0
// Query-row loop: t = 0 .. T-1.
Lab8_row:
    // Load dout + t*128 + head*32. x26 = probs + 4*((head*T + t)*T).
    // x27 / x28 = qkv / dqkv + t*384 + head*32, the Q and dQ slices.
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
    // x9 / x10 = V / dV slices of token 0 (byte 256 of a row).
    add x9,x19,#256
    add x9,x9,x24,lsl #5
    add x10,x22,#256
    add x10,x10,x24,lsl #5
    add x11,sp,#96
    mov x12,#0
// First pass, s = 0 .. t: dA[s] = dot(dout[t], V[s]) is saved on the stack,
// s26 += p[s] * dA[s] (fused), and dV[s] += p[s] * dout[t].
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
    // x9 / x10 = K / dK slices of token 0 (byte 128 of a row).
    add x9,x19,#128
    add x9,x9,x24,lsl #5
    add x10,x22,#128
    add x10,x10,x24,lsl #5
    mov x12,#0
// Second pass, s = 0 .. t: dScore = (dA[s] - s26) * p[s] * scale, then
// dQ sums += dScore * K[s] and dK[s] += dScore * Q[t].
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
    // Store the finished dQ slice of (t, head).
    stp q24,q25,[x28]
    add x25,x25,#1
    cmp x25,x23
    b.lo Lab8_row
    add x24,x24,#1
    cmp x24,#4
    b.lo Lab8_head
    ALEAVE 224

// Apply a register-block operation to a head of 16, 32 or 64 channels.
// Each operation takes register index, byte offset, and dot accumulator
// index. Vector i of the head uses register v(16+i) and byte offset 16*i;
// the accumulator index is i mod 4.
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
// Forward operations (x26 = Q block, x27 = K or V block of the current
// key, x9 = out block):
//   AHLOAD   v(r) = Q vector
//   AHDOT    v(acc) += v(r) * K vector, fused; the accumulators are v0-v3
//   AHZERO   v(r) = 0
//   AHVALUE  v(r) += V vector * v0, fused; v0 = probability in every lane
//   AHSTORE  out vector = v(r)
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
// Backward operations (x27 = Q block, x28 = dQ block; x9 = dout block in
// BHLOAD, V block in BHDV, K block in BHDQ; x10 = dV block in BHDV, dK
// block in BHDQ):
//   BHLOAD   v(r) = dout vector
//   BHDV     v(acc) += v(r) * V vector; dV vector += v(r) * v4 (v4 = p[s])
//   BHDQ     v(r) += K vector * v0; dK vector += Q vector * v0
//            (v0 = dScore[s])
//   BHSTORE  dQ vector = v(r)
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

// AWIDE_FORWARD: forward path for one width with head width 16, 32 or 64.
// Macro expansion specializes strides and register blocks in handwritten
// code. tag = label prefix, w = model width, h = head width,
// lw = log2(model width), lh = log2(head width), scale = float32 constant
// 1/sqrt(h).
// Same algorithm and register roles as Laf8 above, with the Q slice in
// the v16.. register block:
//   x19 = qkv   x20 = out   x21 = probs   x22 = T   x23 = head
//   x24 = query row t   x25 = probs row (head, t)   x26 = Q block of t
//   x27 = K, then V, cursor over keys; probs cursor during the softmax
//   x28 = key index s, or a loop count
// Frame: 112 bytes = 96 saved + running maximum at [sp,#96] + score scale
// at [sp,#100], both float32, + 8 unused. Both are reloaded from the frame
// at each use (s6 and s7 in the score loop, s22 before the softmax); with
// h = 64 the Q block fills v16-v31.
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
// Head loop: head = 0 .. 3.
\tag\()_head:
    mov x24,#0
// Query-row loop: t = 0 .. T-1.
\tag\()_row:
    // x26 = qkv + t*12w + head*4h; load the Q block into v16..
    mov x9,#(\w*12)
    mul x26,x24,x9
    add x26,x19,x26
    add x26,x26,x23,lsl #(\lh+2)
    AHEAD AHLOAD,\h
    // x25 = probs row (head, t). x27 = K block of token 0 (byte 4w of a row).
    // The running maximum starts at -infinity.
    madd x9,x23,x22,x24
    mul x9,x9,x22
    add x25,x21,x9,lsl #2
    add x27,x19,#(\w*4)
    add x27,x27,x23,lsl #(\lh+2)
    mov x28,#0
    mov w9,#0xff800000
    str w9,[sp,#96]
// Score loop: s = 0 .. t. The dot accumulators combine as
// (v0 + v1) + (v2 + v3), then (lane0 + lane1) + (lane2 + lane3).
// score = dot * scale goes to probs[head][t][s] and the running maximum.
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
    // Zero probs[head][t][s] for s = t+1 .. T-1.
    mov x9,x28
// Mask loop: s = t+1 .. T-1, with s in x9.
\tag\()_mask:
    cmp x9,x22
    b.hs \tag\()_exp_start
    str wzr,[x25,x9,lsl #2]
    add x9,x9,#1
    b \tag\()_mask
// Softmax over the t+1 scores: the same sequence as Laf8_exp_start through
// Laf8_norm_tail, with the maximum reloaded from the frame.
\tag\()_exp_start:
    ldr s22,[sp,#96]
    dup v22.4s,v22.s[0]
    movi v23.4s,#0
    mov x27,x25
    lsr x28,x28,#2
    cbz x28,\tag\()_exp_tail
// Group loop: (t+1)/4 groups of four scores.
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
// Group loop: (t+1)/4 groups of four numerators.
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
// Tail loop: the last (t+1) mod 4 numerators, one at a time.
\tag\()_norm_tail:
    ldr s0,[x27]
    fdiv s0,s0,s24
    str s0,[x27],#4
    subs x28,x28,#1
    b.ne \tag\()_norm_tail
// Value mixing: the register block becomes the h output sums,
// sum over s <= t of probs[head][t][s] * V[s]. x27 = V block of token 0.
\tag\()_value_start:
    AHEAD AHZERO,\h
    add x27,x19,#(\w*8)
    add x27,x27,x23,lsl #(\lh+2)
    mov x28,#0
// Key loop: s = 0 .. t, with v0 = probs[head][t][s] in every lane.
\tag\()_value:
    ldr s0,[x25,x28,lsl #2]
    dup v0.4s,v0.s[0]
    AHEAD AHVALUE,\h
    add x27,x27,#(\w*12)
    add x28,x28,#1
    cmp x28,x24
    b.ls \tag\()_value
    // Store the block to out + t*4w + head*4h.
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

// AWIDE_BACKWARD: backward path for one width with head width 16, 32 or
// 64; parameters as for AWIDE_FORWARD. Same algorithm as Lab8 above:
//   x19 = qkv   x20 = probs   x21 = dout   x22 = dqkv   x23 = T
//   x24 = head   x25 = query row t   x26 = probs row (head, t)
//   x27 = Q block of t   x28 = dQ block of t
//   x9 / x10 = V / dV cursors in the first pass, K / dK in the second
//   x11 = dA row   x12 = key index s
// Frame: 240 bytes = 96 saved + 32-float dA row at [sp,#96..223] + score
// scale at [sp,#224] + running sum of p[s] * dA[s] at [sp,#228], both
// float32, + 8 unused. The register block holds the dout block in the
// first pass and the dQ sums in the second, where Q is read from memory.
.macro AWIDE_BACKWARD tag,w,h,lw,lh,scale
\tag\()_start:
    AENTER 240
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x3
    ldr w23,[x4]
    // Zero dqkv: T rows of 3w floats = T * 3w/4 vectors of 16 bytes.
    mov x10,#(\w*3/4)
    mul x10,x10,x23
    mov x11,x22
    movi v0.4s,#0
// Zero loop: T * 3w/4 steps, one vector each.
\tag\()_zero:
    str q0,[x11],#16
    subs x10,x10,#1
    b.ne \tag\()_zero
    AADDR x9,\scale
    ldr s0,[x9]
    str s0,[sp,#224]
    mov x24,#0
// Head loop: head = 0 .. 3.
\tag\()_head:
    mov x25,#0
// Query-row loop: t = 0 .. T-1.
\tag\()_row:
    // Load the dout block, dout + t*4w + head*4h, into v16..
    add x9,x21,x25,lsl #(\lw+2)
    add x9,x9,x24,lsl #(\lh+2)
    AHEAD BHLOAD,\h
    // x26 = probs row (head, t). x27 / x28 = qkv / dqkv + t*12w + head*4h.
    madd x9,x24,x23,x25
    mul x9,x9,x23
    add x26,x20,x9,lsl #2
    mov x9,#(\w*12)
    mul x9,x25,x9
    add x9,x9,x24,lsl #(\lh+2)
    add x27,x19,x9
    add x28,x22,x9
    // Clear the running sum. x9 / x10 = V / dV blocks of token 0.
    str wzr,[sp,#228]
    add x9,x19,#(\w*8)
    add x9,x9,x24,lsl #(\lh+2)
    add x10,x22,#(\w*8)
    add x10,x10,x24,lsl #(\lh+2)
    add x11,sp,#96
    mov x12,#0
// First pass, s = 0 .. t: with v4 = p[s], BHDV forms dA[s] in v0-v3 and
// adds p[s] * dout[t] into dV[s]. dA[s] is reduced as in the forward score
// loop, saved in the stack row and folded into the running sum (fused).
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
    // The block now holds the dQ sums. x9 / x10 = K / dK blocks of token 0.
    AHEAD AHZERO,\h
    add x9,x19,#(\w*4)
    add x9,x9,x24,lsl #(\lh+2)
    add x10,x22,#(\w*4)
    add x10,x10,x24,lsl #(\lh+2)
    mov x12,#0
// Second pass, s = 0 .. t: v0 = dScore[s] = (dA[s] - sum) * p[s] * scale.
// BHDQ adds dScore * K[s] to the dQ sums and dScore * Q[t] to dK[s].
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
    // Store the finished dQ block of (t, head).
    AHEAD BHSTORE,\h
    add x25,x25,#1
    cmp x25,x23
    b.lo \tag\()_row
    add x24,x24,#1
    cmp x24,#4
    b.lo \tag\()_head
    ALEAVE 240
.endm

// Heads larger than 64 channels use bounded feature blocks. A Q or dout
// block occupies v16-v31; its four dot accumulators stay in caller-saved
// v0-v3. The existing small-width paths above are intentionally unchanged.
//
// ABLOCK_SOFTMAX: softmax tail of ABLOCK_FORWARD, the same sequence as
// Laf8_exp_start through Laf8_norm_tail. On entry x25 = probs row holding
// t+1 centered scores, x24 = t, x28 = t+1, and [sp,#96] = the float32
// value subtracted before exp, which ABLOCK_FORWARD sets to zero.
// Execution continues at the _value_start label of the including macro.
.macro ABLOCK_SOFTMAX tag
\tag\()_exp_start:
    ldr s22,[sp,#96]
    dup v22.4s,v22.s[0]
    movi v23.4s,#0
    mov x27,x25
    lsr x28,x28,#2
    cbz x28,\tag\()_exp_tail
// Group loop: (t+1)/4 groups of four scores.
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
// Group loop: (t+1)/4 groups of four numerators.
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
// Tail loop: the last (t+1) mod 4 numerators, one at a time.
\tag\()_norm_tail:
    ldr s0,[x27]
    fdiv s0,s0,s24
    str s0,[x27],#4
    subs x28,x28,#1
    b.ne \tag\()_norm_tail
.endm

// Cache 32 converted query channels in caller-saved v16-v31. Conversion
// before multiplication preserves the exact stored FP32 operands,
// including cancellation residuals; four FP64 chains bound the depth of
// each 32-channel block reduction.
// Each operation takes the FP64 register pair (lo, hi) for one FP32
// vector of four Q channels, its byte offset, and two accumulator indices:
//   AQ64LOAD  (lo, hi) = Q vector at x26, widened to FP64
//   AQ64DOT   K vector at x27 widened into v5, v6;
//             v(acc0) += lo * v5 and v(acc1) += hi * v6, fused
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

// ABLOCK_FORWARD: forward path for width 512 (h = 128) or 1024 (h = 256);
// parameters as for AWIDE_FORWARD, with scale a float64 constant.
//   x19 = qkv   x20 = out   x21 = probs   x22 = T   x23 = head
//   x24 = query row t   x25 = probs row (head, t)   x15 = FP64 score row
//   x26 = Q block   x27 = K, then V, cursor over keys; probs cursor in
//   ABLOCK_SOFTMAX   x28 = key index s; a loop count in ABLOCK_SOFTMAX
// Stack after saved registers: FP32 centered maximum at 96 (= 0),
// feature-byte offset at 104, FP64 scale at 112, and 32 FP64 scores at
// 128..383. One 400-byte frame serves every supported T and head width.
// Future Q/K/V rows never enter a causal sum.
// Per query row: QK scores accumulate in FP64 over 32-channel feature
// blocks; they are scaled and centered on their maximum in FP64, and only
// the centered scores are rounded to FP32; ABLOCK_SOFTMAX turns them into
// probabilities; values are mixed in FP32 over 64-channel feature blocks.
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
// Head loop: head = 0 .. 3.
\tag\()_head:
    mov x24,#0
// Query-row loop: t = 0 .. T-1.
\tag\()_row:
    // x25 = probs row (head, t). x15 = FP64 score row at [sp,#128].
    madd x9,x23,x22,x24
    mul x9,x9,x22
    add x25,x21,x9,lsl #2
    add x15,sp,#128
    mov x9,#0
// Zero all T entries of the probs row and of the score row. Entries with
// s > t are not written again, so the masked probabilities stay zero.
\tag\()_clear_scores:
    str wzr,[x25,x9,lsl #2]
    str xzr,[x15,x9,lsl #3]
    add x9,x9,#1
    cmp x9,x22
    b.lo \tag\()_clear_scores
    // Feature-byte offset = 0.
    str xzr,[sp,#104]
// Score feature-block loop: offset = 0, 128, .. while below 4h bytes
// (32 channels per block).
\tag\()_query_block:
    // x26 = qkv + t*12w + head*4h + offset; v16-v31 = its 32 Q channels in
    // FP64. x27 = the same block of K for token 0; x10 = row stride.
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
// Key loop: s = 0 .. t. The four FP64 accumulators combine as
// (v0 + v1) + (v2 + v3), then lane0 + lane1, and are added to score[s].
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
    // Advance to the next 32-channel block.
    ldr x9,[sp,#104]
    add x9,x9,#128
    str x9,[sp,#104]
    cmp x9,#(\h*4)
    b.lo \tag\()_query_block
    // d6 = running maximum, starting at -infinity. d7 = scale.
    mov x9,#0xfff0000000000000
    fmov d6,x9
    ldr d7,[sp,#112]
    mov x28,#0
// score[s] = score[s] * scale for s = 0 .. t, tracking the maximum.
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
// probs[head][t][s] = float32(score[s] - maximum) for s = 0 .. t.
\tag\()_center_scores:
    ldr d0,[x15,x28,lsl #3]
    fsub d0,d0,d6
    fcvt s0,d0
    str s0,[x25,x28,lsl #2]
    add x28,x28,#1
    cmp x28,x24
    b.ls \tag\()_center_scores
    // The scores are already centered, so the shared softmax subtracts zero.
    str wzr,[sp,#96]
    ABLOCK_SOFTMAX \tag
// Value feature-block loop: offset = 0, 256, .. while below 4h bytes
// (64 channels per block), in FP32.
\tag\()_value_start:
    str xzr,[sp,#104]
\tag\()_value_block:
    // v16-v31 = 64 output sums. x27 = V block of token 0 at this offset.
    AHEAD AHZERO,64
    add x27,x19,#(\w*8)
    add x27,x27,x23,lsl #(\lh+2)
    ldr x9,[sp,#104]
    add x27,x27,x9
    mov x10,#(\w*12)
    mov x28,#0
// Key loop: s = 0 .. t, sums += probs[head][t][s] * V[s] (fused).
\tag\()_value:
    ldr s0,[x25,x28,lsl #2]
    dup v0.4s,v0.s[0]
    AHEAD AHVALUE,64
    add x27,x27,x10
    add x28,x28,#1
    cmp x28,x24
    b.ls \tag\()_value
    // Store to out + t*4w + head*4h + offset, then advance the offset.
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

// dA is a cancellation-sensitive reduction when K has a large common
// component. Retain it in FP64 through the row contraction, scaled dScore
// and dQ sum. dV retains its existing FP32 multiply-add order and
// caller-owned FP32 destination.
//   BHDV64  r = register of an FP32 dout vector, off = byte offset,
//           acc = FP64 accumulator index. The V vector at x9 and v(r) are
//           widened two lanes at a time and v(acc) += V * dout (fused),
//           low pair first. The dV vector at x10 += v(r) * v4 in FP32.
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

// Cache 16 query channels and their 16 dQ accumulators in FP64. Each dQ
// value is rounded only after all causal keys. dK adds each query
// contribution in FP64 before rounding to its caller-owned FP32
// accumulator; no quadratic cache.
// Each operation takes the FP64 pair (qlo, qhi) for four Q channels, the
// FP64 pair (dlo, dhi) for their dQ sums, and a byte offset:
//   BQ64LOAD    (qlo, qhi) = Q vector at x27, widened; (dlo, dhi) = 0
//   BQ64UPDATE  K vector at x9 widened: (dlo, dhi) += K * v0, fused, with
//               v0 = dScore[s]. dK vector at x10 widened, += Q * v0,
//               rounded to FP32 and stored back.
//   BQ64STORE   (dlo, dhi) rounded to FP32 and stored to the dQ block, x28
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

// ABLOCK_BACKWARD: backward path for width 512 (h = 128) or 1024
// (h = 256); parameters as for ABLOCK_FORWARD.
//   x19 = qkv   x20 = probs   x21 = dout   x22 = dqkv   x23 = T
//   x24 = head   x25 = query row t   x26 = probs row (head, t)
//   x27 = Q block   x28 = dQ block   x9 / x10 = V / dV, then K / dK cursors
//   x11 = FP64 dA row   x12 = key index s   x13 = row stride 12w
//   x14 = feature-byte offset
// FP64 dA[32] occupies 96..351 and is replaced by FP64 dScore after
// contraction. FP64 scale at 352, row contraction at 360, feature-byte
// offset at 368; one 384-byte frame.
// Per query row: dA accumulates in FP64 over 64-channel blocks while dV
// is updated in FP32; the row contraction and dScore are formed in FP64;
// then, over 16-channel blocks, dQ accumulates in FP64 and each dK update
// is made in FP64 before rounding.
.macro ABLOCK_BACKWARD tag,w,h,lw,lh,scale
\tag\()_start:
    AENTER 384
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x3
    ldr w23,[x4]
    // Zero dqkv: T * 3w/4 vectors of 16 bytes.
    mov x10,#(\w*3/4)
    mul x10,x10,x23
    mov x11,x22
    movi v0.4s,#0
// Zero loop: T * 3w/4 steps, one vector each.
\tag\()_zero:
    str q0,[x11],#16
    subs x10,x10,#1
    b.ne \tag\()_zero
    AADDR x9,\scale
    ldr d0,[x9]
    str d0,[sp,#352]
    mov x24,#0
// Head loop: head = 0 .. 3.
\tag\()_head:
    mov x25,#0
// Query-row loop: t = 0 .. T-1.
\tag\()_row:
    // x26 = probs row (head, t). Clear the 32 FP64 dA slots and the offset.
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
// dA / dV feature-block loop: offset = 0, 256, .. while below 4h bytes
// (64 channels per block).
\tag\()_first_block:
    // v16-v31 = the FP32 dout block, dout + t*4w + head*4h + offset.
    // x9 / x10 = V / dV blocks of token 0 at this offset. x11 = dA row.
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
// Key loop: s = 0 .. t, with v4 = p[s]. The block dot(dout[t], V[s])
// combines as (v0 + v1) + (v2 + v3), then lane0 + lane1, and is added to
// dA[s].
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
    // Advance to the next 64-channel block.
    ldr x14,[sp,#368]
    add x14,x14,#256
    str x14,[sp,#368]
    cmp x14,#(\h*4)
    b.lo \tag\()_first_block
    // d6 = row contraction = sum over s <= t of dA[s] * p[s], fused in FP64.
    mov x12,#0
    fmov d6,xzr
// Contraction loop: s = 0 .. t.
\tag\()_row_dot:
    ldr d0,[x11,x12,lsl #3]
    ldr s1,[x26,x12,lsl #2]
    fcvt d1,s1
    fmadd d6,d0,d1,d6
    add x12,x12,#1
    cmp x12,x25
    b.ls \tag\()_row_dot
    // Save the contraction. d7 = scale.
    str d6,[sp,#360]
    ldr d7,[sp,#352]
    mov x12,#0
// dScore loop, s = 0 .. t: dA[s] is replaced by
// (dA[s] - contraction) * p[s] * scale.
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
// dQ / dK feature-block loop: offset = 0, 64, .. while below 4h bytes
// (16 channels per block).
\tag\()_second_block:
    // x27 / x28 = qkv / dqkv + t*12w + head*4h + offset, the Q and dQ blocks.
    // v16-v23 = 16 Q channels in FP64; v24-v31 = their dQ sums.
    // x9 / x10 = K / dK blocks of token 0 at this offset.
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
// Key loop: s = 0 .. t, with v0 = dScore[s] in both lanes.
\tag\()_second:
    ldr d0,[x11,x12,lsl #3]
    dup v0.2d,v0.d[0]
    BQ64BLOCK BQ64UPDATE
    add x9,x9,x13
    add x10,x10,x13
    add x12,x12,#1
    cmp x12,x25
    b.ls \tag\()_second
    // Round the 16 dQ sums to FP32 and store them; advance the offset.
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

// Instantiations: label prefix, w, h, log2(w), log2(h), scale constant.
// The two public entries branch to the _start label of each prefix.
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
// Score scales 1/sqrt(h): float32 for h = 8, 16, 32 and 64; float64 for
// h = 128 and 256. Lattention_indices holds the lane numbers 0..3 used to
// mask softmax tails.
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
