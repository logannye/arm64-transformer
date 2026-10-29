// Handwritten FP32 SME matrix kernels for 32-token transformer projections.
// Entry is non-streaming, with no live caller ZA state. The outer dispatcher
// checks hardware/OS support; these entries check shapes and 512-bit streaming
// vector length. No numerical library, allocation routine, or shared scratch.
// Four ZA tiles hold32x32 outputs at smaller compiled widths, or four
// interleaved partial sums for16x16 outputs at compiled widths512/1024.
// Scratch is bounded and page-touched.
#include <layout.inc>
.text
.p2align 4

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

// Every supported scratch size is a positive multiple of4096. Touch each page
// as the stack grows, rather than jumping over Darwin's stack guard interval.
.macro SME_ALLOC bytes
    mov x9,\bytes
991:
    sub sp,sp,#1,lsl #12
    str xzr,[sp]
    subs x9,x9,#1,lsl #12
    b.ne 991b
.endm

// Existing ABI shapes [R,I,O]. Only positive, bounded study-sized tiles enter.
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

// Transpose32 source rows into [columns,32] packed rows. Already streaming,
// p0.s all true, SVL=16float32. x0 src, x1 dst, x2 source stride bytes,
// x3 columns(multiple16). Clobbers x9-x17,ZA0/ZA1; no vector/GPR callee state.
// Named for sampled profiling only: internal streaming ABI, not a public API.
.globl _tf_linear_pack32_sme
_tf_linear_pack32_sme:
Lsme_pack32:
    mov x16,x0
    mov x13,x1
    lsr x14,x3,#4
    lsl x15,x2,#4
Lsme_pack_columns:
    mov x9,x16
    add x10,x9,x15
    mov w12,#0
Lsme_pack_load:
    ld1w {za0h.s[w12,0]},p0/z,[x9]
    ld1w {za1h.s[w12,0]},p0/z,[x10]
    add x9,x9,x2
    add x10,x10,x2
    add w12,w12,#1
    cmp w12,#16
    b.lo Lsme_pack_load
    mov w12,#0
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

// x0 X[32,I],x1 W[O,I],x2 bias[O],x3 Y[32,O],x4 shape.
.globl _tf_linear_forward_sme
_tf_linear_forward_sme:
.if WIDTH >= 512
    SME_SHAPE x4,Lsme_forward_fallback
    SME_SAVE 160
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
    lsl x10,x23,#8 // two32-by-I float32 packs, at most512KiB
    SME_ALLOC x10
    mov x25,sp
    add x26,x25,x23,lsl #7
    mov x1,x25
    mov x2,x28
    mov x3,x23
    bl Lsme_pack32
    mov x27,#0
Lsme_forward_panel:
    madd x0,x27,x28,x20
    mov x1,x26
    mov x2,x28
    mov x3,x23
    bl Lsme_pack32
    // Four subtiles cover the32-by32 panel. Within each16-by16 subtile,
    // ZA0..ZA3 retain I modulo4 partial sums, matching NEON's lane chains.
    mov x0,#0
Lsme_forward_subtile:
    and x1,x0,#1
    lsl x1,x1,#6 // packed output-half byte offset0/64
    lsr x2,x0,#1
    lsl x2,x2,#6 // packed token-half byte offset0/64
    zero {za}
    add x9,x25,x2
    add x10,x26,x1
    lsr x11,x23,#2
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
    add x9,x21,x27,lsl #2
    add x9,x9,x1
    ld1w {z4.s},p0/z,[x9]
    add x9,x22,x27,lsl #2
    add x9,x9,x1
    // Packed token-half offset is rowindex*4, so multiply by O to addressY.
    madd x9,x2,x24,x9
    lsl x11,x24,#2
    mov w12,#0
Lsme_forward_store:
    mov z0.s,p0/m,za0h.s[w12,0]
    mov z1.s,p0/m,za1h.s[w12,0]
    mov z2.s,p0/m,za2h.s[w12,0]
    mov z3.s,p0/m,za3h.s[w12,0]
    // NEON FADDP order: (lane0+lane1)+(lane2+lane3), followed bybias.
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
    lsl x9,x23,#8
    add sp,sp,x9
    smstop
    SME_RESTORE 160
    ret
Lsme_forward_wrong_svl:
    smstop
    SME_RESTORE 160
Lsme_forward_fallback:
    b _tf_linear_forward_neon

.else
    SME_SHAPE x4,Lsme_forward_fallback
    SME_SAVE 160
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
    lsl x10,x23,#8 // two32-by-I float32 packs, at most512KiB
    SME_ALLOC x10
    mov x25,sp
    add x26,x25,x23,lsl #7
    mov x1,x25
    mov x2,x28
    mov x3,x23
    bl Lsme_pack32
    mov x27,#0
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
    add x9,x21,x27,lsl #2
    ld1w {z4.s},p0/z,[x9]
    ld1w {z5.s},p0/z,[x9,#1,mul vl]
    add x9,x22,x27,lsl #2
    lsl x11,x24,#2
    add x10,x9,x11,lsl #4
    mov w12,#0
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
    lsl x9,x23,#8
    add sp,sp,x9
    smstop
    SME_RESTORE 160
    ret
Lsme_forward_wrong_svl:
    smstop
    SME_RESTORE 160
Lsme_forward_fallback:
    b _tf_linear_forward_neon

.endif

// x0 X,x1 W,x2 dY,x3 dX,x4 dW,x5 db,x6 shape[32,I,O].
// dX overwrite, dW/db ADD. Outer products preserve the O/R accumulation order.
.p2align 4
.globl _tf_linear_backward_sme
_tf_linear_backward_sme:
    SME_SHAPE x6,Lsme_backward_fallback
    SME_SAVE 192
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
    lsl x10,x26,#7 //32-by-O float32 pack, at most384KiB
    SME_ALLOC x10
    mov x27,sp
    mov x0,x21
    mov x1,x27
    ldr x2,[x29,#168]
    mov x3,x26
    bl Lsme_pack32
    mov x28,#0
Lsme_dx_panel:
    zero {za}
    mov x9,x27
    add x10,x20,x28,lsl #2
    ldr x13,[x29,#160]
    mov x11,x26
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

    mov x28,#0 // output-channel block
Lsme_dw_outputs:
    str xzr,[x29,#176] // input-channel block
Lsme_dw_inputs:
    ldr x13,[x29,#160]
    ldr x14,[x29,#176]
    madd x9,x28,x13,x23
    add x9,x9,x14,lsl #2
    add x10,x9,x13,lsl #4
    mov w12,#0
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
    add x9,x21,x28,lsl #2
    add x10,x19,x14,lsl #2
    ldr x14,[x29,#168]
    mov x11,#32
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

    add x9,x24,x28,lsl #2
    ld1w {z0.s},p0/z,[x9]
    ld1w {z1.s},p0/z,[x9,#1,mul vl]
    add x10,x21,x28,lsl #2
    ldr x13,[x29,#168]
    mov x11,#32
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
    lsl x9,x26,#7
    add sp,sp,x9
    smstop
    SME_RESTORE 192
    ret
Lsme_backward_wrong_svl:
    smstop
    SME_RESTORE 192
Lsme_backward_fallback:
    b _tf_linear_backward_neon
