// Darwin SME dispatch, immutable once-only policy, and actual-path counters.
// Public interfaces use the normal non-streaming ARM64 ABI (SM=0 on entry).
// Do not enter streaming mode when ZA is active or TPIDR2 has a lazy-save block.
// We leave such state untouched and use NEON instead. SME math entries preserve
// d8-d15 across SMSTART/SMSTOP; no streaming/SVE public calling convention exists.
// Sources: https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst
// https://www.llvm.org/docs/AArch64SME.html
// https://github.com/apple-oss-distributions/xnu/blob/main/doc/arm/sme.md
.text
.p2align 4
.macro BADDR reg, symbol
 adrp \reg,\symbol@PAGE
 add \reg,\reg,\symbol@PAGEOFF
.endm
.macro BCOUNT symbol
 BADDR x9,\symbol
1:
 ldaxr x10,[x9]
 add x10,x10,#1
 stlxr w11,x10,[x9]
 cbnz w11,1b
.endm

// Read one documented Darwin optional-feature selector. No SME instructions.
Lbackend_feature:
 stp x29,x30,[sp,#-32]!
 mov x29,sp
 str wzr,[sp,#16]
 mov x9,#4
 str x9,[sp,#24]
 add x1,sp,#16
 add x2,sp,#24
 mov x3,#0
 mov x4,#0
 bl _sysctlbyname
 cbnz w0,Lfeature_no
 ldr x9,[sp,#24]
 cmp x9,#4
 b.ne Lfeature_no
 ldr w9,[sp,#16]
 cmp w9,#0
 cset w0,ne
 b Lfeature_done
Lfeature_no:
 mov w0,#0
Lfeature_done:
 ldp x29,x30,[sp],#32
 ret

// 0 success, -1 invalid policy or forced SME unavailable. Environment is captured
// once. The acquire/release state protects concurrent explicit initialization.
.globl _tf_backend_init
_tf_backend_init:
 stp x29,x30,[sp,#-48]!
 mov x29,sp
 stp x19,x20,[sp,#16]
 stp x21,x22,[sp,#32]
 BADDR x19,Lbackend_once
Linit_acquire:
 ldaxr w9,[x19]
 cmp w9,#2
 b.eq Linit_complete
 cbnz w9,Linit_wait
 mov w10,#1
 stxr w11,w10,[x19]
 cbnz w11,Linit_acquire
 mov x20,#0
 mov x21,#0
 BADDR x0,Lbackend_env
 bl _getenv
 cbz x0,Linit_features
 mov x22,x0
 BADDR x1,Lbackend_auto
 bl _strcmp
 cbz w0,Linit_features
 mov x0,x22
 BADDR x1,Lbackend_neon
 bl _strcmp
 cbz w0,Linit_neon
 mov x0,x22
 BADDR x1,Lbackend_sme
 bl _strcmp
 cbz w0,Linit_sme
 mov x20,#-1
 mov x21,#3
 b Linit_publish
Linit_neon:
 mov x21,#1
 b Linit_features
Linit_sme:
 mov x21,#2
Linit_features:
 BADDR x0,Lfeature_sme
 bl Lbackend_feature
 cbz x0,Linit_unavailable
 BADDR x0,Lfeature_sme2
 bl Lbackend_feature
 cbz x0,Linit_unavailable
 // RDSVL is permitted outside streaming mode; it does not switch SM or ZA.
 // This is the first SME instruction, after both OS feature checks succeeded.
 rdsvl x10,#1
 BADDR x9,Lbackend_vector_bytes
 str x10,[x9]
 cmp x10,#64
 b.ne Linit_unavailable
 mov x10,#1
 BADDR x9,Lbackend_supported
 str x10,[x9]
 cmp x21,#1
 b.eq Linit_publish
 BADDR x9,Lbackend_kind
 str x10,[x9]
 b Linit_publish
Linit_unavailable:
 cmp x21,#2
 b.ne Linit_publish
 mov x20,#-1
Linit_publish:
 BADDR x9,Lbackend_status
 str x20,[x9]
 BADDR x9,Lbackend_policy
 str x21,[x9]
 mov w9,#2
 stlr w9,[x19]
 b Linit_return
Linit_wait:
 clrex
 yield
 b Linit_acquire
Linit_complete:
 clrex
Linit_return:
 BADDR x9,Lbackend_status
 ldr x0,[x9]
 ldp x21,x22,[sp,#32]
 ldp x19,x20,[sp,#16]
 ldp x29,x30,[sp],#48
 ret

.macro BGET public, symbol
 .globl \public
\public:
 BADDR x9,\symbol
 ldar x0,[x9]
 ret
.endm
BGET _tf_backend_kind,Lbackend_kind
BGET _tf_backend_policy,Lbackend_policy
BGET _tf_sme_supported,Lbackend_supported
BGET _tf_backend_vector_bytes,Lbackend_vector_bytes
BGET _tf_backend_sme_forward_calls,Lbackend_sme_forward
BGET _tf_backend_sme_backward_calls,Lbackend_sme_backward
BGET _tf_backend_neon_forward_calls,Lbackend_neon_forward
BGET _tf_backend_neon_backward_calls,Lbackend_neon_backward

// Output ten uint64 words. Word1 is interpreted as signed init status.
// [1,status,policy,kind,supported,SVL,SME-fwd,SME-bwd,NEON-fwd,NEON-bwd].
.globl _tf_backend_info
_tf_backend_info:
 stp x29,x30,[sp,#-32]!
 mov x29,sp
 str x19,[sp,#16]
 mov x19,x0
 bl _tf_backend_init
 mov x9,#1
 stp x9,x0,[x19]
 BADDR x9,Lbackend_policy
 ldr x10,[x9]
 str x10,[x19,#16]
 BADDR x9,Lbackend_kind
 ldr x10,[x9]
 str x10,[x19,#24]
 BADDR x9,Lbackend_supported
 ldr x10,[x9]
 str x10,[x19,#32]
 BADDR x9,Lbackend_vector_bytes
 ldr x10,[x9]
 str x10,[x19,#40]
 BADDR x9,Lbackend_sme_forward
 ldar x10,[x9]
 str x10,[x19,#48]
 BADDR x9,Lbackend_sme_backward
 ldar x10,[x9]
 str x10,[x19,#56]
 BADDR x9,Lbackend_neon_forward
 ldar x10,[x9]
 str x10,[x19,#64]
 BADDR x9,Lbackend_neon_backward
 ldar x10,[x9]
 str x10,[x19,#72]
 ldr x19,[sp,#16]
 ldp x29,x30,[sp],#32
 ret

// Shape gate shared by forward/backward. This macro touches only scratch GPRs.
.macro BCHECK shape, fallback
 BADDR x9,Lbackend_once
 ldar w10,[x9]
 cmp w10,#2
 b.ne \fallback
 BADDR x9,Lbackend_kind
 ldr x10,[x9]
 cbz x10,\fallback
 ldr w9,[\shape]
 cmp w9,#32
 b.ne \fallback
 ldr w9,[\shape,#4]
 cmp w9,#64
 b.lo \fallback
 cmp w9,#512
 b.hi \fallback
 sub w10,w9,#1
 tst w9,w10
 b.ne \fallback
 ldr w9,[\shape,#8]
 cmp w9,#64
 b.lo \fallback
 cmp w9,#768
 b.hi \fallback
 tst w9,#31
 b.ne \fallback
 // SM=0 is a public-ABI precondition. A live ZA/lazy-save state is supported
 // conservatively through NEON without changing either control or matrix state.
 mrs x9,SVCR
 tst x9,#3
 b.ne \fallback
 mrs x9,TPIDR2_EL0
 cbnz x9,\fallback
 rdsvl x9,#1
 cmp x9,#64
 b.ne \fallback
.endm

.globl _tf_linear_forward
_tf_linear_forward:
 BCHECK x4,Ldispatch_forward_neon
 BCOUNT Lbackend_sme_forward
 b _tf_linear_forward_sme
Ldispatch_forward_neon:
 BCOUNT Lbackend_neon_forward
 b _tf_linear_forward_neon

.globl _tf_linear_backward
_tf_linear_backward:
 BCHECK x6,Ldispatch_backward_neon
 BCOUNT Lbackend_sme_backward
 b _tf_linear_backward_sme
Ldispatch_backward_neon:
 BCOUNT Lbackend_neon_backward
 b _tf_linear_backward_neon

.macro BSEED reg, value
 mov x9,#\value
 fmov \reg,x9
.endm
.macro BVERIFY reg, value
 fmov x9,\reg
 cmp x9,#\value
 csel x20,x20,xzr,eq
.endm
.macro BSEED_ALL
 BSEED d8,0x108
 BSEED d9,0x209
 BSEED d10,0x30a
 BSEED d11,0x40b
 BSEED d12,0x50c
 BSEED d13,0x60d
 BSEED d14,0x70e
 BSEED d15,0x80f
.endm
.macro BVERIFY_ALL
 BVERIFY d8,0x108
 BVERIFY d9,0x209
 BVERIFY d10,0x30a
 BVERIFY d11,0x40b
 BVERIFY d12,0x50c
 BVERIFY d13,0x60d
 BVERIFY d14,0x70e
 BVERIFY d15,0x80f
.endm
.macro BVERIFY_STATE
 BADDR x9,Lbackend_supported
 ldr x9,[x9]
 cbz x9,1f
 mrs x9,SVCR
 cmp x9,x21
 csel x20,x20,xzr,eq
 mrs x9,TPIDR2_EL0
 cmp x9,x22
 csel x20,x20,xzr,eq
1:
.endm

// Independent ABI test: 1 pass, 0 failure. No model parameters or files touched.
// Uses one eligible zero-input forward/backward and increments dispatch counters.
.globl _tf_backend_abi_check
_tf_backend_abi_check:
 stp x29,x30,[sp,#-112]!
 mov x29,sp
 stp x19,x20,[sp,#16]
 stp x21,x22,[sp,#32]
 stp d8,d9,[sp,#48]
 stp d10,d11,[sp,#64]
 stp d12,d13,[sp,#80]
 stp d14,d15,[sp,#96]
 mov x20,#0
 bl _tf_backend_init
 cbnz x0,Labi_return
 mov x0,#57856
 mov x1,#1
 bl _calloc
 cbz x0,Labi_return
 mov x19,x0
 mov x20,#1
 BADDR x9,Lbackend_supported
 ldr x9,[x9]
 cbz x9,Labi_forward
 mrs x21,SVCR
 mrs x22,TPIDR2_EL0
Labi_forward:
 mov x0,x19
 add x1,x19,#2,lsl #12
 add x2,x19,#6,lsl #12
 add x3,x2,#256
 BADDR x4,Labi_shape
 BSEED_ALL
 bl _tf_linear_forward
 BVERIFY_ALL
 BVERIFY_STATE
 mov x0,x19
 add x1,x19,#2,lsl #12
 add x2,x19,#6,lsl #12
 add x2,x2,#256
 add x3,x19,#8,lsl #12
 add x3,x3,#256
 add x4,x19,#10,lsl #12
 add x4,x4,#256
 add x5,x19,#14,lsl #12
 add x5,x5,#256
 BADDR x6,Labi_shape
 BSEED_ALL
 bl _tf_linear_backward
 BVERIFY_ALL
 BVERIFY_STATE
 mov x0,x19
 bl _free
Labi_return:
 mov x0,x20
 ldp d14,d15,[sp,#96]
 ldp d12,d13,[sp,#80]
 ldp d10,d11,[sp,#64]
 ldp d8,d9,[sp,#48]
 ldp x21,x22,[sp,#32]
 ldp x19,x20,[sp,#16]
 ldp x29,x30,[sp],#112
 ret

// Constructors perform detection before public calls; failure remains queryable
// instead of aborting the hosting process. The executable rejects it explicitly.
Lbackend_constructor:
 b _tf_backend_init

.section __DATA,__mod_init_func,mod_init_funcs
.p2align 3
.quad Lbackend_constructor
.section __TEXT,__cstring,cstring_literals
Lbackend_env: .asciz "ARM64_TRANSFORMER_BACKEND"
Lbackend_auto: .asciz "auto"
Lbackend_neon: .asciz "neon"
Lbackend_sme: .asciz "sme"
Lfeature_sme: .asciz "hw.optional.arm.FEAT_SME"
Lfeature_sme2: .asciz "hw.optional.arm.FEAT_SME2"
.section __TEXT,__const
.p2align 2
Labi_shape: .long 32,64,64
.zerofill __DATA,__bss,Lbackend_once,4,2
.zerofill __DATA,__bss,Lbackend_status,8,3
.zerofill __DATA,__bss,Lbackend_policy,8,3
.zerofill __DATA,__bss,Lbackend_kind,8,3
.zerofill __DATA,__bss,Lbackend_supported,8,3
.zerofill __DATA,__bss,Lbackend_vector_bytes,8,3
.zerofill __DATA,__bss,Lbackend_sme_forward,8,3
.zerofill __DATA,__bss,Lbackend_sme_backward,8,3
.zerofill __DATA,__bss,Lbackend_neon_forward,8,3
.zerofill __DATA,__bss,Lbackend_neon_backward,8,3
