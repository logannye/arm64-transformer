// Darwin SME dispatch, immutable once-only policy, and actual-path counters.
// Public interfaces use the normal non-streaming ARM64 ABI (SM=0 on entry).
// Streaming mode is not entered when ZA is active or TPIDR2 has a lazy-save
// block: such state is left untouched and NEON is used instead. SME math
// entries preserve d8-d15 across SMSTART/SMSTOP; no streaming/SVE public
// calling convention exists.
// Sources: https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst
// https://www.llvm.org/docs/AArch64SME.html
// https://github.com/apple-oss-distributions/xnu/blob/main/doc/arm/sme.md
//
// Exported routines:
//   tf_backend_init          choose policy and kind once; returns 0 or -1
//   tf_backend_info          copy ten words of identity and counters
//   tf_backend_kind          selected kind: 0 = NEON, 1 = SME
//   tf_backend_policy        requested policy: 0 auto, 1 neon, 2 sme,
//                            3 = unrecognized value
//   tf_sme_supported         1 if the SME capability checks passed
//   tf_backend_vector_bytes  streaming vector length read by RDSVL
//   tf_backend_sme_forward_calls, tf_backend_sme_backward_calls,
//   tf_backend_neon_forward_calls, tf_backend_neon_backward_calls
//                            the four dispatch counters
//   tf_linear_forward        public dense forward: dispatch to SME or NEON
//   tf_linear_backward       public dense backward: dispatch to SME or NEON
//   tf_backend_abi_check     self-test of register and SME-state preservation
//
// How a path is chosen:
//   1. Once per loaded image (the executable, or each libtransformer.dylib
//      a process loads), tf_backend_init reads ARM64_TRANSFORMER_BACKEND.
//      For auto, neon or sme it asks the OS whether FEAT_SME and FEAT_SME2
//      exist and only then reads the streaming vector length. The selected
//      kind is SME when the policy is auto or sme, both features exist and
//      the length is 64 bytes; in every other case it stays NEON.
//   2. On every tf_linear_forward or tf_linear_backward call, the BCHECK
//      macro tests that selection again together with the shape and the
//      current SME state, then branches to the _sme implementation
//      (linear_sme.S) or the _neon implementation (linear.S).
// The two linear entry points are the only kernels dispatched here.
//
// State lives in the zero-filled globals at the end of the file: the once
// flag, the published status, policy, kind, capability flag and vector
// length, and the four call counters.
.text
.p2align 4
// BADDR reg, symbol: reg = address of a symbol in this image.
.macro BADDR reg, symbol
 adrp \reg,\symbol@PAGE
 add \reg,\reg,\symbol@PAGEOFF
.endm
// BCOUNT symbol: atomically add 1 to a 64-bit counter with an exclusive
// load/store retry loop. Uses only x9, x10 and w11, so the dispatchers can
// count a call without disturbing the argument registers.
.macro BCOUNT symbol
 BADDR x9,\symbol
1:
 ldaxr x10,[x9]
 add x10,x10,#1
 stlxr w11,x10,[x9]
 cbnz w11,1b
.endm

// Lbackend_feature: read one documented Darwin optional-feature selector.
// No SME instructions.
//   x0 = name, a NUL-terminated sysctl name such as hw.optional.arm.FEAT_SME
// Calls sysctlbyname(name, &value, &length, NULL, 0) for a 4-byte value.
// Returns w0 = 1 only if the call succeeds, reports exactly 4 bytes and the
// value is nonzero; otherwise 0. A failing or unknown selector therefore
// reads as "feature absent".
// Frame, 32 bytes: [sp,#0] x29, x30; [sp,#16] the int32 value, preset to 0;
// [sp,#24] the 64-bit length, preset to 4.
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

// tf_backend_init(): choose the backend policy and kind for this image: the
// executable, or one loaded copy of the library. The state is not shared
// between two libraries loaded into the same process.
// Returns signed int64 in x0: 0 on success, -1 for an unrecognized policy
// value or for "sme" when the SME capability checks fail. The environment is
// captured once; later calls return the stored status. The once flag is
// taken with an exclusive load/store pair and released with a release
// store, which protects concurrent explicit initialization: a second caller
// retries until the first has published.
//
// Decision, in the order the code makes it:
//   ARM64_TRANSFORMER_BACKEND unset or "auto" -> policy 0
//   "neon" -> policy 1;  "sme" -> policy 2
//   any other value -> policy 3, status -1, no feature detection
//   supported = FEAT_SME and FEAT_SME2 selectors nonzero and RDSVL == 64
//   kind = SME (1) when supported and the policy is not 1; else NEON (0)
//   status = -1 when the policy is 2 and supported is 0
// Detection also runs for policy 1, so the capability flag and the vector
// length are recorded even when NEON is forced.
//
// Frame, 48 bytes: [sp,#0] x29, x30; [sp,#16] x19, x20; [sp,#32] x21, x22.
//   x19 = address of the once flag   x20 = status being built (0 or -1)
//   x21 = policy being built         x22 = the environment string
.globl _tf_backend_init
_tf_backend_init:
 stp x29,x30,[sp,#-48]!
 mov x29,sp
 stp x19,x20,[sp,#16]
 stp x21,x22,[sp,#32]
 BADDR x19,Lbackend_once
// Try to move the once flag from 0 to 1. A flag of 2 means initialization
// has finished; a flag of 1 means another call is in the middle of it.
Linit_acquire:
 ldaxr w9,[x19]
 cmp w9,#2
 b.eq Linit_complete
 cbnz w9,Linit_wait
 mov w10,#1
 stxr w11,w10,[x19]
 cbnz w11,Linit_acquire
 // This call owns initialization. Start from status 0 and policy 0 (auto),
 // then compare the environment value with "auto", "neon" and "sme".
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
 // Unrecognized value: fail, record policy 3, skip detection.
 mov x20,#-1
 mov x21,#3
 b Linit_publish
Linit_neon:
 mov x21,#1
 b Linit_features
Linit_sme:
 mov x21,#2
// Capability detection, reached for policies 0, 1 and 2. Both OS selectors
// must be nonzero before any SME instruction executes.
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
 // Record the vector length before testing it, so a rejected length is kept.
 BADDR x9,Lbackend_vector_bytes
 str x10,[x9]
 cmp x10,#64
 b.ne Linit_unavailable
 // Capability confirmed. Select SME unless the policy forces NEON: the kind
 // global is zero-filled, so skipping the store leaves NEON (0) selected.
 mov x10,#1
 BADDR x9,Lbackend_supported
 str x10,[x9]
 cmp x21,#1
 b.eq Linit_publish
 BADDR x9,Lbackend_kind
 str x10,[x9]
 b Linit_publish
// SME unavailable: an error only when the policy demanded SME.
Linit_unavailable:
 cmp x21,#2
 b.ne Linit_publish
 mov x20,#-1
// Publish status and policy, then mark the once flag complete (2) with a
// release store.
Linit_publish:
 BADDR x9,Lbackend_status
 str x20,[x9]
 BADDR x9,Lbackend_policy
 str x21,[x9]
 mov w9,#2
 stlr w9,[x19]
 b Linit_return
// Another call is initializing: clear the exclusive monitor, retry.
Linit_wait:
 clrex
 yield
 b Linit_acquire
// Already complete: clear the exclusive monitor set by LDAXR.
Linit_complete:
 clrex
// Every path returns the published status.
Linit_return:
 BADDR x9,Lbackend_status
 ldr x0,[x9]
 ldp x21,x22,[sp,#32]
 ldp x19,x20,[sp,#16]
 ldp x29,x30,[sp],#48
 ret

// BGET public, symbol: define an exported getter that returns one 64-bit
// global in x0 with an acquire load. The getters do not call
// tf_backend_init; before initialization they return the zero-filled
// defaults.
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

// tf_backend_info(out)
//   x0 = out  uint64[10], written
// Calls tf_backend_init (which only returns the stored status after the
// first call), then writes
//   out[0] = 1, the layout version   out[1] = init status, signed
//   out[2] = requested policy        out[3] = selected kind
//   out[4] = SME capability flag     out[5] = streaming vector bytes
//   out[6] = SME forward calls       out[7] = SME backward calls
//   out[8] = NEON forward calls      out[9] = NEON backward calls
// The counters are loaded one at a time, so the copy is not a consistent
// snapshot of calls made concurrently. No return value.
// Frame, 32 bytes: [sp,#0] x29, x30; [sp,#16] x19. x19 = out.
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

// BCHECK shape, fallback: the per-call gate shared by forward and backward.
// shape is the register holding the pointer to int32 [rows, input, output].
// Execution falls through to the SME path only when ALL of the following
// hold, and branches to fallback (the NEON path) otherwise:
//   - initialization is complete (once flag == 2) and selected SME;
//   - rows == 32;
//   - input is a power of two from 64 to 2048;
//   - output is a multiple of 32 from 64 to 3072;
//   - PSTATE.SM and PSTATE.ZA are both clear (SVCR bits 0 and 1);
//   - TPIDR2_EL0 is null, so no lazy ZA save is pending;
//   - the streaming vector length is still 64 bytes.
// The SVCR, TPIDR2_EL0 and RDSVL accesses come after the kind test, so they
// execute only when initialization confirmed SME. The macro touches only the
// scratch registers x9 and x10; the argument registers pass through.
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
 cmp w9,#2048
 b.hi \fallback
 // input & (input - 1) is zero exactly for a power of two.
 sub w10,w9,#1
 tst w9,w10
 b.ne \fallback
 ldr w9,[\shape,#8]
 cmp w9,#64
 b.lo \fallback
 cmp w9,#3072
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

// tf_linear_forward(x, w, b, y, shape): the public dense forward entry.
//   x0 = x  float32[rows][input]     x1 = w  float32[output][input]
//   x2 = b  float32[output]          x3 = y  float32[rows][output], written
//   x4 = shape  int32[3] = [rows, input, output]
// BCHECK selects tf_linear_forward_sme or tf_linear_forward_neon, which do
// the arithmetic and take these arguments unchanged. The call is counted
// under the path taken, then control tail-branches, so the chosen
// implementation returns directly to the caller. No frame, no return value.
.globl _tf_linear_forward
_tf_linear_forward:
 BCHECK x4,Ldispatch_forward_neon
 BCOUNT Lbackend_sme_forward
 b _tf_linear_forward_sme
Ldispatch_forward_neon:
 BCOUNT Lbackend_neon_forward
 b _tf_linear_forward_neon

// tf_linear_backward(x, w, dy, dx, dw, db, shape): the public dense backward
// entry, dispatched the same way to tf_linear_backward_sme or
// tf_linear_backward_neon.
//   x0 = x   float32[rows][input]    x1 = w   float32[output][input]
//   x2 = dy  float32[rows][output]   x3 = dx  float32[rows][input], overwritten
//   x4 = dw  float32[output][input], added to
//   x5 = db  float32[output], added to
//   x6 = shape  int32[3] = [rows, input, output]
// No frame, no return value.
.globl _tf_linear_backward
_tf_linear_backward:
 BCHECK x6,Ldispatch_backward_neon
 BCOUNT Lbackend_sme_backward
 b _tf_linear_backward_sme
Ldispatch_backward_neon:
 BCOUNT Lbackend_neon_backward
 b _tf_linear_backward_neon

// Helpers for tf_backend_abi_check.
// BSEED reg, value: load a known bit pattern into one of d8-d15.
// BVERIFY reg, value: clear the pass flag x20 unless reg still holds it.
// BSEED_ALL, BVERIFY_ALL: apply them to d8-d15, one pattern per register.
// BVERIFY_STATE: when SME is supported, clear x20 unless SVCR and TPIDR2_EL0
// still equal the values saved in x21 and x22.
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

// tf_backend_abi_check(): independent ABI self-test. Returns 1 in x0 on
// pass and 0 on failure, including failed initialization or allocation. No
// model parameters or files are touched.
// It runs one tf_linear_forward and one tf_linear_backward of shape
// [32, 64, 64], which is eligible for SME, on zero-filled inputs. Both calls
// go through the public dispatchers and therefore increment the dispatch
// counters. Around each call it seeds d8-d15 with known bit patterns and
// checks afterwards that they are unchanged; when the SME capability flag is
// set it also checks that SVCR and TPIDR2_EL0 are unchanged.
// Frame, 112 bytes: [sp,#0] x29, x30; [sp,#16] x19, x20; [sp,#32] x21, x22;
// [sp,#48]..[sp,#96] d8-d15 in pairs.
//   x19 = scratch buffer   x20 = result
//   x21, x22 = SVCR and TPIDR2_EL0 before the calls (only with SME present)
// Scratch buffer, 57856 zero-filled bytes from calloc, by byte offset:
//       0  x  [32][64]         8192  w  [64][64]       24576  b  [64]
//   24832  y  [32][64], then passed as dy to the backward call
//   33024  dx [32][64]        41216  dw [64][64]       57600  db [64]
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
 // The result stays 0 until initialization and allocation succeed.
 mov x20,#0
 bl _tf_backend_init
 cbnz x0,Labi_return
 mov x0,#57856
 mov x1,#1
 bl _calloc
 cbz x0,Labi_return
 mov x19,x0
 mov x20,#1
 // With SME present, save SVCR and TPIDR2_EL0 for BVERIFY_STATE.
 BADDR x9,Lbackend_supported
 ldr x9,[x9]
 cbz x9,Labi_forward
 mrs x21,SVCR
 mrs x22,TPIDR2_EL0
// Forward call: x, w, b, y at the offsets listed above.
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
 // Backward call: x, w, dy (the forward output), dx, dw, db.
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
// Common exit: return the pass flag, restore the saved registers.
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

// Constructors perform detection before public calls; failure remains
// queryable instead of aborting the hosting process. The executable rejects
// it explicitly.
// The pointer in __DATA,__mod_init_func below registers this routine as an
// image initializer. It tail-branches to tf_backend_init and nothing reads
// the returned status here.
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
// Shape of the tf_backend_abi_check calls: [rows, input, output].
.section __TEXT,__const
.p2align 2
Labi_shape: .long 32,64,64
// Backend state of this image, zero-filled at load. The labels are private,
// so the executable and each loaded library keep their own copy.
//   Lbackend_once          0 not started, 1 in progress, 2 complete (32-bit)
//   Lbackend_status        0 or -1 (64-bit, signed)
//   Lbackend_policy        0 auto, 1 neon, 2 sme, 3 unrecognized value
//   Lbackend_kind          0 NEON, 1 SME
//   Lbackend_supported     1 when the SME capability checks passed
//   Lbackend_vector_bytes  value read by RDSVL, 0 if RDSVL was never reached
//   the last four          dispatch counters, incremented by BCOUNT
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
