// Standalone handwritten Darwin ARM64 training, selection, evaluation and timing.
// libc calls are restricted to process, memory, files and clocks. No C math.
#include "layout.inc"
.text
.p2align 2
.macro ADDR reg, sym
 adrp \reg, \sym@PAGE
 add \reg, \reg, \sym@PAGEOFF
.endm
.macro LD reg, sym
 ADDR x9,\sym
 ldr \reg,[x9]
.endm
.macro ST reg, sym
 ADDR x9,\sym
 str \reg,[x9]
.endm
.macro CFG reg, off
 LD x9,Rconfig
 ldr \reg,[x9,#\off]
.endm
.macro IMM reg, value
 mov \reg,#(\value & 65535)
 movk \reg,#((\value >> 16) & 65535),lsl #16
.endm
.macro PTR reg, base, offset
 add \reg,\base,#(\offset >> 12),lsl #12
 add \reg,\reg,#(\offset & 4095)
.endm
.macro ENTER
 stp x29,x30,[sp,#-160]!
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
.macro LEAVE
 ldp d14,d15,[sp,#144]
 ldp d12,d13,[sp,#128]
 ldp d10,d11,[sp,#112]
 ldp d8,d9,[sp,#96]
 ldp x27,x28,[sp,#80]
 ldp x25,x26,[sp,#64]
 ldp x23,x24,[sp,#48]
 ldp x21,x22,[sp,#32]
 ldp x19,x20,[sp,#16]
 ldp x29,x30,[sp],#160
 ret
.endm

Rfail:
 ADDR x0,Lfailure
 bl _puts
 mov w0,#2
 bl _exit

Ralloc:
 stp x29,x30,[sp,#-16]!
 mov x1,#1
 bl _calloc
 cbz x0,Rfail
 ldp x29,x30,[sp],#16
 ret

Rnow:
 stp x29,x30,[sp,#-16]!
 bl _mach_absolute_time
 ucvtf d0,x0
 LD d1,Rtick
 fmul d0,d0,d1
 ldp x29,x30,[sp],#16
 ret

// d0 CPU seconds, x0 peak RSS bytes.
Rusage:
 stp x29,x30,[sp,#-16]!
 mov w0,#0
 ADDR x1,Rusagebuf
 bl _getrusage
 cbnz w0,Rfail
 ADDR x9,Rusagebuf
 ldr x0,[x9]
 ldr x1,[x9,#16]
 add x0,x0,x1
 ucvtf d0,x0
 ldr w0,[x9,#8]
 ldr w1,[x9,#24]
 add x0,x0,x1
 ucvtf d1,x0
 ADDR x10,Lmillion
 ldr d2,[x10]
 fdiv d1,d1,d2
 fadd d0,d0,d1
 ldr x0,[x9,#32]
 ldp x29,x30,[sp],#16
 ret

// root, basename, destination (2048 bytes). Roots bounded at startup.
Rpath:
 ENTER
 mov x19,x1
 mov x20,x2
 mov x1,x0
 mov x0,x20
 bl _strcpy
 mov x0,x20
 ADDR x1,Lslash
 bl _strcat
 mov x0,x20
 mov x1,x19
 bl _strcat
 mov x0,x20
 LEAVE

// basename, exact byte count -> owned buffer; require EOF, no short reads.
Rread:
 ENTER
 sub sp,sp,#2048
 mov x19,x1
 mov x1,x0
 LD x0,Rinput
 mov x2,sp
 bl Rpath
 ADDR x1,Lrb
 bl _fopen
 cbz x0,Rfail
 mov x20,x0
 mov x0,x19
 bl Ralloc
 mov x21,x0
 mov x1,#1
 mov x2,x19
 mov x3,x20
 bl _fread
 cmp x0,x19
 b.ne Rfail
 mov x0,x20
 bl _fgetc
 cmn w0,#1
 b.ne Rfail
 mov x0,x20
 bl _ferror
 cbnz w0,Rfail
 mov x0,x20
 bl _fclose
 cbnz w0,Rfail
 mov x0,x21
 add sp,sp,#2048
 LEAVE

Ropenout:
 ENTER
 sub sp,sp,#2048
 mov x1,x0
 LD x0,Routput
 mov x2,sp
 bl Rpath
 ADDR x1,Lwb
 bl _fopen
 cbz x0,Rfail
 add sp,sp,#2048
 LEAVE

// basename, data, bytes.
Rwrite:
 ENTER
 mov x19,x1
 mov x20,x2
 bl Ropenout
 mov x21,x0
 mov x0,x19
 mov x1,#1
 mov x2,x20
 mov x3,x21
 bl _fwrite
 cmp x0,x20
 b.ne Rfail
 mov x0,x21
 bl _fclose
 cbnz w0,Rfail
 LEAVE

Rtokens:
 mov x2,#33
 mul x1,x1,x2
Rtokenloop:
 ldr w2,[x0],#4
 cmp w2,#128
 b.hs Rfail
 subs x1,x1,#1
 b.ne Rtokenloop
 ret

Rcheckparams:
 stp x29,x30,[sp,#-16]!
 mov x1,#PARAM_COUNT
 bl _tf_finite
 cbz w0,Rfail
 ldp x29,x30,[sp],#16
 ret

// x0 data, x1 sequence count, x2 optional FILE* for logits. Return mean NLL d0.
Revaluate:
 ENTER
 mov x19,x0
 mov x20,x1
 mov x21,x2
 mov x22,#0
 fmov d8,xzr
Reval_loop:
 LD x0,Rparams
 mov x1,x19
 LD x2,Rworkspace
 bl _tf_forward
 LD x9,Rworkspace
 PTR x0,x9,W_LOGITS
 mov x1,#4096
 bl _tf_finite
 cbz w0,Rfail
 LD x9,Rworkspace
 PTR x0,x9,W_LOGITS
 add x1,x19,#4
 PTR x2,x9,W_DLOGITS
 mov x3,#CONTEXT
 mov x4,#VOCAB
 bl _tf_cross_entropy
 fmov w9,s0
 and w9,w9,#0x7f800000
 mov w10,#0x7f800000
 cmp w9,w10
 b.eq Rfail
 fcvt d0,s0
 fadd d8,d8,d0
 cbz x21,Reval_next
 LD x9,Rworkspace
 PTR x0,x9,W_LOGITS
 mov x1,#1
 mov x2,#16384
 mov x3,x21
 bl _fwrite
 cmp x0,#4,lsl #12
 b.ne Rfail
Reval_next:
 add x19,x19,#132
 add x22,x22,#1
 cmp x22,x20
 b.lo Reval_loop
 ucvtf d0,x20
 fdiv d0,d8,d0
 LEAVE

// x0 epoch's permutation. Native batching, gradient accumulation and Adam.
Rtrain_epoch:
 ENTER
 mov x19,x0
 CFG w20,16
 CFG w21,12
 mov x22,#0
 fmov d8,xzr
Rbatch:
 sub x23,x20,x22
 cmp x23,x21
 csel x23,x23,x21,lo
 LD x0,Rgrads
 mov w1,#0
 IMM x2,PARAM_BYTES
 bl _memset
 mov x24,#0
Rsample:
 add x9,x22,x24
 ldr w9,[x19,x9,lsl #2]
 mov x10,#132
 mul x25,x9,x10
 LD x9,Rtrain
 add x25,x25,x9
 LD x0,Rparams
 mov x1,x25
 LD x2,Rworkspace
 bl _tf_forward
 LD x9,Rworkspace
 PTR x0,x9,W_LOGITS
 mov x1,#4096
 bl _tf_finite
 cbz w0,Rfail
 LD x0,Rparams
 mov x1,x25
 add x2,x25,#4
 LD x3,Rworkspace
 LD x4,Rgrads
 bl _tf_backward
 fmov w9,s0
 and w9,w9,#0x7f800000
 mov w10,#0x7f800000
 cmp w9,w10
 b.eq Rfail
 fcvt d0,s0
 fadd d8,d8,d0
 add x24,x24,#1
 cmp x24,x23
 b.lo Rsample
 LD x0,Rgrads
 mov x1,#PARAM_COUNT
 ucvtf s1,x23
 fmov s0,#1.0
 fdiv s0,s0,s1
 bl _tf_scale
 LD x0,Rgrads
 bl Rcheckparams
 LD x4,Rstep
 add x4,x4,#1
 ST x4,Rstep
 LD x0,Rparams
 LD x1,Rgrads
 LD x2,Rm
 LD x3,Rv
 mov x5,#PARAM_COUNT
 bl _tf_adam
 LD x0,Rparams
 bl Rcheckparams
 LD x0,Rm
 bl Rcheckparams
 LD x0,Rv
 bl Rcheckparams
 add x22,x22,x23
 cmp x22,x20
 b.lo Rbatch
 ucvtf d0,x20
 fdiv d0,d8,d0
 LEAVE

.globl _main
_main:
 ENTER
 cmp w0,#3
 b.ne Rfail
 ldr x19,[x1,#8]
 ldr x20,[x1,#16]
 ST x19,Rinput
 ST x20,Routput
 mov x0,x19
 bl _strlen
 cmp x0,#1800
 b.hi Rfail
 mov x0,x20
 bl _strlen
 cmp x0,#1800
 b.hi Rfail
 ADDR x0,Rtimebase
 bl _mach_timebase_info
 cbnz w0,Rfail
 ADDR x9,Rtimebase
 ldp w10,w11,[x9]
 cbz w11,Rfail
 ucvtf d0,w10
 ucvtf d1,w11
 fdiv d0,d0,d1
 ADDR x9,Lbillion
 ldr d1,[x9]
 fdiv d0,d0,d1
 ST d0,Rtick
 bl Rnow
 ST d0,Rstart
 ADDR x0,Lconfig
 mov x1,#32
 bl Rread
 ST x0,Rconfig
 mov x23,x0
 ldr w9,[x23]
 IMM w10,0x54465231
 cmp w9,w10
 b.ne Rfail
 ldr w9,[x23,#4]
 cmp w9,#1
 b.ne Rfail
 ldr w9,[x23,#8]
 sub w9,w9,#1
 cmp w9,#100
 b.hs Rfail
 ldr w9,[x23,#12]
 sub w9,w9,#1
 cmp w9,#128
 b.hs Rfail
 ldr w9,[x23,#16]
 sub w9,w9,#1
 cmp w9,#2,lsl #12
 b.hs Rfail
 ldr w9,[x23,#20]
 sub w9,w9,#1
 cmp w9,#2048
 b.hs Rfail
 ldr w9,[x23,#24]
 sub w9,w9,#1
 cmp w9,#2048
 b.hs Rfail
 ADDR x0,Linitial
 IMM x1,PARAM_BYTES
 bl Rread
 ST x0,Rparams
 bl Rcheckparams
 ADDR x0,Ltrain
 CFG w1,16
 mov x10,#132
 mul x1,x1,x10
 bl Rread
 ST x0,Rtrain
 CFG w1,16
 bl Rtokens
 ADDR x0,Lvalidation
 CFG w1,20
 mov x10,#132
 mul x1,x1,x10
 bl Rread
 ST x0,Rvalidation
 CFG w1,20
 bl Rtokens
 ADDR x0,Ltest
 CFG w1,24
 mov x10,#132
 mul x1,x1,x10
 bl Rread
 ST x0,Rtest
 CFG w1,24
 bl Rtokens
 ADDR x0,Lschedule
 CFG w1,16
 CFG w10,8
 mul x1,x1,x10
 lsl x1,x1,#2
 bl Rread
 ST x0,Rschedule
 mov x24,x0
 CFG w20,16
 CFG w21,8
 lsl x0,x20,#2
 bl Ralloc
 mov x25,x0
 mov w22,#1
Rsched_epoch:
 mov x23,#0
Rsched_index:
 ldr w10,[x24],#4
 cmp w10,w20
 b.hs Rfail
 ldr w11,[x25,x10,lsl #2]
 cmp w11,w22
 b.eq Rfail
 str w22,[x25,x10,lsl #2]
 add x23,x23,#1
 cmp x23,x20
 b.lo Rsched_index
 add w22,w22,#1
 cmp w22,w21
 b.ls Rsched_epoch
 mov x0,x25
 bl _free
 // mkdir itself is the no-overwrite gate, including an already empty directory.
 LD x0,Routput
 mov w1,#493
 bl _mkdir
 cbnz w0,Rfail
 IMM x0,WORK_BYTES
 bl Ralloc
 ST x0,Rworkspace
 IMM x0,PARAM_BYTES
 bl Ralloc
 ST x0,Rgrads
 IMM x0,PARAM_BYTES
 bl Ralloc
 ST x0,Rm
 IMM x0,PARAM_BYTES
 bl Ralloc
 ST x0,Rv
 IMM x0,PARAM_BYTES
 bl Ralloc
 ST x0,Rbest
 CFG w0,8
 mov x9,#24
 mul x0,x0,x9
 bl Ralloc
 ST x0,Rhistory
 ADDR x9,Linf
 ldr d0,[x9]
 ST d0,Rbestloss
 bl Rusage
 ST d0,Rcpustart
 ucvtf d1,x0
 ADDR x9,Rperformance
 str d1,[x9,#16]
 bl Rnow
 ST d0,Rtrainstart
 mov x19,#0
Repoch:
 bl Rnow
 fmov d8,d0
 CFG w9,16
 mul x10,x19,x9
 LD x0,Rschedule
 add x0,x0,x10,lsl #2
 bl Rtrain_epoch
 fmov d9,d0
 LD x0,Rvalidation
 CFG w1,20
 mov x2,#0
 bl Revaluate
 fmov d10,d0
 LD d1,Rbestloss
 fcmp d0,d1
 b.ge Rnot_best
 ST d0,Rbestloss
 add x10,x19,#1
 ST x10,Rbestepoch
 LD x0,Rbest
 LD x1,Rparams
 IMM x2,PARAM_BYTES
 bl _memcpy
 ADDR x0,Lbest
 LD x1,Rparams
 IMM x2,PARAM_BYTES
 bl Rwrite
Rnot_best:
 bl Rnow
 fsub d0,d0,d8
 LD x9,Rhistory
 mov x10,#24
 madd x9,x19,x10,x9
 stp d9,d10,[x9]
 str d0,[x9,#16]
 add x19,x19,#1
 CFG w9,8
 cmp x19,x9
 b.lo Repoch
 bl Rnow
 LD d1,Rtrainstart
 fsub d0,d0,d1
 ADDR x9,Rperformance
 str d0,[x9]
 bl Rusage
 LD d1,Rcpustart
 fsub d0,d0,d1
 ADDR x9,Rperformance
 str d0,[x9,#8]
 ucvtf d1,x0
 str d1,[x9,#24]
 ADDR x0,Llast
 LD x1,Rparams
 IMM x2,PARAM_BYTES
 bl Rwrite
 ADDR x0,Lm
 LD x1,Rm
 IMM x2,PARAM_BYTES
 bl Rwrite
 ADDR x0,Lv
 LD x1,Rv
 IMM x2,PARAM_BYTES
 bl Rwrite
 ADDR x0,Lhistory
 LD x1,Rhistory
 CFG w2,8
 mov x10,#24
 mul x2,x2,x10
 bl Rwrite
 LD x0,Rparams
 LD x1,Rbest
 IMM x2,PARAM_BYTES
 bl _memcpy
 ADDR x0,Llogits
 bl Ropenout
 mov x19,x0
 mov x2,x0
 LD x0,Rtest
 CFG w1,24
 bl Revaluate
 mov x0,x19
 bl _fclose
 cbnz w0,Rfail
 mov x19,#0
Rwarm:
 LD x0,Rparams
 LD x1,Rtest
 LD x2,Rworkspace
 bl _tf_forward
 add x19,x19,#1
 cmp x19,#20
 b.lo Rwarm
 mov x19,#0
Rtime:
 bl Rnow
 fmov d8,d0
 LD x0,Rparams
 LD x1,Rtest
 LD x2,Rworkspace
 bl _tf_forward
 bl Rnow
 fsub d0,d0,d8
 ADDR x9,Rtimings
 str d0,[x9,x19,lsl #3]
 add x19,x19,#1
 cmp x19,#100
 b.lo Rtime
 ADDR x0,Ltimings
 ADDR x1,Rtimings
 mov x2,#800
 bl Rwrite
 bl Rusage
 ucvtf d0,x0
 ADDR x10,Rperformance
 str d0,[x10,#32]
 LD x0,Rbestepoch
 ucvtf d0,x0
 str d0,[x10,#40]
 LD x0,Rstep
 ucvtf d0,x0
 str d0,[x10,#48]
 bl Rnow
 LD d1,Rstart
 fsub d0,d0,d1
 ADDR x10,Rperformance
 str d0,[x10,#56]
 ADDR x0,Lperformance
 ADDR x1,Rperformance
 mov x2,#64
 bl Rwrite
 mov w0,#0
 LEAVE

.section __TEXT,__cstring,cstring_literals
Lfailure: .asciz "Transformer native run rejected: invalid input, nonfinite arithmetic, existing output, or I/O failure."
Lslash: .asciz "/"
Lrb: .asciz "rb"
Lwb: .asciz "wb"
Lconfig: .asciz "config.bin"
Linitial: .asciz "initial.f32"
Ltrain: .asciz "train.i32"
Lvalidation: .asciz "validation.i32"
Ltest: .asciz "test.i32"
Lschedule: .asciz "schedule.i32"
Lbest: .asciz "best.f32"
Llast: .asciz "last.f32"
Lm: .asciz "adam_m.f32"
Lv: .asciz "adam_v.f32"
Lhistory: .asciz "history.f64"
Llogits: .asciz "test_logits.f32"
Ltimings: .asciz "timings.f64"
Lperformance: .asciz "performance.f64"
.section __TEXT,__const
.p2align 3
Lmillion: .double 1000000
Lbillion: .double 1000000000
Linf: .quad 0x7ff0000000000000
.zerofill __DATA,__bss,Rinput,8,3
.zerofill __DATA,__bss,Routput,8,3
.zerofill __DATA,__bss,Rconfig,8,3
.zerofill __DATA,__bss,Rparams,8,3
.zerofill __DATA,__bss,Rtrain,8,3
.zerofill __DATA,__bss,Rvalidation,8,3
.zerofill __DATA,__bss,Rtest,8,3
.zerofill __DATA,__bss,Rschedule,8,3
.zerofill __DATA,__bss,Rworkspace,8,3
.zerofill __DATA,__bss,Rgrads,8,3
.zerofill __DATA,__bss,Rm,8,3
.zerofill __DATA,__bss,Rv,8,3
.zerofill __DATA,__bss,Rbest,8,3
.zerofill __DATA,__bss,Rhistory,8,3
.zerofill __DATA,__bss,Rstep,8,3
.zerofill __DATA,__bss,Rbestepoch,8,3
.zerofill __DATA,__bss,Rbestloss,8,3
.zerofill __DATA,__bss,Rtimebase,8,3
.zerofill __DATA,__bss,Rtick,8,3
.zerofill __DATA,__bss,Rstart,8,3
.zerofill __DATA,__bss,Rcpustart,8,3
.zerofill __DATA,__bss,Rtrainstart,8,3
.zerofill __DATA,__bss,Rusagebuf,256,4
.zerofill __DATA,__bss,Rperformance,64,3
.zerofill __DATA,__bss,Rtimings,800,3
