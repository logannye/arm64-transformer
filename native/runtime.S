// Standalone handwritten Darwin ARM64 training, selection, evaluation and
// timing.
// libc calls are restricted to process, memory, files and clocks. No C math.
//
// This file is the whole program behind
//   transformer-arm64 INPUT_DIRECTORY OUTPUT_DIRECTORY
// main reads a prepared fixture, trains for the configured number of epochs,
// keeps the checkpoint with the lowest validation loss, writes that
// checkpoint's test logits, times single-sequence forward passes and writes
// the results. The numerical work is done by tf_forward and tf_backward
// (model.S), tf_adam, tf_cross_entropy, tf_scale and tf_finite; this file
// sequences those calls and does the file handling.
//
// Input files (formats in native/ABI.md; written by transformer/prepare.py):
//   config.bin    nine little-endian uint32 at byte offsets 0, 4, ... 32:
//                 magic, version, epochs, batch, train_sequences,
//                 validation_sequences, test_sequences, seed, width
//   initial.f32   PARAM_COUNT float32 starting parameters
//   train.i32, validation.i32, test.i32
//                 int32 token ids, [sequences][CONTEXT+1]; the first CONTEXT
//                 tokens of a row are the inputs, the last CONTEXT the targets
//   schedule.i32  int32 [epochs][train_sequences], one permutation per epoch
// Output files: best.f32, last.f32, adam_m.f32, adam_v.f32, history.f64,
// test_logits.f32, timings.f64, performance.f64, backend.u64.
//
// Routines (all local except main):
//   Rfail         print the rejection message and exit with status 2
//   Ralloc        zero-filled allocation; never returns NULL
//   Rnow          monotonic time in seconds
//   Rusage        process CPU seconds and peak resident set size
//   Rpath         build "root/basename"
//   Rread         read a whole input file of an exact size
//   Ropenout      create one output file
//   Rwrite        write one output file
//   Rtokens       reject token ids outside 0..VOCAB-1
//   Rcheckparams  reject a non-finite array of PARAM_COUNT floats
//   Revaluate     mean loss over a split, optionally writing logits
//   Rtrain_epoch  one epoch of batched training
//   main          the program
//
// Error handling: every failed check branches to Rfail, which does not
// return. main returns 0 only after every output file has been written.
// Process state lives in the zero-filled globals at the end of the file.
// The pointer and scalar slots are read with the LD macro and written with
// ST; Rtimebase and the larger buffers are addressed with ADDR.
#include <layout.inc>
.text
.p2align 2
// ADDR reg, sym: reg = address of a symbol.
// LD reg, sym and ST reg, sym: load or store one global; both clobber x9.
// CFG reg, off: load the config.bin field at byte offset off; clobbers x9.
// IMM reg, value: load a constant below 2^32 (MOV plus optional MOVK).
// PTR reg, base, offset: reg = base + constant byte offset, as in model.S.
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
 .if (\value >> 16)
 movk \reg,#((\value >> 16) & 65535),lsl #16
 .endif
.endm
.macro PTR reg, base, offset
 // Match model.S: wider parameter offsets need more than one page ADD.
 // Existing layouts retain their original two-instruction address sequence.
 .if (\offset >> 12) > 4095
 add \reg,\base,#4095,lsl #12
 .rept ((\offset >> 12) / 4095) - 1
 add \reg,\reg,#4095,lsl #12
 .endr
 add \reg,\reg,#((\offset >> 12) % 4095),lsl #12
 .else
 add \reg,\base,#(\offset >> 12),lsl #12
 .endif
 add \reg,\reg,#(\offset & 4095)
.endm
// ENTER / LEAVE: a 160-byte frame that saves every callee-saved register.
//   [sp,#0] x29, x30    [sp,#16]..[sp,#80] x19-x28 in pairs
//   [sp,#96]..[sp,#144] d8-d15 in pairs
// LEAVE restores them and returns to the caller.
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

// Rfail: the single failure exit. Prints one fixed message with puts and
// calls exit(2). It is reached by branches, never by a call, from any depth;
// since it does not return, the frames it abandons do not matter.
Rfail:
 ADDR x0,Lfailure
 bl _puts
 mov w0,#2
 bl _exit

// Ralloc: x0 = byte count -> x0 = zero-filled buffer from calloc(count, 1).
// A NULL result goes to Rfail, so callers never test the pointer.
Ralloc:
 stp x29,x30,[sp,#-16]!
 mov x1,#1
 bl _calloc
 cbz x0,Rfail
 ldp x29,x30,[sp],#16
 ret

// Rnow: returns d0 = mach_absolute_time() * Rtick, a monotonic clock in
// seconds. main stores Rtick (seconds per tick) before the first call.
Rnow:
 stp x29,x30,[sp,#-16]!
 bl _mach_absolute_time
 ucvtf d0,x0
 LD d1,Rtick
 fmul d0,d0,d1
 ldp x29,x30,[sp],#16
 ret

// Rusage: getrusage(RUSAGE_SELF) into Rusagebuf; a nonzero result goes to
// Rfail. Returns d0 = user + system CPU seconds, x0 = peak RSS bytes
// (ru_maxrss).
Rusage:
 stp x29,x30,[sp,#-16]!
 mov w0,#0
 ADDR x1,Rusagebuf
 bl _getrusage
 cbnz w0,Rfail
 // struct rusage begins with ru_utime and ru_stime, each a 16-byte timeval
 // (64-bit seconds at +0, 32-bit microseconds at +8); ru_maxrss is at +32.
 // The two seconds fields and the two microsecond fields are summed
 // separately, then d0 = seconds + microseconds / 1e6.
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

// Rpath: x0 = root directory, x1 = basename, x2 = destination buffer of
// 2048 bytes. Writes "root/basename" with strcpy and two strcat calls and
// returns the destination in x0. Roots are bounded at startup: main rejects
// any longer than 1800 bytes, which leaves room for the slash and the
// longest basename used here. x19 = basename, x20 = destination.
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

// Rread: x0 = basename, x1 = exact byte count -> x0 = newly allocated
// buffer holding all of INPUT_DIRECTORY/basename.
// The file must contain exactly that many bytes: a failed open, a short
// read, a byte left after the read, a stream error or a failed close all go
// to Rfail. The buffer is never freed by this routine.
// Frame: ENTER plus 2048 bytes of stack for the path.
//   x19 = byte count, x20 = FILE*, x21 = buffer
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
 // One more read must hit end of file: fgetc returns EOF (-1).
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

// Ropenout: x0 = basename -> x0 = FILE* for OUTPUT_DIRECTORY/basename,
// opened with mode "wb". A failed open goes to Rfail. The path is built in
// 2048 bytes of stack below the ENTER frame.
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

// Rwrite: x0 = basename, x1 = data, x2 = byte count. Creates the output
// file through Ropenout, writes all bytes with one fwrite and closes it; a
// short write or a failed close goes to Rfail.
//   x19 = data, x20 = byte count, x21 = FILE*
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

// Rtokens: x0 = token array, x1 = sequence count. Checks all
// count * (CONTEXT+1) int32 values: an id that is not below VOCAB as an
// unsigned number, which includes every negative id, goes to Rfail. Leaf.
Rtokens:
 mov x2,#(CONTEXT+1)
 mul x1,x1,x2
// One pass per token; x1 counts down from count * (CONTEXT+1).
Rtokenloop:
 ldr w2,[x0],#4
 cmp w2,#VOCAB
 b.hs Rfail
 subs x1,x1,#1
 b.ne Rtokenloop
 ret

// Rcheckparams: x0 = float32[PARAM_COUNT]. Goes to Rfail unless tf_finite
// reports every value finite. Used for the parameters, the gradients and
// both Adam moment arrays, which all have PARAM_COUNT elements.
Rcheckparams:
 stp x29,x30,[sp,#-16]!
 IMM x1,PARAM_COUNT
 bl _tf_finite
 cbz w0,Rfail
 ldp x29,x30,[sp],#16
 ret

// Revaluate: x0 = split data, int32[count][CONTEXT+1], x1 = sequence count,
// x2 = FILE* that receives the logits, or 0. Returns d0 = the mean over
// sequences of each sequence's mean token loss (NLL), summed in FP64.
// For each sequence: tf_forward with the current Rparams, reject non-finite
// logits, tf_cross_entropy against the same row shifted by one token,
// reject a non-finite loss and, if a FILE* was given, append the
// [CONTEXT][VOCAB] float32 logits to it. Parameters are not modified.
//   x19 = current row, x20 = count, x21 = FILE* or 0,
//   x22 = sequences done, d8 = loss sum
Revaluate:
 ENTER
 mov x19,x0
 mov x20,x1
 mov x21,x2
 mov x22,#0
 fmov d8,xzr
// One pass per sequence; x22 counts 0..count-1.
Reval_loop:
 LD x0,Rparams
 mov x1,x19
 LD x2,Rworkspace
 bl _tf_forward
 // Reject non-finite logits before they reach the loss.
 LD x9,Rworkspace
 PTR x0,x9,W_LOGITS
 mov x1,#(CONTEXT*VOCAB)
 bl _tf_finite
 cbz w0,Rfail
 // Targets are the same row one token later (x19 + 4 bytes).
 LD x9,Rworkspace
 PTR x0,x9,W_LOGITS
 add x1,x19,#4
 PTR x2,x9,W_DLOGITS
 mov x3,#CONTEXT
 mov x4,#VOCAB
 bl _tf_cross_entropy
 // Non-finite loss test on the exponent bits, as in tf_finite.
 fmov w9,s0
 and w9,w9,#0x7f800000
 mov w10,#0x7f800000
 cmp w9,w10
 b.eq Rfail
 fcvt d0,s0
 fadd d8,d8,d0
 // Optional output: CONTEXT*VOCAB float32 logits per sequence.
 cbz x21,Reval_next
 LD x9,Rworkspace
 PTR x0,x9,W_LOGITS
 mov x1,#1
 mov x2,#(CONTEXT*VOCAB*4)
 mov x3,x21
 bl _fwrite
 mov x9,#(CONTEXT*VOCAB*4)
 cmp x0,x9
 b.ne Rfail
// Advance to the next row of CONTEXT+1 tokens.
Reval_next:
 add x19,x19,#((CONTEXT+1)*4)
 add x22,x22,#1
 cmp x22,x20
 b.lo Reval_loop
 ucvtf d0,x20
 fdiv d0,d8,d0
 LEAVE

// Rtrain_epoch: x0 = this epoch's permutation, int32[train_sequences].
// Native batching, gradient accumulation and Adam. Returns d0 = the mean
// training loss over the epoch's sequences, each measured with the
// parameters in effect before its own batch's update.
// A batch is the next `batch` permutation entries; the final batch keeps
// whatever remains. For each batch:
//   1. zero Rgrads;
//   2. for each sequence: tf_forward, then tf_backward, which adds that
//      sequence's gradient to Rgrads;
//   3. tf_scale(Rgrads, PARAM_COUNT, 1/n), n being the ACTUAL number of
//      sequences in the batch, to obtain the batch-mean gradient;
//   4. increment Rstep and call tf_adam(params, grads, m, v, step, count).
// Non-finite values go to Rfail: logits and loss per sequence, the scaled
// gradient before Adam, and parameters, m and v after it.
//   x19 = permutation, x20 = train_sequences, x21 = batch size,
//   x22 = sequences consumed, x23 = size of this batch,
//   x24 = index within the batch, x25 = current training row, d8 = loss sum
Rtrain_epoch:
 ENTER
 mov x19,x0
 CFG w20,16
 CFG w21,12
 mov x22,#0
 fmov d8,xzr
// One pass per batch, until x22 reaches train_sequences.
// x23 = min(sequences remaining, batch size).
Rbatch:
 sub x23,x20,x22
 cmp x23,x21
 csel x23,x23,x21,lo
 // Gradients accumulate over the batch, so clear them first.
 LD x0,Rgrads
 mov w1,#0
 IMM x2,PARAM_BYTES
 bl _memset
 mov x24,#0
// One pass per sequence of the batch; x24 counts 0..x23-1.
Rsample:
 // x25 = the Rtrain row named by the next permutation entry.
 add x9,x22,x24
 ldr w9,[x19,x9,lsl #2]
 mov x10,#((CONTEXT+1)*4)
 mul x25,x9,x10
 LD x9,Rtrain
 add x25,x25,x9
 LD x0,Rparams
 mov x1,x25
 LD x2,Rworkspace
 bl _tf_forward
 LD x9,Rworkspace
 PTR x0,x9,W_LOGITS
 mov x1,#(CONTEXT*VOCAB)
 bl _tf_finite
 cbz w0,Rfail
 // Backward pass: inputs at x25, targets one token later.
 LD x0,Rparams
 mov x1,x25
 add x2,x25,#4
 LD x3,Rworkspace
 LD x4,Rgrads
 bl _tf_backward
 // Reject a non-finite loss, then add it to the epoch sum in FP64.
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
 // Batch-mean gradient: scale by 1/x23, formed in float32.
 LD x0,Rgrads
 IMM x1,PARAM_COUNT
 ucvtf s1,x23
 fmov s0,#1.0
 fdiv s0,s0,s1
 bl _tf_scale
 LD x0,Rgrads
 bl Rcheckparams
 // The Adam step number is 1-based and keeps counting across epochs.
 LD x4,Rstep
 add x4,x4,#1
 ST x4,Rstep
 LD x0,Rparams
 LD x1,Rgrads
 LD x2,Rm
 LD x3,Rv
 IMM x5,PARAM_COUNT
 bl _tf_adam
 // The update must leave the parameters and both moments finite.
 LD x0,Rparams
 bl Rcheckparams
 LD x0,Rm
 bl Rcheckparams
 LD x0,Rv
 bl Rcheckparams
 add x22,x22,x23
 cmp x22,x20
 b.lo Rbatch
 // d0 = loss sum / train_sequences.
 ucvtf d0,x20
 fdiv d0,d8,d0
 LEAVE

// main(argc, argv): w0 = argc, x1 = argv. Returns 0; every failure exits
// through Rfail with status 2.
//
// Order of work:
//   1. Arguments: exactly two, each at most 1800 bytes long.
//   2. tf_backend_init, clock calibration, Rstart = origin of total wall time.
//   3. Read and validate config.bin, then initial.f32, the three token files
//      and schedule.i32. The output directory is created only after all of
//      them pass.
//   4. mkdir the output directory, which must not exist yet.
//   5. Allocate the workspace, gradients, Adam moments, best-checkpoint copy
//      and history.
//   6. Timed training scope: for each epoch, Rtrain_epoch, the validation
//      loss, checkpoint selection (only a strictly lower validation loss
//      replaces the best, so the earliest epoch wins exact ties) and a
//      history row.
//   7. Write last.f32, adam_m.f32, adam_v.f32 and history.f64.
//   8. Restore the selected checkpoint and write its test logits.
//   9. 20 untimed, then 100 timed, single-sequence tf_forward calls.
//  10. Write timings.f64, performance.f64 and backend.u64.
//
// Callee-saved registers are reused from phase to phase:
//   x19  INPUT_DIRECTORY, later the epoch index, the logits FILE*, a counter
//   x20  OUTPUT_DIRECTORY, later train_sequences for the schedule check
//   x21  epochs for the schedule check
//   x22  1-based epoch number for the schedule check
//   x23  config pointer, later the index within a schedule row
//   x24  schedule cursor         x25  marks buffer of the schedule check
//   d8   epoch or timing start   d9, d10  training and validation loss
.globl _main
_main:
 ENTER
 // argc must be 3: the program name plus two directories.
 cmp w0,#3
 b.ne Rfail
 ldr x19,[x1,#8]
 ldr x20,[x1,#16]
 ST x19,Rinput
 ST x20,Routput
 // Bound both paths so that the 2048-byte path buffers cannot overflow.
 mov x0,x19
 bl _strlen
 cmp x0,#1800
 b.hi Rfail
 mov x0,x20
 bl _strlen
 cmp x0,#1800
 b.hi Rfail
 // The backend policy must initialize cleanly (0) before any kernel runs.
 bl _tf_backend_init
 cbnz x0,Rfail
 // mach_timebase_info gives numer and denom, nanoseconds per tick as a
 // ratio. Rtick = numer / denom / 1e9 seconds per tick; denom 0 is rejected.
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
 // Rstart: origin of the total native wall time in performance.f64.
 bl Rnow
 ST d0,Rstart
 // config.bin: exactly 36 bytes. Magic 0x54465231, version 2 and the compiled
 // WIDTH are checked first, then each count. "value - 1, unsigned, below N"
 // tests 1 <= value <= N with one compare.
 ADDR x0,Lconfig
 mov x1,#36
 bl Rread
 ST x0,Rconfig
 mov x23,x0
 ldr w9,[x23]
 IMM w10,0x54465231
 cmp w9,w10
 b.ne Rfail
 ldr w9,[x23,#4]
 cmp w9,#2
 b.ne Rfail
 ldr w9,[x23,#32]
 cmp w9,#WIDTH
 b.ne Rfail
 // epochs in 1..100
 ldr w9,[x23,#8]
 sub w9,w9,#1
 cmp w9,#100
 b.hs Rfail
 // batch in 1..128
 ldr w9,[x23,#12]
 sub w9,w9,#1
 cmp w9,#128
 b.hs Rfail
 // train_sequences in 1..8192 (2 << 12)
 ldr w9,[x23,#16]
 sub w9,w9,#1
 cmp w9,#2,lsl #12
 b.hs Rfail
 // validation_sequences in 1..2048
 ldr w9,[x23,#20]
 sub w9,w9,#1
 cmp w9,#2048
 b.hs Rfail
 // test_sequences in 1..2048
 ldr w9,[x23,#24]
 sub w9,w9,#1
 cmp w9,#2048
 b.hs Rfail
 // initial.f32: PARAM_BYTES bytes, every value finite.
 ADDR x0,Linitial
 IMM x1,PARAM_BYTES
 bl Rread
 ST x0,Rparams
 bl Rcheckparams
 // train.i32, validation.i32 and test.i32: sequences * (CONTEXT+1) * 4 bytes
 // each, every token id below VOCAB.
 ADDR x0,Ltrain
 CFG w1,16
 mov x10,#((CONTEXT+1)*4)
 mul x1,x1,x10
 bl Rread
 ST x0,Rtrain
 CFG w1,16
 bl Rtokens
 ADDR x0,Lvalidation
 CFG w1,20
 mov x10,#((CONTEXT+1)*4)
 mul x1,x1,x10
 bl Rread
 ST x0,Rvalidation
 CFG w1,20
 bl Rtokens
 ADDR x0,Ltest
 CFG w1,24
 mov x10,#((CONTEXT+1)*4)
 mul x1,x1,x10
 bl Rread
 ST x0,Rtest
 CFG w1,24
 bl Rtokens
 // schedule.i32: epochs * train_sequences int32. Every epoch's row must be a
 // permutation of 0..train_sequences-1. The check keeps one zero-filled
 // int32 mark per sequence id (x25). A mark equal to the current 1-based
 // epoch number (w22) means the id already occurred in this row; stamping
 // with the epoch number makes it unnecessary to clear the marks per row.
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
// One pass per epoch row; w22 counts 1..epochs.
Rsched_epoch:
 mov x23,#0
// One pass per entry; x23 counts 0..train_sequences-1. The id must be in
// range and not yet seen in this row.
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
 // Mode 493 is octal 0755.
 LD x0,Routput
 mov w1,#493
 bl _mkdir
 cbnz w0,Rfail
 // Zero-filled state: workspace, gradient accumulator, Adam m and v, the
 // best-checkpoint copy and the history table, [epochs][3] doubles.
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
 // The best validation loss starts at +infinity.
 ADDR x9,Linf
 ldr d0,[x9]
 ST d0,Rbestloss
 // Start of the timed training scope: CPU-time origin, peak RSS before
 // training (performance[2], peakRSSbefore in ABI.md) and the wall-clock
 // origin.
 bl Rusage
 ST d0,Rcpustart
 ucvtf d1,x0
 ADDR x9,Rperformance
 str d1,[x9,#16]
 bl Rnow
 ST d0,Rtrainstart
 mov x19,#0
// One pass per epoch; x19 = 0-based epoch index, 0..epochs-1.
Repoch:
 // d8 = epoch start. Train on row x19 of the schedule; d9 = training loss.
 bl Rnow
 fmov d8,d0
 CFG w9,16
 mul x10,x19,x9
 LD x0,Rschedule
 add x0,x0,x10,lsl #2
 bl Rtrain_epoch
 fmov d9,d0
 // d10 = validation loss after the epoch's updates; no logits.
 LD x0,Rvalidation
 CFG w1,20
 mov x2,#0
 bl Revaluate
 fmov d10,d0
 // A new best needs a strictly lower loss. Then record the 1-based epoch,
 // keep a copy in Rbest and rewrite best.f32, all inside the timed scope.
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
// History row x19: training loss, validation loss, epoch wall seconds.
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
 // End of the timed scope, with the ABI.md field names: performance[0] =
 // trainWall (wall seconds), [1] = trainCPU (CPU seconds), [3] = peakRSSafter.
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
 // Final parameters, Adam moments and the history table.
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
 // The test split is evaluated with the selected checkpoint: copy Rbest over
 // Rparams, then write test_logits.f32 sequence by sequence. The loss that
 // Revaluate returns is not stored.
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
// 20 warm-up forward passes on the first test sequence, untimed.
Rwarm:
 LD x0,Rparams
 LD x1,Rtest
 LD x2,Rworkspace
 bl _tf_forward
 add x19,x19,#1
 cmp x19,#20
 b.lo Rwarm
 mov x19,#0
// 100 timed forward passes on the same sequence. Each interval brackets one
// tf_forward call; Rtimings[i] is its duration in seconds.
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
 // timings.f64: 100 doubles.
 ADDR x0,Ltimings
 ADDR x1,Rtimings
 mov x2,#800
 bl Rwrite
 // performance[4] = peakRSSfinal, [5] = bestEpoch (1-based),
 // [6] = optimizerSteps, [7] = totalNativeWall, the wall seconds since Rstart.
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
 // backend.u64: the ten-word backend identity and counters.
 ADDR x0,Rbackend
 bl _tf_backend_info
 ADDR x0,Lbackend
 ADDR x1,Rbackend
 mov x2,#80
 bl Rwrite
 mov w0,#0
 LEAVE

// The failure message, the path separator, fopen modes and file basenames.
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
Lbackend: .asciz "backend.u64"
// Microseconds and nanoseconds per second, and +infinity as raw bits.
.section __TEXT,__const
.p2align 3
Lmillion: .double 1000000
Lbillion: .double 1000000000
Linf: .quad 0x7ff0000000000000
// Process state, zero-filled at load. One 8-byte slot each unless noted.
//   Rinput, Routput        the two directory arguments
//   Rconfig, Rparams, Rtrain, Rvalidation, Rtest, Rschedule
//                          buffers returned by Rread
//   Rworkspace, Rgrads, Rm, Rv, Rbest, Rhistory
//                          buffers returned by Ralloc
//   Rstep                  optimizer steps taken so far
//   Rbestepoch, Rbestloss  selected epoch (1-based), its validation loss
//   Rtimebase              mach_timebase_info: numer, denom (two uint32)
//   Rtick                  seconds per mach_absolute_time tick
//   Rstart, Rcpustart, Rtrainstart   time origins
//   Rusagebuf              256 bytes for struct rusage
//   Rperformance           8 doubles; Rbackend 10 words; Rtimings 100 doubles
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
.zerofill __DATA,__bss,Rbackend,80,3
.zerofill __DATA,__bss,Rtimings,800,3
