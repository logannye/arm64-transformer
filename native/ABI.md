# Native contract

Apple arm64 Darwin ABI, float32 arrays, row-major tensors; symbols have the Darwin
leading underscore in assembly. All integer arguments/pointers are 64-bit unless
otherwise noted. Preserve x19-x28 and the low 64 bits of v8-v15. Never use x18.
Shape arrays are int32. Model tensors and outputs are supplied by the caller.
Kernels can use private stack scratch. The backend has an immutable selection
policy after initialization and mutable diagnostic call counters; model parameters,
gradients, optimizer state, and numerical scratch are never shared globals.
The model/layout is transformer/spec.py; the build-specific layout.inc contains
byte offsets. The supported width ladder is 32, 64, 128, 256, 512, 1024, with context 32,
four heads, feed-forward width twice model width, and vocabulary 128. Each width
has its own compiled executable/library and matching binary fixtures.

## Numerical primitives

- `tf_linear_forward(x,w,b,y,shape)`; shape=[rows,input,output], w=[output,input].
- `tf_linear_backward(x,w,dy,dx,dw,db,shape)`; overwrite dx, ADD to dw/db.
- `tf_ln_forward(x,gamma,beta,y,norm,inv,rows,width)`; epsilon=1e-5, biased variance.
- `tf_ln_backward(dy,norm,inv,gamma,dx,dgamma,dbeta,shape)`; shape=[rows,width],
  overwrite dx; ADD to dgamma/dbeta.
- `tf_attention_forward(qkv,out,probs,shape)`; shape=[tokens,width,heads], qkv is
  [tokens,3*width] with consecutive Q,K,V segments. out=[tokens,width],
  probs=[heads,tokens,tokens]. Causal mask includes diagonal; masked probs ZERO.
  Score scale = 1/sqrt(width/heads). Stable max-subtracted softmax.
  For width 512 or 1024, four heads, and 1..32 tokens, Q/K operands are converted
  from their stored float32 values before FP64 multiply-accumulation. Scores,
  scaling, maximum selection, and maximum subtraction remain FP64; only centered
  scores are rounded to float32. Probabilities and weighted values remain float32.
  This path uses a 400-byte frame including one 32-double private score row.
  Other shapes preserve their existing float32 score arithmetic.
- `tf_attention_backward(qkv,probs,dout,dqkv,shape)`; overwrite dqkv, causal.
  Uses the supplied forward probabilities and the same real-valued attention
  derivative. For the optimized width-512/1024 shapes, dA dot products, the
  softmax row contraction, scaled dScore, and the complete causal dQ sum remain
  FP64. Each dK contribution is added in FP64 before rounding to its float32
  destination; dV retains its float32 accumulation order. The frame is 384 bytes,
  including one 32-double row reused for dA and dScore. Stored probabilities
  are used unchanged, with no renormalization, clamping, or model alteration.
  Other shapes retain their existing derivative arithmetic. Outputs are float32.
  As with standard floating-point autodifferentiation, rounding operations are
  not differentiated. No extra derivative of the subtracted maximum is needed:
  the softmax Jacobian annihilates a constant score shift.
- `tf_relu_forward(x,y,count)` and `tf_relu_backward(x,dy,dx,count)`; derivative
  at zero is zero, overwrite destinations.
- `tf_cross_entropy(logits,targets,dlogits,rows,vocab)` returns float32 mean loss
  in s0, overwrites dlogits with gradient of mean token loss. int32 targets.
- `tf_adam(params,grads,m,v,step,count)`; step>=1, lr=.001, betas=.9/.999,
  eps=1e-8, no decay, float32 state, double scalar bias correction as PyTorch.
- `tf_exp(float)` and `tf_log(float)` return float32. Handwritten math.

All widths used by the full model are multiples of four; primitive kernels must
document smaller-shape restrictions. Test inputs to kernels are already validated.
Native executable validates file sizes, indices, finiteness, and run bounds.

The optimized backends preserve these public signatures. SME, NEON tiles, and
generic kernels are selected according to hardware, execution state, and shape;
see `kernel_notes.md`. The internal `tf_exp4` helper has a private vector-register
ABI. `tf_linear_pack32_sme` is named for profiling but has a private streaming ABI.
Neither helper is a public C-callable interface. The `_neon`, `_sme`, and `_generic`
linear entry points are implementation details; callers use the public dispatchers.

## Backend policy and execution state

Public functions use the ordinary **non-streaming ABI: PSTATE.SM must be zero on
entry**. This is not a streaming-compatible API. SME kernels save d8-d15 before
entering streaming mode and restore them after leaving it; public calls return
with SM zero. The dispatcher selects SME only when ZA is off and TPIDR2_EL0 is
null. With SM zero, active ZA or a pending lazy-save block routes to NEON while
leaving that state untouched. Callers with live SME state must still obey their
platform's ZA calling convention.

`ARM64_TRANSFORMER_BACKEND` is captured once at library/executable initialization:

| Value | Policy |
| --- | --- |
| unset or `auto` | Select SME when the required capability is present; otherwise NEON. |
| `neon` | Use NEON, including its generic shape fallbacks. |
| `sme` | Require SME capability; unsupported shapes or live ZA state still use NEON. |

Any other value, or forced `sme` without the required capability, fails
initialization. A library constructor performs initialization, and FFI callers
must also call `tf_backend_init()` and require a zero result before numerical
calls. It is idempotent and returns signed int64: zero on success, -1 on failure.
The executable rejects initialization failure. A fresh process is required to
choose a different policy; changing the environment after initialization has no
effect on that loaded library.

Detection queries Darwin `hw.optional.arm.FEAT_SME` and
`hw.optional.arm.FEAT_SME2` before executing any SME instruction, including RDSVL.
The current implementation also requires a streaming vector length of **64 bytes**,
or 16 float32 lanes. Before dispatch, it checks the current vector length and
execution state again. SME linear shapes are exactly `rows=32`,
`input in {64,128,256,512,1024,2048}`, and `output` a multiple of 32 in `[64,3072]`.
All five projections in full models of width 64, 128, 256, 512, and 1024 satisfy this rule;
the width-32 model uses NEON. Other positive shapes retain the NEON/generic path.
Callers must provide enough thread stack for the selected shape: forward can use
512 KiB of scratch plus its 160-byte frame, and backward can use 384 KiB plus its
192-byte frame, in addition to the caller's live stack. The standalone runtime
uses the process's main thread. A thread with only a 512 KiB stack is insufficient
for the largest forward shape; custom FFI threads must provision more space or
select the NEON policy before loading the library.
Compiled widths 512 and 1024 use four interleaved FP32 forward accumulators per
output, combined in NEON's pairwise order; smaller compiled models retain their
previous SME reduction order. This changes no public signature, tensor layout,
scratch bound, or backward accumulation contract. See `kernel_notes.md` for the
accuracy and operand-load tradeoff.

`tf_backend_info(out)` writes ten uint64 words (80 bytes), with word 1 interpreted
as signed int64:

`[version=1, init_status, requested_policy, selected_kind, sme_supported,
streaming_vector_bytes, sme_forward_calls, sme_backward_calls,
neon_forward_calls, neon_backward_calls]`

Policies encode `auto=0`, `neon=1`, `sme=2`; selected kind encodes `NEON=0`,
`SME=1`. The capability flag means the checks above succeeded, not merely that
one architectural feature exists. Counters increment atomically on actual public
linear dispatches, never reset, and do not affect arithmetic. A snapshot reads
the counters individually; it is not a transactional snapshot of concurrent
calls. The study uses one compute thread. `tf_backend_abi_check()` is a diagnostic
helper that returns 1 on success and 0 on failure; it allocates temporary buffers
and calls linear forward/backward, so it also increments the counters.

## Complete model

- `tf_model_width()` returns the compiled width as uint64.
- `tf_parameter_count()` returns the compiled parameter count as uint64.
  FFI callers must check both before passing model arrays to the library.
- `tf_forward(params,tokens,workspace)`; one sequence of 32 int32 tokens;
  writes intermediate tensors and logits at W_LOGITS.
- `tf_backward(params,tokens,targets,workspace,grads)`; forward already complete;
  adds parameter gradients of mean token loss, returns loss in s0.
- `tf_scale(x,count,float_scale)`; scale in s0, count in x1, modifies x.
- `tf_finite(x,count)` returns 1 iff all values are finite, else 0.

## Binary fixture/runtime

Executable: `native/build/width-32/transformer-arm64 INPUT_DIRECTORY OUTPUT_DIRECTORY`
for width 32; other widths use the corresponding `width-W` build directory.
Output directory must not exist. Input contains `config.bin` (9 little-endian
uint32, exactly 36 bytes: magic=0x54465231, version=2, epochs, batch,
train_sequences, validation_sequences, test_sequences, seed, width); limits
epochs 1..100, batch 1..128, train 1..8192, validation/test 1..2048. Width must
match the compiled binary and is checked before loading model/data arrays or
performing numerical work. The new runtime accepts version 2 only; preserved
version 1 release binaries remain available for their original 32-byte fixtures.
`initial.f32` has PARAM_COUNT finite floats. `train.i32`, `validation.i32`,
`test.i32` have [sequence_count,33] int32 ASCII IDs in 0..127. First 32 are
inputs, last 32 are next-character targets. `schedule.i32` contains
[epochs,train_sequences] permutations of 0..train_sequences-1.

Each step accumulates mean-sequence gradients, divides by ACTUAL sequence count,
then updates once with Adam. Exactly one CPU compute thread. Final partial batch
is retained. All epoch validation runs are in the timed training scope. Select
lowest validation mean NLL, first epoch wins exact ties. Test is evaluated only
after selection. Save best/last float32 weights and last Adam m/v states.

Outputs: `history.f64` [epochs,3] trainNLL,validationNLL,epochWallSeconds;
`performance.f64` 8 doubles: trainWall,trainCPU,peakRSSbefore,peakRSSafter,
peakRSSfinal,bestEpoch(1based),optimizerSteps,totalNativeWall;
`best.f32`, `last.f32`, `adam_m.f32`, `adam_v.f32`;
`test_logits.f32` [test_sequences,32,128] from selected checkpoint;
`backend.u64` the 80-byte backend identity/counter snapshot described above;
`timings.f64` [100] warmed full-context single-sequence forward times (seconds),
20 warmups. Timer excludes cross entropy/softmax, I/O and input token loading.
Training timer includes epochs, validation, best checkpoint writes and history
bookkeeping; excludes input setup, final test, last checkpoint exports and timings.
Parent runner independently records whole child process wall time.
SME packing, stack-page touching, mode transitions, and dispatch bookkeeping are
inside the relevant numerical calls and therefore inside these measured scopes.
