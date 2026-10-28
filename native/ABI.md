# Native contract

Apple arm64 Darwin ABI, float32 arrays, row-major tensors; symbols have the Darwin
leading underscore in assembly. All integer arguments/pointers are 64-bit unless
otherwise noted. Preserve x19-x28 and the low 64 bits of v8-v15. Never use x18.
Shape arrays are int32. All tensors supplied by the caller; no hidden state.
The model/layout is transformer/spec.py; the build-specific layout.inc contains
byte offsets. The supported width ladder is 32, 64, 128, 256, with context 32,
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
- `tf_attention_backward(qkv,probs,dout,dqkv,shape)`; overwrite dqkv, causal.
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

V2 preserves these public signatures. Specialized tiled/vector paths dispatch to
the original generic implementations when their shape requirements are not met;
see `kernel_notes.md`. The internal `tf_exp4` helper has a private vector-register
ABI and is not part of this C-callable interface.

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
`timings.f64` [100] warmed full-context single-sequence forward times (seconds),
20 warmups. Timer excludes cross entropy/softmax, I/O and input token loading.
Training timer includes epochs, validation, best checkpoint writes and history
bookkeeping; excludes input setup, final test, last checkpoint exports and timings.
Parent runner independently records whole child process wall time.
