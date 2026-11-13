// Complete handwritten one-block transformer: embeddings, forward pass and
// backward pass for one sequence of CONTEXT tokens.
//
// Exported routines:
//   tf_model_width      compiled WIDTH, for the caller's layout check
//   tf_parameter_count  compiled PARAM_COUNT, for the same check
//   tf_forward          tokens -> logits, every intermediate kept in workspace
//   tf_backward         loss and parameter gradients, after tf_forward
//   tf_scale            multiply a float32 array by a scalar in place
//   tf_finite           1 if a float32 array holds no infinity or NaN
// Local helper: Ladd, elementwise float32 addition.
//
// tf_forward and tf_backward sequence calls into the kernels defined in the
// other files (tf_ln_*, tf_linear_*, tf_attention_*, tf_relu_*,
// tf_cross_entropy). The only arithmetic written in this file is the
// embedding sum, its gradient scatter, Ladd, tf_scale and tf_finite. The
// optimizer step is not here: runtime.S calls tf_adam (kernels.S) once per
// batch.
//
// Conventions:
//   - params and grads are flat float32 arrays that share one layout; the P_*
//     constants are byte offsets into either.
//   - workspace is one caller-owned buffer of WORK_BYTES bytes holding one
//     sequence; the W_* constants are byte offsets of its tensors.
//   - P_*, W_* and the sizes come from the build's layout.inc, generated from
//     transformer/spec.py, so each compiled width has its own constants.
//   - Tensors are row-major float32. In the [CONTEXT, ...] workspace tensors
//     and in P_POSITION, row t belongs to token position t.
#include <layout.inc>
#include <precision.h>
.text
.p2align 2
// ADDR reg, symbol: reg = address of a symbol in this image.
.macro ADDR reg, symbol
 adrp \reg, \symbol@PAGE
 add \reg, \reg, \symbol@PAGEOFF
.endm
// PTR reg, base, offset: reg = base + offset for an assemble-time constant
// byte offset (a P_* or W_* value). The offset is split into a multiple of
// 4096 and a remainder below 4096. Offsets of 16 MiB or more need several
// ADDs for the multiple; in the generated layouts only the width-1024
// parameter offsets from P_ATTENTION_BIAS upward are that large.
.macro PTR reg, base, offset
 // A shifted ADD has only 12 immediate bits. Keep the original two ADDs
 // for smaller layouts; wider parameter arrays can cross the 16 MiB boundary.
 .if (\offset >> 12) > 4095
 // Add the page count 4095 pages at a time, then the pages that remain.
 add \reg, \base, #4095, lsl #12
 .rept ((\offset >> 12) / 4095) - 1
 add \reg, \reg, #4095, lsl #12
 .endr
 add \reg, \reg, #((\offset >> 12) % 4095), lsl #12
 .else
 add \reg, \base, #(\offset >> 12), lsl #12
 .endif
 add \reg, \reg, #(\offset & 4095)
.endm
// IMM reg, value: load a constant below 2^32 with MOV plus, when its upper
// 16 bits are nonzero, one MOVK.
.macro IMM reg, value
 mov \reg,#(\value & 65535)
 .if (\value >> 16)
 movk \reg,#((\value >> 16) & 65535),lsl #16
 .endif
.endm
// ENTER / LEAVE: the 96-byte frame shared by tf_forward and tf_backward.
//   [sp,#0]  x29, x30    [sp,#16] x19, x20    [sp,#32] x21, x22
//   [sp,#48] x23, x24    [sp,#64] x25, x26    [sp,#80] d8, d9
// LEAVE restores the same registers and returns to the caller.
.macro ENTER
 stp x29,x30,[sp,#-96]!
 mov x29,sp
 stp x19,x20,[sp,#16]
 stp x21,x22,[sp,#32]
 stp x23,x24,[sp,#48]
 stp x25,x26,[sp,#64]
 stp d8,d9,[sp,#80]
.endm
.macro LEAVE
 ldp d8,d9,[sp,#80]
 ldp x25,x26,[sp,#64]
 ldp x23,x24,[sp,#48]
 ldp x21,x22,[sp,#32]
 ldp x19,x20,[sp,#16]
 ldp x29,x30,[sp],#96
 ret
.endm

// Query the compiled layout before passing any caller-owned arrays.
// tf_model_width() returns the compiled WIDTH as uint64 in x0. Leaf.
.globl _tf_model_width
_tf_model_width:
 mov x0,#WIDTH
 ret

// tf_parameter_count() returns the compiled PARAM_COUNT as uint64 in x0. Leaf.
.globl _tf_parameter_count
_tf_parameter_count:
 IMM x0,PARAM_COUNT
 ret

// tf_forward(params, tokens, workspace)
//   x0 = params     float32[PARAM_COUNT], read only
//   x1 = tokens     int32[CONTEXT], each in 0..VOCAB-1 (not checked here)
//   x2 = workspace  WORK_BYTES bytes, written
// Runs the model on one full-context sequence and leaves every intermediate
// tensor in the workspace for tf_backward. No return value. Builds the ENTER
// frame; x19 = params, x20 = tokens, x21 = workspace throughout.
//
// Stages in order (inputs -> outputs; P_* in params, W_* in workspace):
//   embedding    P_TOKEN, P_POSITION        -> W_X
//   LayerNorm 1  W_X, P_LN1_*               -> W_N1, W_NORM1, W_INV1
//   QKV linear   W_N1, P_QKV_*              -> W_QKV
//   attention    W_QKV                      -> W_ATTENTION, W_PROBS
//   projection   W_ATTENTION, P_ATTENTION_* -> W_PROJECTED
//   residual 1   W_X + W_PROJECTED          -> W_R1
//   LayerNorm 2  W_R1, P_LN2_*              -> W_N2, W_NORM2, W_INV2
//   FF1 linear   W_N2, P_FF1_*              -> W_FF1
//   ReLU         W_FF1                      -> W_RELU
//   FF2 linear   W_RELU, P_FF2_*            -> W_FF2
//   residual 2   W_R1 + W_FF2               -> W_R2
//   LayerNorm 3  W_R2, P_LN3_*              -> W_N3, W_NORM3, W_INV3
//   head linear  W_N3, P_HEAD_*             -> W_LOGITS
// Shapes: W_QKV is [CONTEXT, 3*WIDTH]; W_FF1 and W_RELU are
// [CONTEXT, HIDDEN]; W_LOGITS is [CONTEXT, VOCAB]; W_PROBS is
// [HEADS, CONTEXT, CONTEXT]; W_INV1..3 are [CONTEXT]; every other W_* tensor
// named above is [CONTEXT, WIDTH]. The P_* shapes are in transformer/spec.py.
//
// Precision policy: in builds with WIDTH >= 512, each bit set in
// TF_PRECISION_POLICY (precision.h) replaces one call by its scalar FP64
// variant from precision.S: 1 = LayerNorm 1, 2 = QKV linear,
// 4 = LayerNorm 2, 8 = FF1 linear. Builds with WIDTH < 512 always assemble
// the plain calls, whatever the policy value.
.globl _tf_forward
_tf_forward:
 ENTER
 mov x19,x0
 mov x20,x1
 mov x21,x2
 // Embedding: W_X[t][c] = P_TOKEN[tokens[t]][c] + P_POSITION[t][c].
 // x9 walks the position table and x10 walks W_X, both row after row;
 // x12 is the token cursor and w11 counts the CONTEXT positions left.
 PTR x9,x19,P_POSITION
 PTR x10,x21,W_X
 mov w11,#CONTEXT
 mov x12,x20
// One pass per token position t (CONTEXT passes).
Lembed:
 // x13 = address of token-table row tokens[t]. A row is WIDTH*4 bytes, so the
 // id is shifted left by WIDTH_SHIFT+2; P_TOKEN is offset 0 of params.
 ldr w13,[x12],#4
 add x13,x19,x13,lsl #(WIDTH_SHIFT+2)
 mov w14,#(WIDTH/4)
// One pass per four channels (WIDTH/4 passes).
Lembed_vec:
 ldr q0,[x13],#16
 ldr q1,[x9],#16
 fadd v0.4s,v0.4s,v1.4s
 str q0,[x10],#16
 subs w14,w14,#1
 b.ne Lembed_vec
 subs w11,w11,#1
 b.ne Lembed
 // LayerNorm 1: tf_ln_forward(x=W_X, gamma=P_LN1_WEIGHT, beta=P_LN1_BIAS,
 // y=W_N1, norm=W_NORM1, inv=W_INV1, rows=CONTEXT, width=WIDTH). W_NORM1 and
 // W_INV1 keep the normalized input and the inverse standard deviation of
 // each row for the backward pass.
 PTR x0,x21,W_X
 PTR x1,x19,P_LN1_WEIGHT
 PTR x2,x19,P_LN1_BIAS
 PTR x3,x21,W_N1
 PTR x4,x21,W_NORM1
 PTR x5,x21,W_INV1
 mov x6,#CONTEXT
 mov x7,#WIDTH
 // Policy bit of value 1, WIDTH >= 512 builds only: scalar FP64 LayerNorm.
 .if WIDTH >= 512 && (TF_PRECISION_POLICY & 1)
 bl _tf_ln_forward_precise
 .else
 bl _tf_ln_forward
 .endif
 // QKV projection: tf_linear_forward(x=W_N1, w=P_QKV_WEIGHT, b=P_QKV_BIAS,
 // y=W_QKV, shape=[CONTEXT, WIDTH, 3*WIDTH]). Each W_QKV row holds the Q, K
 // and V segments of one token, WIDTH floats each.
 PTR x0,x21,W_N1
 PTR x1,x19,P_QKV_WEIGHT
 PTR x2,x19,P_QKV_BIAS
 PTR x3,x21,W_QKV
 ADDR x4,Lqkvshape
 // Policy bit of value 2, WIDTH >= 512 builds only: scalar FP64 linear.
 .if WIDTH >= 512 && (TF_PRECISION_POLICY & 2)
 bl _tf_linear_forward_precise
 .else
 bl _tf_linear_forward
 .endif
 // Causal self-attention: tf_attention_forward(qkv=W_QKV, out=W_ATTENTION,
 // probs=W_PROBS, shape=[CONTEXT, WIDTH, HEADS]). W_PROBS keeps the attention
 // probabilities for the backward pass.
 PTR x0,x21,W_QKV
 PTR x1,x21,W_ATTENTION
 PTR x2,x21,W_PROBS
 ADDR x3,Lattshape
 bl _tf_attention_forward
 // Output projection: tf_linear_forward(x=W_ATTENTION, w=P_ATTENTION_WEIGHT,
 // b=P_ATTENTION_BIAS, y=W_PROJECTED, shape=[CONTEXT, WIDTH, WIDTH]).
 PTR x0,x21,W_ATTENTION
 PTR x1,x19,P_ATTENTION_WEIGHT
 PTR x2,x19,P_ATTENTION_BIAS
 PTR x3,x21,W_PROJECTED
 ADDR x4,Lprojshape
 bl _tf_linear_forward
 // Residual 1: W_R1 = W_X + W_PROJECTED.
 PTR x0,x21,W_X
 PTR x1,x21,W_PROJECTED
 PTR x2,x21,W_R1
 mov x3,#(CONTEXT*WIDTH)
 bl Ladd
 // LayerNorm 2: tf_ln_forward(x=W_R1, gamma=P_LN2_WEIGHT, beta=P_LN2_BIAS,
 // y=W_N2, norm=W_NORM2, inv=W_INV2, rows=CONTEXT, width=WIDTH).
 PTR x0,x21,W_R1
 PTR x1,x19,P_LN2_WEIGHT
 PTR x2,x19,P_LN2_BIAS
 PTR x3,x21,W_N2
 PTR x4,x21,W_NORM2
 PTR x5,x21,W_INV2
 mov x6,#CONTEXT
 mov x7,#WIDTH
 // Policy bit of value 4, WIDTH >= 512 builds only: scalar FP64 LayerNorm.
 .if WIDTH >= 512 && (TF_PRECISION_POLICY & 4)
 bl _tf_ln_forward_precise
 .else
 bl _tf_ln_forward
 .endif
 // Feed-forward expansion: tf_linear_forward(x=W_N2, w=P_FF1_WEIGHT,
 // b=P_FF1_BIAS, y=W_FF1, shape=[CONTEXT, WIDTH, HIDDEN]).
 PTR x0,x21,W_N2
 PTR x1,x19,P_FF1_WEIGHT
 PTR x2,x19,P_FF1_BIAS
 PTR x3,x21,W_FF1
 ADDR x4,Lff1shape
 // Policy bit of value 8, WIDTH >= 512 builds only: scalar FP64 linear.
 .if WIDTH >= 512 && (TF_PRECISION_POLICY & 8)
 bl _tf_linear_forward_precise
 .else
 bl _tf_linear_forward
 .endif
 // ReLU: tf_relu_forward(x=W_FF1, y=W_RELU, count=CONTEXT*HIDDEN). W_FF1 is
 // kept as well; tf_backward passes it to tf_relu_backward.
 PTR x0,x21,W_FF1
 PTR x1,x21,W_RELU
 mov x2,#(CONTEXT*HIDDEN)
 bl _tf_relu_forward
 // Feed-forward contraction: tf_linear_forward(x=W_RELU, w=P_FF2_WEIGHT,
 // b=P_FF2_BIAS, y=W_FF2, shape=[CONTEXT, HIDDEN, WIDTH]).
 PTR x0,x21,W_RELU
 PTR x1,x19,P_FF2_WEIGHT
 PTR x2,x19,P_FF2_BIAS
 PTR x3,x21,W_FF2
 ADDR x4,Lff2shape
 bl _tf_linear_forward
 // Residual 2: W_R2 = W_R1 + W_FF2.
 PTR x0,x21,W_R1
 PTR x1,x21,W_FF2
 PTR x2,x21,W_R2
 mov x3,#(CONTEXT*WIDTH)
 bl Ladd
 // LayerNorm 3: tf_ln_forward(x=W_R2, gamma=P_LN3_WEIGHT, beta=P_LN3_BIAS,
 // y=W_N3, norm=W_NORM3, inv=W_INV3, rows=CONTEXT, width=WIDTH). No policy
 // bit selects a precise variant for this stage.
 PTR x0,x21,W_R2
 PTR x1,x19,P_LN3_WEIGHT
 PTR x2,x19,P_LN3_BIAS
 PTR x3,x21,W_N3
 PTR x4,x21,W_NORM3
 PTR x5,x21,W_INV3
 mov x6,#CONTEXT
 mov x7,#WIDTH
 bl _tf_ln_forward
 // Output head: tf_linear_forward(x=W_N3, w=P_HEAD_WEIGHT, b=P_HEAD_BIAS,
 // y=W_LOGITS, shape=[CONTEXT, WIDTH, VOCAB]).
 PTR x0,x21,W_N3
 PTR x1,x19,P_HEAD_WEIGHT
 PTR x2,x19,P_HEAD_BIAS
 PTR x3,x21,W_LOGITS
 ADDR x4,Lheadshape
 bl _tf_linear_forward
 LEAVE

// tf_backward(params, tokens, targets, workspace, grads)
//   x0 = params     float32[PARAM_COUNT], read only
//   x1 = tokens     int32[CONTEXT], the tokens given to tf_forward
//   x2 = targets    int32[CONTEXT], the target token id at each position
//   x3 = workspace  as left by tf_forward for the same params and tokens
//   x4 = grads      float32[PARAM_COUNT] in the params layout
// Returns the mean token loss as float32 in s0 and ADDS the gradient of that
// loss to grads. grads is never cleared here; runtime.S zeroes it once per
// batch. Builds the ENTER frame; x19 = params, x20 = tokens,
// x21 = workspace, x22 = grads, s8 = the loss across the kernel calls.
//
// The kernel calls run in the reverse of the tf_forward order. The two
// residual sums are not at the mirrored positions: each follows the backward
// step of the LayerNorm that opens its sublayer, when both addends exist.
// Each stage reads the gradient of its output together with values the
// forward pass saved and overwrites the gradient of its input; the stages
// with parameters also add to their parameter gradients:
//   cross-entropy  W_LOGITS, targets            -> W_DLOGITS, loss
//   head linear    W_DLOGITS, W_N3              -> W_DN3, grads P_HEAD_*
//   LayerNorm 3    W_DN3, W_NORM3, W_INV3       -> W_DR2, grads P_LN3_*
//   FF2 linear     W_DR2, W_RELU                -> W_DRELU, grads P_FF2_*
//   ReLU           W_DRELU, W_FF1               -> W_DFF1
//   FF1 linear     W_DFF1, W_N2                 -> W_DN2, grads P_FF1_*
//   LayerNorm 2    W_DN2, W_NORM2, W_INV2       -> W_TEMP, grads P_LN2_*
//   residual 2     W_DR2 + W_TEMP               -> W_DR1
//   projection     W_DR1, W_ATTENTION           -> W_DATTENTION,
//                                                  grads P_ATTENTION_*
//   attention      W_DATTENTION, W_QKV, W_PROBS -> W_DQKV
//   QKV linear     W_DQKV, W_N1                 -> W_DN1, grads P_QKV_*
//   LayerNorm 1    W_DN1, W_NORM1, W_INV1       -> W_DX, grads P_LN1_*
//   residual 1     W_DX + W_DR1                 -> W_DX
//   embedding      W_DX                         -> grads P_TOKEN, P_POSITION
// A residual addition in the forward pass passes its output gradient to both
// of its inputs. Each "residual" line above therefore sums the two gradients
// that reach one tensor: the skip path and the path through the sublayer.
// The four policy bits select the same stages as in tf_forward.
.globl _tf_backward
_tf_backward:
 ENTER
 mov x19,x0
 mov x20,x1
 mov x21,x3
 mov x22,x4
 // Loss: s0 = tf_cross_entropy(logits=W_LOGITS, targets, dlogits=W_DLOGITS,
 // rows=CONTEXT, vocab=VOCAB). targets moves from x2 to x1 before x2 is
 // reused. W_DLOGITS receives the gradient of the mean token loss. The loss
 // is parked in s8, which ENTER saved, until the return.
 mov x1,x2
 PTR x0,x21,W_LOGITS
 PTR x2,x21,W_DLOGITS
 mov x3,#CONTEXT
 mov x4,#VOCAB
 bl _tf_cross_entropy
 fmov s8,s0
 // Head: tf_linear_backward(x=W_N3, w=P_HEAD_WEIGHT, dy=W_DLOGITS, dx=W_DN3,
 // dw=grads+P_HEAD_WEIGHT, db=grads+P_HEAD_BIAS, shape as in the forward
 // call). Here and below, weights come from params (x19) and the dw/db
 // destinations from grads (x22) at the same P_* offsets.
 PTR x0,x21,W_N3
 PTR x1,x19,P_HEAD_WEIGHT
 PTR x2,x21,W_DLOGITS
 PTR x3,x21,W_DN3
 PTR x4,x22,P_HEAD_WEIGHT
 PTR x5,x22,P_HEAD_BIAS
 ADDR x6,Lheadshape
 bl _tf_linear_backward
 // LayerNorm 3: tf_ln_backward(dy=W_DN3, norm=W_NORM3, inv=W_INV3,
 // gamma=P_LN3_WEIGHT, dx=W_DR2, dgamma and dbeta in grads,
 // shape=[CONTEXT, WIDTH]). W_DR2 is the gradient of W_R2.
 PTR x0,x21,W_DN3
 PTR x1,x21,W_NORM3
 PTR x2,x21,W_INV3
 PTR x3,x19,P_LN3_WEIGHT
 PTR x4,x21,W_DR2
 PTR x5,x22,P_LN3_WEIGHT
 PTR x6,x22,P_LN3_BIAS
 ADDR x7,Lnormshape
 bl _tf_ln_backward
 // FF2: tf_linear_backward(x=W_RELU, w=P_FF2_WEIGHT, dy=W_DR2, dx=W_DRELU,
 // dw and db in grads). W_DR2 is read again below for the residual sum.
 PTR x0,x21,W_RELU
 PTR x1,x19,P_FF2_WEIGHT
 PTR x2,x21,W_DR2
 PTR x3,x21,W_DRELU
 PTR x4,x22,P_FF2_WEIGHT
 PTR x5,x22,P_FF2_BIAS
 ADDR x6,Lff2shape
 bl _tf_linear_backward
 // ReLU: tf_relu_backward(x=W_FF1, dy=W_DRELU, dx=W_DFF1,
 // count=CONTEXT*HIDDEN); x is the forward input of the ReLU.
 PTR x0,x21,W_FF1
 PTR x1,x21,W_DRELU
 PTR x2,x21,W_DFF1
 mov x3,#(CONTEXT*HIDDEN)
 bl _tf_relu_backward
 // FF1: tf_linear_backward(x=W_N2, w=P_FF1_WEIGHT, dy=W_DFF1, dx=W_DN2,
 // dw and db in grads).
 PTR x0,x21,W_N2
 PTR x1,x19,P_FF1_WEIGHT
 PTR x2,x21,W_DFF1
 PTR x3,x21,W_DN2
 PTR x4,x22,P_FF1_WEIGHT
 PTR x5,x22,P_FF1_BIAS
 ADDR x6,Lff1shape
 // Policy bit of value 8, WIDTH >= 512 builds only: scalar FP64 variant.
 .if WIDTH >= 512 && (TF_PRECISION_POLICY & 8)
 bl _tf_linear_backward_precise
 .else
 bl _tf_linear_backward
 .endif
 // LayerNorm 2: tf_ln_backward(dy=W_DN2, norm=W_NORM2, inv=W_INV2,
 // gamma=P_LN2_WEIGHT, dx=W_TEMP, dgamma and dbeta in grads). W_TEMP is the
 // part of the W_R1 gradient that arrives through the feed-forward branch.
 PTR x0,x21,W_DN2
 PTR x1,x21,W_NORM2
 PTR x2,x21,W_INV2
 PTR x3,x19,P_LN2_WEIGHT
 PTR x4,x21,W_TEMP
 PTR x5,x22,P_LN2_WEIGHT
 PTR x6,x22,P_LN2_BIAS
 ADDR x7,Lnormshape
 // Policy bit of value 4, WIDTH >= 512 builds only: scalar FP64 variant.
 .if WIDTH >= 512 && (TF_PRECISION_POLICY & 4)
 bl _tf_ln_backward_precise
 .else
 bl _tf_ln_backward
 .endif
 // Residual 2 (forward: W_R2 = W_R1 + W_FF2): W_DR1 = W_DR2 + W_TEMP, the
 // skip path plus the feed-forward branch.
 PTR x0,x21,W_DR2
 PTR x1,x21,W_TEMP
 PTR x2,x21,W_DR1
 mov x3,#(CONTEXT*WIDTH)
 bl Ladd
 // Projection: tf_linear_backward(x=W_ATTENTION, w=P_ATTENTION_WEIGHT,
 // dy=W_DR1, dx=W_DATTENTION, dw and db in grads). W_DR1 is also the gradient
 // of W_PROJECTED, because W_R1 = W_X + W_PROJECTED.
 PTR x0,x21,W_ATTENTION
 PTR x1,x19,P_ATTENTION_WEIGHT
 PTR x2,x21,W_DR1
 PTR x3,x21,W_DATTENTION
 PTR x4,x22,P_ATTENTION_WEIGHT
 PTR x5,x22,P_ATTENTION_BIAS
 ADDR x6,Lprojshape
 bl _tf_linear_backward
 // Attention: tf_attention_backward(qkv=W_QKV, probs=W_PROBS,
 // dout=W_DATTENTION, dqkv=W_DQKV, shape=[CONTEXT, WIDTH, HEADS]).
 PTR x0,x21,W_QKV
 PTR x1,x21,W_PROBS
 PTR x2,x21,W_DATTENTION
 PTR x3,x21,W_DQKV
 ADDR x4,Lattshape
 bl _tf_attention_backward
 // QKV: tf_linear_backward(x=W_N1, w=P_QKV_WEIGHT, dy=W_DQKV, dx=W_DN1,
 // dw and db in grads).
 PTR x0,x21,W_N1
 PTR x1,x19,P_QKV_WEIGHT
 PTR x2,x21,W_DQKV
 PTR x3,x21,W_DN1
 PTR x4,x22,P_QKV_WEIGHT
 PTR x5,x22,P_QKV_BIAS
 ADDR x6,Lqkvshape
 // Policy bit of value 2, WIDTH >= 512 builds only: scalar FP64 variant.
 .if WIDTH >= 512 && (TF_PRECISION_POLICY & 2)
 bl _tf_linear_backward_precise
 .else
 bl _tf_linear_backward
 .endif
 // LayerNorm 1: tf_ln_backward(dy=W_DN1, norm=W_NORM1, inv=W_INV1,
 // gamma=P_LN1_WEIGHT, dx=W_DX, dgamma and dbeta in grads). W_DX now holds
 // the part of the W_X gradient that arrives through the attention branch.
 PTR x0,x21,W_DN1
 PTR x1,x21,W_NORM1
 PTR x2,x21,W_INV1
 PTR x3,x19,P_LN1_WEIGHT
 PTR x4,x21,W_DX
 PTR x5,x22,P_LN1_WEIGHT
 PTR x6,x22,P_LN1_BIAS
 ADDR x7,Lnormshape
 // Policy bit of value 1, WIDTH >= 512 builds only: scalar FP64 variant.
 .if WIDTH >= 512 && (TF_PRECISION_POLICY & 1)
 bl _tf_ln_backward_precise
 .else
 bl _tf_ln_backward
 .endif
 // Residual 1 (forward: W_R1 = W_X + W_PROJECTED): W_DX = W_DX + W_DR1, the
 // attention branch plus the skip path, computed in place.
 PTR x0,x21,W_DX
 PTR x1,x21,W_DR1
 PTR x2,x21,W_DX
 mov x3,#(CONTEXT*WIDTH)
 bl Ladd
 // Embedding gradients. Forward row t of W_X is a token row plus a position
 // row, so row t of W_DX is added to both: token row tokens[t] and position
 // row t of grads. x9 walks the position rows of grads, x10 walks W_DX,
 // x12 is the token cursor and w11 counts the CONTEXT positions left.
 PTR x9,x22,P_POSITION
 PTR x10,x21,W_DX
 mov x12,x20
 mov w11,#CONTEXT
// One pass per token position t (CONTEXT passes).
Lscatter:
 // x13 = address of row tokens[t] in the token part of grads.
 ldr w13,[x12],#4
 add x13,x22,x13,lsl #(WIDTH_SHIFT+2)
 mov w14,#(WIDTH/4)
// One pass per four channels (WIDTH/4 passes). Every sum is stored before
// the next position is read, so a token id that occurs at several positions
// accumulates the gradient of each of them.
Lscatter_vec:
 ldr q0,[x10],#16
 ldr q1,[x13]
 ldr q2,[x9]
 fadd v1.4s,v1.4s,v0.4s
 fadd v2.4s,v2.4s,v0.4s
 str q1,[x13],#16
 str q2,[x9],#16
 subs w14,w14,#1
 b.ne Lscatter_vec
 subs w11,w11,#1
 b.ne Lscatter
 // Return the loss saved after tf_cross_entropy.
 fmov s0,s8
 LEAVE

// Ladd: output[i] = a[i] + b[i] for i in 0..count-1, float32.
//   x0 = a, x1 = b, x2 = output, x3 = count, a positive multiple of four
//   (the loop runs at least once and ends only when the count reaches 0).
// output may be the same buffer as a or b: each group of four is loaded
// before it is stored. Leaf; advances x0-x2, consumes x3, uses v0 and v1.
Ladd:
 ldr q0,[x0],#16
 ldr q1,[x1],#16
 fadd v0.4s,v0.4s,v1.4s
 str q0,[x2],#16
 subs x3,x3,#4
 b.ne Ladd
 ret

// tf_scale(x, count, float_scale)
//   x0 = x      float32[count], modified in place
//   x1 = count  zero or a multiple of four
//   s0 = scale
// x[i] = x[i] * scale. No return value. Leaf. The loop takes four elements
// per pass and ends only when the count reaches exactly zero, so a count
// that is not a multiple of four is not supported.
.globl _tf_scale
_tf_scale:
 // Broadcast the scale to all four lanes.
 dup v0.4s,v0.s[0]
 cbz x1,Lscale_done
// One pass per four elements (count/4 passes).
Lscale_loop:
 ldr q1,[x0]
 fmul v1.4s,v1.4s,v0.4s
 str q1,[x0],#16
 subs x1,x1,#4
 b.ne Lscale_loop
Lscale_done:
 ret

// tf_finite(x, count)
//   x0 = x      float32[count]
//   x1 = count  any value, including zero
// Returns 1 in w0 if every element is finite and 0 if any is an infinity or
// a NaN; an empty array is finite. Leaf. A float32 is non-finite exactly
// when its eight exponent bits (mask 0x7f800000) are all ones, so the test
// is done on the integer bit pattern: AND with the mask, compare with it.
.globl _tf_finite
_tf_finite:
 cbz x1,Lfinite_yes
 // v0 = the exponent mask in every lane, v2 = accumulated non-finite flags,
 // x11 = number of whole groups of four.
 mov w10,#0x7f800000
 dup v0.4s,w10
 movi v2.4s,#0
 lsr x11,x1,#2
 cbz x11,Lfinite_tail
// One pass per four elements. Flags are ORed into v2 and tested once after
// the loop, so the vector part has no early exit.
Lfinite_vector:
 ldr q1,[x0],#16
 and v1.16b,v1.16b,v0.16b
 cmeq v1.4s,v1.4s,v0.4s
 orr v2.16b,v2.16b,v1.16b
 subs x11,x11,#1
 b.ne Lfinite_vector
 // A lane flagged in any pass makes the maximum nonzero.
 umaxv s2,v2.4s
 fmov w9,s2
 cbnz w9,Lfinite_no
// The remaining count mod 4 elements, one at a time, same test.
Lfinite_tail:
 and x1,x1,#3
 cbz x1,Lfinite_yes
Lfinite_loop:
 ldr w9,[x0],#4
 and w9,w9,#0x7f800000
 cmp w9,w10
 b.eq Lfinite_no
 subs x1,x1,#1
 b.ne Lfinite_loop
Lfinite_yes:
 mov w0,#1
 ret
Lfinite_no:
 mov w0,#0
 ret

// Shape arrays handed to the kernels by address; int32 as ABI.md requires.
// Linear shapes are [rows, input, output]; Lattshape is [tokens, width,
// heads]; Lnormshape is [rows, width].
.section __TEXT,__const
.p2align 2
Lqkvshape: .long CONTEXT,WIDTH,3*WIDTH
Lprojshape: .long CONTEXT,WIDTH,WIDTH
Lff1shape: .long CONTEXT,WIDTH,HIDDEN
Lff2shape: .long CONTEXT,HIDDEN,WIDTH
Lheadshape: .long CONTEXT,WIDTH,VOCAB
Lattshape: .long CONTEXT,WIDTH,HEADS
Lnormshape: .long CONTEXT,WIDTH
