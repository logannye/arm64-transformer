# Transformer width sweep, v3

This protocol extends the [v2 experiment](PROTOCOL-v2.md) to model widths 32,
64, 128 and 256. It is fixed before the 48 measured fits. The v1 and v2 protocols,
numerical results and archived raw evidence remain unchanged. Published v1/v2
preflight copies normalize an unrelated local interpreter path to `python`, with
an explicit metadata note. No speedup is required for a result to be retained or
reported.

## Question and evaluation status

The question is how the measured PyTorch/ARM64 implementation comparison changes
as this small transformer becomes wider. Each width has its own matched comparison
using ordinary eager, batched PyTorch with built-in scaled-dot-product attention,
autograd and Adam, versus handwritten Apple ARM64/NEON numerical code and runtime.
Both implementations receive one CPU compute thread on the same machine.

V1 and v2 results from this corpus have already been inspected. The sweep therefore
uses a known, fixed engineering benchmark, not a newly untouched assessment set.
No width, seed, training duration, hyperparameter or data subset may be selected
from test results. All four declared widths must be reported, including failed
admission or measured runs. A wider model changes capacity and arithmetic cost;
cross-width quality differences cannot be attributed solely to implementation.

## Fixed model family and data

Every model has one pre-LayerNorm causal decoder block, learned token/position
embeddings, residual connections, a ReLU feed-forward network, final LayerNorm
and an untied output head. Context stays 32 characters, vocabulary stays ASCII-128,
and the number of attention heads stays four. Feed-forward width is twice model
width. Parameter counts include all embeddings, weights and biases.

| Model width | Head width | Feed-forward width | Parameters |
|---:|---:|---:|---:|
| 32 | 8 | 64 | 17,952 |
| 64 | 16 | 128 | 52,160 |
| 128 | 32 | 256 | 169,728 |
| 256 | 64 | 512 | 601,472 |

The same pinned Tiny Shakespeare source and contiguous partition boundaries are
retained. Each fit uses 65,536 training, 8,192 validation and 8,192 test target
characters: 2,048 training sequences and 256 sequences in each evaluation split,
with 32 predictions per sequence. Contexts never cross partition boundaries.
Token rows and training schedules are identical across widths; their hashes must
be recorded. Within each width/seed pair, both implementations start from identical
exported float32 parameters and consume identical batches. Different widths have
different parameter shapes and therefore different initial parameter artifacts.

Seeds are 42, 43 and 44. Training uses eight epochs, batch size 16 and exactly
1,024 Adam updates per fit. Adam uses learning rate .001, betas (.9, .999), epsilon
1e-8 and no weight decay. LayerNorm epsilon is 1e-5. There is no dropout, clipping,
learning-rate scheduler, mixed precision, pretrained initialization or KV cache.
Each update averages over its actual batch tokens. The lowest epoch validation
NLL selects the checkpoint, with the first epoch winning exact ties. Test
evaluation occurs only after checkpoint selection.

## Width and binary identity

`ARM64_TRANSFORMER_WIDTH` selects one of the four declared widths at process
startup. Python modules are imported in a separate process for each width;
changing an already imported module's width is not a supported execution path.
Each native model is compiled into its own `native/build/width-WIDTH/` directory.
Only layout constants are generated. Numerical and training instructions remain
handwritten assembly, without BLAS, Accelerate, PyTorch or compiled C numerical
implementations in the native executable.

The library exports its model width and parameter count. Validation must reject
any mismatch before passing model buffers to native code. Historical width-32
libraries without these identity symbols may be explicitly admitted only for
separate historical diagnostics; they cannot satisfy the v3 validation gate.
New fixture configs use version 2 with nine little-endian uint32 fields, adding
the width after the seed. Each native executable must reject unsupported or
mismatched widths, obsolete versions and incorrect config lengths before creating
an output directory.

## Admission before timing

Every width must pass the complete numerical and executable preflight against
its exact linked library and executable. All original tolerance values remain
unchanged. Existing width-32/v2 fixture streams are retained; additional v3
boundary probes use independent NumPy random seed 20261029.

The gate includes primitive results and derivatives, every full-model parameter
gradient, repeated-token embedding accumulation, finite differences away from
ReLU boundaries, causal intervention checks, and several complete Adam steps
with parameter and moment-state comparisons. Standalone training is compared
with PyTorch on the same synthetic fixture, including a final partial batch,
validation selection, repeatability, corrupt input files and output refusal.

Additional width probes cover every actual linear projection shape, row
remainders and feature tails, normalization at the selected width and adjacent
scalar fallback width, and optimized attention at head widths 8/16/32/64.
Attention tests include short token counts, vector boundaries, context 31/32,
a longer generic context, non-specialized feature/head shapes, uniform and large
scores, exact masked zeros and exact zero future gradients. Guard sentinels detect
writes beyond output, gradient and whole-model workspace buffers; input arrays
must remain unchanged. Nonzero parameter-gradient accumulators are reused twice.

The gate authenticates all native numerical/runtime sources, the Makefile, and
the Python specification, fixture, reference, training and validation sources.
Each generated layout must exactly match the selected specification. Record its
path and SHA256 separately from tracked-source hashes. Source, generated layout,
library and executable must remain unchanged during validation. The measured
source must then be committed and clean. Timing requires passing gates for all
four widths and rejects stale or mismatched evidence before launching a child.

Failures retain their evidence. Correct implementation defects and rerun the
affected admission checks without relaxing thresholds. Any further numerical
source change requires new exact-binary gates before measured runs. Development
checks use synthetic inputs; test quality does not guide implementation choices.

## Execution order and timing

Run four widths × three seeds × two repeats × two implementations: **48 fits**.
The first repeat visits widths in ascending order; the second visits them in
descending order. Within each width, the initial implementation order alternates
by seed and reverses in the second repeat. Record the complete deterministic order
before the first fit. This balances width position and implementation order across
repeats; it does not experimentally fix operating-system scheduling or thermals.

Only one measured child runs at a time. Use AC power, the same one-thread
environment variables and a fixed 15-second pause between fits. Record power,
thermal and load snapshots before and after each child. Do not run kernel timing,
compilation or other validation concurrently with measured fits. Authenticate
the selected binary, generated layout and frozen source for every child.

Training wall/CPU timers include forward/backward, Adam, finite checks, epoch
validation, checkpoint selection/writes and history bookkeeping. Setup, final
test evaluation, last-state exports and inference samples remain outside that
timer. Whole child-process wall time is measured independently by the coordinator.
Memory is process peak resident set size at declared boundaries, including the
runtime/framework and data, rather than isolated model storage.

Inference remains a batch-one forward pass over a full 32-character context, with
20 warmups and 100 samples per run. It excludes token loading, loss, softmax,
sampling and I/O. It is not cached single-token generation or application latency.
No performance admission threshold, early termination for slow results or removal
of inconvenient observations is permitted. Stop on an invalid child and preserve
the stopped campaign; a replacement campaign requires a new output directory.

## Analysis and retained evidence

For each width, the primary training comparison is the geometric mean of its six
paired PyTorch/ARM64 wall-time ratios; values greater than one favor assembly.
Report all four ratios individually and all 48 run observations. Average timing
repeats within each seed, then report the mean and sample SD of the three seed
means. Repeats are not independent quality seeds. Forward p50 ratios use the
corresponding implementation means and remain distinct from training ratios.

Report NLL, bits per character, perplexity, accuracy, paired quality differences
and prediction disagreement at every width, together with the same uniform and
training-only add-one unigram baselines. Exact within-implementation repeated
checkpoints, optimizer states, logits and non-timing histories are required before
pooling quality. Similar aggregate quality does not establish statistical
equivalence. Floating-point grouping can change long training trajectories.

Independently recompute every selected-checkpoint test logit and selected
validation NLL with the separate functional reference and unchanged tolerances.
The complete sweep contains 393,216 retained test prediction rows, including
repeats. Preserve fixture manifests, inputs, schedules, initial and selected/last
weights, final Adam states, raw predictions, histories, latency samples, source
and binary identities, generated layouts, every admission check and post-run
audit, all plots, and failed attempts under distinct v3 paths.

Plots of cost or implementation ratio versus width describe this fixed model
family. Historical comparisons with v1/v2 are separate campaigns whose machine
conditions may differ; primary v3 conclusions come from the fresh paired runs
within each width. The scope remains short-context float32 eager PyTorch versus
handwritten assembly on one CPU machine. It does not establish GPU performance,
graph-compiled PyTorch performance, long-context scaling, energy consumption,
cached decoding speed or production serving behavior. Native sequence-by-sequence
gradient accumulation remains an allowed implementation; batch-wide native work
must not be claimed without separate implementation and verification.
