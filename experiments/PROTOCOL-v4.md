# Transformer matrix-backend study, v4

This protocol retains the complete v3 width study and its evidence unchanged,
then evaluates a new handwritten matrix backend in a fresh 48-fit campaign.
All v1/v2/v3 source, protocols, measured results and release evidence remain
historical records. V4 uses distinct input, preflight, result and report paths;
existing evidence is never overwritten. No speedup is required for publication.

## Question and evaluation status

The question is whether restructuring the dominant dense operations around
hardware matrix instructions improves the measured implementation comparison
across the same four transformer widths. Each width has its own matched comparison
using ordinary eager, batched PyTorch with built-in scaled-dot-product attention,
autograd and Adam, versus handwritten Apple ARM64 numerical code and runtime, including NEON and
a capability-guarded SME matrix backend.
Both implementations receive one CPU compute thread on the same machine.

V1, v2 and v3 results from this corpus have already been inspected. The sweep therefore
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
separate historical diagnostics; they cannot satisfy the v4 validation gate.
New fixture configs use version 2 with nine little-endian uint32 fields, adding
the width after the seed. Each native executable must reject unsupported or
mismatched widths, obsolete versions and incorrect config lengths before creating
an output directory.

## Backend policy and observed execution

`ARM64_TRANSFORMER_BACKEND` accepts `auto`, `neon`, or `sme` and is captured once
at process initialization. Changing the environment after loading the library
must not change its policy. Each policy therefore receives a separate process.
The formal 48-fit campaign uses `auto`; forced policies are correctness gates,
not additional formal performance arms.

SME admission requires Darwin's SME and SME2 capability flags and the required
64-byte streaming vector length. Forced `neon` always selects NEON. Forced `sme`
rejects unsupported hardware before creating output; `auto` safely chooses NEON
there. Invalid policy strings fail explicitly. The dispatcher uses NEON for
unsupported shapes and when SME state requirements are not met. A policy that
enables SME does not mean every operation used an SME instruction. Width 32
continues to use NEON dense kernels; eligible projections at widths 64/128/256
use the matrix backend on the admitted machine. Attention, normalization, loss,
Adam and other non-dense operations retain their declared implementation.

Every preflight records the library's backend identity. Every native training
process writes `backend.u64`: ten little-endian 64-bit words containing format
version, signed initialization status, requested policy, selected backend kind,
SME capability, streaming vector bytes, and separate SME/NEON forward/backward
dense-call counters. The runner checks these bytes against the selected preflight
and the exact workload. A completed formal native run makes 94,040 dense forward
and 81,920 dense backward calls, including validation, final test, inference
warmups and samples. Actual counters must match the expected dispatch for its
width and capability; recording a requested policy alone is insufficient.
Backend metadata and its SHA256 are retained in the raw manifest and report.

Matrix instructions use float32 operands and accumulation. Packing, transpose,
scratch allocation and state-transition work that occurs in public dense calls
remains inside the actual training and inference timers. Model parameter files,
gradient accumulation semantics and Adam state remain canonical row-major
float32. There is no persistent packed-weight representation exempted from
update or checkpoint costs. The normal ABI returns with caller-preserved state
intact; live matrix-state ownership must not be overwritten.

## Admission before timing

Every width and required backend policy must pass the complete numerical and executable preflight against
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

V4 appends random seed 20261030 without perturbing any earlier fixture stream.
Additional cases exercise 4/8/12-byte pointer offsets from vector alignment,
row and feature counts immediately around tile boundaries, nonzero gradient
accumulators, repeated calls, and changing weights at the same address. Both
SME-eligible calls and NEON fallbacks execute within the same process. An assembly
ABI probe verifies the preserved low 64 bits of d8-d15 and streaming/ZA/TPIDR2
state across eligible forward/backward calls. The mathematical checks still cover
every full-model gradient, multiple Adam steps, causality and partial batches.
No tolerance may be relaxed to admit a changed matrix reduction order.

The primary auto-policy gates are `width-W.json`. Forced NEON gates are
`width-W-neon.json` at every width. On supported hardware, forced SME gates are
`width-W-sme.json` at every width. Each runs the complete numerical and runtime
gate against the same library, executable, generated layout and source hashes.
The runner rejects missing or mismatched forced-mode evidence. On unsupported
hardware, the forced-SME failure path is checked and recorded; unsupported hosts
must not silently report an SME correctness pass. CI always runs forced NEON,
then runs forced SME only when its actual capability query admits it. Reports
separate primary-auto, supplemental-forced and combined preflight check counts.

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
audit, all plots, and failed attempts under distinct v4 paths.

Plots of cost or implementation ratio versus width describe this fixed model
family. Historical comparisons with v1/v2/v3 are separate campaigns whose machine
conditions may differ; primary v4 conclusions come from the fresh paired runs
within each width. The scope remains short-context float32 eager PyTorch versus
handwritten assembly on one CPU machine. It does not establish GPU performance,
graph-compiled PyTorch performance, long-context scaling, energy consumption,
cached decoding speed or production serving behavior. Native sequence-by-sequence
gradient accumulation remains an allowed implementation; batch-wide native work
must not be claimed without separate implementation and verification.

## Profiling and interpretation

Development uses deterministic synthetic activations and training fixtures,
never final-test quality to select an optimization. Public-API microbenchmarks
report complete kernel calls, including packing and dispatch. Fresh comparisons
pair the preserved v3 library with the candidate in rotated/reversed blocks and
retain call counts, shapes, source/binary hashes and environment. Microbenchmarks
include FFI overhead and are diagnostics, not the formal training endpoint.
Whole-forward/backward diagnostics retain their separate scope.

Actual native runtime profiling uses synthetic training and separates startup
from the training phase. Sample counts or explicitly instrumented timings state
their denominator and scope. Inclusive call-stack samples are never summed as
exclusive contributions; ambiguous symbols are reported at a defensible family
level. Profiler overhead and sampling resolution limit attribution. Formal fits
run without profiler instrumentation. A before/after improvement across a bundle
of changes does not identify an individual instruction's causal contribution;
no isolated ablation is implied. The historical v3 and fresh v4 formal campaigns
may differ in machine conditions, so current PyTorch/assembly paired comparisons
remain primary.
