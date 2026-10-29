# Wider transformer matrix-backend study, v5

This protocol declares a new matched campaign at model widths **512 and 1024**.
It preserves all v1–v4 protocols, measurements, reports and releases. New fixtures,
admission gates, raw runs and reports use distinct v5 paths. Earlier results
remain historical evidence; no speedup or quality improvement is required for a
result to be retained or reported.

## Question and evaluation status

The question is whether the handwritten CPU implementation remains competitive
with ordinary eager PyTorch when this short-context transformer becomes wider.
The two implementations receive the same architecture, float32 initial parameters,
data, updates and one CPU compute thread. The comparison is against batched
PyTorch with built-in scaled-dot-product attention, autograd and Adam, and the
handwritten ARM64 runtime using guarded SME dense kernels and NEON/scalar work.

The corpus and earlier test results are already known. This is a reused fixed
engineering benchmark, not a fresh untouched scientific assessment. No test
quality may guide kernel development, width selection, training duration, seed
selection, checkpoint rules or tolerances. Both declared widths and all valid
fits must be reported, including unfavorable performance. A wider model changes
capacity and compute cost; quality differences across widths cannot be attributed
solely to implementation or interpreted as evidence of convergence.

## Fixed model, data and training

The architecture and rules from [v4](PROTOCOL-v4.md) remain unchanged: one
pre-LayerNorm causal decoder block, learned token/position embeddings, residual
connections, four attention heads, ReLU feed-forward width twice the model width,
final LayerNorm and an untied ASCII-128 output head. Context is 32 characters.

| Model width | Head width | Feed-forward width | Parameters |
|---:|---:|---:|---:|
| 512 | 128 | 1024 | 2,251,392 |
| 1024 | 256 | 2048 | 8,696,960 |

The same pinned Tiny Shakespeare corpus and contiguous partitions supply 2,048
training sequences and 256 sequences in each evaluation split. Each sequence
contains 33 characters and yields 32 targets: **65,536 training, 8,192 validation
and 8,192 test targets**. Contexts never cross partitions. Corpus, rows and
per-seed schedules must match across widths and their identities are retained.
Initial parameter files differ by width but match exactly between implementations
within each width and seed.

Seeds are 42, 43 and 44. Every fit uses eight epochs, batch size 16 and exactly
1,024 Adam updates. Learning rate is .001, betas are (.9, .999), Adam epsilon is
1e-8 and LayerNorm epsilon is 1e-5. There is no decay, dropout, clipping, scheduler,
mixed precision, pretrained initialization or KV cache. The lowest epoch
validation NLL selects the checkpoint, with the first epoch winning ties. Test
prediction happens after selection. Each fit processes 524,288 training targets.

## Supported widths and admission

The model supports widths 32, 64, 128, 256, 512 and 1024. Process-local
`ARM64_TRANSFORMER_WIDTH` selection, model-width/parameter-count exports and
version-2 nine-word fixture configs remain mandatory. The native executable
rejects mismatched widths and malformed input before output creation. Only layout
constants are generated; numerical and training instructions remain handwritten.

The **timed protocol width set is only (512, 1024)**. V3 and v4 keep their original
(32, 64, 128, 256) width sets even though the implementation now supports more
models. The v5 runner must also authenticate complete correctness regression gates
at **all six supported widths** before any measured child is admitted. Old-width
regression gates are current binaries and current source, not historical v4 gates.
They do not represent new timed fits or additional quality seeds.

Automatic dispatch, forced NEON and hardware-admitted forced SME use separate
processes. Gate names remain `width-W.json`, `width-W-neon.json` and
`width-W-sme.json`. Forced NEON is always required. Supported hardware also requires
the full forced-SME gate; unsupported hardware must exercise clean rejection of
forced SME and may not claim an SME correctness pass. CI applies this rule at all
six widths. The measured campaign requests `auto` and records actual dispatch.

The existing forward, gradient, optimizer, mathematical, trajectory and finite-
difference tolerances remain unchanged. All historical random streams are
preserved. Every width retains primitive comparisons, every full-model parameter
gradient, multiple Adam steps with moments, finite differences away from ReLU
boundaries, exact causal masking, future-token intervention, repeated embeddings,
nonzero gradient accumulation, buffer guards, input immutability and standalone
partial-batch training against PyTorch. Invalid inputs, duplicate output refusal,
backend rejection and exact within-implementation repeats remain covered.

V5 appends independent random seed 20261031 for matrix-limit tests. These include
maximum supported input/output dimensions, one-past limits, non-power-of-two
inputs and row remainders. Expected SME/NEON dispatch-counter changes are checked
alongside results and derivatives. Actual wide QKV, attention, feed-forward and
output projections are all exercised by the selected-width gates. Existing
misalignment, changed-weights-at-the-same-address and ABI/state-preservation checks
remain active. Unmodified malformed-input payloads may use hard links to bound
local storage; corruption replaces its own directory entry, and original fixture
hashes must remain unchanged.

The public matrix gate admits supported 32-token shapes with input widths that
are powers of two from 64 through 2048 and output widths divisible by 32 from 64
through 3072. Unsupported shapes retain NEON fallbacks. Packing scratch reaches
512 KiB for forward and 384 KiB for backward; page-touching bounds stack growth.
The main training thread has adequate stack. An arbitrary 512 KiB worker stack is
not an adequate call contract for the largest forward kernel. There is no hidden
persistent packed-weight cache or excluded repacking after Adam.

## Prospective admission amendment, before formal timing

Synthetic development exposed a limitation of the original cross-implementation
free-running trajectory criterion at the two new widths. Small float32 cancellation
residuals in ordinary, well-defined weight gradients can be amplified by Adam
and later cross a ReLU boundary. The retained diagnosis found that an initial native
gradient can be closer to float64 while the two float32 training paths subsequently
differ. Identical-gradient Adam replay and common-checkpoint forward comparisons
address different numerical questions from independent trajectory agreement.

This amendment is informed by those synthetic failures and is declared **before
any formal v5 measured fit**. It is a change of admission contract, not evidence
that the original wide-model criterion passed. Initial failures, their exact
source and binaries are preserved under `work/v5/initial-admission` and retained
with the development evidence. The analysis in `experiments/numerics-v5` records
the observed mechanism and its limits. Neither thresholds, fixtures, random
streams, model, data, training budget nor optimizer settings change.

Widths 32–256 retain `strict-free-running-v1`: their original trajectory and
runtime cross-implementation comparisons remain blocking. Widths 512/1024 use
`v5-common-state-replay-v1`. At each of the original three synthetic updates,
validation evaluates **both engines' evolving parameter states separately**. At
each identical state and identical token row, every saved intermediate, loss and
parameter gradient must agree at the original forward/gradient limits. Both
engines' gradients are then independently replayed through native and PyTorch Adam
from identical parameters, first moments and second moments, with the same step
index. All resulting parameters/moments must pass the original optimizer limits;
step counts and finite states remain mandatory.

Wide-model free-running loss, parameter, gradient and moment comparisons retain
their original names, thresholds and true/false results as `legacy_diagnostics`.
The original runtime comparisons of best/last weights, moments, logits, history
and cross-implementation selected epoch are also retained there. A false result
remains false and appears in `legacy_failed_checks`; these records are **excluded
from passing admission counts**. Shape, cardinality and finiteness failures always
block, even for a diagnostic comparison. The emitted `legacy_contract_passed`
field states whether the retained old comparisons happened to pass; it does not
replace the named contract or erase earlier failures.

The runtime gate replays each implementation's own selected and last checkpoint
through **both complete implementations** on the unchanged synthetic test and
validation rows. Logits must agree at the original forward tolerance. Each selected
checkpoint must reproduce its own saved logits exactly in its originating engine,
and both engines must reproduce those logits within the original forward limits.
Each checkpoint's validation loss must match its own selection/final-epoch history
at the original limits. A separate functional PyTorch audit checks saved predictions,
file sizes, finite states, nonnegative second moments, validation selection and
optimizer step counts. Same-input/order controls, exact within-implementation
repeats, partial batches, malformed-input rejection and all other gates remain.

An independent Python controller also replays the entire four-update smoke schedule
for each implementation, including actual batch sizes `[3, 2, 3, 2]`, sequence
order, accumulation, normalization, validation and checkpoint selection. Native
arithmetic advances the native replay, while each identical-input Adam update is
checked separately against PyTorch at the original optimizer tolerance. The Python
replay uses the production model and standard Adam in an independent control loop.
Both replays must reproduce their own saved best/last parameters, moments and
non-timing histories exactly, in addition to passing the original magnitude limits.
This directly tests partial-batch normalization and runtime update orchestration;
fault-injection tests require wrong normalization, overwritten accumulation,
incorrect schedules and broken optimizer updates to be rejected.

The v5 runner, reports and packager require the explicit new contract identifier,
complete common-state/replay evidence and separate diagnostic records. Older
passed flags or missing diagnostics cannot satisfy this admission rule. Published
results must disclose the revision, original failures and diagnostic failure counts;
they may claim passing the declared common-state contract, never silently claim
that the original wide free-running contract passed.

## Evidence identity and backend scope

Every primary and forced gate authenticates its library, executable, generated
layout and all numerical/validation sources. Model exports must match before
buffers cross the FFI boundary. Gate sources, tolerances and binaries must agree
across policies; generated layouts must match the selected specification. The
runner rejects stale evidence and checks all six admitted binaries/gates plus
measured inputs and the committed clean source before and after each child.

`backend.u64` retains the v4 ten-word identity: format version, signed init status,
requested policy, selected kind, SME capability, streaming vector bytes, and SME/
NEON dense forward/backward counters. Runtime capability requires the admitted
OS feature flags and 64-byte streaming vectors. Initialization policy is captured
once. Live ZA or pending TPIDR2 state retains its guarded fallback behavior.

The fixed workload still makes **94,040 dense forward calls and 81,920 dense
backward calls per native fit**, including validation, final test and inference
samples. On the supported measured host, all five projections at widths 512/1024
are SME eligible. Actual counters must demonstrate the path; an enabled policy
alone does not establish execution. Attention, normalization and other operations
have their own handwritten implementations. Neither capability nor a sample count
establishes an elapsed-time contribution by itself.

## Execution and endpoints

Run **two widths × three seeds × two repeats × two implementations = 24 fits**.
The first repeat visits widths 512 then 1024; the second visits 1024 then 512.
Adjacent implementation pairs alternate their starting implementation by width
index plus seed index, then reverse on repeat two. The entire deterministic order
is recorded before the first fit. There is no performance-based early termination
or omission of unfavorable observations.

Only one measured child runs at a time, with AC power, the frozen single-thread
environment and a fixed 15-second pause between fits. Record power, thermal and
load conditions before and after each child. Builds, correctness runs, profiling
and microbenchmarks must not run concurrently with formal timing. Invalid runs
stop the campaign and preserve its failure evidence; retries use fresh paths.

Training wall/CPU time includes forward/backward, Adam, finite checks, epoch
validation, checkpoint selection/writes and history bookkeeping. Setup, final
test, last-state exports and inference samples are excluded. Public matrix-call
packing, scratch and streaming transitions remain inside the timers. Native
batch training still accumulates per-sequence gradients. Whole-process wall time
is independently observed. Peak RSS includes runtime/framework, data and state;
training-boundary peak and final-process peak are separately labeled.

Forward latency is batch one over the complete 32-character context, after 20
warmups, with 100 samples per fit. It includes causal attention softmax; it excludes
loss, output-probability softmax, sampling, token loading and I/O. This is not cached
single-token generation or application latency.

## Analysis and publication

The primary statistic at each width is the geometric mean of its six paired
PyTorch/ARM64 training-time ratios. Values above one favor assembly. Forward p50
ratios use the implementation means of per-run p50, a distinct statistic. Average
repeats within seed, then report mean and sample SD over three seed means. Report
all 24 observations; there is no pooled-width primary ratio. Throughput divides
524,288 training targets by training-plus-validation time, with validation targets
excluded from the numerator. Repeats are not independent quality seeds.

Report NLL, accuracy, perplexity, bits per character, paired differences and
prediction disagreement against the same uniform and training-only add-one
unigram baselines. Exact repeated checkpoints, Adam states, logits and non-timing
history are required before quality aggregation. Similar mean quality is not
statistical equivalence. Independently recompute every selected-checkpoint test
logit and selected validation loss using the unchanged functional-reference
tolerances. V5 retains **196,608 test prediction rows**, including repeats.

Reports distinguish the two measured widths' automatic/forced/combined admission
counts from the complete six-width regression counts. The release retains all
six admitted binaries/layouts and policy gates, plus the two measured widths'
fixtures, schedules, initial/selected/last weights, moments, predictions, latency
samples, conditions and source revision. Archives authenticate all payload bytes.
Preserve failed attempts and development diagnostics separately from formal fits.

Use static plots showing both widths and all outcomes, with clearly labeled ratio
statistics. Historical v4 versus v5 timings are separate campaigns and cannot
isolate a width effect from machine conditions. Conclusions remain limited to this
short-context float32 CPU model family against eager PyTorch. They do not establish
GPU, graph-compiled PyTorch, long-context, quantized, energy, cached-decoding or
production-serving performance.
