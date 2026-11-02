# Maintained precision library comparison, prospective GPU v5

This successor uses protocol `gpu-v5-library-five-engine-v1`. It selects the
explicit maintained precision APIs and the repaired native/Metal arithmetic
before new admission outcomes. It is a comparison of these implementations,
with FP32 stored state and declared internal higher precision. Reports must
identify the project-maintained framework builds and APIs. Standard released
PyTorch/MLX performance is a separate claim.

The new admission stream is **20261112, reserved and unopened**. Before this
declaration, a bounded search of 38,254 project text files up to 10 MiB found
no use of this stream. It covered this checkout, the retained CPU benchmark
checkout and the owned-runtime workspace, excluding expanded framework source,
runtime, dependencies, builds, capsules, immutable snapshots and raw arrays.
The retained search receipt is `work/library-successor-stream-search-v1.json`,
SHA-256 `a8e04a84d8504760574b44b2aa7ef406997ceb5bf3888c2f6319c95f19d749f8`.
This establishes the bounded selection basis, not absence in external files.
Development continues on the already exposed stream 20261103. Reserved streams
20261104 and 20261108 and both historical study contracts remain unchanged.

## Fixed workload and denominator

Retain the GPU v4 model, original CPU v6 corpus partitions, initialization and
schedules: one causal decoder block, four heads, T32, FFN 2W with ReLU, three
LayerNorms with epsilon 1e-5, and ASCII-128 vocabulary. Widths are 32, 64, 128,
256, 512 and 1024. Train B16/T32 for eight epochs, 1,024 updates and 524,288
processed targets per fit, using Adam .001, betas .9/.999 and epsilon 1e-8.
Store parameters, activations, probabilities, gradients and both moments as
FP32. Retain 65,536 training, 8,192 validation and 8,192 test targets. Choose
the lowest own-engine epoch validation NLL; earliest epoch wins an exact tie.

Run all five engines for seeds 42/43/44 and two timing repeats: **180 fits**,
72 CPU and 108 GPU, and 1,474,560 test-token predictions when complete. Keep
the original engine rotation, ascending/descending width traversals, AC power,
one host numerical thread, 15-second inter-fit cooldowns, power/thermal/load
records and one measured process at a time. The existing training and resident
B1/T32 inference timing boundaries remain explicit. Setup, graph compilation,
restored warmup and final test evaluation are reported separately. Preserve
exact parameters/moments/step around every warmup that takes an update.

## Select implementations explicitly

`experiments/gpu-v5-library/framework-selection.json` pins the full package
selection, NumPy 2.2.6 ABI, source revisions, ordered native patches, hosted
builds, runtime manifests and API derivation receipts. Its SHA-256 is
`f1fb641386e1a2cfd5afd49981618969c32b61f7b07a38704e7d347e42b77909`.
The same PyTorch installation and native identity serve CPU and MPS.

This is selection revision 2. Preserve the original selection at
`experiments/gpu-v5-library/framework-selection-v1.json`. Its MPS full sweep
executed all 10,357 methods but failed with 88 stock conformance extension errors:
the unchanged upstream loader emitted an unquoted library path containing a
space. Revision 2 places an independently authenticated byte-identical package
at a path without spaces. Only the Torch derivation receipt and manifest identity
change; every native, API and source payload file remains identical. The old
failure and positive primitive evidence keep their original runtime identity.
All required gates must be observed again for the new selection before admission.

| Engine | Selected implementation |
|---|---|
| CPU PyTorch | Maintained `torch.nn.precise_fp32` at LN1, QKV and causal attention |
| MPS | Same selected PyTorch API and stages, real MPS with CPU fallback disabled |
| MLX | Maintained `mlx.nn.precise_fp32` at LN1, QKV, causal attention and FF1; actual compiled GPU training |
| ARM64 CPU | Policy 0 at widths 32–256 and policy 15 at widths 512/1024 |
| Metal | Precision-normalization policy 1 on the reference shader path for training and inference |

The selected CPU numerical source is
`bcf0d5deb25c468ac1751d61f05caf75d23fd07e`, including explicit library selection
for CPU fit/admission/replay. Native binaries must independently match their
selected source, width, layout and precision policy. The frozen Metal shader
and derivation are recorded in `experiments/gpu-v5-library/metal-selection.json`.
This Metal arm makes no optimized-kernel performance claim. Its factory fixes
the reference dispatch policy and does not accept an ambient optimized policy.
Stock public framework operators remain unchanged; selected arithmetic lives in
the maintained API modules. No checkpoint-specific or width-specific framework
arithmetic or undeclared host tensor fallback is allowed.

## Gates before admission and measurement

`transformer.gpu_library_study.plan()` enumerates the required gates and selection
without executing a framework or granting formal authority. All full-model
accuracy groups at every width and engine are blocking in this successor. Keep
every original admission group and member, all existing common-state, optimizer,
finite-state, lifecycle, exact repetition, checkpoint, causal/sequence isolation,
partial-batch/tail/guard and malformed-input controls. Preserve the original
isolated stock-framework stress diagnostic and wide trajectory diagnostic scope;
these do not absorb required model, API, runtime or optimizer failures.

Required development evidence includes complete NumPy 2 framework suites, all
18 independently audited installed-API primitive cells, complete exposed trained
state forward/every-gradient replay, full embedding/model repeatability with ten
executions in two fresh processes, fresh same-distribution CPU anchors, all-width
common-state and identical-gradient optimizer accuracy, all 18 optimizer repeat
and restart cells, and independently qualified selected native/Metal policies.
The framework suite inventories and their exact existing skip/expected-failure
eligibility cannot shrink. Preserve the earlier failed receipts and logs.

Retain all numerical thresholds: forward 2e-5 + 2e-4|reference|, gradients
3e-5 + 3e-3|reference|, probability/math 2e-7 + 2e-6|reference|, optimizer
3e-6 + 3e-4|reference|, trajectory 2e-4 + 2e-3|reference| and independent
FP64 oracles 1e-10 + 1e-10|reference|. Repetition and restoration require byte
equality. Retain raw arrays and independent audits of every required decision.

Only an implemented, authenticated successor route may freeze all source,
runtime, policy, binary, fixture, prerequisite and 180-cell identities and then
open this reserved stream. Existing standard/owned launchers reject this new
protocol. A plan, hosted build or development pass alone cannot authorize fits.
The prospective plan currently reports `formal_execution_supported=false`;
dedicated dispatch, admission and checkpoint auditing must be implemented and
validated before it can become executable. No formal fits have started.

Use hosted native builds and one local numerical worker. Reuse authenticated
packages and shared lossless raw-array storage. Check disk reserves at validation
boundaries and each training update; stop before exhausting them. Preserve useful
failure evidence and user data. Do not create a new environment or build tree for
each width, process or engine.
