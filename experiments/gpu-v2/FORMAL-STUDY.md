# Matched CPU and GPU study machinery

The formal comparison is implemented but has not run. The standard MPS and MLX
baselines failed the required exact-repeatability gate at all six widths in
`work/gpu-v2/repeatability/full-v3`. MLX also raised errors for the T3 full-model
and training cases at widths 256, 512, and 1024. The admission seed remains
unopened. The committed plan retains all 180 cells, with zero completed formal
fits and no qualified speedup claim.

`planned-study.json` is a plan, not an executable frozen manifest.
`blocked-study.json` records the outcome and the original baseline summary hash.
Development profiling, candidate selection, CPU bridge smoke tests, and mock
control-plane tests do not substitute for formal admission.

## The comparison

The study crosses six widths (32–1024), three original seeds (42–44), two timing
repeats, and five engines: frozen ARM64 CPU, PyTorch CPU, handwritten Metal,
standard PyTorch MPS, and standard MLX. Every fit uses the original CPU v6 input
bytes, initial state, schedule, B16/T32 workload, eight epochs and 1,024 updates.
The CPU source stays at `2c5191047068c16b12e3312652b118d48a5f3744`.
CPU Torch and MPS use the same installed Python and Torch distribution.

Each width/seed/repeat block rotates the engine order by
`(width_index + 2 * seed_index + repeat_index) % 5`, using zero-based indices.
The first repeat traverses widths ascending; the second traverses descending.
Fits run serially in fresh processes on AC power, with a 15-second cooldown.
A new exact-repeat failure stops later cells immediately and retains the full
planned denominator and partial artifacts.

Full training timing includes forward, backward, Adam, finite checks, scalar
loss reads, validation, checkpoint writes, packing and synchronization.
Framework import and setup, including Metal shader compilation, are recorded
separately. The primary inference timing uses resident B1/T32 operands, 20
warmups, and 100 forward/completion samples. It excludes host readback and
validation checks. Warmup must restore the complete optimizer state exactly.

The audit replays each of 180 saved checkpoints through all five engines and
the FP64 oracle on every validation and test sequence. It retains raw replay
arrays and authenticates the original integer rows. The reporter recomputes
numerical checks, quality and prediction differences from those arrays; it
does not trust a summary's claimed pass or quality numbers. Own-engine saved
outputs, both repeat checkpoints, Adam moments and update counts are checked
separately from cross-engine numerical tolerances.

## Admission and evidence scope

The named inventory has 2,088 gate groups and 9,858 members. It includes 270
custom Metal attention/normalization stress cases, 540 nonblocking standard
framework stress diagnostics, full-model gradients, common-state comparisons,
all 25 identical-gradient optimizer source/destination pairs at steps 1, 2,
and 10, three-update trajectories, independent FP64 validation and lifecycle
controls. Framework diagnostic runtime errors are retained explicitly and
cannot waive a blocking gate. The trajectory gates at widths 512 and 1024
retain the CPU protocol's diagnostic scope.

Handlers are registered for every named gate. Formal execution of that
inventory remains unvalidated because the baseline prerequisite fails.
The runtime-failure control exercises rejected use after close or malformed
CPU bridge operands; it does not inject an actual GPU device fault. Barrier
tests exercise completed tails and guard checks; they do not prove race
freedom. Frozen CPU warmup/restore controls exercise serialized-state reload
and checkpoint replay; the immutable standalone CPU runner has no persistent
GPU-style warmup API. CPU same-process training repeats do execute both
six-update sequences in one frozen-source child, with exact parameter, moment,
step and checkpoint-logit comparisons.

Every formal boundary authenticates source, frozen CPU source/binaries, input
bytes, installed runtime, selected policy, and baseline evidence. Admission
requires a complete receipt tied to those identities and raw numerical arrays.
The manifest and admission receipt are anchored by externally supplied SHA-256
digests. This is hash-bound local evidence, not a public-key signature or an
independent third-party attestation.

Runtime discovery follows the actual `.venv/bin/python` symlink into its uv
installation, inventories the whole base installation and virtual environment,
and includes the explicit MLX add-on directory. It binds canonical file bytes,
symlink resolutions, package origins and the OS build. Verification discovers
the inventory again, so an omitted or newly added dependency fails. It copies
nothing and imports no GPU framework. macOS dyld shared-cache bytes are not
independently hashed; their scope is the recorded OS build. Formal processes
disable writing new Python bytecode.

The disk preflight estimates retained evidence, allows 25% overhead and keeps
a 2 GiB reserve. Current conservative free-space estimates are approximately
105.3 GiB for admission, 22.2 GiB for the fit campaign plus its later audit, and
14.5 GiB for a separate audit. These are estimates, not storage reservations.
No gate is opened merely to discover whether disk space is sufficient.

## Commands

Run these from the repository root with the existing environment. No command
below builds a native library. Outputs are collision-safe and must be fresh.
The planning, inventory and storage commands neither create reserved operands
nor initialize a GPU:

```sh
.venv/bin/python -m transformer.gpu_study plan --output work/gpu-v2/new-plan.json
.venv/bin/python -m transformer.gpu_study_admission inventory
.venv/bin/python -m transformer.gpu_study storage
.venv/bin/python -m pytest tests/test_gpu_study.py tests/test_gpu_study_runtime.py tests/test_gpu_admission_checks.py -q
```

The remaining commands describe a future qualified run. They must not be run
to bypass the current baseline failure. First finish policy selection, preserve
the frozen CPU archive and its existing binaries, commit the complete source,
and select a directory containing exactly one final policy JSON. A new complete
baseline repetition run must pass on the same discovered runtime; its CLI
accepts `--runtime-identity-sha256`, equal to the canonical JSON SHA-256 of the
runtime inventory's `files` mapping. Baseline evidence from another environment
or an old hash schema cannot qualify.

```sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/work/gpu-v1/python-site"
.venv/bin/python -m transformer.gpu_study runtime \
  --runtime-root / \
  --runtime-python-path "$PWD/work/gpu-v1/python-site" \
  --output work/gpu-v2/runtime-inventory.json
```

Set `GPU_STUDY_ROOTS` to a JSON object with absolute paths for `source`, `cpu`,
`runtime`, `inputs`, `policy`, and `baseline`. A typical mapping uses this repo,
`work/gpu-v2/cpu-reference`, `/`, `results/inputs-v6`, the directory containing
only the selected policy JSON, and the directory of a new passing baseline
suite, respectively. The broad runtime root permits canonical files in both uv
and the repo to be authenticated without copying the environment.

```sh
.venv/bin/python -m transformer.gpu_study freeze \
  --roots "$GPU_STUDY_ROOTS" \
  --runtime-inventory work/gpu-v2/runtime-inventory.json \
  --output work/gpu-v2/frozen-before-admission.json
GPU_PRE_HASH=$(shasum -a 256 work/gpu-v2/frozen-before-admission.json | cut -d ' ' -f 1)
.venv/bin/python -m transformer.gpu_study_admission run \
  --manifest work/gpu-v2/frozen-before-admission.json --manifest-sha256 "$GPU_PRE_HASH" \
  --roots "$GPU_STUDY_ROOTS" --output work/gpu-v2/formal-admission
.venv/bin/python -m transformer.gpu_study_admission seal \
  --manifest work/gpu-v2/frozen-before-admission.json --manifest-sha256 "$GPU_PRE_HASH" \
  --roots "$GPU_STUDY_ROOTS" --receipt work/gpu-v2/formal-admission/admission.json \
  --output work/gpu-v2/frozen-measurement.json
GPU_MEASURE_HASH=$(shasum -a 256 work/gpu-v2/frozen-measurement.json | cut -d ' ' -f 1)
.venv/bin/python -m transformer.gpu_study run \
  --manifest work/gpu-v2/frozen-measurement.json --manifest-sha256 "$GPU_MEASURE_HASH" \
  --roots "$GPU_STUDY_ROOTS" --admission work/gpu-v2/formal-admission/admission.json \
  --output work/gpu-v2/formal-campaign
.venv/bin/python -m transformer.gpu_study_audit \
  --manifest work/gpu-v2/frozen-measurement.json --manifest-sha256 "$GPU_MEASURE_HASH" \
  --roots "$GPU_STUDY_ROOTS" --admission work/gpu-v2/formal-admission/admission.json \
  --campaign work/gpu-v2/formal-campaign --output work/gpu-v2/formal-audit
.venv/bin/python -m transformer.gpu_study_report \
  --manifest work/gpu-v2/frozen-measurement.json \
  --campaign work/gpu-v2/formal-campaign --audit work/gpu-v2/formal-audit \
  --output work/gpu-v2/formal-report
```

If execution stops, run the reporter with `--campaign` and omit `--audit` to
retain completed and failed cells without issuing qualified ratios. A complete
campaign requires the complete audit. Reusing an existing output directory,
editing a bound dependency, removing evidence, or presenting only a `passed`
boolean fails closed.
