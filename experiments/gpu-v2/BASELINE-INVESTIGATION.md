# Standard GPU baseline repeatability investigation

This development diagnostic preserves ordinary PyTorch MPS and compiled MLX
arithmetic. It does not replace embedding gradients, introduce a hybrid
baseline, loosen tolerances, or open formal admission. The v1 failure record is
immutable. A baseline failure does not prevent custom Metal kernel development.

## Bounded diagnostic

`scripts/check_gpu_repeatability.py` runs one engine and width per fresh Python
process, serially. Defaults cover both standard engines, all six widths, batches
1/3/16, lengths 3/32, ten executions per case, and two independent processes.
The process-local width is selected before importing model definitions. T1/T31
edge coverage belongs to the separate admission/lifecycle inventory.

At each width the embedding inventory declares 72 cases: six shapes, four token
patterns, and three cotangents. The patterns are identical, alternating, skewed,
and globally unique token IDs. The three B16/T32 unique-token cases are marked
unsupported because 512 unique positions cannot fit the 128-token vocabulary;
they are never silently converted to repeated-token controls. The 69 executable
cases use zero, random, and cancellation-heavy exact stored FP32 cotangents.

Each process also executes six full-model cases and six six-update training
cases, using a fixed skewed repeated-token pattern. Full-model checks compare
pure forward outputs and every parameter gradient while proving that parameters
and Adam state remain unchanged. Training replays restore the initial parameters,
moments, and step, run six ordinary updates, then export and reload the final
checkpoint and require exact forward replay. No corpus validation/test data or
reserved admission randomness is accessed. The diagnostic seed is 20261103.

An engine/shape runtime error is retained with its traceback and unexecuted
repetitions. Other declared cases continue. In particular, the retained MLX
T3/head-width-256 VJP rejection must remain visible if it still occurs.
An invalid lazy graph can also fail during backend cleanup. The diagnostic
retains that cleanup error, retires the backend, and constructs an ordinary new
backend before the next independent case. A cleanup exception cannot suppress
the worker's final receipt or stop later workers from reporting their outcomes.

Every execution retains complete array hashes, shapes, dtypes, exact mismatch
counts, finiteness, and separate FP64 accuracy results. The first output and the
first divergent output for each array are retained in full compressed NPZ files;
later divergences retain hashes and counts rather than duplicate wide states.
Both processes' first arrays are retained and compared byte for byte. Initial
parameters/state, source hashes before/after, framework capabilities, package
versions, environment controls, and fixture hashes accompany the evidence.
The report explicitly identifies a selected subset; a subset cannot claim the
complete declared inventory passed.
For prospective admission, `--runtime-identity-sha256` must be supplied with the
canonical digest of the frozen installed-runtime binding before execution. A
development run without that field remains useful diagnostic evidence, but
cannot be retroactively relabeled as authenticated admission evidence. The
admission verifier independently checks all expected worker/case names, raw
array fingerprints, process exit statuses, source hashes, and runtime identity.

Embedding FP64 reductions use independent NumPy and PyTorch CPU calculations.
Full-model gradients use PyTorch FP64 autograd. Forward and final-checkpoint
logits additionally cross-check independent NumPy FP64 equations, including
partial sequence lengths. Numerical limits remain forward 2e-5/2e-4, gradient
3e-5/3e-3, and independent-oracle agreement 1e-10/1e-10. Accuracy does not waive an
exact-repeat failure, and exact repetition does not imply numerical accuracy.

## Commands and cost

Inspect the prospective inventory without importing GPU runtimes:

```sh
PYTHONPATH=. .venv/bin/python scripts/check_gpu_repeatability.py --inventory
```

First run a narrow, explicitly labeled screen, then the complete inventory if
needed. Each output directory must be new. For MLX, retain the installed task
site on PYTHONPATH; do not recreate or silently upgrade its environment.

```sh
PYTHONPATH=work/gpu-v1/python-site:. .venv/bin/python scripts/check_gpu_repeatability.py \
  --engines mps mlx --widths 32 --stages embedding \
  --output work/gpu-v2/repeatability/embedding-screen-32

PYTHONPATH=work/gpu-v1/python-site:. .venv/bin/python scripts/check_gpu_repeatability.py \
  --output work/gpu-v2/repeatability/full-standard-baselines
```

One full engine/width/process has 690 isolated embedding VJPs, 60 full-model
gradient controls, and 360 training steps, plus checkpoint replay and CPU FP64
audits. The full declared diagnostic has 24 workers and at most 8,640 training
steps. Rejected cases reduce executed work but remain incomplete failures.
These are correctness diagnostics, not performance benchmarks. Run them serially
and separately from profiling, candidate timings, and formal fits. Check free
disk space first: retained wide checkpoint references require several GB for
the complete inventory, even with bounded divergence retention.

The optional custom Metal verification uses only the full-model/training stages;
the existing Metal backend exposes no isolated embedding probe:

```sh
PYTHONPATH=. .venv/bin/python scripts/check_gpu_repeatability.py \
  --engines metal --stages full_model training \
  --output work/gpu-v2/repeatability/metal-full-model
```

## Upstream evidence and stop rule

MPS already requests `torch.use_deterministic_algorithms(True)`. The pinned
[PyTorch 2.14.1 embedding Metal kernel](https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/aten/src/ATen/native/mps/kernels/Embedding.metal)
uses floating-point atomic accumulation for repeated-index weight gradients.
The [corresponding host dispatch](https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/aten/src/ATen/native/mps/operations/Embedding.mm)
contains no deterministic alternate branch. The pinned
[MLX 0.32.4 scatter kernel](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.4/mlx/backend/metal/kernels/indexing/scatter.h)
also uses atomic updates. This is consistent with the v1 observed embedding
gradient differences; exact dispatch attribution remains a diagnostic finding,
not an assumption about all wider-model discrepancies.

Current [PyTorch determinism documentation](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html)
lists MPS embedding backward among operations that can reject strict
deterministic execution. An upgrade may make rejection explicit without adding
a deterministic implementation. No reviewed official evidence currently
justifies selecting a package upgrade as a demonstrated repair. No package was
installed or upgraded during this investigation.

Permit at most one prospectively pinned stable upgrade per framework only when
official source or release evidence identifies a relevant repair. Record its
exact version, wheel hashes, and justification before executing the diagnostic.
Preserve the original environment and failures. Do not hunt across versions,
switch off strict determinism, enable silent CPU fallback, or rewrite baseline
embedding arithmetic. If any required exact-repeatability or runtime gate still
fails, the matched formal campaign stays closed. Publish the baseline limitation
alongside completed custom engineering work; claim no qualified GPU comparison.
