# Reproduce the protected-QKV development experiment

The [protocol](../PROTOCOL-gpu-v3-qkv.md) fixes the arithmetic, candidate grid,
qualification, timing, and selection rules. This is a development experiment
against the selected GPU v2 Metal implementation. It cannot admit the formal
five-engine study or establish a CPU/framework speed ratio.

## Environment and evidence

Use Apple silicon with Metal GPU access and AC power. The measured device is an
M4 Max; actual compiler, device limits, native/shader hashes and environment
receipts are retained with every search. Numerical host libraries use one
thread. Run only one project numerical process at a time. Build into the single
`native/metal/build` directory, using `make -C native/metal`.

The evidence release retains the original seed-42 initialization and training
files for every width, their CPU fixture manifests, the installed runtime
identity receipt, native binary, complete new-phase evidence, and source
snapshots for measured and published revisions. It links historical releases
rather than duplicating their data. No original validation/test arrays or
reserved admission operands are included or read by this phase's packager.

Restoring evidence for inspection does not install a runtime or qualify a new
machine. A fresh execution needs a verified runtime inventory at
`work/gpu-v2/runtime-identity.json`, compatible installed packages, and the
training fixtures under `results/inputs-v6/width-W/seed-42/`. The
[v2 release tools](../gpu-v2/RELEASE-TOOLS.md) describe runtime identity handling.
Any changed source, binary, runtime, fixture or policy requires a new recorded
freeze. Commit the numerical sources, protocol and ledger before scoring;
the runner rejects uncommitted source bytes. Never overwrite an old attempt.

## Serial development commands

Set `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=.`, and the numerical thread
environment variables recorded in the protocol. The runner also enforces the
framework precision/fallback settings through the existing environment contract.
Execute these widths in order: **512, 1024, 32, 64, 128, 256**.

```sh
ARM64_TRANSFORMER_WIDTH=512 .venv/bin/python -m transformer.gpu_qkv_optimize \
  --width 512 --output work/NEW-ATTEMPT/width-512.json
```

Each process compiles the fixed candidates, freezes eligibility, checks accuracy
before timing, and writes immutable ledgers and per-candidate evidence. It
rechecks selected bundles against all stress families and the original v2
implementation. A failed process preserves partial evidence and stops the
attempt. Change the width and output name for the remaining five processes.

Assemble a width-indexed policy file from the six `selected_policy` fields,
preserving their exact values and recording the search hashes. Repeatability
must then bind that file and the exact runtime inventory hash:

```sh
ARM64_TRANSFORMER_METAL_POLICY=experiments/gpu-v3-qkv/policies/selected.json \
.venv/bin/python scripts/check_gpu_repeatability.py \
  --engines metal --widths 32 64 128 256 512 1024 \
  --stages full_model training \
  --runtime-identity-sha256 RUNTIME_INVENTORY_SHA256 \
  --output work/NEW-REPLAY

MPLCONFIGDIR=work/matplotlib-cache .venv/bin/python scripts/report_gpu_qkv.py \
  --searches work/NEW-ATTEMPT --repeatability work/NEW-REPLAY \
  --output results/NEW-REPORT
```

The reporter recomputes selections and paired ratios, validates complete check
inventories and input/source/runtime bindings, and reads the saved replay
arrays. Omitting repeatability produces an explicitly interim report. Figures
separate candidate comparisons, final unchanged/selected policies, and intrusive
stage profiling. Inspect final figures before publishing.

## Package and verify

Run after all evidence writers have stopped. Keep the archive output outside
the repository, and the packaging log outside the evidence directory being
archived. The output directory must be new, with an existing parent.

```sh
.venv/bin/python scripts/package_gpu_qkv.py create \
  --evidence work/gpu-v3-qkv \
  --searches work/gpu-v3-qkv/search/selection-v1 \
  --repeatability work/gpu-v3-qkv/repeatability \
  --report results/gpu-v3-qkv --output ../arm64-transformer-gpu-v3-evidence

.venv/bin/python scripts/package_gpu_qkv.py verify \
  --index ../arm64-transformer-gpu-v3-evidence/gpu-v3-qkv-archive.json
```

The archive is one gzip/tar stream split into parts below 1.9 GB. Its index
records ordered part hashes; its manifest records member hashes, measured
source bindings, runtime and training-input identities. Verification rejects
missing, reordered, truncated or changed parts and members. Remote release
asset hashes are checked separately after upload. Integrity verification does
not rerun numerical checks or turn a blocked study into an admitted study.
