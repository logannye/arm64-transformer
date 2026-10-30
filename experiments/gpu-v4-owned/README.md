# An independent route to the CPU/GPU study

This phase compares our hardware-specific transformer implementations with
**project-maintained PyTorch and MLX builds**. Keeping the repairs in pinned
full-source frameworks lets us continue without submitting patches upstream or
waiting for a framework release. Every eventual result must identify those
patched baselines explicitly.

The [prospective protocol](../PROTOCOL-gpu-v4-owned.md) preserves the six widths,
original data and initialization, eight-epoch budget, three seeds, two repeats,
180 planned fits, exact training repeatability and FP64 accuracy requirements.
PyTorch CPU and MPS must use the same selected distribution. The full checkpoint
audit and per-width reporting requirements remain in place.

**Status: preparation, not admission. Zero owned-study formal fits have run.**
The original standard-release study also remains at zero of 180, with its
reserved stream `20261104` unopened. The new contract reserves `20261108` and
does not open it during planning or preflight.

## What is already available

The [integration release](https://github.com/logannye/arm64-transformer/releases/tag/transformer-upstream-integration-v2)
contains authenticated full-source framework binaries, patches, regression
evidence and reproduction instructions. Patched MLX passes the declared Release
integration checks in both shader modes. Patched MPS passes targeted embedding
and matched-transformer checks. The custom Metal implementation and its selected
production policies already have separate development evidence.

These results are starting evidence for an isolated owned runtime. Combining or
relocating components creates a new execution identity and requires new checks;
old receipts cannot be reused as admission passes.

## What still blocks formal execution

The broad patched MPS suite retains **25 failures and two errors**. They also
reproduce in a pristine matched build, but remain blocking integration failures
for this new route. This contract does not remove tests or reclassify those
failures to permit measurement. We can investigate and repair them ourselves.

An independently executable study also needs verified runtime assembly, fresh
baseline and CPU checks, dedicated admission identities, the complete passing
admission inventory, and finally all 180 fits and checkpoint audits. Existing
planning or preflight output is not an executable manifest or a passing admission
receipt. The old standard-study runner must reject this new protocol.

## Planning and preflight

The additive `transformer.gpu_owned_study` command can produce the complete plan
and inspect the retained evidence. It imports no framework, creates no admission
operands and has no formal-run command. Run from the repository with the existing
Python environment and use fresh output paths:

```sh
.venv/bin/python -m transformer.gpu_owned_study plan \
  --output work/gpu-v4-owned/plan.json
.venv/bin/python -m transformer.gpu_owned_study preflight \
  --repository "$PWD" \
  --evidence-root /ABSOLUTE/VERIFIED/EVIDENCE/ROOT \
  --output work/gpu-v4-owned/preflight.json
```

The evidence root is the common reconstructed `work/` and `outputs/` parent
described in the [integration reproduction guide](../upstream-integration-v2/REPRODUCTION.md).
Preflight authenticates the selected patch identities and retained failed broad
suite; it reports blockers without turning them into passes. An optional
hash-bound joint-runtime development probe remains a smoke check. A dedicated
owned-study freeze, admission, execution and audit integration still has to be
implemented and validated before formal measurement can begin.

The preserved historical CPU numerical failures remain unresolved. A successful
owned-build study could establish a matched comparison against these maintained
frameworks; it would not establish performance against standard released packages.
