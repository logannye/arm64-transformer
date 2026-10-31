# Maintained-library common-state accuracy gate

This separate development gate checks numerical accuracy after the optimizer
repeat/restart gate has produced its exposed CPU trajectories. It is not a
formal fit or an opening of the reserved admission stream.

For each CPU PyTorch/MPS/compiled MLX width 32, 64, 128, 256, 512 and 1024,
pin the runtime capture, immutable worker source and seed-42 CPU development
trajectory receipt before execution. Reuse complete exposed CPU parameters,
first/second moments and step at updates **0, 8 and 64**. The previous capture
must identify the same runtime bytes and exact API capsule receipts and
derivation helper bytes. Recorded source paths may differ between immutable
checkouts; this does not transfer qualification to the new source. These saved
states are common operands for the new comparisons.

At each state, use the first B1/B3/B16 training rows at T32, with shifted targets.
Exercise the persistent GPU model actually used for training, or the explicit
library CPU admission entrypoint. Repeat each forward/loss/every-gradient
operation twice, require byte equality, and verify it has not modified the
parameters, moments or step. Retain first raw outputs and both fingerprints.
Compare against independent NumPy FP64 equations/analytic backward and Torch
FP64 autograd; require agreement between the two FP64 oracles as well.

For identical-gradient Adam, start from states **0 and 64** and apply the same
supplied FP32 gradient for **1, 2 and 10** updates. Generate that fixed gradient
from the exposed development stream, NumPy SeedSequence `[20261103, width, 91]`,
normal mean zero and standard deviation .01. Use the actual production framework
optimizer path; do not substitute the independent reference update. Compare all
parameters, both moments and step against the FP64 reference with FP32 state
rounding after every update. The nonzero step-64 moments exercise continuation
and bias correction separately from a fresh optimizer.

All 15 cases in every one of the 18 cells are required. Forward tolerance is
2e-5 + 2e-4|reference|, gradients 3e-5 + 3e-3|reference|, optimizer
3e-6 + 3e-4|reference|, independent FP64 agreement 1e-10 + 1e-10|reference|,
and step/repetition/restoration equality is exact. State remains FP32; CPU loss
reads may widen the already-computed FP32 scalar for reporting.

`audit_library_common_state.py` authenticates captured origins and API modules,
rehashes source states and outputs, independently recomputes every decision and
rejects changed shapes, storage, moments, counts, fingerprints or comparisons.
It records its separate oracle environment and never imports the selected
precision API into its independent FP64 model. Complete numerical failures remain
auditable negative results and cannot qualify a cell.

Run one numerical worker at a time. Reserve 16 GiB beyond a 1.5 GiB per-worker
output budget, check the floor between cases, reuse existing package/environment
trees and retain compressed raw outputs. All full-suite, primitive, exposed
replay, full repeatability, selected native/Metal and successor admission gates
remain independently mandatory before the 180 formal fits.
