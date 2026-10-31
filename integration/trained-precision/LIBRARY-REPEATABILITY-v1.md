# Captured maintained-library repeatability development gate

This is the separate `maintained-library-repeatability-development-v1` contract
in the `trained-precision-development-v1` namespace. It grants no admission or
formal measurement authority. The standard and GPU v4 studies remain unchanged.

Use one externally hash-bound runtime capture and immutable source snapshot.
For each MPS/MLX width 32, 64, 128, 256, 512 and 1024, run two fresh serial
processes using `library_repeatability.py`. Select only the capture's explicit
maintained precision API and the unchanged `library_candidate.STAGES` table.
Record actual Python/native module provenance before and after execution.

The retained complete `gpu_owned_repeatability.run_worker` inventory, operands,
ten repetitions, oracles, tolerances and storage bounds apply unchanged. The
backend factory is an explicit dependency of that worker; the ordinary route
continues to instantiate its original authenticated backend. Candidate factories
cannot emit the historical owned namespace. There is no framework monkeypatch.

The inventory includes isolated embedding cases for batches 1/3/16, lengths
3/32, repeated/alternating/skewed and eligible unique tokens, with zero, random
and cancelling cotangents. Impossible globally unique cases at B*T > 128 keep
their original explicit exclusion. Full-model forward/loss/every-gradient and
six-update training cover every batch/length pair. Every training execution
restores exact starting parameters/moments/step and checks final checkpoint
restoration and forward replay. All ten executions and both fresh processes
must agree byte for byte. MLX retains the actual compiled training graph and
requires one Python training trace across changed inputs and state restoration.

`audit_library_repeatability.py` independently authenticates both workers,
their exact artifact inventories, captured module origins and selected APIs.
It checks every execution fingerprint against retained raw arrays, reconstructs
embedding fixtures, initialization and model operands, recomputes independent
NumPy FP64 forward and analytic backward, checks agreement with Torch FP64
autograd, evaluates checkpoint logits and compares raw arrays across processes.
It does not import or execute the selected candidate implementation. The audit
requires the width environment variable to match before importing model layout.

Every declared case is mandatory, with existing tolerances: forward 2e-5 +
2e-4|reference|, gradients 3e-5 + 3e-3|reference|, independent FP64 oracles
1e-10 + 1e-10|reference|, and byte equality for repetition/restoration.
Retain all failures and incomplete evidence. Prototype tests and a successful
worker do not substitute for the independently audited complete pair.

Use the existing development seed 20261103. Neither reserved admission stream
is opened. Full framework suites, generic primitives, exposed trained states,
CPU anchors, common-state/optimizer comparisons, optimizer repetition/restart,
native/Metal qualification and an implemented successor admission contract
remain separate prerequisites for the 180 formal fits.
