# Maintained-library optimizer development gate v1

This prospective gate retains the trained-precision development stream 20261103.
It opens neither reserved admission stream and does not authorize formal fits.
It applies to explicitly captured `torch.nn.precise_fp32` CPU/MPS and
`mlx.nn.precise_fp32` compiled GPU models, at widths 32, 64, 128, 256, 512 and
1024. The model-stage table is the unchanged `library_candidate.STAGES` table.
The package identities, input manifest and source snapshot are pinned before
execution. Prototype CI results cannot qualify installed packages.

For each of the 18 engine/width cells, use the authenticated seed-42 initial
checkpoint and training rows from the original CPU fixtures. Use the existing
development schedule: NumPy SeedSequence `[20261103, width, 73]`, 64 B16/T32
updates sampled from the 2,048 training sequences. Validation/test files and
formal admission operands are excluded. Adam remains the framework optimizer,
learning rate 0.001, betas 0.9/0.999, epsilon 1e-8, with FP32 parameters,
gradients and first/second moments. CPU/MPS uses ordinary non-fused,
non-foreach `torch.optim.Adam`; MLX uses its production compiled stateful
`optimizers.Adam` update with bias correction.

Each cell requires three separate serial processes: two complete trajectories
and a fresh-process continuation from the first process's step-32 checkpoint.
Every process first performs two warmup updates and restores its exact starting
parameters, both moments and step. It then runs its entire assigned trajectory
twice in that same process, restoring the start between repetitions. Require
byte-exact equality at every corresponding state, including step 0 or 32.
Save complete parameters and both moments at steps 0, 1, 8, 32 and 64, restricted
to steps at or beyond the process's start. Retain all state digests and the exact
schedule. Require finite outputs/gradients/state and nonnegative second moments.
MLX additionally requires a retained actual compiled training graph and exactly
one Python trace across changed-input updates and state restoration.

The independent auditor rehashes saved arrays, verifies the captured imports and
native package provenance, reconstructs the schedule, checks every state digest
across both same-process and fresh-process repetitions, and binds restart to the
first trajectory's complete checkpoint. Each cell must pass. Failures remain
failures; no tolerance or inventory changes may reclassify them.

This checks optimizer execution, repeatability and checkpoint continuation.
It does not establish numerical accuracy. Separate all-width FP64 model,
common-state/optimizer, primitive, exposed-state replay, complete framework,
repeatability and successor-admission gates remain required before 180 fits.

The CPU fit, supplied-operand admission bridge and saved-checkpoint replay expose
an explicit `--library-api` selection for this same maintained-package model.
The fit records its API origin, stage table, FP32 storage and non-stock identity.
Default invocations retain ordinary Torch arithmetic. The flag is rejected for
the native arm and FP64 oracle. Legacy frozen-study launchers continue to reject
the extra argument; this capability does not amend either historical protocol
or grant successor admission. A successor must bind the revised CPU source,
use the explicit selection for fit/admission/replay, and pass its own gates.

Run one numerical process at a time. Each worker reserves 16 GiB beyond its
bounded checkpoint allocation, checks the reserve before every update, writes
no bytecode and builds no native framework. At width 1024 the full trajectory
checkpoints require approximately 498 MiB per process; the continuation saves
only steps 32 and 64. Keep useful raw checkpoints, digests and failure receipts.
