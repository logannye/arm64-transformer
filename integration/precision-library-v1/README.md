# Prospective maintained-library precision APIs, v1

Implement generic `linear`, last-axis `layer_norm` and causal self-attention
operations in the maintained PyTorch and MLX packages. These are explicitly named
experimental APIs, separate from stock operations and from the existing custom
diagnostic models. No transformer architecture, checkpoint, width selector or
vocabulary belongs in the library implementations.

Preserve FP32 parameters, outputs, saved activations, probabilities and gradients.
CPU operations use FP64 intermediate arithmetic with explicit first derivatives
and FP32 saved state. MPS and MLX use the existing declared two-float FP32
arithmetic; dense derivatives retain only FP32 inputs and weights. Device
operations stay on the selected GPU. Dense temporaries retain fixed row/output
tiles of 32/64. The initial APIs support nonempty, finite moderate operands,
first derivatives, positive LayerNorm epsilon and matching 4D causal
self-attention operands with zero dropout.

Before integrating an API into a model, require primitive agreement with FP64,
all returned derivatives, FP32 saved-tensor checks, causal isolation, changed
inputs and exact repeated execution. Include noncontiguous views and malformed
inputs. Preserve all existing tolerances. MLX additionally requires real graph
compilation and changed-input trace proof.

The first model integration must reproduce the already declared arithmetic:
CPU v2, MPS policy 7 and compiled MLX policy 15. Generic library code must not
contain those model-stage selectors. Keep ordinary controls and prior custom
models identifiable. New package/source identities require fresh full suites,
complete historical and fresh same-state inventories, all-width repeatability
and optimizer/checkpoint checks. Merely relocating code into a package does not
qualify a library repair or a formal comparison arm.

This directory is prospective source preparation. No maintained runtime has
loaded these files, and no performance or formal eligibility is established.

`../gpu-owned/derive_precision_api.py` now authenticates each original full build,
creates a bounded APFS clone, adds only the three Python files to source and
runtime, and verifies the complete allowed manifest delta. Existing native,
Python, stock-operator and test files are preserved byte for byte. The new
`PRECISION-API.json` receipt confers no old qualification. The MLX copy has been
derived and authenticated; no model or primitive has loaded it yet. PyTorch's
copy awaits the hosted NumPy 2 build. Explicit capture selection records the
API module and derivation separately from unchanged native patches.

`../trained-precision/library_candidate.py` selects the model stages while calling
only the package's generic precision APIs for repaired arithmetic. It reproduces
CPU v2 and MPS 7 stages (LN1/QKV/attention) and MLX 15 stages (also FF1).
Its prospective CPU source test requires byte-equal output and every parameter
gradient against v2 on changed tokens, with only FP32 floating-point saved state,
at each width selected by the hosted six-width CI loop. This test is separate
from actual installed-package, GPU, historical-state and optimizer validation.

`validate_runtime.py` and `audit_runtime.py` prepare a separate all-width primitive
execution/audit route under an explicitly captured package selection. They retain
both FP64 references and raw FP32 outputs/derivatives, exact repeats, changed
inputs, noncontiguous operands and rejected malformed inputs. CPU/MPS use saved
state hooks; MLX records its actual compiled graphs and changed-input trace proof
without claiming a saved-tensor hook. No real package/device execution has yet
occurred. Prototype tests of the gate and both independent oracles run on hosted
CI for all six widths; they cannot confer captured-package qualification.
