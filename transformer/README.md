# The `transformer` package

The Python side of the project: the model's layout, the PyTorch reference, the
bindings and audits for the native code, the GPU engines, and the runners that
produced each study in [experiments/](../experiments/README.md). The native
engine itself is assembly, in [native/](../native/README.md).

Modules with a command line run as `python -m transformer.<name>`. The model
width is fixed per process: set `ARM64_TRANSFORMER_WIDTH` before Python starts.

Most of this package is experiment control: it authenticates the source,
runtime and inputs a study runs on. A reader who wants the model should start
with the first two groups.

## Model, layout and data

| Module | Role |
|---|---|
| [spec.py](spec.py) | The model layout for the selected width: dimensions, parameter order and byte offsets, with no model math. Also writes the assembly's `layout.inc`. |
| [reference.py](reference.py) | The ordinary batched PyTorch reference for the fixed transformer. |
| [demo.py](demo.py) | Informal side-by-side run: trains the model once with the assembly engine and once with stock PyTorch and prints both times. |
| [fetch_data.py](fetch_data.py) | Fetches and verifies the pinned public text source. |
| [prepare.py](prepare.py) | Prepares the matched binary inputs both engines read and verifies their format. |

## Native engine: bindings and audits

| Module | Role |
|---|---|
| [validate.py](validate.py) | The `ctypes` binding to the native library, and independent numerical and executable audits of it. Never builds the native code. |
| [backend.py](backend.py) | Authenticates the native backend policy and the dense-kernel dispatch that was actually observed. |
| [audit.py](audit.py) | Independent checks of saved run artifacts and language-model quality. |
| [runtime_replay.py](runtime_replay.py) | Independent smoke-training control loops; not a timing or training entry point. |

## Two-engine CPU studies (v1–v6)

| Module | Role |
|---|---|
| [python_train.py](python_train.py) | Runs the standard batched PyTorch implementation on one matched fixture. |
| [run.py](run.py) | Executes the frozen twelve-run comparison after its preflight gates. |
| [report.py](report.py) | Audits all twelve matched runs and writes the result report. |
| [sweep.py](sweep.py) | Coordinates a width study under a named protocol; never builds or prepares fixtures. |
| [sweep_report.py](sweep_report.py) | Audits every fit a width-sweep protocol declares and writes per-width and aggregate reports. |

## Precision studies

| Module | Role |
|---|---|
| [precision.py](precision.py) | Independent FP64 attention equations and the v6 stress fixtures. |
| [precision_audit.py](precision_audit.py) | v6 checkpoint accuracy and compatibility evidence. |
| [torch_precision_probe.py](torch_precision_probe.py) | Explicit CPU diagnostic substitutions; never a standard-framework baseline. |
| [trained_precision.py](trained_precision.py) | Development replay of exposed trained states; never admission or timing. |

## GPU engines and development tools

| Module | Role |
|---|---|
| [gpu_contract.py](gpu_contract.py) | The shared host interface for the GPU engines. |
| [gpu_metal.py](gpu_metal.py) | Adapter for the handwritten Metal backend; the host performs no tensor math. |
| [gpu_mps.py](gpu_mps.py) | Standard eager PyTorch MPS baseline, with no CPU tensor-math fallback. |
| [gpu_mlx.py](gpu_mlx.py) | Compiled standard MLX baseline with explicit state and full FP32 inputs. |
| [gpu_mlx_compile.py](gpu_mlx_compile.py) | Fail-closed MLX compilation check, importable without loading any ML framework. |
| [gpu_policy.py](gpu_policy.py) | Immutable Metal dispatch policies. |
| [gpu_environment.py](gpu_environment.py) | The environment controls for GPU runs, including variables that must be absent. |
| [gpu_inputs.py](gpu_inputs.py) | Reads existing fixtures without importing a model or GPU runtime. |
| [gpu_prepare.py](gpu_prepare.py) | Creates tiny GPU development fixtures from CPU training rows only. |
| [gpu_precision.py](gpu_precision.py) | CPU FP64 oracles and GPU development stress operands. |
| [gpu_run.py](gpu_run.py) | Exercises the shared GPU fit and output contract on training-only development data. |
| [gpu_repeatability.py](gpu_repeatability.py) | Bounded, development-only GPU repeatability diagnostics. |
| [gpu_optimize.py](gpu_optimize.py) | Bounded Metal development profiling and policy selection. |
| [gpu_qkv_optimize.py](gpu_qkv_optimize.py) | The prespecified, precision-preserving QKV development search. |
| [gpu_admission_checks.py](gpu_admission_checks.py) | Raw-array producers for GPU admission checks. |

## Five-engine study control

| Module | Role |
|---|---|
| [gpu_study.py](gpu_study.py) | The frozen five-engine control plane; never builds and never waives admission. |
| [gpu_study_context.py](gpu_study_context.py) | Immutable namespace selection for the standard and project-owned studies. |
| [gpu_study_runtime.py](gpu_study_runtime.py) | Discovers the executing Python installation without importing a GPU framework. |
| [gpu_study_case_inventory.py](gpu_study_case_inventory.py) | Metadata for the GPU stress inventory, with no numerical imports. |
| [gpu_study_admission.py](gpu_study_admission.py) | The exact admission inventory and independently evaluated raw-array receipts. |
| [gpu_study_cpu_admission.py](gpu_study_cpu_admission.py) | Admission-only bridge into the unchanged CPU v6 arithmetic, in an isolated process. |
| [gpu_study_worker.py](gpu_study_worker.py) | One authenticated formal fit. |
| [gpu_study_replay.py](gpu_study_replay.py) | Isolated checkpoint replay; also runs against the immutable CPU source tree. |
| [gpu_study_audit.py](gpu_study_audit.py) | The all-origin, all-engine checkpoint audit, separate from measured execution. |
| [gpu_study_report.py](gpu_study_report.py) | Reports all 180 planned cells, including a zero-fit outcome. |

## Project-maintained framework route

The accepted five-engine phase ran PyTorch and MLX builds maintained by this
project. Its driver is
[integration/library-study-v1](../integration/library-study-v1/README.md), which
calls into this package.

| Module | Role |
|---|---|
| [gpu_owned_study.py](gpu_owned_study.py) | Plans and inspects the independently maintained five-engine study. |
| [gpu_owned_command.py](gpu_owned_command.py) | The owned-study entry point, with explicit authenticated import roots. |
| [gpu_owned_backend.py](gpu_owned_backend.py) | Provenance-only adapters for the authenticated project-maintained frameworks. |
| [gpu_owned_origins.py](gpu_owned_origins.py) | Authenticates loaded framework modules without importing or executing a framework. |
| [gpu_owned_eligibility.py](gpu_owned_eligibility.py) | Authenticates framework eligibility from full-suite raw evidence. |
| [gpu_owned_admission.py](gpu_owned_admission.py) | Inventory of the frozen-CPU calls that owned admission makes. |
| [gpu_owned_cpu.py](gpu_owned_cpu.py) | Provenance launcher for the unchanged frozen CPU Python entry points. |
| [gpu_owned_repeatability.py](gpu_owned_repeatability.py) | Owned-runtime development repeatability; never formal admission. |
| [gpu_library_study.py](gpu_library_study.py) | The maintained-library successor selection; carries no execution authority. |
| [library_metal.py](library_metal.py) | The precision-corrected reference Metal backend. |
| [library_training.py](library_training.py) | Persistent models for the selected maintained-library candidate. |
