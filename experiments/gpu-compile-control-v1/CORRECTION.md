# GPU baseline environment correction

Two environment controls were interpreted incorrectly in earlier development runs. This correction preserves all historical evidence and requires fresh qualification of the intended baselines. No formal study admission or fit has occurred.

| Control | Historical setting | Actual effect | Prospective requirement |
|---|---|---|---|
| `MLX_DISABLE_COMPILE` | String `"0"` | Its presence disables MLX graph compilation | Variable absent |
| `PYTORCH_MPS_PREFER_METAL` | String `"0"` | Its presence forces Metal matrix-multiplication dispatch | Variable absent, preserving ordinary framework dispatch |

MLX 0.32.4 checks whether `MLX_DISABLE_COMPILE` exists, without parsing its value. When disabled, its compile wrapper returns the original function. The project's original MLX backend called `mx.compile` but never overrode this global mode, so the affected fresh-process runs executed with graph compilation disabled. A capability field that simply stated `compiled: true` did not establish actual compilation. [Pinned stable MLX source](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/compile.cpp#L203).

The installed historical PyTorch 2.14.1 identifies source revision `5c4886908584029761b579af026dcfb627c84070` in `torch/version.py`. That exact revision checks the presence of `PYTORCH_MPS_PREFER_METAL` in its matrix-multiplication dispatcher. The historical value therefore selected forced Metal dispatch, even though the experiment intended the ordinary dispatcher. [Exact installed-version source](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/operations/LinearAlgebra.mm#L497).

The other four controls have different semantics. `PYTORCH_ENABLE_MPS_FALLBACK=0` and `PYTORCH_MPS_FAST_MATH=0` correctly disable those options in the installed PyTorch revision. MLX's `MLX_ENABLE_TF32=0` and `MLX_SDPA_VJP_FALLBACK=0` use integer-valued controls; their zero settings remain appropriate. The prospective change preserves those four zero values. [PyTorch fallback](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/mps/MPSFallback.mm#L91), [PyTorch fast math](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/OperationUtils.mm#L789), [MLX TF32](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/utils.h#L208), [MLX attention fallback](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/scaled_dot_product_attention.cpp#L1532).

## Scope of the historical correction

The incorrect settings entered the GPU backend and runner at project commit `d9cf70a`, then propagated into the GPU v2 repeatability and prospective study environments. Historical GPU-development-v1/v2 baseline results establish behavior for **graph-disabled MLX and forced-Metal PyTorch MPS**. They do not establish repeatability, accuracy, or performance for the intended compiled-MLX/ordinary-MPS pair. Existing failures remain failures for their recorded execution configurations; no absent result becomes a pass.

The isolated patched-MLX full-model regression and the first independent 18-shape FP64 diagnostic inherited `MLX_DISABLE_COMPILE="0"` without calling `enable_compile`. Their retained accuracy and exact-replay observations apply to graph-disabled execution. They need fresh runs to support compiled-execution claims. Separately, the retained attention microbenchmark explicitly calls `mx.enable_compile`; its compiled cases are not subject to this specific disabled-mode inference. Runtime Metal shader JIT and Python graph compilation are distinct settings.

Historical CPU measurements and custom Metal-only kernel measurements do not execute MLX or MPS graphs and are unaffected by these two dispatch controls. Their existing precision limits and study boundaries remain unchanged. Raw states, failures, source snapshots, released reports, and old manifests are preserved, including the old QKV reporter's original control contract. This document supersedes misleading baseline configuration descriptions without rewriting the underlying records.

## Prospective execution safeguards

The central environment contract records both required absences as JSON `null`. Launchers remove these keys from fresh child environments; direct backend imports and worker checks reject their presence, including empty strings and `"0"`. Removing a variable after a framework has cached its mode is insufficient, so fresh workers remain mandatory.

MLX backend initialization now executes a GPU compilation probe before constructing the model. Four evaluations with changing input values must share one Python trace, retain a `Compiled` graph primitive, and produce exact dyadic FP32 results. The probe does not call `enable_compile` to conceal a cached disabled state. Its graph, trace count, and output hashes enter the capability receipt; admission requires this evidence. Existing tolerances are unchanged.

The prospective repeatability and five-engine protocol identifiers advance to `gpu-v2-standard-baseline-repeatability-v2` and `gpu-v2-five-engine-v2`. Old environment receipts cannot satisfy these contracts. CPU-only negative controls test forbidden values, child-environment removal, stale manifests, eager execution, missing compiled primitives, incorrect outputs, and unavailable GPU execution. They do not constitute real-device numerical qualification.

Fresh baseline diagnostics, required framework repairs in eligible standard releases, a new bound freeze, and complete admission remain prerequisites for the 180-fit campaign. The correction changes neither the selected engines nor the strict repeatability requirement. Admission stream `20261104` remains unopened.
