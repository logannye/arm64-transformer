# Scripts

Standalone tools that sit beside the studies: development checks, profiling,
reports, figures and evidence packaging. None of them is part of a measured
fit. The study runners are in [transformer/](../transformer/README.md) and
[integration/](../integration/README.md).

Run them from the repository root as `python scripts/<name>.py`; the ones that
import the package need `PYTHONPATH=.` or an editable install. The descriptions
below follow each script's own docstring where it has one.

Two cautions. The packaging scripts, and several of the report scripts, read a
study's raw evidence, which is not in the repository. The figure scripts write
into `docs/` and `experiments/`, where committed images are recorded by hash in
evidence summaries; `draw_architecture.py` and `draw_kernel_tiling.py` take no
arguments and overwrite their image when run.

## Checks

| Script | What it does |
|---|---|
| [check_native_build_identity.py](check_native_build_identity.py) | Builds the native sources of the measured commit and of the working tree in temporary directories and compares the SHA-256 of every output. See [native/](../native/README.md#relationship-to-the-measured-source). |
| [check_gpu_repeatability.py](check_gpu_repeatability.py) | Runs the bounded GPU development repeatability diagnostic in fresh processes. |
| [check_metal_attention.py](check_metal_attention.py) | Runs every development attention case on Metal, retaining failures without tuning gates. |
| [check_metal_development.py](check_metal_development.py) | Smoke-checks Metal linear, attention and LayerNorm against FP64 references at the selected width. |
| [check_metal_norm_qkv.py](check_metal_norm_qkv.py) | Development-only Metal LayerNorm and QKV probes; never run alongside GPU timing. |
| [check_metal_qkv_optimization.py](check_metal_qkv_optimization.py) | Development-only qualification of the ordered, protected QKV contractions. |
| [audit_gpu_development.py](audit_gpu_development.py) | Audits saved development-fit logits against independent FP64 equations. |
| [capture_gpu_environment.py](capture_gpu_environment.py) | Authenticates the installed GPU development dependencies without importing them. |

## Profiling and microbenchmarks

| Script | What it does |
|---|---|
| [benchmark_kernels.py](benchmark_kernels.py) | Measures synthetic native kernels and PyTorch forward operations on one CPU thread. |
| [profile_native.py](profile_native.py) | Samples a native training process on seeded synthetic data, outside benchmarks. |
| [profile_gpu.py](profile_gpu.py) | Serial, development-only Metal profiling; never opens held-out or admission data. |

## Reports

| Script | What it does |
|---|---|
| [compare_versions.py](compare_versions.py) | Plots two audited campaigns without treating their timings as a paired study; drew `experiments/v1-v2.png`. |
| [publish_sweep_view.py](publish_sweep_view.py) | Creates a compact repository view of an unchanged, complete v6 report. |
| [report_failed_sweep.py](report_failed_sweep.py) | Post-study descriptive reporting for the failed v5 checkpoint audit. |
| [report_kernel_optimization.py](report_kernel_optimization.py) | Renders the separately scoped synthetic v4 profiling evidence. |
| [report_gpu_v2_engineering.py](report_gpu_v2_engineering.py) | Reports retained Metal engineering diagnostics; never executes model arithmetic. |
| [summarize_gpu_v2_baselines.py](summarize_gpu_v2_baselines.py) | Summarizes completed baseline diagnostics from saved arrays; runs no model. |
| [report_gpu_qkv.py](report_gpu_qkv.py) | Validates and reports saved QKV development evidence, without model execution. |

## Figures

| Script | What it does |
|---|---|
| [draw_architecture.py](draw_architecture.py) | Draws the static architecture diagram, `docs/architecture.png`. |
| [draw_kernel_tiling.py](draw_kernel_tiling.py) | Draws the four-token, four-output NEON register tile, `docs/kernel-tiling.png`. |
| [render_sme_dataflow.py](render_sme_dataflow.py) | Renders the SME matrix dataflow, `docs/sme-dataflow.png`. |
| [draw_width_summary.py](draw_width_summary.py) | Renders a publication layout from an audited sweep summary. |
| [render_width_overview.py](render_width_overview.py) | Plots separate campaigns together; a failed audit must be labeled as descriptive. |
| [render_gpu_development.py](render_gpu_development.py) | Renders the retained precision-development inventory; no model or GPU execution. |

The figures in the root README come from a separate renderer,
[docs/render_readme_figures.py](../docs/render_readme_figures.py).

## Evidence packaging

| Script | What it does |
|---|---|
| [package_evidence.py](package_evidence.py) | Packages and independently checksums a completed matched study for release. |
| [package_sweep_evidence.py](package_sweep_evidence.py) | Streams a completed width sweep into a deterministic, verified release archive. |
| [package_failed_sweep.py](package_failed_sweep.py) | Archives an authenticated, completed sweep whose post-run numerical audit failed. |
| [package_gpu_development.py](package_gpu_development.py) | Archives GPU development evidence without claiming formal admission. |
| [package_gpu_v2_engineering.py](package_gpu_v2_engineering.py) | Streams and verifies a split archive of the retained, blocked GPU v2 engineering work. |
| [package_gpu_qkv.py](package_gpu_qkv.py) | Packages the GPU v3 QKV development release after every evidence writer has stopped. |
