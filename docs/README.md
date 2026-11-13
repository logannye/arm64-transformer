# Docs

Reproduction guides, figures and pointers to the design notes. The design notes
themselves sit beside the code they describe.

## Reproduction guides

Each guide belongs to one study. They are not interchangeable: the studies ran
under different protocols, baselines and source revisions.

| Guide | Study | What it covers |
|---|---|---|
| [REPRODUCE-five-engine.md](REPRODUCE-five-engine.md) | The accepted five-engine phase, [gpu-v5-library-phase1](../experiments/gpu-v5-library-phase1/results/README.md) | What can be checked from the public files, with a command that recomputes the headline ratios, and what a full rerun would need. |
| [REPRODUCE.md](REPRODUCE.md) | The two-engine attention-precision study against stock PyTorch 2.14.1, [width-sweep-v6](../experiments/width-sweep-v6/README.md) | A full rerun: six widths, 72 fits, pinned to that study's commit. |
| [REPRODUCE-v5.md](REPRODUCE-v5.md) | The wider two-engine study, [width-sweep-v5](../experiments/width-sweep-v5/README.md) | A rerun of widths 512 and 1024, 24 fits. That candidate failed its trained-checkpoint audit. |
| [experiments/gpu-v3-qkv/REPRODUCE.md](../experiments/gpu-v3-qkv/REPRODUCE.md) | The protected-QKV Metal development experiment, [gpu-v3-qkv](../experiments/gpu-v3-qkv/README.md) | The QKV kernel search and its report. |
| [experiments/upstream-integration-v2/REPRODUCTION.md](../experiments/upstream-integration-v2/REPRODUCTION.md) | The framework integration work, [upstream-integration-v2](../experiments/upstream-integration-v2/README.md) | The GPU engineering evidence, with its [public evidence catalog](../experiments/upstream-integration-v2/PUBLICATION.md). |

## Design notes

| Note | Subject |
|---|---|
| [native/README.md](../native/README.md) | Map of the assembly sources, the build and the runtime switches. |
| [native/ABI.md](../native/ABI.md) | The native contract: every public function and the executable's file formats. |
| [native/kernel_notes.md](../native/kernel_notes.md) | The CPU kernels: layout, SME dispatch, tiling and arithmetic. |
| [native/metal/README.md](../native/metal/README.md) | The handwritten Metal backend. |
| [native/metal/OPTIMIZED-KERNELS.md](../native/metal/OPTIMIZED-KERNELS.md) | The optimized Metal kernels and their dispatch contract. |
| [native/metal/QKV-KERNELS.md](../native/metal/QKV-KERNELS.md) | The protected QKV contractions. |
| [experiments/README.md](../experiments/README.md) | Every study in order, with its protocol and outcome. |

## Figures

| File | What it shows | Source |
|---|---|---|
| [figures/](figures/README.md) | The two diagrams in the root README, as PNG and SVG. The performance chart beside them is in the accepted study's results folder. | [render_readme_figures.py](render_readme_figures.py), from retained summaries. |
| [sme-dataflow.png](sme-dataflow.png) | The SME matrix dataflow, linked from the root README. | [scripts/render_sme_dataflow.py](../scripts/render_sme_dataflow.py); [sme-dataflow.json](sme-dataflow.json) records the image and renderer hashes. |
| [sme-dataflow-v4.png](sme-dataflow-v4.png) | The earlier v4 version of that diagram, kept as the historical image. | Recorded in `sme-dataflow.json`. |
| [precision-dataflow.png](precision-dataflow.png) | Selective FP64 precision through attention, shown in the [v6 numerics report](../experiments/numerics-v6/README.md). | Not named by any script here. |
| [architecture.png](architecture.png) | The static architecture diagram from the two-engine studies. | [scripts/draw_architecture.py](../scripts/draw_architecture.py) |
| [kernel-tiling.png](kernel-tiling.png) | The NEON register tile of the dense kernels. | [scripts/draw_kernel_tiling.py](../scripts/draw_kernel_tiling.py) |
| [width-overview.png](width-overview.png) | Separate two-engine campaigns plotted across widths. | [scripts/render_width_overview.py](../scripts/render_width_overview.py); [width-overview.json](width-overview.json) records its sources. |

Several of these images and JSON files are linked from no other page. They stay
where they are because evidence summaries under `experiments/` record them by
path and hash; moving or regenerating one would break that record.
