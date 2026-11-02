# README figures

The diagrams and chart share an engine palette and export to PNG and SVG.
Numerical inputs come from retained reports; rendering does not execute model,
training, qualification or audit workers.

| Figure | Source | Vector download |
|---|---|---|
| Model and engines | [Reference equations](../../transformer/reference.py) and [frozen selections](../../experiments/gpu-v5-library-phase1/results/summary.json) | [Model and engines](model-and-engines.svg) |
| Accepted performance | The 20 engine/width medians in the [accepted summary](../../experiments/gpu-v5-library-phase1/results/summary.json) | [Performance](../../experiments/gpu-v5-library-phase1/results/performance.svg) |
| ReLU precision diagnostic | Recorded preactivation values and failure count in the [historical localization report](../../experiments/trained-precision-v1/mlx-branch-localization.json) | [Precision gate](precision-gate.svg) |

The ReLU curve illustrates its backward multiplier; its two marked inputs are
observations from the historical diagnostic. That unselected width-1024 case
is development evidence, not part of the accepted four-width formal results.
Performance labels at the right of each curve show width-256 values. The full
curves use the retained six-fit medians without smoothing or refitting.

To regenerate, use an existing Python environment with the project's `report`
dependencies installed, from the repository root:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=work/readme-mpl python docs/render_readme_figures.py
```

The [renderer](../render_readme_figures.py) writes only these publication images.
The measurement data, historical evidence, experiment and validation harness are
unchanged by rendering.
