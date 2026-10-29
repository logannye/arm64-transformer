# Reproduce the wider transformer study

V5 times widths 512 and 1024. It requires current correctness gates at all six
supported widths before launching its 24 matched fits. V3 and v4 retain their
original four-width protocols and published evidence.

The published v5 candidate passed synthetic admission but failed the later
trained-checkpoint audit. Reproducing that candidate should retain the failures;
a nonzero report exit is an experimental result. The [outcome report](../experiments/width-sweep-v5-outcome/README.md)
separates descriptive measurements from numerical acceptance. Its reporting and
archive tools were added after measurement and never change failed flags.

Use an Apple silicon Mac with Xcode Command Line Tools that can assemble SME.
The measured machine is an M4 Max. Runtime feature checks select SME only when
the OS and 64-byte streaming vector length support it; older hardware uses NEON.
The largest dense call needs more than a 512 KiB worker stack. The standalone
runtime and these validation commands execute on the main thread.

Python reference tests also run on Linux. Use prebuilt Python packages; the
recorded versions are in `requirements-benchmark.txt`.

```sh
git clone https://github.com/logannye/arm64-transformer.git
cd arm64-transformer
git checkout c12874c2fe6af3fd1612046950aa56d756f4d0a0
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: -r requirements-benchmark.txt
python -m pip install --no-deps -e .
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0
python -m transformer.fetch_data --output data/tinyshakespeare.txt
make -C native widths PYTHON=python
```

The pinned package versions describe the measured macOS environment. Other hosts
may require compatible prebuilt wheels; record their versions and treat their
measurements as a new campaign. A compatible installed environment can be reused.

Run each backend policy in a separate process. The native library captures its
policy at initialization, so changing an environment variable after loading it is
not supported. Every output path below must be fresh.

```sh
for width in 32 64 128 256 512 1024; do
  export ARM64_TRANSFORMER_WIDTH=$width
  export ARM64_TRANSFORMER_BACKEND=auto
  python -m pytest -q
  python -m transformer.validate --output results/preflight-v5/width-$width.json \
    --runtime --work-dir results/smoke-v5/width-$width
  ARM64_TRANSFORMER_BACKEND=neon python -m transformer.validate \
    --output results/preflight-v5/width-$width-neon.json \
    --runtime --work-dir results/smoke-v5/width-$width-neon
  supported=$(python -c "import json; print(json.load(open('results/preflight-v5/width-$width.json'))['backend']['sme_supported'])")
  if [ "$supported" = 1 ]; then
    ARM64_TRANSFORMER_BACKEND=sme python -m transformer.validate \
      --output results/preflight-v5/width-$width-sme.json \
      --runtime --work-dir results/smoke-v5/width-$width-sme
  fi
done
for width in 512 1024; do
  ARM64_TRANSFORMER_WIDTH=$width python -m transformer.prepare \
    --source data/tinyshakespeare.txt --output results/inputs-v5/width-$width
done
unset ARM64_TRANSFORMER_WIDTH
python -m transformer.sweep --protocol v5 --inputs results/inputs-v5 \
  --preflight-dir results/preflight-v5 --output results/width-sweep-v5
python -m transformer.sweep_report --inputs results/inputs-v5 \
  --results results/width-sweep-v5 --output results/report-v5
```

The measured source must be committed and clean. Use AC power and do not run
builds, validation, profiling, or other benchmarks concurrently with timing. The
coordinator authenticates the source, all six admitted binaries/layouts/gates,
and measured fixtures around each child. It uses one compute thread, records
conditions, pauses 15 seconds between fits, and preserves failed campaigns.

The [v5 protocol](../experiments/PROTOCOL-v5.md) distinguishes strict comparisons
at common model/optimizer states from diagnostics of independently evolving
training trajectories. The original failed wide-model trajectory checks remain
in the numerical investigation; their tolerances were not raised. A passing v5
gate must identify its validation contract and retain those diagnostic outcomes.
The [native contract](../native/ABI.md) and [kernel notes](../native/kernel_notes.md)
describe the numerical interfaces and hardware paths.

The report independently recomputes predictions from saved selected checkpoints,
checks selected validation losses and exact within-implementation repeats, and
retains every timing observation. It never overwrites an existing destination.
Archive packaging verifies source, inputs, binaries, raw runs and report identities
and rehashes the streamed payload. Release checksums authenticate the downloadable
bytes separately from the numerical correctness checks.

The original successful-study reporter stops at the first failed width. To retain
both widths' failures, the published outcome also ran the unchanged per-width
reporter separately at width 1024. The publication revision's
`scripts/report_failed_sweep.py` then authenticated all saved evidence and computed
descriptive statistics without new model execution. Its separate
`scripts/package_failed_sweep.py` archives that failed outcome and explicitly sets
`numerical_validation_passed` to false. The successful-study packager still rejects
failed audits. The release includes the exact post-study tools and measured source
separately, plus instructions for reconstructing the archive if delivered in parts.
