# Reproduce the attention-precision study

V6 measures all six widths with the same dataset and training budget in **72
matched fits**. Wide native attention uses selective FP64 accumulation in forward
and backward while model and optimizer state remain stored in FP32. The
[protocol](../experiments/PROTOCOL-v6.md) fixes tests, thresholds, execution, and
reporting before measurement.

The post-run report separates evidence integrity, each engine's accuracy against
independent FP64 equations, and mutual FP32 compatibility. It continues through
all widths when numerical requirements fail and reports `baseline_precision_limit`
or `native_precision_limit` explicitly. A zero report-process exit means complete
intact evidence, **not necessarily numerical qualification**: inspect the named
flags and status. Original v5 failures remain available through the
[historical reproduction guide](REPRODUCE-v5.md).

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
git checkout 2c5191047068c16b12e3312652b118d48a5f3744
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
  python -m transformer.validate --output results/preflight-v6/width-$width.json \
    --runtime --work-dir results/smoke-v6/width-$width
  ARM64_TRANSFORMER_BACKEND=neon python -m transformer.validate \
    --output results/preflight-v6/width-$width-neon.json \
    --runtime --work-dir results/smoke-v6/width-$width-neon
  supported=$(python -c "import json; print(json.load(open('results/preflight-v6/width-$width.json'))['backend']['sme_supported'])")
  if [ "$supported" = 1 ]; then
    ARM64_TRANSFORMER_BACKEND=sme python -m transformer.validate \
      --output results/preflight-v6/width-$width-sme.json \
      --runtime --work-dir results/smoke-v6/width-$width-sme
  fi
done
for width in 32 64 128 256 512 1024; do
  ARM64_TRANSFORMER_WIDTH=$width python -m transformer.prepare \
    --source data/tinyshakespeare.txt --output results/inputs-v6/width-$width
done
unset ARM64_TRANSFORMER_WIDTH
python -m transformer.sweep --protocol v6 --inputs results/inputs-v6 \
  --preflight-dir results/preflight-v6 --output results/width-sweep-v6
python -m transformer.sweep_report --inputs results/inputs-v6 \
  --results results/width-sweep-v6 --output results/report-v6
```

The measured source must be committed and clean. Use AC power and do not run
builds, validation, profiling, or other benchmarks concurrently with timing. The
coordinator authenticates the source, all six admitted binaries/layouts/gates,
and measured fixtures around each child. It uses one compute thread, records
conditions, pauses 15 seconds between fits, and preserves failed campaigns.

The [v6 protocol](../experiments/PROTOCOL-v6.md) distinguishes strict comparisons
at common model/optimizer states from diagnostics of independently evolving
training trajectories. The original failed wide-model trajectory checks remain
in the numerical investigation; their tolerances were not raised. A passing v6
gate must identify its validation contract and retain those diagnostic outcomes.
The [native contract](../native/ABI.md) and [kernel notes](../native/kernel_notes.md)
describe the numerical interfaces and hardware paths.

The report independently recomputes predictions from saved selected checkpoints,
checks selected validation losses and exact within-implementation repeats, and
retains every timing observation. It never overwrites an existing destination.
Archive packaging verifies source, inputs, binaries, raw runs and report identities
and rehashes the streamed payload. Release checksums authenticate the downloadable
bytes separately from the numerical correctness checks.


Package the complete saved evidence, including failed numerical comparisons:

```sh
python scripts/package_sweep_evidence.py --help
```

The packager supports v6 numerical-limit outcomes only when evidence-integrity
checks pass and all qualification flags agree with the complete comparison
records. It preserves exact measured source, gates, binaries, inputs, checkpoints,
moments, predictions, reports, and checksums. It does not rerun models.
