# Native code

The ARM64 engine: the complete model, its backward pass, Adam and a standalone
training executable, written by hand in Darwin AArch64 assembly. Nothing here
calls BLAS, Accelerate or a C math library. The handwritten Metal engine lives
in [metal/](metal/README.md).

Three notes carry the detail this page leaves out:

- [ABI.md](ABI.md) is the contract: every public function, the backend
  policy, the model entry points and the executable's file formats.
- [kernel_notes.md](kernel_notes.md) explains the numerical implementation:
  layout, SME dispatch, tiling, width specialization and arithmetic.
- [metal/README.md](metal/README.md) covers the Metal host and shaders, with
  [OPTIMIZED-KERNELS.md](metal/OPTIMIZED-KERNELS.md) and
  [QKV-KERNELS.md](metal/QKV-KERNELS.md) for the kernel designs.

## Source files

| File | What it holds |
|---|---|
| [kernels.S](kernels.S) | The generic primitives the optimized paths fall back to: linear, LayerNorm, attention and cross entropy, plus ReLU, Adam, `tf_exp` and `tf_log`. |
| [linear.S](linear.S) | NEON dense layers: forward and backward matrix tiles of four rows by sixteen columns held in registers. Shapes that do not tile tail-call the generic kernels. |
| [linear_sme.S](linear_sme.S) | SME dense layers for the 32-token projections: operand packing, then forward and backward products accumulated in four ZA tiles. |
| [backend.S](backend.S) | Backend selection. Detects SME, fixes the policy once per process, and provides the public `tf_linear_forward` and `tf_linear_backward` dispatchers with their call counters. |
| [attention.S](attention.S) | Causal self-attention, forward and backward: the width-32 path, unrolled blocks for wider heads, and FP64 scores at widths 512 and 1024. |
| [softmax.S](softmax.S) | The four-lane exponential `tf_exp4` and the ASCII-128 cross entropy with its gradient. |
| [normalization.S](normalization.S) | Four-wide LayerNorm, forward and backward. |
| [precision.S](precision.S) | Scalar FP64 versions of LayerNorm and the linear layer behind FP32 interfaces, and `tf_precision_policy`, which reports the compiled policy. Always assembled; called only when a precision policy selects them, at widths 512 and 1024. |
| [model.S](model.S) | The complete one-block transformer: embeddings, `tf_forward`, `tf_backward`, and the `tf_scale` and `tf_finite` helpers. |
| [runtime.S](runtime.S) | `main` for the standalone executable: reads the binary fixtures, trains, selects the best checkpoint, evaluates, times inference and writes the results. Its libc calls are limited to process, memory, files and clocks. |

## Building

The build needs an Apple silicon Mac and Xcode Command Line Tools whose
assembler accepts SME instructions.

```sh
make -C native widths PYTHON=python        # all six widths
make -C native WIDTH=64 all                # one width
make -C native WIDTH=1024 PRECISION=15 all # one width, one precision policy
make -C native WIDTH=64 clean
```

Model dimensions are assembly-time constants, so each width gets its own
binaries. The [Makefile](Makefile) has three targets: `all` builds one width
(`WIDTH`, default 32), `widths` builds 32, 64, 128, 256, 512 and 1024 in turn,
and `clean` removes one width's objects and binaries. Everything lands in the
git-ignored `native/build/width-W/`:

- nine objects, linked into `libtransformer.dylib`, which Python loads through
  `ctypes`;
- `runtime.o`, linked with those nine into the executable `transformer-arm64`.

Two generated headers come first:

- `layout.inc` holds the dimensions, the parameter count and the byte offset of
  every parameter and workspace tensor for that width. It is written by
  `python -m transformer.spec`, which is why the build takes a `PYTHON`. The
  same module gives the Python side its layout.
- `precision.h` defines `TF_PRECISION_POLICY` from `PRECISION`, an integer from
  0 to 15 (default 0); any other value stops the build. The bits select FP64
  arithmetic for one stage each: 1 the first LayerNorm, 2 the QKV projection,
  4 the second LayerNorm, 8 the first feed-forward projection. The policy takes
  effect only in binaries of width 512 or 1024. The accepted results, at widths
  32–256, use policy 0.

The compiler is `xcrun clang -arch arm64 -O2`. Only `linear_sme.S` and
`backend.S` are assembled with `-march=armv9-a+sme`; no other file is assembled
with SME enabled.

## Runtime switches

| Variable | Read by | Effect |
|---|---|---|
| `ARM64_TRANSFORMER_WIDTH` | [transformer/spec.py](../transformer/spec.py), at import | Selects which width's layout and binaries the Python side uses: 32 (default), 64, 128, 256, 512 or 1024. Anything else raises an error. Set it before Python starts. |
| `ARM64_TRANSFORMER_BACKEND` | [backend.S](backend.S), once at initialization | Unset or `auto`: SME when the required capability is present, otherwise NEON. `neon`: NEON, including its generic fallbacks. `sme`: require the SME capability; unsupported shapes still use NEON. Any other value, or `sme` without the capability, fails initialization. |

The backend choice cannot change within a process. SME needs Darwin to report
the feature and a 64-byte streaming vector length; it serves the projections of
the width-64 and wider models, and the width-32 model uses NEON.
[ABI.md](ABI.md#backend-policy-and-execution-state) has the exact shapes, stack
requirements and the counters that record which path actually ran.

## Relationship to the measured source

The accepted five-engine measurements were taken with these files as they stood
at commit `bd2cd2823038729779a217ee5b084fa06191d3a9`, the `math` source revision
in the [accepted summary](../experiments/gpu-v5-library-phase1/results/summary.json).
Later commits to these files add comments only. Every object file, the library
and the executable build byte-identical to that commit at all six widths.

The check is:

```sh
python scripts/check_native_build_identity.py
```

It builds the measured commit's sources and the working tree's sources in
temporary directories and compares the SHA-256 of every output. It needs a full
clone, so that the measured commit is available, and the Xcode Command Line
Tools.
