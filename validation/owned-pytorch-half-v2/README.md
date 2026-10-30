# Combined owned PyTorch candidate, build recipe v2

This prospective candidate combines the previous 12-file embedding, layout and
ordered FP16 indexing patches with the two-header explicit Half storage proposal.
It preserves every earlier candidate, failed build, source tree and test result.
The new 14-file patch has its own identity; prior binary qualification does not
transfer. It is development work, not numerical admission or a formal study.

Two build-check repairs follow observed failures. The previous AST fingerprint
used `ast.dump`, whose empty-field formatting differs between Python 3.12 and
3.14. This recipe hashes canonical structural fields and independently pins
exact source bytes. Bodies, decorators, arguments, constants and nonempty type
parameters remain protected. The old retained compiler database also shows that
Objective-C++ omitted `-g0`: this recipe sets CFLAGS, CXXFLAGS, OBJCFLAGS and
OBJCXXFLAGS and checks all four CMake variables plus every changed translation
unit. It does not relax the debug-information check.

The explicit Half conversion patch makes the already existing FP32-to-FP16
storage conversions explicit. Before building the complete framework, hosted
C++17 probes compile against both original and proposed headers. They retain
result bits, NaN payloads and floating-point exception flags for all 65,536 Half
encodings, 16 fixed arithmetic partners, midpoint neighbors, complex operations
and self-aliasing. Size/alignment and public return types have static assertions.
Both optimized assembly listings and binary outputs are retained. This is ARM64
CPU evidence only; CUDA/HIP/SYCL/x86 and the full upstream C++ test suite remain
separate requirements. The unchanged authoritative `spin lint` remains required,
including its C++17 compatibility and formatter checks; no suppressions are added.

The complete framework is built only by the inherited editable-install command
on hosted macOS ARM64, using one source tree, one environment, two workers,
Release and no debug information. The original 45 dependency wheel identities,
40 GiB initial free-space threshold, 8 GiB reserve, 60 GiB allocation bound and
360-minute limit remain. Source/runtime archives, failures, compiler arguments,
objects and logs are retained. Local compilation is not used.

All 10,353 original MPS test IDs, 744 skips and 382 expected failures remain
bound, plus the same four new tests: 10,357 total. The only permitted old test
body change remains the reviewed direct-FP16 indexing expectation. This static
inventory is not a claim of actual collection or execution.

Run the static checks with CPython 3.12 and 3.14:

```sh
python3 -I -B -m unittest discover -s validation/owned-pytorch-half-v2 -p 'test_*.py'
```

Pushing this recipe or its workflow on `codex/trained-state-precision` starts the
bounded hosted build. The shared PyTorch concurrency group prevents overlapping
framework builds. A successful build still requires authenticated archive
transport, fresh focused and full-suite device tests, trained-state and gradient
checks, and a successor comparison contract before formal admission. The old
reserved streams remain unopened.
