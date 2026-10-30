# Owned PyTorch build recipe v3

This successor preserves the v2 14-file source patch, 45 dependency wheels,
Half differential probe, unchanged authoritative lint and all test inventory
requirements. Run 41110294834 completed compilation but failed its explicit
`-g0` command check. Its four changed Objective-C++ commands omit `-g0` even
though CMakeCache.txt contains it. That failed artifact remains separate.

This recipe additionally passes CMAKE_C_FLAGS, CMAKE_CXX_FLAGS, CMAKE_OBJC_FLAGS
and CMAKE_OBJCXX_FLAGS explicitly through the upstream environment forwarding
mechanism. This initializes their cache entries before dependency configuration
can create normal variables that shadow later compiler-environment defaults.
The actual generated commands remain authoritative: all five changed translation
units must contain `-g0` and no other debug flag. The supervisor checks them as
soon as CMake emits its compile database and again after compilation. A mismatch
terminates the build early. No source patch, compiler diagnostic or numerical
tolerance is waived. This mechanism awaits native hosted validation.

The workflow uses the same one-tree, two-worker, 40 GiB initial / 8 GiB reserve /
60 GiB allocation / 360-minute limits. Compilation is hosted only. A successful
build still requires fresh runtime authentication, upstream conformance and
numerical qualification; no earlier runtime evidence transfers.

Static checks: `python3 -I -B -m unittest discover -s validation/owned-pytorch-half-v3 -p 'test_*.py'`.
