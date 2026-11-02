# MLX test compatibility with a PyTorch prerelease

The selected MLX Python inventory cannot collect with the shared PyTorch
`2.16.0a0+git71d2a08` distribution: `test_array.py` converts `0a0` to an integer.
The declared patch uses the existing `packaging.version.Version.release` parser.
All test functions, class bodies, decorators, assertions and skip conditions
retain identical ASTs. The complete 1,053-case inventory, seven eligible
skips, 17,701 passing subtests and 27 added GPU regressions remain required.

`derive_mlx_test_compat.py` authenticates the original Release/JIT-on capsule,
then creates an APFS clone. It changes that one Python test file and records the
patch and derivation in a new manifest. Every native source, binary, shader,
library Python file and original build-provenance byte must remain unchanged.
The original capsule is retained. No compilation or installation is performed.
The original native build remains run `40978071574`; this test-only derivation
must never be described as a new native build.

The tool requires enough free space for the entire logical capsule plus a
6 GiB reserve even though APFS sharing can reduce its physical allocation.
Its ten tests cover exact native reuse, real release/prerelease/local version
parsing, source/binary mutations, forged manifests, rebound test assertions,
changed contracts and overlapping output paths.

Capture the new capsule only with its external
`--mlx-test-compatibility-sha256` receipt binding. The explicit MLX suite contract
is `--suite-contract torch-dev-compat`. It binds the new test-source hash while
preserving every method, assertion, eligibility rule and denominator. A fresh
complete execution remains mandatory. Historical failed attempts, the original
framework selection and all formal gates remain unchanged.

The upstream pytest teardown deliberately clears all MLX streams. The suite
producer now verifies that GPU remains the default, creates a fresh GPU stream,
and verifies that stream before its separate postflight compilation probe. This
does not alter a test result. An incomplete child receipt keeps the parent
summary incomplete, including collection failures.
