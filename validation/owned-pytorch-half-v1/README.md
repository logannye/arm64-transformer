# Combined owned PyTorch candidate: hosted build

This prospective build combines the deterministic embedding repair, matrix and
convolution layout dispatch repairs, and the general FP16 `index_put` accumulation
candidate. The FP16 candidate preserves CPU per-add half rounding in original
TensorIterator order. It targets compatibility and repeatability; it does not
claim improved FP64 accuracy, performance, or completed device validation.

The combined patch changes exactly 12 files against the pinned PyTorch revision.
Its contract binds the component patches, every resulting source hash, all 45
original dependency wheels, the original runtime verifier and new static checks.
The preceding layout builder, its workflow, original framework source trees and
all earlier evidence remain unchanged.

The build uses one complete checkout, one environment and one build tree on the
project's macOS ARM64 hosted runner. It requires two build workers, Release,
`DEBUG=0`, `REL_WITH_DEB_INFO=0`, and C/C++ `-g0`. The actual compiler arguments and
object bytes must be retained for all five affected translation units, including
`Indexing.mm` and `TensorAdvancedIndexing.cpp`, plus both offline Metal objects.
The shared indexing headers are bound by the complete source archive. Compilation
uses only the existing full-framework editable installation command; no partial
operator library can qualify this candidate.

The limits remain 40 GiB free initially, an 8 GiB reserve, 60 GiB total allocated
source/build/cache/output, 8 GiB each for source and runtime payloads, and 360
minutes. Dependency wheels and Homebrew OpenMP are binary-only. Source and
runtime archives are retained before downstream checks, and failures retain all
available logs, receipts, input bytes, dependencies and partial archives.

The local native formatting check used the retained clang-format 21.1.8 binary,
verified against its installed RECORD and historical version receipt. Changes
were confined to changed-line formatting and comment reflow; noncomment code
tokens remained identical. Revision 2 remains preserved. This is not the
upstream lint result: the hosted build requires unmodified `spin lint`, no rule
exclusions, and the pinned upstream clang-format 19.1.4 binary.

The prospective inventory contains all original 10,353 test IDs with unchanged
744 skip and 382 expected-failure entries, plus four new tests: 10,357 total.
Static AST checks preserve the existing test module except the explicitly
reviewed direct-FP16 `index_put` expectation body and the four new methods.
Existing decorators, eligibility and tolerance declarations remain bound.
Strict `index_add_` and `put_` rejection remains in place. Actual native test
collection and execution have not occurred; this inventory is a requirement,
not a claim that tests passed.

Run the non-numerical checks locally:

```sh
python3 -I -B -m unittest discover \
  -s validation/owned-pytorch-half-v1 -p 'test_hosted_build.py' -v
python3 -I -B -m unittest discover \
  -s integration/gpu-owned/half-index-candidate-v1 -p 'test_reference.py' -v
```

Both commands are included in the hosted workflow. The second command is also
recommended for regular CPU-only CI: it checks all 15 scalar ordering, address,
word-ownership, rounding and negative controls without importing a framework.

Pushing changes to this directory, the FP16 candidate directory or the new
workflow on `codex/gpu-v4-owned-baselines` starts this bounded build. Manual
dispatch is also available after workflow registration. Use only one route for a
revision. The shared PyTorch validation concurrency group prevents overlapping
builds. This preparation has neither pushed nor dispatched the workflow.

A successful hosted receipt establishes build, authoritative lint, import and
transport checks only. The new binary requires newly authenticated transport,
focused FP16 and layout checks, all 10,357 full-suite tests, fresh model/runtime
checks, numerical admission and the complete matched study. Earlier runtime
results cannot be transferred to this candidate. No upstream submission occurs;
formal eligibility remains false and the admission stream stays unopened.
