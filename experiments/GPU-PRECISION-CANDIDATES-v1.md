# Bounded GPU precision development, v1

This candidate inventory is recorded before the proposed compensated variants
are implemented or evaluated. It uses only the development fixture stream
20261103 and training-only development rows. Admission seed 20261104 remains
unopened. No formal GPU timing or held-out checkpoint selection has started.

The retained reference C0 uses FP32 arithmetic with safe, precise Metal compiler
settings and a fixed 16×16 dense tile. Its width-1024 attention stress fails 17
of 23 development cases, including large common score offsets with small gaps.
Its source/binary/receipts are retained under
`work/gpu-v1/metal-development/reference-v0/`. These observations motivate the
following finite precision sequence; they are not performance-selection results.

| Candidate | Change from C0 | Admission to further development |
|---|---|---|
| C0 | Existing reference FP32 path | Preserve all failures and controls |
| P1 | Two-float attention products/sums, score scaling and centering, and sensitive backward reductions | All attention development checks at every width must pass |
| P2 | P1 plus compensated first/all LayerNorm reductions and sensitive normalization arithmetic | Evaluate only if normalization or full-model development failures remain |
| P3 | P2 plus compensated QKV projection and its necessary gradient reductions | Evaluate only if QKV development failures remain |

Each variant preserves the same real-valued equations, parameter/state FP32
storage, workload, initial bytes and error limits. Two-float expansions are
temporary arithmetic representations made from FP32 components; they do not
provide native FP64 hardware or promise full FP64 equivalence. FMA product
residuals, error-free transforms and scale constants must be documented and
tested under the actual compiler mode. All failures, source identities and
diagnostic traces are retained before a variant is superseded.

Use unchanged forward/gradient/math tolerances and independent FP64 references.
Test forward and backward, including the derivative conditioned on rounded saved
probabilities, causal controls, guards and input immutability. Local accuracy
does not guarantee a model-level error bound. No test-set outcomes may select
these candidates, and no tolerance may be loosened to admit a variant.

Stop this sequence if P3 still cannot pass its declared development accuracy
requirements, retaining the precision limit. A new algorithm, storage format or
additional candidate search requires a new prospective development declaration.
Tile/dispatch/fusion performance tuning is separate and must have its own finite
grid frozen before it is evaluated. This file does not open formal measurement
or resolve the standard frameworks' observed embedding-gradient repeatability
failures.
