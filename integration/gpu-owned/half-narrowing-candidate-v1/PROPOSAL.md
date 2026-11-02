# Explicit Half storage conversions: prospective lint repair

This is an uncompiled two-header proposal. It has not passed upstream lint or
native numerical tests and does not qualify any framework or study run. The two
existing hosted candidate recipes, their patch bytes, and every retained source
tree remain unchanged.

## Finding

The authoritative retained `111-compare-base-spin-lint.log` reports 13 narrowing
errors: five in `torch/headeronly/util/Half.h` and eight in
`torch/headeronly/util/complex.h`. Their paths, lines, messages and source hashes
match the pristine, original embedding-patch and formatted embedding-patch
comparisons. The original unformatted variant also has a separate
`Embedding.metal` formatting finding; this proposal neither removes nor relabels
that historical failure.

On `__aarch64__ && !__CUDACC__`, `Half` takes `float16_t` (`__fp16`) in its
constructor. Elsewhere, the corresponding constructor takes `float`. The 13
expressions already compute in FP32, then implicitly narrow the result for Half
storage. The diagnostic concerns that implicit storage conversion; it is not
evidence that these expressions should compute in FP16 or use a different
rounding algorithm.

## Proposed change

Add one inline `c10::detail::half_from_float(float)` helper. It explicitly casts
to `float16_t` under the same existing ARM constructor condition and otherwise
passes the FP32 value to the existing `Half(float)` constructor. Replace only
the 13 affected result conversions with calls to that helper.

The existing FP32 operand conversions, expression grouping, operation order,
complex multiplication temporaries and final Half storage remain the same. The
CUDA/HIP intrinsic and SYCL unary-negation branches remain untouched. No
constructor signature, conversion operator, field, alignment, return type,
optimizer equation, numerical tolerance or compiler math option changes.
`Half` remains the same two-byte representation; `complex<Half>` retains its
existing two-field, four-byte alignment declaration. The new function is inline
and does not replace an existing ABI entry point. Those source-level observations
still require compilation and behavioral verification on the intended target.

An outer `static_cast<Half>(expression)` alone is not a reliable repair: the
builtin float-to-`__fp16` conversion may still be an implicit constructor argument.
The pinned [LLVM 21.1.0 checker](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.0/clang-tools-extra/clang-tidy/bugprone/NarrowingConversionsCheck.cpp)
excludes implicit casts with an immediate cast-expression parent. The proposal
makes the actual builtin storage conversion explicit, without changing the
constructor interface or suppressing the rule.

## Completed static checks

- Both original headers match Git base
  `71d2a08507ab47dd6b2fa9b66319d17ce2377de4` and the retained diagnostic hashes.
- `git apply --check` passes against isolated copies of those original headers.
- There are exactly five and eight replaced conversions. Removing only the new
  helper and its wrappers recovers every original noncomment token.
- The authenticated upstream clang-format **19.1.4** binary formatted the changed
  ranges; applying the same formatting again is idempotent. This is distinct
  from the pinned clang-tidy **21.1.0** checker.
- No native compilation, installed-library mutation, framework import, GPU
  execution or fresh modeling fixture access occurred.

`contract.json` binds the patch, original and proposed source hashes, diagnostic
comparisons and formatting evidence. Complete before/after header copies,
formatter receipts, the licensed LLVM checker source and the static preparation
failure are retained under
`work/gpu-v4-owned/half-narrowing-candidate-v1/` in the workspace. The preparation
failure records an initial mistaken assumption that the original unformatted
variant had only 13 total findings; the corrected comparison explicitly retains
its fourteenth formatting finding.

## Next required checks

1. Review this isolated proposal. If adopted, declare a new combined candidate
   identity and full-source recipe; do not alter either running historical
   candidate or reuse its runtime qualification. These two header paths are
   disjoint from the existing 12-file combined patch.
2. Run the unchanged complete upstream lint command in the hosted environment,
   retaining every result. Success must include the C++17 header compatibility
   checks and the ordinary formatter, with no suppressions or narrowed rules.
3. In hosted native checks, compare original and proposed conversion results at
   the bit level: all 65,536 Half encodings for conversion/unary operations,
   prespecified arithmetic pairs and complex cases, signed zeros, subnormals,
   halfway cases, overflow, infinities and NaNs. Retain original/proposed NaN
   payloads and floating-point exception behavior rather than assuming them
   equal. Test complex self-aliasing and cancellation. Assert size/alignment and
   public type signatures; run the existing Half and complex C++ tests.
4. Inspect optimized ARM64 code for additional conversions, calls or changed
   contraction. Check host and device branch compilation where environments are
   available. Unchecked CUDA/HIP/SYCL or x86 branches remain explicitly unvalidated.
5. Rebuild the complete selected PyTorch distribution and rerun all required
   focused/full-suite, fresh-process, matched-model and admission checks under
   its new identity. A source-level equivalence argument cannot transfer results
   from the previous binary or resolve the existing FP16 index-update issue.
