# Combined FP16 candidate: fresh validation only

These tools select own-project hosted run `41099457320`, built from study commit
`64dce37fe686bd52d8aa7b1886ea6526bc69bdc9`. They are separate from the layout-only
candidate tools and cannot accept that earlier build. The selection does not
claim that the new hosted build or any numerical checks have passed.

`candidate_half_validation.py` requires an externally checked hosted receipt
hash, the exact selected run and commit, the 12-file build contract, five changed
C++/Objective-C++ translation units, both Metal objects, authoritative upstream
lint, and source-preservation evidence. It authenticates both source and runtime
archives before expansion, retains exact dependency-wheel checks, rejects
foreign framework origins, and prevents unverified dependency bytecode/site
hooks from running. Every subprocess and its logs, imports and outcome are
bound into retained evidence. Failed earlier evidence is never overwritten.

The focused sequence contains 34 complete methods: all 27 original failed
methods, all four new FP16 regressions, and the original three parameterized
nondeterminism methods for FP16, BF16 and FP32. These checks exercise the revised
direct FP16 `index_put` expectation and retained strict rejection for its
unqualified indirect callers. No method is selected only at a favorable sample
index. The seven additional focused checks do not replace the original failures.

`candidate_half_full_suite.py` requires a passing and fully re-audited focused
sequence. Its prospective inventory contains 10,357 exact IDs: all 10,353 old
IDs plus four new methods. The same 744 skip and 382 expected-failure identities
remain required. Missing, additional, reordered or reclassified execution is a
failure. Old expected failures do not become new exclusions, and new failures
cannot be waived.

Preparation and execution are separate. Full-suite preparation creates one new
runtime-bound stock conformance-fixture cache but does not compile it. Only an
explicit full-suite run may invoke the unchanged upstream C++ fixture, with two
workers, no debug flags, a 1 GiB / 64-file cache cap and a 5 GiB free-space
reserve. No framework build or operator replacement occurs on the local device.
Real-device runs must be serial and must wait until other project numerical work
has stopped.

Static checks, requiring no framework imports or compilation:

```sh
python3 -I -B -m unittest discover \
  -s integration/gpu-owned -p 'test_candidate_half_validation.py' -v
python3 -I -B -m unittest discover \
  -s integration/gpu-owned -p 'test_candidate_half_full_suite.py' -v
```

Once the selected hosted build passes and its artifacts have been independently
verified, prepare a fresh path under the workspace's `work/gpu-v4-owned` tree:

```sh
python3 -I -B integration/gpu-owned/candidate_half_validation.py prepare \
  --workspace /ABSOLUTE/WORKSPACE \
  --artifact /ABSOLUTE/VERIFIED-HOSTED-ARTIFACT \
  --receipt-sha256 EXTERNALLY_VERIFIED_RECEIPT_SHA256 \
  --hosted-head 64dce37fe686bd52d8aa7b1886ea6526bc69bdc9 \
  --output /ABSOLUTE/WORKSPACE/work/gpu-v4-owned/half-candidate-preparation-v1
```

Execution requires the resulting immutable plan and its independently recorded
hash. The full-suite preparation further requires the exact focused summary and
its hash, and must use the authenticated CPython executable with `-I -S -B`.
There is no flag that converts a static, transport or focused pass into admission.
The runtime/model checks, broader scientific admission and matched campaign still
require fresh evidence for this new binary. Earlier runtime and full-shape model
results remain attached to their original framework build.
