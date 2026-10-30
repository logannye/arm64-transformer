# Complete MPS conformance in the owned joint runtime

`owned_pytorch_suites.py` is a new execution adapter for the unchanged full
`test/test_mps.py` suite. It authenticates the same interpreter, dependencies and
complete maintained PyTorch distribution intended for both CPU and MPS study
workers. This is framework eligibility evidence, not numerical admission or a
formal fit. No GPU suite or native compilation was performed while implementing
the adapter. **This adapter is prepared but not qualification-ready.** Owned MPS
eligibility is explicitly blocked until a separately reviewed archive-safe audit
validates its complete descendant, fixture, helper and source evidence. The
generic full-test count check cannot qualify MPS in the meantime.

The current version binds the original 10,353 test identities, their order, skips,
and expected failures through `owned-suite-inventory-v1.json`. All failures remain
blocking. The four additional half-precision regressions in the prospective
candidate produce a different 10,357-test inventory. That candidate requires an
explicit prospective framework selection and suite-contract revision before it
can use this qualification route. No filtering or automatic test-count adjustment
is permitted, and old results are not promoted to a new runtime.

Preparation authenticates an externally SHA-bound complete source manifest and
compares its base revision and combined patch with the selected native build. It
rehashes every retained source file and checks the complete tree, including
source symlink identities and excluding only Git administration entries. The
unchanged test source must also match the independent suite inventory. A read-only compatibility audit passed against the actual retained source:
146,435 files and 2,510,710,463 bytes, using source-receipt SHA256
`bedb94cf99cc11b057f21c1b9f654b9b910e8fa0432788d3b8ddba08e9594afc`.
This was a source inventory/hash check, not a framework import or device check.
Preparation
binds the original `RetainedResult` and result-serialization helpers without
calling their historical runtime bootstrap or fixture policy.

The shared owned-runtime bootstrap verifies the whole captured project and
runtime before installing any dependency import roots. It checks actual import
resolution and excludes site processing, editable hooks and unbound bytecode.
The producer and shared helper are checked against the external runtime receipt
before either helper is imported. Every worker uses canonical CPython
`-P -S -B`, `PYTHONHASHSEED=0`, and a fresh nonexistent bytecode-cache prefix.
`-I` is deliberately excluded because it would ignore the fixed hash seed.

The current frozen candidate full-suite helper is a separate development tool.
It uses the earlier isolation contract and its unchanged fixture policy pins the
old `test_mps.py` bytes, while the half candidate binds the newly extended file.
Those mismatched inputs cannot qualify the owned study. They remain unchanged as
historical preparation evidence; this adapter does not silently modify them.

## Bounded stock conformance extension

`owned_pytorch_fixture.py` is a new, versioned policy for the single unchanged
upstream `mps_iter_conformance` C++ test extension. A frozen AST hash covers its
source strings, helper functions and configuration. The exact relevant source
fragments are retained for static regression testing and are never imported.
Other test methods may change only through a separately approved prospective
suite contract; changing the extension requires a new fixture policy as well.

The fixture policy binds source/header inputs, the selected runtime manifest,
compiler and Ninja bytes, and the toolchain environment. It retains the original
bounds: two compiler jobs, no debug symbols, one dedicated cache, at most 1 GiB and
64 files, and at least 5 GiB free. The full suite has a six-hour timeout and a
separate 2 GiB output bound. The supervisor checks storage each second, so these
are monitored limits rather than filesystem quotas. It terminates the whole
worker process group on timeout or resource failure.

Preparation only hashes files and creates a cache marker. Compilation can occur
only when a separately invoked `run` executes the unchanged upstream extension
test. There is no framework installation, source build, source patching or
operator replacement. The native extension and generated Ninja file are retained
and checked after the suite. Any precompiled runtime header that the stock loader
would delete is rejected before execution.

## Conformance subprocesses and workload separation

The full suite includes two source-bound subprocess calls. One tests an ordinary
`import torch` with warnings enabled. The other deliberately enables fallback
inside the unchanged upstream warning test to check an unsupported operation.
These are conformance controls. The full suite is not described as exclusively
GPU execution: it also contains CPU reference work and this deliberate fallback
control. All model, benchmark and formal workers retain fallback disabled.

Only those exact two `-W always -c SCRIPT` invocations can use the temporary
interpreter launcher. It preserves the original scripts, their stdout and exit
behavior. The first Torch import occurs inside each original script, so import
warnings are not hidden by a preliminary framework import. Each child starts
from the same authenticated runtime, records its PID/parent, verifies native and
Python origins afterward, and retains a separate receipt. Worker bytecode
routing is installed by `-X pycache_prefix` before the producer executes; child
routing is installed by its hash-checked launcher before producer execution. The fallback override
is accepted only for that exact warning-test script; an ordinary worker with
fallback enabled fails its controls.

The parent installs `sys.executable` routing only around the unchanged test
runner, with unconditional restoration before postflight authentication. Native
provenance checks use the canonical interpreter. The wrapper does not replace
subprocess functions, modify upstream test source, or substitute an embedding or
attention implementation.

## Evidence and invocation

The parent checks raw full-suite results and invokes the same production
eligibility validator as the prospective study. At this stopping point that
validator explicitly rejects owned MPS qualification, even if all raw methods
pass. The remaining archive auditor must independently revalidate relative raw
bindings for both descendant receipts and the launcher; complete imported-file
proofs, process intervals and parent identity; the bounded stock fixture and
its generated native files; retained helper identities; and the full selected
source manifest. No absolute local path may substitute for archived evidence.
Until that code and its mutation tests are completed and reviewed, this producer
cannot return a qualified passing study receipt. A worker's `passed` boolean cannot substitute
for the expected test inventory, exact skip policy, raw results, source/runtime
hashes, native module and library origins, complete child receipts or successful
postflight checks. Failed and interrupted attempts retain available logs and
receipts; they cannot produce passing eligibility. Formal fits remain zero.

Use the retained interpreter for CPU-only tests:

```sh
/ABSOLUTE/SELECTED/PYTHON -I -S -B \
  integration/gpu-owned/test_owned_pytorch_suites.py
```

After all sources and the joint runtime have been frozen, run `prepare` with
explicit `--runtime-root`, `--runtime-inventory`, `--selection`, `--receipt`,
`--receipt-sha256`, `--source`, `--source-manifest`,
`--source-manifest-sha256`, `--driver`, `--reference`, and a new `--output` path.
The driver/reference paths select the already-retained immutable result helpers,
not another framework implementation. The two external SHA256 arguments must
come from verified publication/capture evidence.

The 45 CPU-only regression checks pass under the retained CPython 3.12 interpreter,
including actual fresh-process launcher
checks, source mutations, cache bounds, native-origin mutations, and unconditional
interpreter restoration after a test-runner exception. Retained Python import maps
are checked against authenticated runtime, project, full-source and explicit
fixture authorities without accessing their original paths. Child receipts must
have distinct integer process identities and serial, finite intervals contained
within the parent process. Missing origins, changed hashes, foreign imports and
overlapping or escaped intervals are rejected. Ruff passes for the new
Python files. These tests do not validate native execution or complete the
pending archive auditor. The original AST fingerprints bind CPython 3.12's AST
representation; a different Python AST version does not inherit this check.

Both `prepare` and `run` require the same sanitized environment constructed by
`controls`, including its exact cache path and selected Ninja search path. The
launch is `PYTHONHASHSEED=0 /ABSOLUTE/SELECTED/PYTHON -P -S -B
integration/gpu-owned/owned_pytorch_suites.py ...`; the complete control map in
source is mandatory. Preparation creates a versioned plan with all environment
and cache inputs. Run it separately with `run --plan ... --plan-sha256 ...
--output ...`, after acquiring the project's serial device lease. Do not invoke
`worker` or `descendant` independently.

A source/runtime selection mismatch, stale source, missing dependency, changed
native origin, failed full-suite method, changed skip inventory or incomplete
proof blocks qualification. None is waived because development model probes
passed.
