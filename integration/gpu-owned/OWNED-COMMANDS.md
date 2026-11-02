# Independently maintained study commands

The owned study has its own immutable namespace, `gpu-v4-owned-five-engine-v1`, and reserved admission stream **20261108**. The standard release study and its stream **20261104** remain separate. Both streams remain unopened. Creating a plan, capturing a runtime, or passing a development probe grants no admission.

The shared implementation now routes the owned study through the same 180 cells, training budgets, numerical tolerances, raw admission checks, training runner, all-engine checkpoint audit, and report calculations. Owned framework construction delegates to the unchanged production MPS and MLX adapters. A source-build capability wrapper supplies provenance without changing tensor operations or inventing package metadata.

The selected PyTorch build still has retained broad-suite failures. The new combined repair candidate has not been selected for a formal study. An eligibility receipt must bind fresh, complete framework-suite results to the exact runtime and dependencies that will execute the study. Earlier NumPy 2.2.6 MLX suite results cannot qualify the joint NumPy 1.26.4 runtime. The selected old MPS inventory has 10,353 methods; the four prospective FP16 additions remain explicitly unselected until a reviewed contract and runtime selection change.

## Interpreter and source identity

Owned processes use the canonical CPython executable, `-P -S -B`, a fresh nonexistent bytecode prefix, and `PYTHONHASHSEED=0`. The bootstrap supplies authenticated import roots explicitly. `-I` is deliberately absent because it ignores `PYTHONHASHSEED`; a same-string environment setting would otherwise conceal differing actual hash behavior. `-B` alone would still read existing bytecode, which is why the fresh prefix is also required.

Each owned child has an externally hashed bootstrap authority. Its standard-library bootstrap verifies every bound runtime and source file before activating dependency import paths. A changed dependency therefore fails before it can shadow a standard-library or framework import. Runtime discovery and provenance work are setup costs outside measured training and inference.

Current source, frozen CPU source, framework source/native manifests, selected patches, interpreter files, dependencies and runtime paths are bound separately. CPU PyTorch uses the same selected distribution as MPS. Its launcher verifies actual Torch module/native-library origins before and after running the unchanged frozen CPU entrypoint. It performs raw admission validation in a separate control-plane process so the current project cannot shadow the frozen CPU numerical package. Provenance checks sit outside the runner's recorded training and inference timing. The native CPU anchor retains its authenticated executable and native backend receipt.

Standard study launches retain their previous behavior. System dyld shared-cache bytes remain represented by OS identity, not by an independently copied image.

## Entry point

Use `transformer/gpu_owned_command.py` by absolute path. Set the canonical Python path and repository/work paths to the actual captured installation. For example, a plan can be created without any dataset or framework execution:

```sh
env -i PATH="$PATH" HOME="$HOME" TMPDIR="${TMPDIR:-/tmp}" \
  PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
  "$OWNED_PYTHON" -P -S -B \
  -X "pycache_prefix=$OWNED_WORK/plan-unused-cache" \
  "$OWNED_REPO/transformer/gpu_owned_command.py" \
  plan --output "$OWNED_WORK/prospective-owned-manifest.json"
```

The `storage` command reports conservative admission, campaign and audit storage estimates without creating operands.

Executing commands additionally require `--runtime-root`, `--runtime-inventory`, `--runtime-receipt`, and an externally retained `--runtime-receipt-sha256` before the subcommand. Use the files emitted by `integration/gpu-owned/capture_runtime.py`. Do not hand-edit a passing claim or substitute an older runtime's receipt.

| Command | Existing implementation and required authority |
|---|---|
| `repeatability prepare …` | Freeze the complete two-engine, six-width development diagnostic against the selected runtime. |
| `repeatability run …` | Twenty-four fresh workers; ten executions per case; exact embedding, model, six-step training and checkpoint replay controls. No formal fixtures. |
| `freeze …` | Freeze all six roots, complete source/runtime bytes, original CPU fixture bytes, policy and baseline evidence. |
| `admit …` | Recheck complete framework eligibility and raw baseline repeatability before opening the owned reserved stream. |
| `seal …` | Independently re-evaluate every admitted raw-array check and bind the passing receipt. |
| `verify …` | Recheck the sealed manifest, actual bound files, baseline proofs and raw admission. |
| `run …` | Execute the exact 180-cell order with unchanged timing and condition checks. No boolean flag bypasses admission. |
| `audit …` | Replay each selected checkpoint through all engines and independent FP64 equations; retain per-implementation failures. |
| `report …` | Authenticate admission, baseline, campaign, complete checkpoint audits and repeat evidence before computing qualified summaries. |

The dispatcher forwards remaining arguments to the corresponding shared command. It rejects standard-study manifests and any attempt to override the owned context. The underlying `transformer.gpu_study_report` module also retains its archive-only reporting interface, which does not require the historical framework installation to be present.

The full MLX suite has a separate prepare/run producer, `integration/gpu-owned/owned_mlx_suites.py`. Framework conformance execution and any declared interpreter routing for unchanged upstream subprocess tests are distinct from formal baseline execution. Every deviation in launch plumbing must remain explicit and retain original test source, eligibility and numerical behavior.

The complete repeatability diagnostic reserves 37 GiB before starting, keeps a 5 GiB free-space floor, caps retained output at 32 GiB, and bounds each worker at four hours. It records actual native/module origins again after all operations. A resource or identity failure preserves partial evidence and blocks qualification.

## Evidence and remaining gates

`owned-runtime.json` and `owned-eligibility.json` belong in the frozen baseline evidence root alongside the complete owned repeatability output. The eligibility verifier re-parses all selected MPS method outcomes and MLX raw logs/XML; verifies exact case, skip and expected-failure inventories; authenticates compiled-GPU proof and native identities; and retains the original failed MPS audit. It rejects missing suites, stale dependency identities, omitted failure records, narrowed inventories, and late integrity failures.

Owned admission prespecifies every CPU Torch operation, including CPU reference work inside GPU checks. Each operation binds its retained inputs, outputs and origin proof to its owning gate. Malformed-input controls retain the exact expected rejection together with post-execution provenance. Removing ordinary proofs while retaining only checkpoint-tie proofs cannot satisfy the inventory. Formal cell and checkpoint auditors reject absent proofs, changed checkpoint inputs, substituted native origins and wrong optimizer counts. The original CPU precision limitations remain separate numerical results.

This implementation has CPU-only orchestration and mutation tests. It has not executed an owned admission, formal fit, or full checkpoint campaign. Real-device development repeatability, fresh framework integration suites, prospective candidate selection, complete numerical admission and the 180 fits remain required before study results can be claimed. Raw developer/model probes are useful engineering evidence, not replacements for those gates.

One control-plane gap remains explicit: the owned MPS archive auditor must recheck its two upstream conformance subprocesses, stock C++ fixture, retained helper/source inventories and actual native proofs. The live suite adapter does not fill that archival role. `validate_suite` therefore unconditionally rejects MPS qualification until this auditor is implemented and reviewed; generic passing MPS test totals cannot open admission.
