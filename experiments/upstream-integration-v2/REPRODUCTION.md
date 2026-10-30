# Reproducing the GPU engineering evidence

Use this guide to verify the archived evidence, inspect its results and prepare a newly bound execution. The paired PyTorch control and its independent evidence audit are complete; the original broad-suite failure remains visible. Final source revision, asset IDs, package hashes, remote verification and publication status are recorded separately in the release catalog for tag `transformer-upstream-integration-v2` (release ID `428513598`). Consult the [repository releases page](https://github.com/logannye/arm64-transformer/releases) for that catalog. This guide does not substitute local verification for remote publication evidence.

Start with the patched MLX numerical report (archived `outputs/upstream-integration-v2/release-integration-v2/RELEASE-INTEGRATION.md`) and matched MLX operator report (archived `outputs/upstream-integration-v2/operator-results-v3/README.md`). The former checks synthetic initial-state FP64 accuracy and separate six-step/checkpoint replay. The latter measures individual operators in ordinary full-source main/patched **Release builds**. Release is a compiler configuration, not a supported stable framework release. Neither result is a complete CPU/GPU training study.

There are three different reproduction tasks:

| Task | What it establishes | What it does not establish |
|---|---|---|
| Verify downloaded assets and members | The published bytes match the authenticated evidence and alias map | A new numerical execution or a passing historical failed job |
| Inspect or rederive saved results | Recorded inventories, comparisons and report calculations reproduce within their stated audit scope | Fresh FP64 equations or repeatable execution on another machine |
| Run new integration checks or timings | New evidence for explicitly selected source, binaries, dependencies and hardware | Retroactive qualification of the old records or formal-study admission |

## Finding the archived files

The Git repository contains this summary and selected figures; it does not contain the
complete native runtimes, raw tensors or every historical harness. Paths beginning with
`work/` or `outputs/` below are **paths inside the reconstructed evidence archive**, not
links to files in this repository. Obtain the final asset catalog and complete packages
from the [project releases page](https://github.com/logannye/arm64-transformer/releases),
using tag `transformer-upstream-integration-v2` after publication. The catalog supplies the publication identities separately from these
fixed technical instructions.

| Evidence or recipe | Path after verified reconstruction | Use |
|---|---|---|
| Layer verifier | `work/upstream-integration-v2/package_selected_evidence.py` | Verify each complete package and its member inventory before layering |
| Remote asset verification | `work/upstream-integration-v2/RELEASE-ASSET-VERIFICATION-V1.md` | Distinguish downloaded-byte verification from the final whole-release inventory |
| MLX integration report | `outputs/upstream-integration-v2/release-integration-v2/RELEASE-INTEGRATION.md` | Completed combined-patch Release numerical evidence |
| MLX operator report | `outputs/upstream-integration-v2/operator-results-v3/README.md` | Qualified per-operator costs and retained failures |
| MLX integration commands | `work/upstream-integration-v2/mlx-release-validation-v1/RUN.md` | Rebind and rerun the declared model/suite checks |
| MLX operator commands | `work/upstream-integration-v2/mlx/operator-bench-release-v1/RUN.md` | Rebind the matched operator protocol before new timings |
| Selected PyTorch patch | `outputs/upstream-integration-v2/pytorch/formatted-v2/changes.patch` | Review the exact candidate, with recorded base and patch hash |
| PyTorch model/suite commands | `work/upstream-integration-v2/pytorch/formatted-v2-runtime-v5/RUN.md` | Match the full source, runtime, dependencies and selected test scope |
| Original broad-suite audit | `work/upstream-integration-v2/pytorch/closed-five-family-audit-v2/AUDIT-SUMMARY.md` | Preserve targeted passes and the failed broad suite |
| Source-byte postflight | `work/upstream-integration-v2/pytorch/source-postflight-v2/execution-v2/result.json` | Closed full source-byte identity, separate from numerical qualification |
| Content-identity guard | `work/upstream-integration-v2/pytorch/paired-transport-content-v1/README.md` | Original-hash verification when ctime invalidates the cache |
| Notice preservation | `work/upstream-integration-v2/notice-record-closure-v1/closure.json` | Closed declared notice inventory and original archive/wheel locations |
| Paired attribution commands | `work/upstream-integration-v2/pytorch/paired-runtime-v4/README.md` | Run the separate 54-worker retrospective control after new authentication |
| Paired closed-output audit | `work/upstream-integration-v2/pytorch/paired-control-audit-v3/README.md` | Audit final receipt hashes and source-normalized comparisons; closed replay recorded; closed audit passed; original tests remain failed |

Do not manufacture repository-relative links to these archive members. Restore the
common `work/` and `outputs/` roots first, keep original receipts unchanged, and follow
the relocation and new-contract rules below before a new execution.

## Restore and authenticate the evidence

Use the final catalog's asset IDs, byte counts and SHA-256 values. Verify both the downloaded bytes and their package/member manifests. Keep each package's `package.json`, `input-selection.json` and complete shard set together; the verifier rejects missing or additional package files. The command form is:

```sh
python3 -I -S -B work/upstream-integration-v2/package_selected_evidence.py verify \
  --package ABSOLUTE_COMPLETE_PACKAGE_DIRECTORY \
  --expected-package-sha256 TRUSTED_FINAL_PACKAGE_RECEIPT_SHA256
```

Verify each original layer separately before reconstructing the common `work/` and `outputs/` layout in a fresh tree: `evidence-major-v1`, `evidence-mlx-remainder-v1`, then the final supplement selected by the coordinator. The closed external-payload preflight (archived `work/upstream-integration-v2/supplement-v3/external-assets-v1/summary.json`) verified 64 canonical payloads and 201 original paths, totaling 2,260,100,980 bytes. It includes all four original 45-wheel inventories, including the same-host pair's shared wheelhouse. This is the native/wheel/bytecode subset, not a final supplement package or whole-release inventory; the catalog separately binds the final selection and publication identities. Keep the original archives and receipts immutable. A later layer may replace a changed path only when its selection and `replaced_paths` receipt explicitly authenticate that replacement; reject undeclared differing bytes. Verify each installed member and its executable bits against the layer that supplies it. The resulting tree is the declared latest view. To inspect or rerun a historical version, reconstruct its matching layer versions separately instead of combining old receipts with later source bytes. Package verification itself reads archives; it is not a source-checkout or runtime-eligibility check.

Native archives and dependency wheels are canonical assets stored once. Restore their original locations through the verified alias map using regular files with the recorded bytes and modes. Preserve all aliases and original receipts; matching filenames alone do not prove equivalence. Expanded build/runtime trees, virtual environments and caches are intentionally omitted. The historical native MLX archives, shared archive and eager-FP64 archive still belong in the final catalog, alongside the four matched Release bundles and the authenticated PyTorch runtimes. Do not silently drop older failure evidence because a newer runtime exists.

Some hosted ZIPs are represented by their complete authenticated extracted members and canonical-asset mappings. Those receipts establish member preservation, not regeneration of the original ZIP compression bytes. Original archive identities and failure statuses remain recorded. Two original lint-helper bytecode members are explicit `.bin` evidence assets with original-path aliases; they do not authorize restoring unrelated caches. The closed notice-preservation record (archived `work/upstream-integration-v2/notice-record-closure-v1/closure.json`) resolves the older companion's pending runtime inspection: each of the two earlier and two same-host PyTorch runtimes retains 63 declared package-license files within 69 retained notice paths, with source LICENSE/NOTICE companions; the four MLX Release archives retain their source licenses. Preserve the ten-file notice companion, each exact wheel's embedded notices, and all source/archive identities. The record is limited to preservation of the declared inventory, not legal clearance or unrecorded transitive terms. No license has been invented for the user's project.

The final asset-verification recipe (archived `work/upstream-integration-v2/RELEASE-ASSET-VERIFICATION-V1.md`) separates streamed remote hash checks from the final complete release-inventory check. A local package pass does not establish that every remote asset was uploaded correctly.

## Preserve old identities; bind new executions

Original receipts contain absolute paths, source/helper hashes, process identities and, where applicable, inode/stat records. Keep those bytes unchanged. Replacing `/Users/logannye/...` with a new path inside old JSON would invalidate its hash and misrepresent the execution. Restoring the same pathname also cannot recreate the original inode or process history.

At another location, verify the archive as evidence first. Record the relocation/reconstruction separately, then declare new path-aware source/runtime contracts, eligibility receipts, manifests and output namespaces before executing checks. Existing strict verifiers may deliberately reject a relocated tree. Do not bypass that rejection, substitute a new receipt into an old run, or present a changed helper as the historical one. Rebuilding at the same source commit also produces a new binary identity that requires new qualification.

The selected-model PyTorch v5 validation requires a complete matching checkout, not four patched files over an unrelated tree. The selected patch is formatted revision 2 (archived `outputs/upstream-integration-v2/pytorch/formatted-v2/changes.patch`), SHA-256 `3834dca844ea3de6b40467ce214b99bce83fa06dae2bc454623516dafc0fdd71`, on base `71d2a08507ab47dd6b2fa9b66319d17ce2377de4`, with all 61 recorded recursive submodule revisions. Runtime package v5 (archived `work/upstream-integration-v2/pytorch/formatted-v2-runtime-v5/RUN.md`) uses `source_auth_v2.py`: it retains raw working-tree and canonical Git identities, including the exact declared CRLF expansion for 38 upstream paths. Do not normalize files or reuse the earlier raw-blob-only authenticator. The complete source receipt is retained; the full upstream Git object mirror is not, so source reconstruction depends on availability of the pinned objects. The separate paired-control recipe uses its two compact authenticated test-source archives and normal runtimes for the original 27-test scope. Those archives do not replace the selected v5 model-validation checkout.

The historical v5 guide was written prospectively. A later source freeze and development gate passed; that does not change its original text or the failed hosted lint status. The separate lint comparison retained 13 shared pristine/formatted findings and no formatted-only findings. Development eligibility is distinct from transport readiness, an overall hosted CI pass, and formal admission.

## Source and runtime integrity follow-through

The selected-source byte postflight (archived `work/upstream-integration-v2/pytorch/source-postflight-v2/execution-v2/result.json`) completed a fresh rehash of all **146,435 tracked files / 2,510,710,463 bytes**, including the exact 38 declared checkout transformations and 61 recursive submodules. The new receipt matched the original in all fields except permitted ctime entries. Raw and canonical source identities, patch bytes, other stat fields and checkout declarations stayed equal. The original receipt and failed postflight attempts remain unchanged; this established current source-byte identity and a new stat binding, not a new numerical pass or historical requalification. Preserve both full source receipts, including the fresh 105,440,769-byte receipt, as ordinary evidence files.

The paired runtime later had ctime-only invalidations under a fixed-stat guard. The stopped v2 replay retained 53 completed failed workers and one incomplete worker; stopped v3 retained 44 completed failed workers, one incomplete worker and nine unstarted workers. Both attempts remain incomplete and unqualified, with no salvaged worker results. The origin of the metadata changes remains unproven.

The reviewed content-identity transport (archived `work/upstream-integration-v2/pytorch/paired-transport-content-v1/README.md`) keeps the passing immutable rebound receipt as its hash-cache anchor. A later ctime change invalidates a hash entry and requires a stable read against the **original** size and SHA-256; it never accepts a newly observed hash as authority. All non-ctime identity fields and path topology remain strict, and the five API sidecars without original raw-byte hashes must retain all original stats including ctime. Before/after authentication events and their append-only journals record every check. Bounded races or missing evidence fail. The closed preflight (archived `work/upstream-integration-v2/pytorch/paired-content-preflight-v1/execution.json`) passed by revalidating 11 files / 1,512 bytes; it did not execute a framework or GPU test.

The authenticated hosted pair used runner image `20261021.0445.8` and OpenMP 23.1.4; the selected model runtime used `20261013.0424.1` and OpenMP 23.1.3. These are distinct runtimes. The pair retained both native variants and the original failed hosted-lint status.

The fresh v4 replay executed all 54 workers anew and closed with `complete:true` and `evidence_valid:true`. All 27 test IDs failed in each runtime. Its final auditor verified both authentication phases for the parent and each child, **110 execution events and their 110 journals**, plus two separately labeled audit-time checks and journals. It also retains every original test, import, source, state and source-normalized assertion comparison. This integrity repair changes neither tensor equations nor test eligibility. The closed paired summary (archived `work/upstream-integration-v2/pytorch/paired-device-control-v4/summary.json`) records 27 matching failure kinds, 27 matching source-bound assertion locations and 81 matching framework-state snapshots. Raw traceback line comparisons differed in all 27 pairs because the patch inserts lines into the test module; normalization uses the authenticated unchanged method ASTs. This shows the selected tests also fail in the pristine runtime under the recorded controls. It does not establish a shared numerical cause or clear the original broad suite. The closed-output audit (archived `work/upstream-integration-v2/pytorch/paired-control-audit-actual-v3/audit.json`) passed with SHA-256 `0c650134d2c4cd8ac5aacd27d1fc530fdad63237da6371d1d47785faa24be578`. It independently authenticated the retained evidence and recomputed the 27 comparisons; it did not rerun numerical equations or convert failed tests into passes.

## Run a newly bound diagnostic

Use the existing exact wheel/lock inventories in an isolated environment; do not resolve a fresh dependency set silently. Use normal full-source framework binaries, verify loaded module/dylib origins, and prove real GPU execution. `MLX_DISABLE_COMPILE` and `PYTORCH_MPS_PREFER_METAL` must be absent: setting either to `0` still activates its presence-based behavior. Graph compilation requires the actual trace/Compiled-primitive and changing-input proofs; shader-library JIT is a separate switch.

The following retained recipes specify the checks and command interfaces. Their old output paths and embedded historical gates are examples of their original execution contract, not reusable destinations for a new run.

| Work | Recipe and preserved scope |
|---|---|
| Patched MLX model integration | Release validation (archived `work/upstream-integration-v2/mlx-release-validation-v1/RUN.md`): both shader modes, all six widths, complete six-step/fresh-process and 18-shape FP64 inventories, saved step-3 continuation, full-suite gates and raw integrity |
| Main versus combined-patch MLX operators | Frozen operator protocol (archived `work/upstream-integration-v2/mlx/operator-bench-release-v1/RUN.md`): 26 embedding/112 attention cases, 1,104 ABBA workers, 20 warmups and 100 synchronized observations, exact replay and accuracy gates; a new manifest and exclusive numerical lease are required |
| Formatted PyTorch integration | Runtime v5 (archived `work/upstream-integration-v2/pytorch/formatted-v2-runtime-v5/RUN.md`): focused upstream tests, embedding matrix, legacy decoder, matched transformer and full unchanged `test_mps.py`; the legacy decoder's vocabulary/FF width differ from the matched model |
| Paired failure attribution | Paired replay v4 (archived `work/upstream-integration-v2/pytorch/paired-runtime-v4/README.md`): 27 original failed IDs × two normal runtimes, each in a fresh process; compare exact source-bound assertion locations and three recorded framework-state snapshots, retaining failures |
| Closed paired evidence | Metadata audit (archived `work/upstream-integration-v2/pytorch/paired-control-audit-v3/README.md`): after closure only, supply the final summary hash, reauthenticate 54 worker receipts and recompute all 27 comparisons; closed audit passed; original tests remain failed |
| Closed PyTorch evidence | Saved-result audit (archived `work/upstream-integration-v2/pytorch/result-audit-v2/README.md`): explicit attempt inventory, source/runtime/fixture bindings, all retained arrays, failure records and documented transient-checkpoint limits |

Run serially on the declared hardware, with bounded storage and one numerical process. Full framework rebuilds belong in the authorized hosted validation route. PyTorch's stock conformance test may compile its small extension in the single bounded two-job cache; that is separate from rebuilding PyTorch. Stop on changed inputs or invalid conditions and retain a fresh linked failure/retry path.

Complete MLX results include all baseline failures: only 180 of 328 operator comparisons qualify for ratios; the other 148 retain costs/errors without ratios. Do not pool these into a full-model speedup. The patched numerical report's tolerance margins describe initial synthetic fixtures, not post-training checkpoint precision. The closed PyTorch audit (archived `work/upstream-integration-v2/pytorch/closed-five-family-audit-v2/AUDIT-SUMMARY.md`) authenticated all five result families. The 79 focused tests, 4,320 embedding evaluations, legacy decoder and 36-worker matched model passed. The full `test_mps.py` run executed all 10,353 tests but retained 25 failures and two errors, with 744 skips and 382 expected failures. Overall development qualification is false. Its `coverage_complete:false` means the auditor rejected the failed worker's success-validation path, not that tests were missing. The separate exact-ID replay reproduced all 27 failing tests in fresh processes; that observation did not establish pristine-base attribution. The closed same-host control reproduced all 27 failures in each variant, with matching failure kinds, source-bound assertion locations and all 81 recorded state comparisons. Its independent evidence audit passed, retaining all 54 failed test results and both stopped attempts. [Source and runtime integrity](#source-and-runtime-integrity-follow-through) preserves the stopped attempts and authentication repair; the paired-build summary (archived `work/upstream-integration-v2/pytorch/paired-hosted-summary-v1/summary.json`) records its separate runner image and OpenMP version. Any attribution is scoped to that internally matched pair and cannot clear the original suite or replace the selected model runtime.

Formal admission remains separate: supported stable packages within the agreed update allowance, unchanged strict gates, fresh CPU anchors, the five-engine manifest and every-checkpoint audit are still required. If PyTorch changes, use the same selected package version for CPU and MPS. Combined MLX embedding-plus-attention validation does not establish coverage or performance for either independently submitted patch; each actual submission state needs applicable exact-revision validation. This handoff authorizes neither opening seed `20261104` nor starting the 180 formal fits.

See the companion [human review handoff](HUMAN-REVIEW.md) before any upstream action. That checklist identifies the human-authored material and public actions required by the retained upstream policies.
