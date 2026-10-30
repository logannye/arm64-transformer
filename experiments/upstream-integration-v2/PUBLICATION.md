# Public evidence catalog

The [GPU engineering release](https://github.com/logannye/arm64-transformer/releases/tag/transformer-upstream-integration-v2) is public. The final check at **2026-10-30 10:55 UTC** verified exactly **92 assets, totaling 16,897,047,897 bytes**. Five retained download checks independently streamed every asset and matched its byte count and SHA-256. The final public inventory matched those exact asset IDs, names, sizes and server digests, with no missing or extra assets.

The release tag identifies tested source **`695afbfebcae976e315125f9cbddbf4186dd2fe2`**. Both [push CI](https://github.com/logannye/arm64-transformer/actions/runs/41038541163) and [PR CI](https://github.com/logannye/arm64-transformer/actions/runs/41038825815) passed on that revision. This later catalog records publication bookkeeping; it does not claim that its own containing commit was the tested numerical source.

**Formal fits remain 0/180.** The strict standard-release, accuracy and exact-repeatability requirements are unchanged, and admission stream `20261104` remains unopened. Publishing the evidence does not admit a private patched framework or clear a recorded failure. See the [results](README.md), [reproduction guide](REPRODUCTION.md) and [human upstream handoff](HUMAN-REVIEW.md).

## What to download

[PUBLICATION.json](PUBLICATION.json) is the complete machine-readable catalog. Every asset entry gives its public URL, GitHub ID, bytes, SHA-256, original path, restoration paths, executable bits and streamed-download proof. Its `packages` section gives the precise release-name to on-disk-name mapping. Its `canonical_assets` section maps 64 payloads to all 201 original locations, including the four original 45-wheel inventories.

| Evidence | Release filenames | Restore under a fresh evidence root |
|---|---|---|
| Major layer: 18,282 members, 12 shards | `package.json`, `input-selection.json`, `evidence-0001.tar.gz` through `evidence-0012.tar.gz` | `outputs/upstream-integration-v2/evidence-major-v1/`, with the same filenames |
| MLX remainder: 1,672 members, one shard | `mlx-remainder-package.json`, `mlx-remainder-selection.json`, `mlx-remainder-evidence.tar.gz` | `outputs/upstream-integration-v2/evidence-mlx-remainder-v1/`, renamed to `package.json`, `input-selection.json`, `evidence-0001.tar.gz` |
| Final supplement: 17,124 members, six shards | `upstream-supplement-v3-package.json`, `upstream-supplement-v3-input-selection.json`, `upstream-supplement-v3-evidence-0001.tar.gz` through `…0006.tar.gz` | `outputs/upstream-integration-v2/evidence-supplement-v3/`, removing the `upstream-supplement-v3-` prefix |
| Remainder restoration map | `mlx-remainder-asset-aliases.json` | `outputs/upstream-integration-v2/release-assets/mlx-remainder-asset-aliases.json`, outside the verified package directory |
| Canonical native archives, wheels and two retained bytecode records | The 64 explicit names in `canonical_assets` | Every listed `restore_paths` entry, as regular files with the recorded bytes and executable bits |
| Supplemental selection history | `upstream-supplement-v3-selection-receipt.json` | `work/upstream-integration-v2/final-supplement-selection-v1/selection.receipt.json` |
| Supplemental local verification | `upstream-supplement-v3-verification.json` | `work/upstream-integration-v2/final-supplement-verification-v1.json` |

The final two files are separate post-packaging metadata assets. They preserve the layer-selection history and local verification without changing the frozen package, its selections or its scientific results. The original alias catalog and upload receipts retain their earlier `remote_verification_passed:false` or `published:false` fields. Later download and public-inventory proofs establish the subsequent publication state; the earlier records have not been rewritten.

## Verify and reconstruct

1. Download the assets from their catalog URLs and verify each byte count and SHA-256 before use. Preserve the original release filenames until verification is complete. Check the catalog against the retained public inventory rather than trusting a filename alone.
2. Restore each package into its own directory using the table and exact JSON mappings. A package directory must contain only its `package.json`, `input-selection.json` and complete shard set. Keep aliases and the two post-packaging metadata files outside it.
3. Run the preserved standard-library package verifier on each complete package. The source is available here as an unchanged [text copy](publication/work/upstream-integration-v2/package_selected_evidence.py.txt); Python can execute that text filename directly. From this repository root, the command form is:

   ```sh
   python3 -I -S -B experiments/upstream-integration-v2/publication/work/upstream-integration-v2/package_selected_evidence.py.txt verify \
     --package ABSOLUTE_PACKAGE_DIRECTORY \
     --expected-package-sha256 HASH_FROM_THE_TABLE_BELOW
   ```

   | Package | Exact `package.json` SHA-256 |
   |---|---|
   | Major | `3c239abfeaeb15f0e9b7719f2d3bf66b767d3406e21f9b6166d1a9c95459f024` |
   | MLX remainder | `77e8501e76e7d7fd1b55d1d2a64275b0b400051a9cd13599baa23edf10b98ad6` |
   | Supplement | `fb21c76bab9559ac8b0532211685accb7bc1bf5135fda500892921b768c829e6` |

4. Inspect or extract verified layers separately, then reconstruct the common `work/` and `outputs/` roots in order: major, MLX remainder, supplement. Validate member hashes and executable bits. Identical repeated paths can share bytes; a differing replacement requires an explicit authenticated `replaced_paths` record. The final supplement declares **zero differing replacements**. Do not blindly overwrite a conflicting path or extract into the live repository used to inspect this catalog.
5. Restore the 64 canonical payloads to their 201 aliases as regular files, following the catalog and recorded modes. Keep original archives intact. The retained wrapper representations authenticate extracted members and aliases; they do not regenerate the original ZIP compression bytes. Expanded runtime trees, virtual environments and disposable caches are not replacement evidence assets.

The [full reproduction guide](REPRODUCTION.md) describes native-runtime reconstruction, source checkout requirements, notices and newly bound execution. Original absolute paths, source hashes, process identities and stat records remain historical evidence. Relocation or a rebuild requires a new contract; editing an old receipt to fit a new machine would invalidate it. Verification here authenticates the saved evidence, not a fresh GPU experiment.

## Verification records kept in Git

The `publication/` directory contains byte-identical copies of the small records needed to inspect the publication chain:

- [Public inventory result](publication/work/upstream-integration-v2/remote-final-public-inventory-v1/summary.json) and [exact 92-asset contract](publication/work/upstream-integration-v2/final-public-inventory-contract-v1.json).
- [Release snapshot](publication/work/upstream-integration-v2/published-state-v1/release.json) and [tag snapshot](publication/work/upstream-integration-v2/published-state-v1/tag.json), binding the public release to the tested commit.
- [Five-download index](publication/work/upstream-integration-v2/final-download-receipts-v1.json), with all five complete proof directories, their adjacent contracts, API metadata and per-asset stream receipts.
- [Push CI](publication/work/upstream-integration-v2/final-ci-v1/summary.json), [PR CI](publication/work/upstream-integration-v2/final-pr-ci-v1/summary.json), their original run/job records and the [586-file tested-source inventory](publication/work/upstream-integration-v2/final-own-project-source-v1.json).
- [Canonical alias catalog](publication/work/upstream-integration-v2/final-supplement-selection-v1/asset-catalog.json) and [closed two-asset metadata upload](publication/work/upstream-integration-v2/final-publication-metadata-v1/upload-v1/summary.json).
- Post-packaging uploader source, tests, inspection records, retained failed inspection/style attempts and both independent reviews. Historical Python files use `.py.txt` so normal project CI does not treat preserved failed source as active implementation. `proof_copies` records each original filename, unchanged SHA-256, original executable bits and copied location. Renaming is transport bookkeeping, not a source edit.

The proof copies total about 2 MB and include no native binaries or raw tensor arrays. The catalog verifies those copy hashes separately from the 16.9 GB of release assets. Its original public-inventory result has SHA-256 `052a96a2050a0ebbc66195dfac73ff3de6fa12316f77d9a4930dadb21bb232b5`. No catalog or commit is assigned a self-referential hash.
