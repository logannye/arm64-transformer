# Read-only release asset verification

The verifier is prepared for `logannye/arm64-transformer`, release ID `428513598`, tag `transformer-upstream-integration-v2`. Preparation and tests make no GitHub requests. Actual downloads begin only when the coordinator runs the `download` command after the selected uploads finish.

`verify_release_assets_v1.py` uses only Python's standard library and the existing authenticated `gh` executable. Every API request is a read. Payloads flow from `gh api` stdout directly into SHA-256 in bounded chunks; no downloaded payload file is created. The helper checks the authenticated login, release identity and draft/publication state, the exact asset ID and name, uploaded state, byte count, GitHub's server digest, and streamed content hash. Selected asset metadata must remain identical before and after download. Local asset bytes are checked before network work and their file identities are checked again afterward.

The selected contract is explicit. Its hashes and byte counts must come from the verified local artifacts, not from GitHub's digest field. Asset IDs and names can be taken from the completed upload metadata. Save JSON with exactly these fields, replacing the example placeholders:

```json
{
  "schema": "release-asset-download-contract-v1",
  "repository": "logannye/arm64-transformer",
  "release_id": 428513598,
  "tag_name": "transformer-upstream-integration-v2",
  "draft": true,
  "prerelease": false,
  "expected_login": "logannye",
  "workspace_root": "/Users/logannye/Documents/Codex/2026-10-28/we-x20",
  "assets": [
    {
      "name": "ACTUAL-UPLOADED-NAME",
      "asset_id": 123456789,
      "local_path": "outputs/ACTUAL/LOCAL/ASSET",
      "bytes": 123,
      "sha256": "REPLACE_WITH_THE_64_LOWERCASE_HEX_DIGITS_FROM_LOCAL_VERIFICATION"
    }
  ]
}
```

The `assets` list must be unique and sorted by name. Duplicate names, IDs, paths, JSON keys, or reordered expected entries fail. GitHub's listing order is deliberately irrelevant: matching uses exact name and ID. Local paths must stay within the declared workspace under `work/` or `outputs/`, without symlink components.

Run one selected subset at a time, always using a new output directory:

```sh
python3 -I -S -B work/upstream-integration-v2/verify_release_assets_v1.py download \
  --contract work/upstream-integration-v2/EXPLICIT-ASSET-CONTRACT.json \
  --output work/upstream-integration-v2/remote-download-NEW
```

The helper permits other assets on the release during this selected-subset check. It verifies one asset at a time, has bounded metadata and payload byte limits, and stops on the first failure. Receipts retain completed, failed, and unstarted assets, partial streamed byte counts/hashes, filtered API identities, and authentication/error classifications. A retry requires a new output directory; the previous failed attempt stays intact. Every run returns the SHA-256 of its final `summary.json`.

No tokens, environment variables, raw CLI stderr, or signed download URLs are recorded. CLI stderr contributes only a byte count, digest, and coarse error classification. The release is accessed through authenticated API asset IDs; the helper does not use draft browser-download URLs.

After all assets are uploaded and individually verified, create a new contract containing the complete final release inventory. Also create a receipt index with the exact hashes returned by successful download runs:

```json
{
  "schema": "release-download-receipts-v1",
  "receipts": [
    {
      "path": "/ABSOLUTE/PATH/TO/remote-download-NEW/summary.json",
      "sha256": "REPLACE_WITH_THE_TRUSTED_64_HEX_SUMMARY_DIGEST"
    }
  ]
}
```

Keep each receipt's adjacent `contract.json`. The final checker authenticates those original contracts, complete expected orders, successful streams, stable asset identities, and before/after authentication and release records. It requires successful download proof for every final expected asset. Download receipts from the draft can remain valid after publication when the same asset IDs, names, sizes, and digests persist; the final contract must explicitly match the release's current draft/prerelease state.

```sh
python3 -I -S -B work/upstream-integration-v2/verify_release_assets_v1.py inventory \
  --contract work/upstream-integration-v2/COMPLETE-FINAL-ASSET-CONTRACT.json \
  --download-receipts work/upstream-integration-v2/DOWNLOAD-RECEIPT-INDEX.json \
  --output work/upstream-integration-v2/remote-final-inventory-NEW
```

This separate final command rejects missing, duplicate, unexpected, changed, or incompletely verified assets. It reads the complete paginated inventory twice and rechecks the account and release state. It downloads no payload again. Publication, additions, replacements, or renames must not occur concurrently with the final inventory check.

Sixteen synthetic test methods cover the explicit contract, remote metadata, exact final inventory, altered/reordered/missing receipts, local corruption, mid-download changes, partial failure retention, authentication failures, byte limits, timeout cleanup, and absence of sensitive stderr in receipts. Tests use a fake GitHub client and small local subprocess streams. They do not establish that any real remote asset has been verified; that requires successful actual run receipts.
