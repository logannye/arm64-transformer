# Read-only verification of the owned release

`verify_release_assets.py` is restricted to `logannye/arm64-transformer`, tag
`transformer-gpu-owned-v4`, authenticated GitHub account `logannye`. It has no
upload, release-creation, editing or deletion capability. Importing the helper
does not contact GitHub. CPU tests use fake clients and local Python child
processes; they make no network requests.

**The release ID is not known or guessed here.** After the actual release is
created and its assets uploaded, record the returned release ID and each asset
ID. Prepare an explicit contract and freeze its canonical SHA-256. The verifier
requires the release ID and contract hash as separate command-line anchors;
neither is inferred from the contract or looked up by an unbound tag request.

The contract has exactly these fields:

| Field | Required value |
|---|---|
| `schema` | `gpu-owned-release-assets-v1` |
| `repository` | `logannye/arm64-transformer` |
| `release_id` | The positive integer returned for the actual new release |
| `tag_name` | `transformer-gpu-owned-v4` |
| `expected_login` | `logannye` |
| `draft`, `prerelease` | Explicit booleans matching the intended release state |
| `assets` | Nonempty, exact asset inventory sorted by unique asset name |

Each asset contains exactly `name`, `asset_id`, `local_path`, `bytes`, and
`sha256`. IDs come from the actual upload receipts. `local_path` is a safe
workspace-relative path beneath `work/` or `outputs/`; bytes and SHA-256 come
from the retained local file. Each asset must be smaller than 2,000,000,000
bytes. Duplicate IDs, names or paths are rejected.

Canonical JSON uses Python's `json.dumps(value, sort_keys=True,
separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")` with
**no trailing newline**. Duplicate JSON keys and nonfinite constants are
rejected before validation. The external SHA-256 hashes these canonical bytes;
the receipt separately records the original contract file's byte hash. Record
the canonical hash during the reviewed contract-freeze step, then supply that
fixed value to the verification command.

```sh
.venv/bin/python -m unittest discover -s integration/gpu-owned \
  -p test_verify_release_assets.py -v
.venv/bin/python integration/gpu-owned/verify_release_assets.py \
  --contract /ABSOLUTE/FROZEN/ASSET-CONTRACT.json \
  --workspace /ABSOLUTE/RECONSTRUCTED/WORKSPACE \
  --expected-release-id ACTUAL_RECORDED_RELEASE_ID \
  --expected-contract-sha256 RECORDED_CANONICAL_CONTRACT_SHA256 \
  --output /ABSOLUTE/FRESH/VERIFICATION/DIRECTORY
```

The command first hashes every local expected asset. It then authenticates the
account, checks the release identity, enumerates all remote assets using pages
of 100, and requires an exact inventory. Every remote asset is downloaded using
an explicit read-only `gh api --method GET` request, with bytes streamed directly
into SHA-256. No downloaded payload is copied to disk. The byte count and hash
must agree with both the local contract and GitHub's server digest. Asset
metadata is checked before and after each download; the full inventory,
release, authenticated identity, local file hashes/stat identities, contract
and verifier source are checked again before a passing receipt is written.

Metadata replies are capped at 4 MiB. Stderr is capped at 1 MiB; only its byte
count, hash and fixed error classifications are retained. A 16 KiB prefix is
used in memory for those classifications and discarded. Raw stderr, command
arguments, signed download URLs and credentials are never written to receipts.
Public API origin URLs are validated in memory; arbitrary labels, release
text and metadata URLs appear only inside metadata hashes. Unexpected error
messages are represented by their exception type, preventing remote error
text from leaking into saved evidence. Requests have total and inactivity
timeouts; a failed or interrupted run retains its failed receipt.

The 27 no-network tests cover external-anchor mismatches, wrong accounts and
release identities, changed local/remote assets, one-byte hash differences,
truncation, symlinks, unsafe paths, duplicate/reordered/missing/extra assets,
metadata changes, pagination bounds, stderr and metadata limits, timeout
cleanup and suppression of synthetic secret-bearing URLs.

This helper has been tested locally without contacting GitHub. It does not
establish that the forthcoming release exists or that any asset is published.
A passing future receipt proves the explicitly bound release bytes and exact
inventory at verification time; it does not qualify the frameworks or open
the formal study. Verifying a draft also does not establish public visibility.
