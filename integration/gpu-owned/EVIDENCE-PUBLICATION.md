# Publishing closed owned-study evidence

`prepare_evidence.py` packages a fixed list of completed or terminated
development attempts. It imports only the Python standard library. It does not
run numerical work, import frameworks, compile code, extract archives or upload
anything. A successful archive receipt establishes evidence preservation, not
numerical qualification or admission.

The [frozen plan](evidence-plan-v1.json) contains every original workspace-relative
member name, SHA-256, byte count and file mode. Its SHA-256 is
`eeaacae7de524030806096a50745ab114e27e8751de9a596c53ae7bc426f5cdc`.
The archive helper itself is bound by SHA-256
`2a30d78a30cfe75e56bf3a133f7611e0544706c4b1171a40732eb113b0b4ba51`.

| Archive group | Included closed evidence | Files | Raw bytes | Gzip bytes |
|---|---|---:|---:|---:|
| `joint-runtime-v1-through-v4` | All four attempts, two failed-helper snapshots and retry declaration | 31 | 223,920 | 29,937 |
| `layout-diagnostic-v1` | Four workers, all 328 raw NPZ files, case records, plan and logs | 998 | 13,399,774 | 6,224,205 |
| `half-index-v1` | Two workers, all 80 raw NPZ files, records and logs; independent audit and its helper | 252 | 6,664,645 | 206,933 |
| `failed-model-v1` | Failed model attempt, all 26 raw NPZ files, original manifest and helper snapshot | 43 | 5,134,450 | 3,055,238 |

The input namespace is explicitly restricted to listed roots under
`work/gpu-v4-owned/`. Every named root is included in full. The still-active
`model-compatibility-v2` attempt was excluded when this plan was prepared.
The candidate framework build is a separate pending identity. Framework
runtime/source trees, NumPy overlays, wheel payloads and native libraries are
not duplicated here; their existing authenticated references remain inside the
original receipts. A later completed group needs a separately reviewed,
new plan and output directory, preserving this plan and its archives.

The helper rejects missing or changed closure receipts, active attempts,
symlinks (including ancestor components), unsafe member names, native binary
payloads and undeclared file types. The input and compressed output of each
group must each be strictly smaller than 2,000,000,000 bytes. It checks free
storage before creating archives and uses bounded streaming buffers.

Source contents and inventories are checked before and after archiving, with
file identity checks around each read. Verification checks every member's
SHA-256, size and mode, exact member order/coverage, safe regular-file type,
canonical metadata and final zero padding. Duplicate, reordered, extra or
missing members, symlink/hardlink members and payloads hidden after the first tar
end marker fail verification. No paths are rewritten inside source files.
Failed archive attempts retain their partial bytes and a failed receipt; the
helper refuses to overwrite them.

Run from the repository using a fresh archive output directory. Replace the
workspace argument with the reconstructed parent of `work/` and `outputs/`:

```sh
.venv/bin/python -m unittest discover -s integration/gpu-owned \
  -p test_prepare_evidence.py -v
.venv/bin/python integration/gpu-owned/prepare_evidence.py build \
  --workspace /ABSOLUTE/RECONSTRUCTED/WORKSPACE \
  --plan integration/gpu-owned/evidence-plan-v1.json \
  --plan-sha256 eeaacae7de524030806096a50745ab114e27e8751de9a596c53ae7bc426f5cdc \
  --output /ABSOLUTE/FRESH/ARCHIVE/DIRECTORY
.venv/bin/python integration/gpu-owned/prepare_evidence.py verify \
  --plan integration/gpu-owned/evidence-plan-v1.json \
  --plan-sha256 eeaacae7de524030806096a50745ab114e27e8751de9a596c53ae7bc426f5cdc \
  --archive-directory /ABSOLUTE/ARCHIVE/DIRECTORY
```

The original four local archives passed complete member verification, covering
1,324 files and 25,422,789 raw bytes. The [local packaging receipt](evidence-archives-v1.json)
records their outer SHA-256 hashes. Twenty-one CPU-only regression and mutation
tests pass. No upload or remote verification has occurred in this packaging
step. Publication still requires the source revision to be frozen, release
assets to be uploaded, and downloaded asset hashes to match this receipt before
claiming remote availability. Both formal study streams remain unopened.
