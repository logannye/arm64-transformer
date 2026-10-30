# Two post-package publication metadata assets

This additive uploader preserves the frozen selector, selection, package, shards,
and earlier upload receipts. It adds only the exact two files in the already
frozen `plan.json` (SHA256
`14a57d584d2616a2ae4228618c3ca9cba44f10c1c978a232d541308d3be7863d`): the original
selection receipt containing layer replacements/exclusions, and the original
local package-verification result. Each asset retains its declared original
workspace-relative restore path. This phase makes no numerical, admission, or
publication claim.

`upload.py` authenticates the closed complete-delta summary
`1362f56e74da6850eb353cfd89c6a12ce45d3860c9b0ed63dfd6b4f2392aebe0` and its exact
90-asset contract, then checks the account, still-draft release, and every full
remote asset identity before and during serial uploads. It reuses the reviewed
`UploadClient` (5400-second total/silent-response bound), API identity checks,
and streamed-download contract validators. It never replaces an asset or
rehashes an earlier payload. Both prospective rows exist in the saved summary
before remote execution; errors and unstarted rows remain visible.

The two local JSON files are hashed and checked against their original full
stat identities before and after upload. Their original JSON links the same
selection/package already uploaded. The reviewed 64 MiB metadata reader bound
applies; the unrelated delta helper's 8 MiB evidence-binding limit is not used.

Run only after the preceding 14-asset remote download verification closes and
with no other release writer:

```sh
python3 -I -S -B work/upstream-integration-v2/final-publication-metadata-v1/upload.py inspect
python3 -I -S -B work/upstream-integration-v2/final-publication-metadata-v1/upload.py upload \
  --output work/upstream-integration-v2/final-publication-metadata-v1/upload-v1
```

A successful upload writes a two-asset `download-contract.json` and a
92-asset `final-inventory-contract.json`. Root must run the unchanged streamed
remote verifier on the former, then authenticate the whole latter inventory
against all prior passing download receipts plus the new receipt, before
publishing the release. Upload success alone leaves `published` and
`remote_payloads_verified` false. Preserve the complete summary, API responses,
and both contracts separately in the final publication bookkeeping catalog.
