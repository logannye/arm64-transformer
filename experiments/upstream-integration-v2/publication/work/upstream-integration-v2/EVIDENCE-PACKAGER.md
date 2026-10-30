# Explicit evidence packaging

The packager is prepared and tested on tiny temporary fixtures. No real project evidence has been packaged by this work.

Create a new final selection only after its files are immutable. The planning inventories are insufficient because they deliberately contain missing hashes and active-run placeholders. The selection JSON has exactly this schema:

```json
{
  "schema": "explicit-evidence-selection-v1",
  "workspace_root": "/Users/logannye/Documents/Codex/2026-10-28/we-x20",
  "files": [
    {
      "path": "work/example/result.json",
      "bytes": 123,
      "sha256": "64 lowercase hexadecimal characters",
      "executable_bits": 0
    }
  ]
}
```

`executable_bits` is the integer `stat.S_IMODE(mode) & 0o111`, preserving owner, group, and other execute permissions separately. A typical executable has value 73 (octal 0111); a nonexecutable file has value 0. Every file is explicitly listed with its real size and SHA-256. Paths must start with `work/` or `outputs/`. Absolute paths, traversal, duplicate paths, symlink files, symlink parents, and special files are rejected. Nothing is recursively discovered or selected.

Run with the existing compatible Python; no installation or environment change is needed:

```sh
work/upstream-integration-v2/mlx/runtime-env/bin/python -S \
  work/upstream-integration-v2/package_selected_evidence.py pack \
  --selection work/final-evidence-selection.json \
  --output work/fresh-evidence-package
```

The output directory must not exist and its parent must already exist. Default shards target 1,073,741,824 uncompressed tar bytes, including headers and padding. A single file larger than the target remains one explicitly marked oversize shard. `--shard-bytes` can set a different declared target before packaging.

Members are sorted by their relative POSIX path. PAX tar headers have zero time/owner IDs, empty owner names, and permissions 0644 OR the selected executable bits. Gzip uses level 1, zero timestamp, and no stored filename. Python and zlib identities are recorded; deterministic compressed bytes were verified by repeated packaging in the same runtime. Relative `work/` and `outputs/` locations are preserved.

Each source is opened through directory descriptors with `O_NOFOLLOW`. Its inode, mode, size, modification time, change time, and link count must agree before and after streaming; payload size and SHA-256 must match the selection. All source stats and the original selection are checked again after packaging. Each compressed archive is reopened, fully decompressed with bounded reads, and every member's order, metadata, size, and SHA-256 is verified. Gzip checksum/end-of-stream validation and the compressed archive hash are checked too. The final package is then verified again against the copied selection and declared shard plan.

Successful output contains `input-selection.json`, `package.json`, and numbered `evidence-*.tar.gz` shards. The command returns the package-receipt SHA-256 and each archive SHA-256 for independent publication/download checks. A failure keeps an explicitly failed receipt, any completed shards, and partial archives; it cannot qualify as verified.

To verify a copied package, use the trusted receipt hash returned by the packing command:

```sh
work/upstream-integration-v2/mlx/runtime-env/bin/python -S \
  work/upstream-integration-v2/package_selected_evidence.py verify \
  --package work/copied-evidence-package \
  --expected-package-sha256 TRUSTED_SHA256
```

Verification requires no source workspace or numerical framework. It authenticates the copied selection, complete shard/member inventory, every streamed payload, compressed archive hashes, and retained source-stat evidence. It does not extract files or execute their contents. This is an integrity mechanism; scientific qualification remains in the original evidence and admission/audit machinery.
