"""Process-local byte authentication with complete immutable-tree guards.

The original authority still hashes every runtime and framework file. Cached
byte identities are reused only while every file, directory and symlink has
the same identity, mode, size, mtime and ctime. No cache crosses a process.
"""

import json
import os
import stat
from pathlib import Path

MAX_ENTRIES = 250_000


def metadata(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def snapshot(roots, aliases=()):
    """Enumerate complete trees, following and separately recording symlinks."""
    records, visited = {}, set()

    def put(path, value):
        name = str(path)
        if name in records and records[name] != value:
            raise ValueError("Runtime entry changed during inventory: " + name)
        records[name] = value
        if len(records) > MAX_ENTRIES:
            raise ValueError("Runtime guard inventory budget exceeded")

    def walk(path):
        path = Path(path)
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode):
            target = path.resolve(strict=True)
            put(path, ("link", metadata(before), os.readlink(path), str(target)))
            if metadata(path.lstat()) != metadata(before):
                raise ValueError("Runtime symlink changed during inventory")
            walk(target)
            return
        if stat.S_ISREG(before.st_mode):
            put(path, ("file", metadata(before)))
            return
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError("Nonregular runtime guard entry: " + str(path))
        put(path, ("directory", metadata(before)))
        if str(path) in visited:
            return
        visited.add(str(path))
        with os.scandir(path) as directory:
            for entry in directory:
                # The parent is canonical; ordinary entries need no repeated
                # resolution of every ancestor of every one of 75k files.
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISREG(info.st_mode):
                    put(entry.path, ("file", metadata(info)))
                else:
                    walk(entry.path)
        if metadata(path.lstat()) != metadata(before):
            raise ValueError("Runtime directory changed during inventory")

    for root in sorted(set(map(str, roots))):
        path = Path(root)
        if not path.is_absolute() or path.resolve(strict=True) != path:
            raise ValueError("Noncanonical runtime guard root")
        walk(path)
    for alias in sorted(aliases):
        walk(alias)
    return tuple(sorted(records.items()))


class RuntimeGuard:
    """No flags or caller-supplied cached hashes can replace full verification."""

    def __init__(self, authority, inventory, roles, contract):
        self._authority, self._contract = authority, contract
        self._pid = os.getpid()
        self._inventory = json.loads(contract.canonical(inventory))
        self._roles, self._roots = json.loads(contract.canonical(roles)), None
        self._aliases = tuple(self._inventory["layout"]["aliases"])
        self._snapshot, self._verified = None, None
        self._full, self._checks, self._fingerprint = 0, 0, None

    def __getattr__(self, name):
        return getattr(self._authority, name)

    def verify(self, *, full=False):
        C = self._contract
        C.require(os.getpid() == self._pid, "Runtime authentication cache crossed a process")
        if self._roots is None:
            C.require(
                C.canonical(self._inventory) == self._authority.inventory_json,
                "Runtime guard inventory differs from the authenticated authority",
            )
            receipt = C.read(self._authority.receipt_path, self._authority.receipt_sha256)
            self._roots = tuple(
                sorted(
                    {
                        *self._inventory["layout"]["scan_roots"],
                        *(r["root"] for r in self._roles.values()),
                        str(Path(self._authority.receipt_path).parent),
                        *(r["payload_root"] for r in receipt["frameworks"].values()),
                        *(
                            str(Path(r["manifest"]["path"]).parent)
                            for r in receipt["frameworks"].values()
                        ),
                    }
                )
            )
        before = snapshot(self._roots, self._aliases)
        if self._snapshot is not None:
            C.require(
                before == self._snapshot, "Authenticated runtime inventory or metadata changed"
            )
        if self._verified is None or full:
            receipt, payloads = self._authority.verify()
            verified = (C.canonical(receipt), C.canonical(payloads))
            after = snapshot(self._roots, self._aliases)
            C.require(before == after, "Runtime changed during full authentication")
            C.require(
                self._verified is None or verified == self._verified,
                "Fully authenticated runtime bytes changed",
            )
            self._snapshot, self._verified = after, verified
            self._fingerprint = C.identity(after)
            self._full += 1
        self._checks += 1
        # Return independent values, as the original verifier does; callers
        # cannot mutate the stored byte identity through a returned dictionary.
        return tuple(json.loads(value) for value in self._verified)

    def witness(self):
        self._contract.require(self._verified is not None, "Runtime guard has not authenticated")
        return dict(
            schema="process-local-runtime-guard-v1",
            pid=self._pid,
            mode="full_hash_then_complete_unchanged_inventory",
            full_authentications=self._full,
            guard_checks=self._checks,
            entries=len(self._snapshot),
            roots=list(self._roots),
            metadata_fingerprint_sha256=self._fingerprint,
            runtime_identity_sha256=json.loads(self._verified[0])["runtime_identity_sha256"],
            inventory_identity_sha256=self._contract.identity(self._inventory),
            source_roles_sha256=self._contract.identity(self._roles),
        )
