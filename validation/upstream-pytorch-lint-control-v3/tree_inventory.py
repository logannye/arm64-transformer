"""Bounded, stable tool-tree closure with explicit declared-root symlink traversal.

Only regular .pyc files are excluded. __pycache__ directories are inspected and
must contain only regular .pyc files; aliases or other hidden payloads fail.
Directory aliases are expanded at their logical path, so all reachable source
and native bytes appear even in the single-root compatibility interface.
"""

import hashlib
import os
import stat
from pathlib import Path

MAX_ENTRIES = 500_000
MAX_UNIQUE_BYTES = 16 * 1024**3
MAX_DEPTH = 96
BLOCK = 1024 * 1024
EXCLUSIONS = {
    "regular_file_suffix": ".pyc",
    "bytecode_directory": "__pycache__",
    "bytecode_directory_content": "regular .pyc files only",
    "aliases_to_excluded_paths": "rejected",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def stamp(value):
    return tuple(
        getattr(value, key)
        for key in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    )


class _Scanner:
    def __init__(self, roots, max_entries, max_unique_bytes, max_depth):
        for name, limit in (
            ("max_entries", max_entries),
            ("max_unique_bytes", max_unique_bytes),
            ("max_depth", max_depth),
        ):
            require(type(limit) is int and limit > 0, f"Invalid inventory bound {name}")
        require(isinstance(roots, dict) and roots, "Declare at least one tool root")
        self.roots = {}
        for name, value in roots.items():
            require(
                isinstance(name, str)
                and name
                and not name.startswith("/")
                and "\\" not in name
                and not any(part in ("", ".", "..") for part in name.split("/"))
                and not any(ord(c) < 32 or ord(c) == 127 for c in name),
                "Invalid declared root name",
            )
            path = Path(value).absolute()
            require(path.resolve(strict=True) == path, f"Declared root is not canonical: {path}")
            require(stat.S_ISDIR(path.lstat().st_mode), f"Declared root is not a directory: {path}")
            require(path.name != "__pycache__", f"Excluded directory cannot be a root: {path}")
            require(path not in self.roots.values(), f"Duplicate declared root: {path}")
            self.roots[name] = path
        self.max_entries, self.max_unique_bytes, self.max_depth = (
            max_entries,
            max_unique_bytes,
            max_depth,
        )
        self.entries = self.unique_bytes = 0
        self.watches = {}
        self.directory_lists = {}
        self.links = {}
        self.files = {}
        self.excluded = set()

    def owner(self, path):
        matches = [(name, root) for name, root in self.roots.items() if path.is_relative_to(root)]
        require(matches, f"Path escapes declared tool roots: {path}")
        return max(matches, key=lambda item: len(item[1].parts))

    def excluded_path(self, path):
        _, root = self.owner(path)
        relative = path.relative_to(root)
        return "__pycache__" in relative.parts or path.name.endswith(".pyc")

    def watch(self, path):
        require(path.parent.resolve(strict=True) == path.parent, f"Unresolved parent alias: {path}")
        try:
            value = path.lstat()
        except FileNotFoundError as error:
            raise ValueError(f"Missing/broken tool path: {path}") from error
        identity = stamp(value)
        require(
            path in self.watches or len(self.watches) < self.max_entries,
            f"Distinct-node entry bound exceeded: {path}",
        )
        require(
            path not in self.watches or self.watches[path] == identity, f"Tool path changed: {path}"
        )
        self.watches[path] = identity
        return value

    def count(self, path, depth):
        self.entries += 1
        require(
            self.entries <= self.max_entries, f"Expanded inventory entry bound exceeded: {path}"
        )
        require(depth <= self.max_depth, f"Inventory depth bound exceeded: {path}")

    def read_link(self, path):
        before = self.watch(path)
        require(stat.S_ISLNK(before.st_mode), f"Expected symlink: {path}")
        raw = os.readlink(path)
        require(path not in self.links or self.links[path] == raw, f"Tool alias changed: {path}")
        self.links[path] = raw
        require(stamp(path.lstat()) == stamp(before), f"Tool alias changed while reading: {path}")
        # Preserve .. until preceding components and aliases have been resolved.
        target = path.parent / raw
        self.owner(target)
        require(not self.excluded_path(target), f"Alias targets excluded payload: {path} -> {raw}")
        return raw, target

    def resolve(self, path, seen=()):
        _, root = self.owner(path)
        require(not self.excluded_path(path), f"Alias resolves to excluded payload: {path}")
        current = root
        for component in path.relative_to(root).parts:
            if any(current.is_relative_to(r) for r in self.roots.values()):
                require(
                    stat.S_ISDIR(self.watch(current).st_mode),
                    f"Non-directory path prefix: {current}",
                )
            if component == "..":
                current = current.parent
                continue
            current = current / component
            self.owner(current)
            value = self.watch(current)
            if stat.S_ISLNK(value.st_mode):
                require(current not in seen, f"Symlink cycle at {current}")
                require(len(seen) < self.max_depth, f"Symlink depth bound exceeded: {current}")
                _, target = self.read_link(current)
                current = self.resolve(target, (*seen, current))
        self.owner(current)
        require(not self.excluded_path(current), f"Alias resolves to excluded payload: {current}")
        return current

    def names(self, path):
        names = []
        with os.scandir(path) as entries:
            for entry in entries:
                names.append(entry.name)
                require(len(names) <= self.max_entries, f"Directory entry bound exceeded: {path}")
        return tuple(sorted(names))

    def listing(self, path):
        before = self.watch(path)
        require(stat.S_ISDIR(before.st_mode), f"Expected directory: {path}")
        names = self.names(path)
        require(
            path not in self.directory_lists or self.directory_lists[path] == names,
            f"Directory contents changed: {path}",
        )
        self.directory_lists[path] = names
        require(stamp(path.lstat()) == stamp(before), f"Directory changed while listing: {path}")
        return names

    def hash_file(self, path, before):
        key = (before.st_dev, before.st_ino)
        if key in self.files:
            identity, row = self.files[key]
            require(identity == stamp(before), f"Cached physical file changed: {path}")
            return dict(row)
        self.unique_bytes += before.st_size
        require(self.unique_bytes <= self.max_unique_bytes, f"Unique-byte bound exceeded: {path}")
        digest = hashlib.sha256()
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            require(
                stamp(os.fstat(stream.fileno())) == stamp(before),
                f"File changed before read: {path}",
            )
            count = 0
            for block in iter(lambda: stream.read(BLOCK), b""):
                count += len(block)
                require(count <= before.st_size, f"File grew while reading: {path}")
                digest.update(block)
            require(
                count == before.st_size and stamp(os.fstat(stream.fileno())) == stamp(before),
                f"File changed while reading: {path}",
            )
        require(stamp(path.lstat()) == stamp(before), f"File path changed while reading: {path}")
        row = {
            "bytes": count,
            "sha256": digest.hexdigest(),
            "executable_bits": before.st_mode & 0o111,
        }
        self.files[key] = (stamp(before), row)
        return dict(row)

    def bytecode_directory(self, path, depth):
        for name in self.listing(path):
            child = path / name
            self.count(child, depth + 1)
            value = self.watch(child)
            require(
                stat.S_ISREG(value.st_mode) and name.endswith(".pyc"),
                f"Unexpected payload in excluded bytecode directory: {child}",
            )
            self.excluded.add(child)

    def walk(self, physical, logical, rows, ancestors, depth):
        self.count(physical, depth)
        value = self.watch(physical)
        if stat.S_ISLNK(value.st_mode):
            require(not self.excluded_path(physical), f"Alias occupies excluded path: {physical}")
            raw, _ = self.read_link(physical)
            resolved = self.resolve(physical)
            require(
                physical.resolve(strict=True) == resolved,
                f"Actual symlink resolution differs from authenticated traversal: {physical}",
            )
            target = self.watch(resolved)
            owner, root = self.owner(resolved)
            alias = {
                "link": raw,
                "resolved": str(resolved),
                "target_root": owner,
                "target_relative": resolved.relative_to(root).as_posix(),
            }
            if stat.S_ISREG(target.st_mode):
                rows[logical] = {**self.hash_file(resolved, target), "kind": "file_alias", **alias}
                return
            require(
                stat.S_ISDIR(target.st_mode),
                f"Alias targets special node: {physical} -> {resolved}",
            )
            rows[logical] = {"kind": "directory_alias", "mode": target.st_mode & 0o7777, **alias}
            self.walk_directory(resolved, logical, rows, ancestors, depth)
            return
        if stat.S_ISREG(value.st_mode):
            if physical.name.endswith(".pyc"):
                self.excluded.add(physical)
                return
            rows[logical] = self.hash_file(physical, value)
            return
        require(stat.S_ISDIR(value.st_mode), f"Special tool-tree node is forbidden: {physical}")
        if physical.name == "__pycache__":
            self.bytecode_directory(physical, depth)
            return
        rows[logical] = {"kind": "directory", "mode": value.st_mode & 0o7777}
        self.walk_directory(physical, logical, rows, ancestors, depth)

    def walk_directory(self, path, logical, rows, ancestors, depth):
        value = self.watch(path)
        key = (value.st_dev, value.st_ino)
        require(key not in ancestors, f"Directory alias cycle: {path}")
        for name in self.listing(path):
            require(name not in (".", "..") and "/" not in name, f"Unsafe directory member: {name}")
            child_logical = name if not logical else logical + "/" + name
            self.walk(path / name, child_logical, rows, (*ancestors, key), depth + 1)

    def finish(self):
        # Cached payloads and aliases are rechecked after the entire forest, not
        # only when first hashed. Listing checks include explicitly excluded files.
        for path, expected in self.watches.items():
            require(
                path.parent.resolve(strict=True) == path.parent, f"Parent alias changed: {path}"
            )
            require(stamp(path.lstat()) == expected, f"Tool path changed before closure: {path}")
        for path, names in self.directory_lists.items():
            require(
                self.names(path) == names,
                f"Directory changed before closure: {path}",
            )
        for path, raw in self.links.items():
            require(os.readlink(path) == raw, f"Alias changed before closure: {path}")
        for path, expected in self.watches.items():
            require(stamp(path.lstat()) == expected, f"Tool path changed during closure: {path}")

    def run(self):
        result = {}
        for name, root in sorted(self.roots.items()):
            self.count(root, 0)
            root_stat = self.watch(root)
            rows = {
                "": {
                    "kind": "inventory_root",
                    "path": str(root),
                    "mode": root_stat.st_mode & 0o7777,
                    "exclusions": dict(EXCLUSIONS),
                    "bounds": {
                        "expanded_entries": self.max_entries,
                        "unique_file_bytes": self.max_unique_bytes,
                        "depth": self.max_depth,
                    },
                }
            }
            self.walk_directory(root, "", rows, (), 0)
            result[name] = dict(sorted(rows.items()))
        self.finish()
        return result


def inventory_forest(
    roots, *, max_entries=MAX_ENTRIES, max_unique_bytes=MAX_UNIQUE_BYTES, max_depth=MAX_DEPTH
):
    """Hash a closed forest, expanding aliases only within explicit canonical roots."""
    return _Scanner(roots, max_entries, max_unique_bytes, max_depth).run()


def inventory(root, allowed_roots=None, **limits):
    """Compatibility map; canonical file rows retain bytes/SHA/executable bits.

    Optional allowed_roots is an explicit name->Path mapping. All its trees are
    checked; aliases beneath root are expanded in the returned relative map.
    The forest-only root/policy row is omitted for flat helper/wheel callers.
    """
    roots = dict(allowed_roots or {})
    root = Path(root).absolute()
    matches = [name for name, path in roots.items() if Path(path).absolute() == root]
    if matches:
        name = matches[0]
    else:
        name = "__inventory_primary__"
        require(name not in roots, "Reserved inventory root name conflicts")
        roots[name] = root
    rows = inventory_forest(roots, **limits)[name]
    return {path: row for path, row in rows.items() if path}
