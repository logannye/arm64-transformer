"""Durable fresh-output export; input/source authentication is never retried."""

import hashlib
import os
from pathlib import Path


def write_exact(path, data, contract, *, attempts=3):
    """Accept only source-exact bytes and two fully stable metadata reads."""
    C, path = contract, Path(path).absolute()
    C.require(
        type(attempts) is int and 1 <= attempts <= 3, "Bounded artifact export attempts required"
    )
    C.require(path.resolve() == path and not path.exists(), "Fresh canonical artifact required")
    expected = hashlib.sha256(data).hexdigest()
    with path.open("xb", buffering=0) as stream:
        view = memoryview(data)
        while view:
            count = stream.write(view)
            C.require(count is not None and count > 0, "Artifact write did not advance")
            view = view[count:]
        os.fsync(stream.fileno())
        created = os.fstat(stream.fileno())
    errors = []
    for _ in range(attempts):
        current = path.lstat()
        C.require(
            (current.st_dev, current.st_ino, current.st_mode, current.st_size)
            == (created.st_dev, created.st_ino, created.st_mode, len(data)),
            "Exported artifact identity or length changed",
        )
        try:
            first = C.binding(path)
            middle = path.stat()
            second = C.binding(path)
            last = path.stat()
        except ValueError as exc:
            if str(exc) != "Bound file changed during read":
                raise
            errors.append(str(exc))
            continue
        C.require(
            first["sha256"] == second["sha256"] == expected
            and first["bytes"] == second["bytes"] == len(data),
            "Exported artifact differs from the in-memory source bytes",
        )
        if C._stat(middle) != C._stat(last):
            errors.append("Metadata changed between independently stable reads")
            continue
        return first, dict(
            binding=first,
            expected_sha256=expected,
            expected_bytes=len(data),
            policy="fsync_then_two_stable_exact_content_bindings",
            fsync=True,
            stable_reads=2,
            metadata_retries=len(errors),
            metadata_failures=errors,
        )
    raise ValueError("Fresh artifact metadata did not stabilize within export budget")
