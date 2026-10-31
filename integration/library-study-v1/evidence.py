"""Bounded lossless admission arrays and independent numerical decisions."""

import hashlib
import importlib.util
import io
import math
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("library_evidence_contract", HERE / "contract.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)
MAX_BYTES = 512 * 1024**2


def storage(math_root):
    path = Path(math_root) / "integration/trained-precision/zip_evidence.py"
    spec = importlib.util.spec_from_file_location("library_exact_zip_storage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decision(actual, expected, kind):
    import numpy as np

    from transformer.gpu_study_admission import TOLERANCES

    a, b = np.asarray(actual), np.asarray(expected)
    C.require(a.shape == b.shape and a.size > 0, "Admission result shape differs or is empty")
    C.require(
        a.dtype in (np.float32, np.float64, np.int32, np.int64)
        and b.dtype in (np.float32, np.float64, np.int32, np.int64),
        "Wrong result dtype",
    )
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if kind == "exact":
        passed = finite and a.dtype == b.dtype and a.tobytes() == b.tobytes()
        return dict(passed=passed, finite=finite, kind=kind, exact_bytes=passed)
    C.require(a.dtype.kind == b.dtype.kind == "f", "Numerical evidence must be floating point")
    atol, rtol = TOLERANCES[kind]
    a, b = a.astype(np.float64), b.astype(np.float64)
    difference = np.abs(a - b)
    failed = ~np.isfinite(a) | ~np.isfinite(b) | (difference > atol + rtol * np.abs(b))
    return dict(
        passed=not bool(failed.any()),
        finite=finite,
        kind=kind,
        atol=atol,
        rtol=rtol,
        failed_coordinates=int(failed.sum()),
        coordinates=int(b.size),
    )


def save_pair(actual, expected, store, math_root):
    import numpy as np

    values = dict(
        actual=np.array(actual, copy=True, order="C"),
        expected=np.array(expected, copy=True, order="C"),
    )
    C.require(sum(v.nbytes for v in values.values()) <= MAX_BYTES, "Raw pair exceeds memory budget")
    C.require(
        all(v.dtype in (np.float32, np.float64, np.int32, np.int64) for v in values.values()),
        "Wrong raw pair dtype",
    )
    with io.BytesIO() as stream:
        np.savez_compressed(stream, **values)
        C.require(stream.tell() <= MAX_BYTES, "Compressed raw pair exceeds budget")
        stream.seek(0)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        row = storage(math_root).pack_stream(stream, digest, store)
    return {k: row[k] for k in ("sha256", "bytes")}


def load_pair(row, store, math_root):
    C.require(
        set(row) == {"sha256", "bytes"}
        and type(row["bytes"]) is int
        and 0 < row["bytes"] <= MAX_BYTES,
        "Wrong raw pair binding",
    )
    with io.BytesIO() as stream:
        recipe = storage(math_root).reconstruct(store, row["sha256"], stream, max_bytes=MAX_BYTES)
        C.require(recipe["bytes"] == row["bytes"], "Raw pair byte count differs")
        stream.seek(0)
        values = decode_arrays(stream, ("actual", "expected"))
    return values["actual"], values["expected"]


def decode_arrays(stream, names, *, max_bytes=MAX_BYTES):
    """Check every numerical NPY header before allocating its declared array."""
    import numpy as np

    C.require(type(max_bytes) is int and 0 < max_bytes <= MAX_BYTES, "Wrong array budget")
    with zipfile.ZipFile(stream) as archive:
        entries = archive.infolist()
        C.require(
            len(entries) == len(names)
            and {e.filename for e in entries} == {name + ".npy" for name in names}
            and sum(e.file_size for e in entries) <= max_bytes + 4096 * len(names),
            "Unexpected or oversized array inventory",
        )
        values, payload_bytes = {}, 0
        for entry in entries:
            with archive.open(entry) as raw:
                version = np.lib.format.read_magic(raw)
                headers = {
                    (1, 0): np.lib.format.read_array_header_1_0,
                    (2, 0): np.lib.format.read_array_header_2_0,
                }
                C.require(version in headers, "Unsupported numerical NPY header version")
                shape, fortran, dtype = headers[version](raw, max_header_size=4096)
                C.require(
                    dtype in (np.float32, np.float64, np.int32, np.int64)
                    and not fortran
                    and all(type(n) is int and n >= 0 for n in shape),
                    "Unexpected array type or layout",
                )
                size = math.prod(shape) * dtype.itemsize
                payload_bytes += size
                C.require(
                    payload_bytes <= max_bytes and raw.tell() + size == entry.file_size,
                    "NPY header exceeds actual array bounds",
                )
                raw.seek(0)
                values[entry.filename[:-4]] = np.load(raw, allow_pickle=False, max_header_size=4096)
    return values


def audit_gate(specification, row, store, math_root):
    C.require(row["id"] == specification["id"], "Wrong admission group identity")
    C.require(set(row["members"]) == set(specification["members"]), "Missing admission member")
    decisions = {}
    for name, kind in specification["members"].items():
        actual, expected = load_pair(row["members"][name], store, math_root)
        decisions[name] = decision(actual, expected, kind)
    passed = all(d["passed"] for d in decisions.values())
    C.require(
        C.canonical(decisions) == C.canonical(row["decisions"]) and row["passed"] is passed,
        "Reported admission decision differs from raw evidence",
    )
    return passed
