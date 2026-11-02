"""Verify selected hosted native assets without compiling or executing a fit."""

import ctypes
import hashlib
import importlib.util
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("library_native_authority", HERE / "authority.py")
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
C = A.C
BUILD_REVISION = "00829baaa8acc74fda617f547e85d8dba65b71c6"
BUILD_RUN = 41233052374
WIDTHS = (32, 64, 128, 256, 512, 1024)
FILES = ("libtransformer.dylib", "transformer-arm64", "layout.inc", "precision.h")


def verify(receipt_binding, math):
    path = C.bound(receipt_binding)
    value = C.read(path, receipt_binding["sha256"])
    C.require(
        value["schema"] == "hosted-selected-native-assets-v1"
        and value["source_revision"] == BUILD_REVISION
        and value["math_revision"] == math["revision"] == C.MATH_REVISION
        and value["run_id"] == BUILD_RUN
        and value["numerical_source_diff"] == []
        and value["native_builds"] == value["formal_fits"] == 0
        and value["qualification_transferred"] is False,
        "Unselected hosted native asset derivation",
    )
    C.bound(value["metadata"])
    metadata = C.read(value["metadata"]["path"], value["metadata"]["sha256"])
    C.require(len(metadata) == 6, "Native transport inventory is incomplete")
    C.require([row["width"] for row in value["widths"]] == list(WIDTHS), "Missing native width")
    root = Path(math["root"])
    sources = [n for n in math["files"] if n.startswith("native/") and n.endswith(".S")]
    sources += ["native/Makefile", "transformer/spec.py"]
    for name in sources:
        original = A.git(root, "show", BUILD_REVISION + ":" + name)
        C.require(
            hashlib.sha256(original).hexdigest() == math["files"][name],
            "Hosted native build source differs from the frozen numerical source",
        )
    for row in value["widths"]:
        width, policy = row["width"], row["policy"]
        C.require(
            type(width) is int and type(policy) is int and policy == (15 if width >= 512 else 0),
            "Wrong selected native precision policy",
        )
        artifact = row["artifact"]
        C.require(
            artifact in metadata
            and artifact["workflow_run"]["id"] == BUILD_RUN
            and artifact["workflow_run"]["head_sha"] == BUILD_REVISION
            and artifact["name"] == f"precision-native-candidates-width-{width}",
            "Native transport came from a different hosted build",
        )
        C.bound(row["archive"])
        C.require(
            "sha256:" + row["archive"]["sha256"] == artifact["digest"]
            and row["archive"]["bytes"] == artifact["size_in_bytes"],
            "Native archive changed",
        )
        C.require(set(row["selected"]) == set(FILES), "Incomplete selected native assets")
        with zipfile.ZipFile(row["archive"]["path"]) as archive:
            entries = archive.infolist()
            C.require(
                len(entries) <= 100
                and sum(i.file_size for i in entries) < 16 * 1024**2
                and len({i.filename for i in entries}) == len(entries),
                "Unbounded native archive",
            )
            C.require(
                archive.read("source-revision.txt").decode().strip() == BUILD_REVISION,
                "Native archive source revision differs",
            )
            for name in FILES:
                selected = row["selected"][name]
                expected = root / f"native/build/width-{width}" / name
                C.require(
                    Path(selected["path"]) == expected, "Selected native asset has another origin"
                )
                C.bound(selected)
                data = archive.read(f"width-{width}/policy-{policy}/{name}")
                C.require(
                    selected["bytes"] == len(data)
                    and selected["sha256"] == hashlib.sha256(data).hexdigest(),
                    "Selected native bytes differ from the hosted archive",
                )
        C.require(
            (root / f"native/build/width-{width}/precision.h").read_text()
            == f"#define TF_PRECISION_POLICY {policy}\n",
            "Native policy header differs",
        )
    C.bound(receipt_binding)
    return value


def load(runtime, receipt_binding, width):
    from transformer import spec
    from transformer.validate import Native

    C.require(type(width) is int and width == spec.WIDTH, "Select native width before imports")
    value = verify(receipt_binding, runtime.roles["math"])
    row = next(row for row in value["widths"] if row["width"] == width)
    native = Native(row["selected"]["libtransformer.dylib"]["path"])
    precision = native.lib.tf_precision_policy
    precision.argtypes, precision.restype = [], ctypes.c_uint64
    C.require(int(precision()) == row["policy"], "Actual native precision policy differs")
    C.require(
        (Path(native.path).parent / "layout.inc").read_text() == spec.assembly_constants(),
        "Hosted native layout differs from the frozen model specification",
    )
    runtime.origins("cpu_arm64")
    verify(receipt_binding, runtime.roles["math"])
    return native
