"""Authenticate original published fixture bytes without reading tensor values."""

import importlib.util
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("library_fixture_authority", HERE / "authority.py")
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
C = A.C
PUBLICATION = "experiments/width-sweep-v6/publication.json"
PUBLICATION_SHA256 = "095aa9b73e1155f38031c931a513220d0b777468a3b379bf8dd16d950d4039d7"
WIDTHS, SEEDS = (32, 64, 128, 256, 512, 1024), (42, 43, 44)
FILES = ("config.bin", "initial.f32", "train.i32", "validation.i32", "test.i32", "schedule.i32")
CONFIG = (
    "magic",
    "version",
    "epochs",
    "batch",
    "train_sequences",
    "validation_sequences",
    "test_sequences",
    "seed",
    "width",
)


def published_inputs(summary, seed):
    C.require(type(seed) is int and seed in SEEDS, "Unplanned fixture seed")
    rows = [r for r in summary["runs"] if type(r["seed"]) is int and r["seed"] == seed]
    C.require(
        len(rows) == 4 and all(r["fixture"] == f"seed-{seed}" for r in rows),
        "Incomplete published fixture denominator",
    )
    expected = rows[0]["fixture_sha256"]
    C.require(
        set(expected) == set(FILES) and all(r["fixture_sha256"] == expected for r in rows),
        "Published fixture bytes differ across original engine/repeat arms",
    )
    return expected


def capture(math, summary_root, input_root):
    summary_root, input_root = A.directory(summary_root), A.directory(input_root)
    publication_path = Path(math["root"]) / PUBLICATION
    C.require(math["files"][PUBLICATION] == PUBLICATION_SHA256, "Unselected fixture publication")
    publication = C.read(publication_path, PUBLICATION_SHA256)
    summaries, fixtures = [], []
    for width in WIDTHS:
        name = f"width-{width}/summary.json"
        bound = C.binding(summary_root / name)
        C.require(
            {k: bound[k] for k in ("sha256", "bytes")} == publication["compact_output_files"][name],
            "Unpublished compact summary",
        )
        summary = C.read(bound["path"], bound["sha256"])
        C.require(len(summary["runs"]) == 12, "Published three-seed fixture denominator changed")
        summaries.append(bound)
        for seed in SEEDS:
            directory = input_root / f"width-{width}/seed-{seed}"
            expected = published_inputs(summary, seed)
            files = {name: C.binding(directory / name) for name in (*FILES, "manifest.json")}
            C.require(
                {name: files[name]["sha256"] for name in FILES} == expected,
                "Original published fixture bytes changed",
            )
            config = (0x54465231, 2, 8, 16, 2048, 256, 256, seed, width)
            C.require(
                struct.unpack("<9I", (directory / "config.bin").read_bytes()) == config,
                "Formal fixture budget changed",
            )
            manifest = C.read(directory / "manifest.json", files["manifest.json"]["sha256"])
            C.require(
                manifest["sha256"] == expected
                and C.canonical(manifest["config"])
                == C.canonical(dict(zip(CONFIG, config, strict=True)))
                and manifest.get("gpu_development_fixture") is not True,
                "Original fixture metadata or scope differs",
            )
            sizes = dict(
                zip(
                    FILES,
                    (
                        36,
                        (8 * width**2 + 301 * width + 128) * 4,
                        2048 * 33 * 4,
                        256 * 33 * 4,
                        256 * 33 * 4,
                        8 * 2048 * 4,
                    ),
                    strict=True,
                )
            )
            C.require(
                all(files[name]["bytes"] == size for name, size in sizes.items()),
                "Original fixture length differs from the fixed workload",
            )
            fixtures.append(dict(width=width, seed=seed, directory=str(directory), files=files))
    return dict(
        schema="frozen-library-original-fixtures-v1",
        publication=C.binding(publication_path),
        math_revision=math["revision"],
        summary_root=str(summary_root),
        input_root=str(input_root),
        summaries=summaries,
        fixtures=fixtures,
        qualification_transferred=False,
        scope="published_fixture_bytes_only",
    )


def verify(value, math):
    C.require(
        C.canonical(value)
        == C.canonical(capture(math, value["summary_root"], value["input_root"])),
        "Original fixture binding changed",
    )
    return value
