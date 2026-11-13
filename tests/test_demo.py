"""The demo runs both engines for real wherever the native executable is built."""

import math
import platform
import stat
import tempfile

import numpy as np
import pytest

from transformer import demo, spec
from transformer.validate import DEFAULT_EXECUTABLE, TOLERANCES

NATIVE_AVAILABLE = (
    platform.system() == "Darwin" and platform.machine() == "arm64" and DEFAULT_EXECUTABLE.is_file()
)
# The schedule the validation suite's runtime smoke already runs at every width.
SMOKE = ["--epochs=2", "--batch=3", "--train-sequences=5"]
SMOKE += ["--validation-sequences=2", "--test-sequences=2"]


def corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("The quick brown fox jumps over the lazy dog.\n" * 120)
    return path


def placeholder_executable(tmp_path, body="exit 0"):
    path = tmp_path / "transformer-arm64"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.mark.skipif(
    not NATIVE_AVAILABLE, reason="Requires an explicitly built macOS ARM64 executable"
)
def test_demo_trains_both_engines_and_prints_the_ratio_it_measured(tmp_path, monkeypatch, capsys):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    results = []
    compare = demo.compare
    monkeypatch.setattr(
        demo, "compare", lambda *args: results.append(compare(*args)) or results[-1]
    )

    assert demo.main(["--corpus", str(corpus(tmp_path)), *SMOKE]) == 0

    captured = capsys.readouterr()
    (result,) = results
    assembly, pytorch = result["assembly"], result["pytorch"]
    for run in (assembly, pytorch):
        assert run["seconds"] > 0
        assert run["steps"] == 4
        assert math.isfinite(run["train_loss"]) and math.isfinite(run["validation_loss"])
    if spec.WIDTH < 512:
        # Same fixture and initial weights: the two short trajectories must meet the
        # tolerance the validation suite holds this smoke schedule to at these widths.
        np.testing.assert_allclose(
            [assembly["train_loss"], assembly["validation_loss"]],
            [pytorch["train_loss"], pytorch["validation_loss"]],
            **TOLERANCES["trajectory"],
        )
    assert result["ratio"] == pytorch["seconds"] / assembly["seconds"]
    assert f"PyTorch time / assembly time = {result['ratio']:.2f}" in captured.out
    assert f"Width {spec.WIDTH} " in captured.out
    assert "handwritten ARM64 assembly" in captured.out
    assert f"stock PyTorch {result['torch']}, one thread" in captured.out
    assert result["threads"] == 1
    assert "one informal run" in captured.out
    assert "not the accepted measurement" in captured.out
    assert "(shortened)" in captured.out
    assert "not comparable with the published ones" in captured.out
    assert captured.err == ""
    assert list(scratch.iterdir()) == []


def test_demo_missing_corpus_names_the_fetch_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(demo, "DEFAULT_EXECUTABLE", placeholder_executable(tmp_path))
    monkeypatch.setattr(demo, "DEFAULT_CORPUS", tmp_path / "data/tinyshakespeare.txt")

    assert demo.main([]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "Corpus not found" in captured.err
    assert "python -m transformer.fetch_data --output data/tinyshakespeare.txt" in captured.err


def test_demo_missing_executable_names_the_build_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(demo, "DEFAULT_EXECUTABLE", tmp_path / "absent/transformer-arm64")

    assert demo.main(["--corpus", str(corpus(tmp_path))]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "Native executable not found" in captured.err
    assert "make -C native widths PYTHON=python" in captured.err


def test_demo_rejects_a_corpus_too_short_for_the_schedule(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(demo, "DEFAULT_EXECUTABLE", placeholder_executable(tmp_path))

    assert demo.main(["--corpus", str(corpus(tmp_path))]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "too short" in captured.err


def test_demo_reports_a_failed_engine_instead_of_a_ratio(tmp_path, monkeypatch, capsys):
    failing = placeholder_executable(tmp_path, "echo 'engine refused the run' >&2\nexit 3")
    monkeypatch.setattr(demo, "DEFAULT_EXECUTABLE", failing)

    assert demo.main(["--corpus", str(corpus(tmp_path)), *SMOKE]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "assembly run failed with exit status 3" in captured.err
    assert "engine refused the run" in captured.err
