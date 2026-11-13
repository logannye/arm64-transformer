"""The build-identity check's bookkeeping; like every test here it never invokes a compiler."""

import subprocess

from scripts import check_native_build_identity as identity


def complete_build(directory, content=b"object"):
    for name in identity.REQUIRED_OUTPUTS:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content + name.encode())
    return identity.hash_outputs(directory)


def test_identical_builds_have_no_differences(tmp_path):
    base = complete_build(tmp_path / "base")
    head = complete_build(tmp_path / "head")
    assert len(base) == len(identity.REQUIRED_OUTPUTS) == 12
    assert identity.differences(base, head) == []


def test_one_changed_output_is_named(tmp_path):
    base = complete_build(tmp_path / "base")
    head = complete_build(tmp_path / "head")
    (tmp_path / "head/width-64/libtransformer.dylib").write_bytes(b"changed")
    head = identity.hash_outputs(tmp_path / "head")
    assert identity.differences(base, head) == ["differs: width-64/libtransformer.dylib"]


def test_missing_and_extra_outputs_are_named(tmp_path):
    base = complete_build(tmp_path / "base")
    head = complete_build(tmp_path / "head")
    (tmp_path / "head/width-32/transformer-arm64").unlink()
    (tmp_path / "head/width-32/extra.o").write_bytes(b"extra")
    head = identity.hash_outputs(tmp_path / "head")
    assert identity.differences(base, head) == [
        "only in the working tree build: width-32/extra.o",
        "only in the base build: width-32/transformer-arm64",
    ]


def test_two_empty_builds_do_not_pass():
    problems = identity.differences({}, {})
    assert len(problems) == len(identity.REQUIRED_OUTPUTS)
    assert problems[0] == "base build did not produce width-32/libtransformer.dylib"


def test_working_tree_export_leaves_build_outputs_behind(tmp_path):
    root = tmp_path / "root"
    for name, content in (
        ("native/model.S", "ret\n"),
        ("native/Makefile", "all:\n"),
        ("native/build/width-32/model.o", "stale"),
        ("native/metal/build/kernel.air", "stale"),
        ("transformer/spec.py", "WIDTH = 32\n"),
        ("transformer/__pycache__/spec.cpython-312.pyc", "stale"),
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    tree = tmp_path / "tree"
    identity.export_working_tree(root, tree)
    exported = sorted(str(p.relative_to(tree)) for p in tree.rglob("*") if p.is_file())
    assert exported == ["native/Makefile", "native/model.S", "transformer/spec.py"]


def test_commit_export_matches_the_commit_not_the_working_tree(tmp_path):
    root = tmp_path / "root"
    (root / "native").mkdir(parents=True)
    (root / "transformer").mkdir()
    (root / "native/model.S").write_text("ret\n")
    (root / "transformer/spec.py").write_text("WIDTH = 32\n")
    git = ["git", "-C", str(root), "-c", "user.name=test", "-c", "user.email=test@example.invalid"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "native/model.S", "transformer/spec.py"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "base"], check=True)
    (root / "native/model.S").write_text("nop\nret\n")
    tree = tmp_path / "tree"
    identity.export_commit(root, identity.resolve_commit(root, "HEAD"), tree)
    assert (tree / "native/model.S").read_text() == "ret\n"
    assert identity.resolve_commit(root, "no-such-ref") is None


def test_unsupported_host_is_reported_without_building(monkeypatch, capsys):
    monkeypatch.setattr(identity.platform, "system", lambda: "Linux")
    assert identity.main([]) == 2
    assert "macOS on Apple silicon" in capsys.readouterr().err
