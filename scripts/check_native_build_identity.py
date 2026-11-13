"""Check that the native build is byte-identical to the build of another commit.

The accepted measurements were taken with the native sources at commit bd2cd28.
Commits made to native/*.S after that add comments only. This script checks that
claim directly: it builds the base commit's sources and the working tree's
sources with the same commands at every width, and compares the SHA-256 of
every file the build produces (objects, library, executable and the two
generated headers).

Example:
    python scripts/check_native_build_identity.py
    python scripts/check_native_build_identity.py --base origin/main --precision 0

Requires macOS on Apple silicon with the Xcode command line tools and a clone
that contains the base commit. Both builds happen in a temporary directory; the
repository and its own native/build are not touched.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEASURED_SOURCE = "bd2cd2823038729779a217ee5b084fa06191d3a9"
WIDTHS = (32, 64, 128, 256, 512, 1024)
# The build reads these two directories and nothing else.
BUILD_INPUTS = ("native", "transformer")
REQUIRED_OUTPUTS = tuple(
    f"width-{width}/{name}"
    for width in WIDTHS
    for name in ("libtransformer.dylib", "transformer-arm64")
)


def resolve_commit(root: Path, ref: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def export_commit(root: Path, commit: str, tree: Path) -> None:
    tree.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "-C", str(root), "archive", "--format=tar", commit, *BUILD_INPUTS],
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(tree)], input=archive, check=True)


def export_working_tree(root: Path, tree: Path) -> None:
    ignore = shutil.ignore_patterns("build", "__pycache__", "*.pyc")
    for name in BUILD_INPUTS:
        shutil.copytree(root / name, tree / name, ignore=ignore)


def build(tree: Path, precision: int) -> dict[str, str]:
    for width in WIDTHS:
        subprocess.run(
            [
                "make",
                "-s",
                "-C",
                str(tree / "native"),
                f"WIDTH={width}",
                f"PRECISION={precision}",
                f"PYTHON={sys.executable}",
                "all",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    return hash_outputs(tree / "native/build")


def hash_outputs(build_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(build_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(build_dir.rglob("*"))
        if path.is_file()
    }


def differences(base: dict[str, str], head: dict[str, str]) -> list[str]:
    """Return every way the two builds differ; an empty list means identical.

    A base build that lacks a library or executable is itself a difference, so
    two empty or failed builds can never compare as equal.
    """
    problems = [
        f"base build did not produce {name}" for name in REQUIRED_OUTPUTS if name not in base
    ]
    for name in sorted(base.keys() | head.keys()):
        if name not in base:
            problems.append(f"only in the working tree build: {name}")
        elif name not in head:
            problems.append(f"only in the base build: {name}")
        elif base[name] != head[name]:
            problems.append(f"differs: {name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--base",
        default=MEASURED_SOURCE,
        help="commit to compare against (default: the measured source, %(default).7s)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        action="append",
        choices=range(16),
        metavar="POLICY",
        help="precision policy to build, 0-15; repeatable (default: 0 and 15)",
    )
    args = parser.parse_args(argv)
    if platform.system() != "Darwin" or platform.machine() != "arm64" or not shutil.which("xcrun"):
        print(
            "This check needs macOS on Apple silicon with the Xcode command line tools.",
            file=sys.stderr,
        )
        return 2
    commit = resolve_commit(ROOT, args.base)
    if commit is None:
        print(
            f"Commit {args.base} is not in this clone. A shallow clone lacks it: "
            "run `git fetch --unshallow` and try again.",
            file=sys.stderr,
        )
        return 2
    print(f"base commit: {commit}")
    failed = False
    with tempfile.TemporaryDirectory() as temporary:
        # Both sides are built at the same path, one after the other, so the
        # comparison does not depend on the outputs being path-independent.
        tree = Path(temporary) / "tree"
        for precision in args.precision or (0, 15):
            export_commit(ROOT, commit, tree)
            base = build(tree, precision)
            shutil.rmtree(tree)
            export_working_tree(ROOT, tree)
            head = build(tree, precision)
            shutil.rmtree(tree)
            problems = differences(base, head)
            if problems:
                failed = True
                print(f"PRECISION={precision}: {len(problems)} differences")
                for problem in problems:
                    print(f"  {problem}")
            else:
                print(f"PRECISION={precision}: {len(base)} build outputs, all byte-identical")
    if failed:
        print("FAIL: the working tree does not build the same native binaries as the base commit")
        return 1
    print(f"PASS: the working tree builds the same native binaries as {commit[:7]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
