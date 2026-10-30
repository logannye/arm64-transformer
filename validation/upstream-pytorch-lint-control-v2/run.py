"""Hosted, same-environment pristine/original/formatted upstream lint control.

No framework build, linter replacement, rule exclusion, or pass waiver is used.
"""

import argparse
import collections
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
GIB = 1024**3
VARIANTS = ("base", "original", "formatted")
CONFIGS = (
    ".lintrunner.toml",
    ".spin/cmds.py",
    "tools/linter/adapters/s3_init_config.json",
    "tools/linter/adapters/ruff_linter.py",
    "tools/linter/adapters/clangtidy_linter.py",
)
EXTRAS = ("pytest", "pytest-xdist", "pytest-subtests", "pytest-rerunfailures", "uv")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def inventory(root):
    root = Path(root)
    result = {}
    if not root.exists():
        return result
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or path.name.endswith(".pyc"):
            continue
        if path.is_symlink():
            require(path.exists(), "Broken executable/dependency link")
            require(
                not path.is_dir(), "Directory symlink would hide executable dependency payloads"
            )
        if path.is_file():
            before = path.stat()
            row = {
                "bytes": before.st_size,
                "sha256": sha(path),
                "executable_bits": before.st_mode & 0o111,
            }
            if path.is_symlink():
                row["link"] = os.readlink(path)
                row["resolved"] = str(path.resolve(strict=True))
            after = path.stat()
            require(
                all(
                    getattr(before, key) == getattr(after, key)
                    for key in (
                        "st_dev",
                        "st_ino",
                        "st_size",
                        "st_mode",
                        "st_mtime_ns",
                        "st_ctime_ns",
                    )
                ),
                "File changed while recording tool/dependency bytes",
            )
            result[path.relative_to(root).as_posix()] = row
    return result


def hosted_guard():
    require(
        os.environ.get("GITHUB_ACTIONS") == "true"
        and os.environ.get("GITHUB_REPOSITORY") == "logannye/arm64-transformer",
        "This helper runs only in the authorized own-repository hosted job",
    )


def tool_failure_markers(text):
    """Identify tool failures before attempting to attribute source diagnostics."""
    patterns = {
        "general_linter_failure": r"^>>> General linter failure:",
        "python_traceback": r"Traceback \(most recent call last\)",
        "runtime_exception": (
            r"(?:ModuleNotFoundError|ImportError|RuntimeError|CalledProcessError|"
            r"FileNotFoundError|TimeoutError):"
        ),
        "failed_tool_execution": (
            r"failed to (?:run|execute|initialize)|"
            r"(?:command|linter|process) (?:failed|timed out)|"
            r"command-failed|command-not-found|cannot execute"
        ),
    }
    return [
        {"kind": kind, "log_lines": lines}
        for kind, pattern in patterns.items()
        if (
            lines := [
                index
                for index, line in enumerate(text.splitlines(), 1)
                if re.search(pattern, line, re.I)
            ]
        )
    ]


def lint_execution_eligible(text, exit_code, findings):
    two_phases = (
        len(re.findall(r"^\$ uvx --python 3\.10 lintrunner@0\.12\.7 --take ", text, re.M)) == 2
    )
    runtime_failure = bool(tool_failure_markers(text))
    eligible = (
        two_phases
        and type(exit_code) is int
        and exit_code in (0, 1)
        and not runtime_failure
        and (exit_code == 0 or (bool(findings) and "Lint failed!" in text))
    )
    return {
        "two_phases_reached": two_phases,
        "runtime_failure_detected": runtime_failure,
        "eligible": eligible,
    }


def parse_findings(text):
    """Keep all full blocks; keys compare duplicate-preserving diagnostic identities."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    path, rows, current = None, [], None
    for line in text.splitlines():
        file_match = re.match(r"^>>> Lint for (.+):$", line)
        finding = re.match(r"^  (Error|Warning|Advice|Disabled) \(([^)]+)\) (.*)$", line)
        if file_match or finding or line.startswith("$ ") or line == "Lint failed!":
            if current is not None:
                current["block"] = "\n".join(current.pop("lines")).rstrip()
                current["line"] = next(
                    (
                        int(m.group(1))
                        for m in re.finditer(r"^\s*>>>\s+(\d+)\s*\|", current["block"], re.M)
                    ),
                    None,
                )
                rows.append(current)
                current = None
        if file_match:
            path = file_match.group(1)
        elif finding:
            require(path is not None, "Finding lacks a file identity")
            current = {
                "path": path,
                "severity": finding.group(1),
                "code": finding.group(2),
                "name": finding.group(3),
                "lines": [line],
            }
        elif current is not None:
            current["lines"].append(line)
    if current is not None:
        current["block"] = "\n".join(current.pop("lines")).rstrip()
        current["line"] = next(
            (int(m.group(1)) for m in re.finditer(r"^\s*>>>\s+(\d+)\s*\|", current["block"], re.M)),
            None,
        )
        rows.append(current)
    require(
        len(rows) == len(re.findall(r"^  (?:Error|Warning|Advice|Disabled) \(", text, re.M)),
        "Diagnostic parser dropped a finding",
    )
    return rows


def findings_counter(rows):
    return collections.Counter(json.dumps(row, sort_keys=True) for row in rows)


def compare_findings(base, candidate):
    left, right = findings_counter(base), findings_counter(candidate)

    def expand(values):
        return [
            {"finding": json.loads(key), "count": count} for key, count in sorted(values.items())
        ]

    return {
        "shared": expand(left & right),
        "base_only": expand(left - right),
        "candidate_only": expand(right - left),
        "scope": "Exact diagnostic blocks, retaining multiplicity; shared errors remain failures",
    }


class Run:
    def __init__(self, source, output, contract):
        self.source, self.output, self.contract = source, output, contract
        self.environment = source / ".venv"
        self.cache = output / "cache"
        self.env = os.environ.copy()
        for name in list(self.env):
            if name.startswith(
                ("UV_", "PIP_", "PYTHON", "DYLD_", "LINTRUNNER", "LINT_", "GIT_CONFIG_")
            ) or name in {
                "CMAKE_ARGS",
                "VIRTUAL_ENV",
                "CFLAGS",
                "CXXFLAGS",
                "CPPFLAGS",
                "LDFLAGS",
                "CPATH",
                "C_INCLUDE_PATH",
                "CPLUS_INCLUDE_PATH",
                "LIBRARY_PATH",
                "CC",
                "CXX",
                "SDKROOT",
                "MACOSX_DEPLOYMENT_TARGET",
            }:
                self.env.pop(name, None)
        self.env.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONHASHSEED": "0",
                "MAX_JOBS": "2",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "UV_NO_BUILD": "1",
                "UV_PYTHON_PREFERENCE": "only-managed",
                "PIP_CACHE_DIR": str(self.cache / "pip"),
                "UV_CACHE_DIR": str(self.cache / "uv"),
                "UV_PYTHON_INSTALL_DIR": str(self.cache / "uv-python"),
            }
        )
        self.current = "base"
        self.refs_sha256 = None
        self.build_directory_identity = None
        self.record = {
            "schema": contract["schema"],
            "complete": False,
            "evidence_valid": False,
            "all_lints_passed": False,
            "status": "running",
            "formal_fits": 0,
            "framework_build_executed": False,
            "lint_failures_waived": False,
            "source": str(source),
            "output": str(output),
            "contract_sha256": sha(HERE / "contract.json"),
            "helpers": inventory(HERE),
            "commands": [],
            "provisioning": [],
            "variants": [],
            "historical_run": contract["historical_run"],
            "previous_failed_control": contract.get("previous_failed_control"),
            "historical_environment_identity_claim": (
                "New hosted comparison; historical tool/compiler equality reported separately"
            ),
            "github": {
                key: os.environ.get(key)
                for key in (
                    "GITHUB_REPOSITORY",
                    "GITHUB_SHA",
                    "GITHUB_RUN_ID",
                    "GITHUB_RUN_ATTEMPT",
                    "RUNNER_OS",
                    "RUNNER_ARCH",
                    "ImageOS",
                    "ImageVersion",
                )
            },
        }

    def save(self):
        write(self.output / "receipt.json", self.record)

    def command(self, label, argv, *, check=True, timeout=1800):
        require(shutil.disk_usage(self.output).free >= 8 * GIB, "8 GiB storage reserve unavailable")
        log = self.output / f"{len(self.record['commands']):03d}-{label}.log"
        row = {
            "label": label,
            "argv": list(map(str, argv)),
            "cwd": str(self.source),
            "log": log.name,
            "status": "running",
            "exit_code": None,
            "timed_out": False,
        }
        self.record["commands"].append(row)
        self.save()
        started = time.monotonic()
        with log.open("w") as stream:
            child = subprocess.Popen(
                row["argv"],
                cwd=self.source,
                env=self.env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                while child.poll() is None:
                    require(time.monotonic() - started < timeout, "Hosted command timed out")
                    require(
                        shutil.disk_usage(self.output).free >= 8 * GIB, "8 GiB storage reserve lost"
                    )
                    time.sleep(2)
            except BaseException:
                row["timed_out"] = time.monotonic() - started >= timeout
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=10)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                raise
            finally:
                row.update(
                    exit_code=child.returncode,
                    elapsed_seconds=time.monotonic() - started,
                    status="passed" if child.returncode == 0 else "failed",
                )
                self.save()
        row["log_sha256"] = sha(log)
        self.save()
        require(
            not check or row["exit_code"] == 0, f"Hosted command failed: {label}; see retained log"
        )
        return row, log.read_text(errors="replace")

    def git(self, label, args):
        return self.command(label, ["git", *args])[1]

    def empty_build_directory(self):
        build = self.source / "build"
        require(
            build.is_dir() and not build.is_symlink() and build.resolve(strict=True) == build,
            "The clang-tidy working directory must be a real contained build directory",
        )
        require(not list(build.iterdir()), "The clang-tidy build directory is no longer empty")
        stat = build.stat()
        identity = {
            "path": "build",
            "kind": "empty_directory",
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "mode": stat.st_mode & 0o7777,
            "entries": [],
        }
        require(
            self.build_directory_identity is None or identity == self.build_directory_identity,
            "The clang-tidy working-directory identity changed",
        )
        return identity

    def prepare_build_directory(self):
        expected = self.contract["filesystem_prerequisite"]
        require(
            expected["path"] == "build"
            and expected["kind"] == "empty_directory"
            and expected["create_once_before_initial_source_identity"] is True
            and expected["empty_before_and_after_every_lint"] is True
            and all(
                expected[key] is False
                for key in (
                    "compile_database_created",
                    "generated_headers_created",
                    "framework_build_executed",
                )
            ),
            "Unexpected clang-tidy filesystem prerequisite contract",
        )
        require(
            {name: sha(self.source / name) for name in CONFIGS}
            == self.contract["lint_configuration"],
            "Upstream source/configuration differs before directory setup",
        )
        build = self.source / "build"
        require(not build.exists() and not build.is_symlink(), "Build path must start absent")
        build.mkdir()
        self.build_directory_identity = self.empty_build_directory()
        self.record["filesystem_setup"] = {
            "operation": "mkdir",
            "path": "build",
            "before": {"exists": False},
            "after": self.build_directory_identity,
            "source_configuration_sha256": self.contract["lint_configuration"],
            "compile_database_created": False,
            "generated_headers_created": False,
            "framework_build_executed": False,
        }
        self.save()

    def source_identity(self, label):
        build_directory = self.empty_build_directory()
        head = self.git(label + "-head", ["rev-parse", "HEAD"]).strip()
        diff = self.git(label + "-diff", ["diff", "--name-only"]).splitlines()
        staged = self.git(label + "-staged", ["diff", "--cached", "--name-only"]).strip()
        require(head == self.contract["base"] and not staged, "Base or staged source state changed")
        expected_names = (
            [] if self.current == "base" else list(self.contract["sources"][self.current])
        )
        require(
            sorted(diff) == sorted(expected_names),
            "Tracked diff differs from declared source variant",
        )
        sources = {name: sha(self.source / name) for name in self.contract["sources"][self.current]}
        require(sources == self.contract["sources"][self.current], "Variant source bytes differ")
        configs = {name: sha(self.source / name) for name in CONFIGS}
        require(
            configs == self.contract["lint_configuration"], "Upstream lint configuration changed"
        )
        refs = self.git(label + "-refs", ["show-ref"])
        refs_sha256 = hashlib.sha256(refs.encode()).hexdigest()
        require(
            self.refs_sha256 is None or self.refs_sha256 == refs_sha256,
            "Git reference/merge-base inventory changed between variants",
        )
        self.refs_sha256 = refs_sha256
        require(
            not (self.source / ".lintrunner.private.toml").exists(),
            "Unexpected private lint configuration",
        )
        merge = self.git(
            label + "-merge-base", ["merge-base", "HEAD", "refs/remotes/origin/main"]
        ).strip()
        require(merge == self.contract["base"], "Pinned base is not origin/main's ancestor")
        return {
            "head": head,
            "merge_base_origin_main": merge,
            "refs_sha256": hashlib.sha256(refs.encode()).hexdigest(),
            "diff_names": sorted(diff),
            "sources": sources,
            "lint_configuration": configs,
            "build_directory": build_directory,
        }

    def transition(self, variant):
        self.source_identity("before-transition-" + self.current)
        if self.current != "base":
            patch = HERE / self.contract["patches"][self.current]["file"]
            self.git("reverse-check-" + self.current, ["apply", "--reverse", "--check", str(patch)])
            self.git("reverse-" + self.current, ["apply", "--reverse", str(patch)])
        self.current = "base"
        if variant != "base":
            patch = HERE / self.contract["patches"][variant]["file"]
            self.git("apply-check-" + variant, ["apply", "--check", str(patch)])
            self.git("apply-" + variant, ["apply", str(patch)])
        self.current = variant
        return self.source_identity("after-transition-" + variant)

    def tools(self):
        roots = {
            "venv": self.environment,
            "lintbin": self.source / ".lintbin",
            "managed_python": self.cache / "uv-python",
        }
        # Archive and environment directories hold installed executable/package
        # bytes. Resolver timestamp/msgpack indexes are not executable payloads.
        uv = self.cache / "uv"
        for path in sorted(uv.iterdir()) if uv.exists() else []:
            if path.is_dir() and path.name.startswith(("archive-", "environments-", "builds-")):
                roots["uv/" + path.name] = path
        result = {name: inventory(path) for name, path in roots.items()}
        binaries = {
            name: {key: row[key] for key in ("sha256", "bytes")}
            for name, row in result["lintbin"].items()
            if not name.startswith(".")
        }
        require(
            binaries == self.contract["lint_binaries"],
            "Upstream lint binaries differ from original failed run",
        )
        require(
            result["venv"]
            and result["managed_python"]
            and any(name.startswith("uv/") for name in result),
            "Incomplete installed lint-tool payload inventory",
        )
        return result

    def lint(self, phase, variant):
        before = self.transition(variant)
        row, text = self.command(
            phase + "-" + variant + "-spin-lint",
            [str(self.environment / "bin/spin"), "lint"],
            check=False,
        )
        after = self.source_identity(phase + "-" + variant + "-after")
        require(before == after, "Lint changed declared tracked inputs or git scope")
        # A general tool failure has no source-file identity. Retain it as a
        # tool failure instead of either parsing it as a finding or hiding it
        # behind the parser's missing-path exception.
        markers = tool_failure_markers(text)
        parse_error = None
        findings = []
        if not markers:
            try:
                findings = parse_findings(text)
            except ValueError as error:
                parse_error = {"type": type(error).__name__, "message": str(error)}
        for finding in findings:
            path = Path(finding["path"])
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and (self.source / path).resolve(strict=True).is_relative_to(self.source),
                "Lint finding does not identify a contained source file",
            )
            finding["source_sha256"] = sha(self.source / path)
        eligibility = lint_execution_eligible(text, row["exit_code"], findings)
        eligible = eligibility["eligible"] and parse_error is None
        classification = (
            "tool_runtime_failure"
            if markers
            else "diagnostic_parse_failure"
            if parse_error is not None
            else "incomplete_lint_execution"
            if not eligible
            else "passed"
            if row["exit_code"] == 0
            else "source_diagnostics"
        )
        result = {
            "variant": variant,
            "command": row,
            "source_before": before,
            "source_after": after,
            "upstream_two_phase_lint_reached": eligibility["two_phases_reached"],
            "runtime_failure_detected": eligibility["runtime_failure_detected"],
            "classification": classification,
            "tool_failure_markers": markers,
            "diagnostics_parsed": not markers and parse_error is None,
            "diagnostic_parse_error": parse_error,
            "findings": findings,
            "lint_passed": eligible and row["exit_code"] == 0,
            "eligible_for_comparison": eligible,
        }
        write(self.output / f"{phase}-{variant}.json", result)
        return result

    def run(self):
        require(
            platform.system() == "Darwin"
            and platform.machine() == "arm64"
            and sys.version_info[:3] == (3, 12, 10),
            "Use the declared macOS arm64 Python3.12.10 hosted environment",
        )
        hosted_guard()
        require(
            shutil.disk_usage(self.output).free >= 40 * GIB,
            "40 GiB free storage required before preparation",
        )
        require(
            not (self.source / "build").exists()
            and not (self.source / "build").is_symlink()
            and not self.environment.exists()
            and not self.environment.is_symlink(),
            "Use one fresh unbuilt source checkout",
        )
        require(self.contract["variants"] == list(VARIANTS), "Variant order changed")
        require(
            sha(HERE / "requirements-full.txt") == self.contract["requirements_sha256"],
            "Dependency requirements changed",
        )
        for patch in self.contract["patches"].values():
            require(sha(HERE / patch["file"]) == patch["sha256"], "Patch bytes changed")
        self.prepare_build_directory()
        self.source_identity("initial")
        self.command(
            "submodules",
            ["git", "submodule", "update", "--init", "--recursive", "--depth", "1", "--jobs", "2"],
            timeout=3600,
        )
        submodules = self.git("submodule-identities", ["submodule", "status", "--recursive"])
        require(
            not any(line.startswith(("-", "+", "U")) for line in submodules.splitlines()),
            "Submodule identity differs",
        )
        host = {}
        for label, argv in (
            ("os", ["sw_vers"]),
            ("xcode-selection", ["xcode-select", "-p"]),
            ("xcode-version", ["xcodebuild", "-version"]),
            ("clang-version", ["xcrun", "clang", "--version"]),
            ("sdk-version", ["xcrun", "--sdk", "macosx", "--show-sdk-version"]),
        ):
            host[label] = self.command(label, argv)[1]
        clang = self.command("clang-path", ["xcrun", "--find", "clang"])[1].strip()
        host["clang_path"], host["clang_sha256"] = clang, sha(clang)
        host["python_path"], host["python_sha256"] = (
            str(Path(sys.executable).resolve()),
            sha(sys.executable),
        )
        self.record["host"] = host
        self.command("create-environment", [sys.executable, "-m", "venv", str(self.environment)])
        self.env["PATH"] = str(self.environment / "bin") + os.pathsep + self.env["PATH"]
        self.env["VIRTUAL_ENV"] = str(self.environment)
        pip = str(self.environment / "bin/pip")
        wheels = self.output / "wheels"
        wheels.mkdir()
        requirements = str(HERE / "requirements-full.txt")
        self.command(
            "download-exact-wheels",
            [
                pip,
                "download",
                "--only-binary=:all:",
                "--no-deps",
                "--require-hashes",
                "--requirement",
                requirements,
                "--dest",
                str(wheels),
            ],
        )
        actual = {
            name: {key: row[key] for key in ("sha256", "bytes")}
            for name, row in inventory(wheels).items()
        }
        require(
            actual == self.contract["wheel_inventory"],
            "Exact45 historical wheels did not reproduce",
        )
        self.record["wheel_inventory"] = actual
        self.command(
            "install-exact-wheels",
            [
                pip,
                "install",
                "--no-index",
                "--find-links",
                str(wheels),
                "--only-binary=:all:",
                "--no-deps",
                "--require-hashes",
                "--requirement",
                requirements,
            ],
        )
        self.command("dependency-check", [pip, "check"])
        freeze = self.command("dependency-freeze", [pip, "freeze", "--all"])[1]
        self.record["dependency_freeze"] = freeze
        # Provision every variant via the normal unchanged command. A finding
        # is retained; only runtime/setup failures invalidate paired evidence.
        for variant in VARIANTS:
            result = self.lint("provision", variant)
            self.record["provisioning"].append(result)
            self.save()
            require(
                result["eligible_for_comparison"],
                "Upstream lint provisioning did not complete both phases",
            )
        self.transition("base")
        self.env["UV_OFFLINE"] = "1"
        allowed = (
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONNOUSERSITE",
            "PYTHONHASHSEED",
            "MAX_JOBS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "PIP_DISABLE_PIP_VERSION_CHECK",
            "UV_NO_BUILD",
            "UV_PYTHON_PREFERENCE",
            "PIP_CACHE_DIR",
            "UV_CACHE_DIR",
            "UV_PYTHON_INSTALL_DIR",
            "UV_OFFLINE",
            "PATH",
            "VIRTUAL_ENV",
        )
        self.record["comparison_environment"] = {key: self.env[key] for key in allowed}

        sealed = self.tools()
        write(self.output / "installed-tool-payloads.json", sealed)
        self.record["installed_tool_payload_sha256"] = sha(
            self.output / "installed-tool-payloads.json"
        )
        for variant in VARIANTS:
            require(
                self.tools() == sealed and sha(clang) == host["clang_sha256"],
                "Paired tool/compiler state changed before lint",
            )
            require(
                self.command("before-" + variant + "-freeze", [pip, "freeze", "--all"])[1]
                == freeze,
                "Dependencies changed before lint",
            )
            result = self.lint("compare", variant)
            unchanged = self.tools() == sealed and sha(clang) == host["clang_sha256"]
            result["tool_payloads_unchanged"] = unchanged
            result["dependency_freeze_unchanged"] = (
                self.command("after-" + variant + "-freeze", [pip, "freeze", "--all"])[1] == freeze
            )
            self.record["variants"].append(result)
            self.save()
            require(
                result["eligible_for_comparison"]
                and unchanged
                and result["dependency_freeze_unchanged"],
                "Same-environment paired lint controls failed",
            )
        require(
            [row["variant"] for row in self.record["variants"]] == list(VARIANTS),
            "Missing paired lint variant",
        )
        rows = self.record["variants"]
        self.record["comparisons"] = {
            variant: compare_findings(rows[0]["findings"], rows[index]["findings"])
            for index, variant in enumerate(VARIANTS)
            if index
        }
        self.record.update(
            complete=True,
            evidence_valid=True,
            all_lints_passed=all(row["lint_passed"] for row in rows),
            status="passed"
            if all(row["lint_passed"] for row in rows)
            else "completed_with_lint_failures",
            same_host_compiler_dependencies_tools=True,
        )
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hosted_guard()
    source, output = args.source.resolve(strict=True), args.output.absolute()
    require(
        not output.exists() and output.resolve(strict=False) == output,
        "Output must be a new canonical directory",
    )
    output.mkdir(parents=True)
    run = Run(source, output, json.loads((HERE / "contract.json").read_text()))
    try:
        run.run()
    except BaseException as error:
        run.record.update(
            status="failed",
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        run.save()
    return (
        0
        if run.record["complete"]
        and run.record["evidence_valid"]
        and run.record["all_lints_passed"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
