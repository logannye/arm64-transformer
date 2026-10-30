"""Hosted, same-environment pristine/original/formatted upstream lint control.

No framework build, linter replacement, rule exclusion, or pass waiver is used.
"""

import argparse
import collections
import hashlib
import importlib.util
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
    "tools/linter/clang_tidy/generate_build_files.py",
    "tools/setup_helpers/generate_code.py",
    ".github/scripts/lintrunner.sh",
    "cmake/EnvVarForwarding.cmake",
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


def load_tree_inventory():
    spec = importlib.util.spec_from_file_location(
        "lint_tree_inventory_v3", HERE / "tree_inventory.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


trees = load_tree_inventory()
inventory = trees.inventory


def read_cache(path):
    values = {}
    for line in path.read_text().splitlines():
        if line.startswith(("//", "#")) or ":" not in line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.split(":", 1)[0]] = value
    return values


def read_effective_forwarded(path):
    values = {}
    for line in path.read_text().splitlines():
        require("=" in line, "Malformed effective forwarded setting")
        name, value = line.split("=", 1)
        require(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) and name not in values,
            "Missing/duplicate effective forwarded name",
        )
        values[name] = value
    require(values, "Empty effective forwarded settings")
    return values


def require_tool_payloads(result):
    def has_payload(rows):
        return any("sha256" in row and "bytes" in row for row in rows.values())

    required = ("venv", "managed_python", "host_python", "lintbin", "libomp")
    require(
        all(name in result and has_payload(result[name]) for name in required)
        and any(name.startswith("uv/") and has_payload(rows) for name, rows in result.items()),
        "Incomplete installed lint-tool payload inventory (root policy is not payload)",
    )


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
                (
                    "UV_",
                    "PIP_",
                    "PYTHON",
                    "DYLD_",
                    "LINTRUNNER",
                    "LINT_",
                    "GIT_CONFIG_",
                    "CMAKE_",
                    "BUILD_",
                    "USE_",
                    "TORCH_",
                    "SKBUILD_",
                )
            ) or name in {
                "CMAKE_ARGS",
                "DEBUG",
                "REL_WITH_DEB_INFO",
                "WERROR",
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
        self.env.update(contract.get("configure_prerequisite", {}).get("fullbuild_environment", {}))
        self.current = "base"
        self.refs_sha256 = None
        self.generated_inputs = None
        self.generated_inputs_sha256 = None
        self.libomp_root = None
        self.tool_root_identity = None
        self.record = {
            "schema": contract["schema"],
            "complete": False,
            "evidence_valid": False,
            "all_lints_passed": False,
            "status": "running",
            "formal_fits": 0,
            "framework_build_executed": False,
            "compiler_configuration_probes_possible": True,
            "lint_failures_waived": False,
            "source": str(source),
            "output": str(output),
            "contract_sha256": sha(HERE / "contract.json"),
            "helpers": inventory(HERE),
            "commands": [],
            "provisioning": [],
            "variants": [],
            "historical_run": contract["historical_run"],
            "previous_failed_controls": contract.get("previous_failed_controls"),
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

    def generated_roots(self):
        names = self.contract["configure_prerequisite"]["generated_roots"]
        require(
            names
            == [
                "build",
                "torch/csrc/autograd/generated",
                "torch/csrc/functionalization/generated",
                "torch/csrc/lazy/generated",
                "torch/testing/_internal/generated",
            ],
            "Generated-input inventory scope changed",
        )
        roots = {name: self.source / name for name in names}
        require(
            all(path.is_dir() and not path.is_symlink() for path in roots.values()),
            "Missing or aliased generated input root",
        )
        return roots

    def generated_identity(self):
        if self.generated_inputs is None:
            require(
                not (self.source / "build").exists() and not (self.source / "build").is_symlink(),
                "Build tree exists before official preparation",
            )
            return None
        actual = trees.inventory_forest(self.generated_roots())
        require(actual == self.generated_inputs, "Generated lint inputs changed between variants")
        require(
            sha(self.output / "generated-inputs.json") == self.generated_inputs_sha256,
            "Saved generated-input inventory changed",
        )
        return {
            "inventory_sha256": self.generated_inputs_sha256,
            "roots": self.contract["configure_prerequisite"]["generated_roots"],
        }

    def prepare_generated_inputs(self):
        declaration = self.contract["configure_prerequisite"]
        require(
            declaration["command"] == ["spin", "regenerate-clangtidy-files"]
            and declaration["full_framework_build_executed"] is False
            and declaration["compiler_configuration_probes_possible"] is True
            and declaration["generated_inputs_unchanged_between_variants"] is True,
            "Unexpected configure/codegen prerequisite contract",
        )
        before = self.source_identity("before-official-configure")
        self.command(
            "regenerate-clangtidy-files",
            [str(self.environment / "bin/spin"), "regenerate-clangtidy-files"],
            timeout=3600,
        )
        cache_path = self.source / "build/CMakeCache.txt"
        database_path = self.source / "build/compile_commands.json"
        effective_path = self.source / "build/env_forwarded.txt"
        require(
            cache_path.is_file() and database_path.is_file() and effective_path.is_file(),
            "Official configure outputs missing",
        )
        cache = read_cache(cache_path)
        effective = read_effective_forwarded(effective_path)
        for key, accepted in {
            "CMAKE_BUILD_TYPE": {"Release"},
            "CMAKE_EXPORT_COMPILE_COMMANDS": {"ON", "1", "TRUE"},
            "USE_MPS": {"ON", "1", "TRUE", "YES"},
            "USE_CUDA": {"OFF", "0", "FALSE", "NO"},
            "USE_DISTRIBUTED": {"OFF", "0", "FALSE", "NO"},
            "USE_OPENMP": {"ON", "1", "TRUE", "YES"},
            "USE_NCCL": {"OFF", "0", "FALSE", "NO"},
            "USE_PRECOMPILED_HEADERS": {"ON", "1", "TRUE", "YES"},
        }.items():
            require(cache.get(key) in accepted, "Unexpected cached configure setting: " + key)
            require(
                effective.get(key) in accepted, "Unexpected effective configure setting: " + key
            )
        database = json.loads(database_path.read_text())
        require(
            isinstance(database, list)
            and database
            and all(
                isinstance(row, dict)
                and isinstance(row.get("file"), str)
                and isinstance(row.get("directory"), str)
                and (isinstance(row.get("command"), str) or isinstance(row.get("arguments"), list))
                for row in database
            ),
            "Invalid official compilation database",
        )
        self.generated_inputs = trees.inventory_forest(self.generated_roots())
        inventory_path = self.output / "generated-inputs.json"
        write(inventory_path, self.generated_inputs)
        self.generated_inputs_sha256 = sha(inventory_path)
        shutil.copyfile(cache_path, self.output / "CMakeCache.txt")
        shutil.copyfile(database_path, self.output / "compile_commands.json")
        shutil.copyfile(effective_path, self.output / "env_forwarded.txt")
        after = self.source_identity("after-official-configure")
        require(
            {k: v for k, v in before.items() if k != "generated_inputs"}
            == {k: v for k, v in after.items() if k != "generated_inputs"},
            "Official configure/codegen changed declared tracked source or git scope",
        )
        self.record["configure_prerequisite"] = {
            "command": [str(self.environment / "bin/spin"), "regenerate-clangtidy-files"],
            "inherited_fullbuild_environment": declaration["fullbuild_environment"],
            "official_generator_overrides": declaration["official_generator_overrides"],
            "cmake_prefix_path": self.env["CMAKE_PREFIX_PATH"],
            "cmake_cache": cache,
            "cmake_cache_sha256": sha(cache_path),
            "effective_forwarded": effective,
            "effective_forwarded_sha256": sha(effective_path),
            "compile_commands_sha256": sha(database_path),
            "compile_commands_count": len(database),
            "generated_inputs_sha256": self.generated_inputs_sha256,
            "source_before": before,
            "source_after": after,
            "full_framework_build_executed": False,
            "compiler_configuration_probes_possible": True,
        }
        self.save()

    def source_identity(self, label):
        generated_inputs = self.generated_identity()
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
            "generated_inputs": generated_inputs,
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
        roots["host_python"] = Path(sys.base_prefix).resolve(strict=True)
        require(self.libomp_root is not None, "OpenMP root not recorded")
        roots["libomp"] = self.libomp_root
        identities = {name: str(path.resolve(strict=True)) for name, path in roots.items()}
        require(
            self.tool_root_identity is None or self.tool_root_identity == identities,
            "Tool-payload root identities changed",
        )
        self.tool_root_identity = identities
        self.record["tool_roots"] = identities
        result = trees.inventory_forest(roots)
        binaries = {
            name: {key: row[key] for key in ("sha256", "bytes")}
            for name, row in result["lintbin"].items()
            if not name.startswith(".") and "sha256" in row
        }
        require(
            binaries == self.contract["lint_binaries"],
            "Upstream lint binaries differ from original failed run",
        )
        require_tool_payloads(result)
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
            ("metal-path", ["xcrun", "--find", "metal"]),
            ("metal-version", ["xcrun", "metal", "--version"]),
            ("libomp-package", ["brew", "info", "--json=v2", "libomp"]),
        ):
            host[label] = self.command(label, argv)[1]
        clang = self.command("clang-path", ["xcrun", "--find", "clang"])[1].strip()
        host["clang_path"], host["clang_sha256"] = clang, sha(clang)
        host["python_path"], host["python_sha256"] = (
            str(Path(sys.executable).resolve()),
            sha(sys.executable),
        )
        omp = self.command("libomp-prefix", ["brew", "--prefix", "libomp"])[1].strip()
        self.libomp_root = Path(omp).resolve(strict=True)
        require((self.libomp_root / "lib/libomp.dylib").is_file(), "OpenMP binary bottle missing")
        self.env["CMAKE_PREFIX_PATH"] = omp
        host["libomp_root"] = str(self.libomp_root)
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
        self.env["PIP_NO_INDEX"] = "1"
        self.env["PIP_FIND_LINKS"] = str(wheels)
        self.prepare_generated_inputs()
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
        allowed = tuple(
            sorted(
                set(allowed)
                | set(self.contract["configure_prerequisite"]["fullbuild_environment"])
                | {"CMAKE_PREFIX_PATH", "PIP_NO_INDEX", "PIP_FIND_LINKS"}
            )
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
