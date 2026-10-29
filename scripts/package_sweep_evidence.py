"""Stream a completed protocol-declared width sweep into a deterministic, verified release archive.

This utility reads saved evidence only. It performs no training or numerical audit.
Temporary disk use is one Git source tar plus the eventual compressed archive;
raw checkpoints and predictions are never copied into a staging tree or read whole.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import struct
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

WIDTHS = (32, 64, 128, 256)
PROTOCOL_WIDTHS = {"v3": WIDTHS, "v4": WIDTHS, "v5": (512, 1024)}
REGRESSION_WIDTHS = (*WIDTHS, 512, 1024)
WIDE_VALIDATION_CONTRACT = "v5-common-state-replay-v1"
STRICT_VALIDATION_CONTRACT = "strict-free-running-v1"
SEEDS = (42, 43, 44)
CHUNK = 1024 * 1024
FIXTURE_FILES = {
    "config.bin",
    "initial.f32",
    "train.i32",
    "validation.i32",
    "test.i32",
    "schedule.i32",
}
RUN_FILES = {
    "best.f32",
    "last.f32",
    "adam_m.f32",
    "adam_v.f32",
    "history.f64",
    "performance.f64",
    "timings.f64",
    "test_logits.f32",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stream_digest(handle):
    value = hashlib.sha256()
    while block := handle.read(CHUNK):
        value.update(block)
    return value.hexdigest()


def digest(path):
    with Path(path).open("rb") as handle:
        return stream_digest(handle)


def tool_is_measured_source(tool, repo, source_hashes):
    tool, repo = Path(tool).resolve(), Path(repo).resolve()
    return tool.is_relative_to(repo) and source_hashes.get(
        tool.relative_to(repo).as_posix()
    ) == digest(tool)


def read_json(path):
    return json.loads(Path(path).read_text())


def safe_name(name):
    path = PurePosixPath(name)
    require(
        bool(name)
        and not path.is_absolute()
        and path.as_posix() == name
        and ".." not in path.parts
        and "." not in path.parts
        and not any(character in name for character in ("\n", "\r", "\0", "\\")),
        f"Unsafe archive or evidence path: {name!r}",
    )
    return name


def file_in(directory, name):
    path = Path(directory) / safe_name(name)
    require(path.is_file() and not path.is_symlink(), f"Missing or linked evidence file: {path}")
    for parent in path.parents:
        require(not parent.is_symlink(), f"Symlink in evidence path: {path}")
        if parent == Path(directory):
            break
    return path


def check_hash(path, expected):
    require(digest(path) == expected, f"Evidence hash mismatch: {path}")


def check_audit(audit, label):
    records = audit.get("checks", [])
    require(
        audit.get("passed") is True
        and bool(records)
        and audit.get("check_count") == len(records)
        and all(row.get("passed") is True for row in records)
        and audit.get("failed_checks") == []
        and audit.get("error") is None,
        f"Incomplete or failed audit: {label}",
    )


def expected_order(protocol="v3"):
    widths = PROTOCOL_WIDTHS[protocol]
    result = []
    for repeat in (1, 2):
        for width in widths if repeat == 1 else widths[::-1]:
            for index, seed in enumerate(SEEDS):
                pair = ("python", "arm64")
                if (widths.index(width) + index) % 2:
                    pair = pair[::-1]
                if repeat == 2:
                    pair = pair[::-1]
                result.extend((width, seed, repeat, impl) for impl in pair)
    return result


def identity(row, width=None):
    return (
        row["width"] if width is None else width,
        row["seed"],
        row["repeat"],
        row["implementation"],
    )


PARAMETER_NAMES = (
    "token",
    "position",
    "ln1_weight",
    "ln1_bias",
    "qkv_weight",
    "qkv_bias",
    "attention_weight",
    "attention_bias",
    "ln2_weight",
    "ln2_bias",
    "ff1_weight",
    "ff1_bias",
    "ff2_weight",
    "ff2_bias",
    "ln3_weight",
    "ln3_bias",
    "head_weight",
    "head_bias",
)
TRACE_NAMES = (
    "x",
    "norm1",
    "inv1",
    "norm2",
    "inv2",
    "norm3",
    "inv3",
    "n1",
    "qkv",
    "attention",
    "projected",
    "r1",
    "n2",
    "ff1",
    "relu",
    "ff2",
    "r2",
    "n3",
    "logits",
    "probs",
)


def expected_legacy_names(runtime=True):
    names = []
    for step in (1, 2, 3):
        names.append(f"trajectory/{step}/loss")
        names.extend(f"trajectory/{step}/gradient/{name}" for name in PARAMETER_NAMES)
        names.extend(f"trajectory/{step}/{name}" for name in ("parameters", "m", "v"))
    if runtime:
        names.extend(
            "runtime/matched/" + name
            for name in ("best.f32", "last.f32", "adam_m.f32", "adam_v.f32", "test_logits.f32")
        )
        names.extend(("runtime/history", "runtime/python/best_epoch"))
    return names


def required_contract_records(runtime=True):
    """Complete owned-name/tolerance schema, independently reconstructed from gate data."""
    required = {"audit/legacy_diagnostics_complete": None}
    for step in (1, 2, 3):
        for origin in ("native", "python"):
            tag = f"common_state/{step}/{origin}"
            required[tag + "/input_state_shape_finite"] = None
            required[tag + "/loss"] = "forward"
            required.update({tag + "/trace/" + name: "forward" for name in TRACE_NAMES})
            required.update({tag + "/gradient/" + name: "gradient" for name in PARAMETER_NAMES})
            for gradient in ("native", "python"):
                replay = tag + "/adam/" + gradient
                required[replay + "/step"] = None
                required[replay + "/all_state_finite"] = None
                required.update(
                    {replay + "/" + name: "optimizer" for name in ("parameters", "m", "v")}
                )
    for name in expected_legacy_names(runtime):
        if name != "runtime/python/best_epoch":
            required[name + "/finite_shape"] = None
    if runtime:
        required["runtime/shared_checkpoints_complete"] = None
        required["runtime/python/own_best_epoch"] = None
        for owner in ("native", "python"):
            for checkpoint in ("best", "last"):
                for split in ("test", "validation"):
                    tag = f"runtime/shared_checkpoint/{owner}/{checkpoint}/{split}"
                    required[tag + "/native_vs_python"] = "forward"
                    required[tag + "/all_logits_finite"] = None
                    if checkpoint == "best" and split == "test":
                        required[tag + "/native_vs_own_saved"] = "forward"
                        required[tag + "/python_vs_own_saved"] = "forward"
                        required[tag + "/own_engine_saved_logits_exact"] = None
                    if split == "validation":
                        required[tag + "/native_vs_own_history"] = "trajectory"
                        required[tag + "/python_vs_own_history"] = "trajectory"
            for name in (
                "fixture_manifest_files",
                "second_moment_nonnegative",
                "history_nonnegative",
                "timers_positive",
                "latencies_positive",
                "rss_monotone",
                "native_wall_includes_training",
                "best_validation_epoch",
                "optimizer_step_count",
                "training_includes_epoch_time",
            ):
                required[owner + "/" + name] = None
            for filename in (
                "config.bin",
                "initial.f32",
                "train.i32",
                "validation.i32",
                "test.i32",
                "schedule.i32",
            ):
                required[owner + "/fixture_hash/" + filename] = None
            for sequence in (0, 1):
                required[f"{owner}/selected_checkpoint/sequence_{sequence}"] = "forward"
            required[owner + "/selected_checkpoint_validation_nll"] = "forward"
            replay = f"runtime/replay/{owner}"
            for name in (
                "batch_sizes",
                "scheduled_rows",
                "steps",
                "best_epoch",
                "state_finite",
                "history_exact",
            ):
                required[replay + "/" + name] = None
            required[replay + "/history"] = "trajectory"
            for name in ("best.f32", "last.f32", "adam_m.f32", "adam_v.f32"):
                required[replay + "/" + name] = "trajectory"
                required[replay + "/" + name + "_exact"] = None
            for step in (1, 2, 3, 4):
                prefix = f"{replay}/step-{step}"
                required[prefix + "/gradient_finite"] = None
                required[prefix + "/state_finite"] = None
                if owner == "native":
                    required[prefix + "/adam/step"] = None
                    required.update(
                        {prefix + "/adam/" + name: "optimizer" for name in ("parameters", "m", "v")}
                    )
        required["runtime/replay/complete"] = None
    return required


def check_validation_contract(gate, width):
    if gate.get("validation_contract") != (
        WIDE_VALIDATION_CONTRACT if width >= 512 else STRICT_VALIDATION_CONTRACT
    ):
        raise ValueError("Missing or incompatible prospective validation contract")
    diagnostics = gate.get("legacy_diagnostics", [])
    failed = [record["name"] for record in diagnostics if record.get("passed") is not True]
    if (
        gate.get("legacy_diagnostic_count") != len(diagnostics)
        or gate.get("legacy_failed_checks") != failed
        or gate.get("legacy_contract_passed") is not (not failed)
    ):
        raise ValueError("Legacy diagnostic records/counts are incomplete or inconsistent")
    if width >= 512:
        runtime = gate.get("runtime_included") is True
        if [record.get("name") for record in diagnostics] != expected_legacy_names(runtime):
            raise ValueError("Original legacy diagnostic sequence is incomplete or reordered")
        for record in diagnostics:
            expected_tolerance = (
                None if record["name"] == "runtime/python/best_epoch" else "trajectory"
            )
            if (
                record.get("admission_role") != "diagnostic"
                or type(record.get("passed")) is not bool
                or record.get("tolerance") != expected_tolerance
            ):
                raise ValueError("Original legacy diagnostic semantics were altered")
        records = gate.get("checks", [])
        for name, tolerance in required_contract_records(runtime).items():
            selected = [record for record in records if record.get("name") == name]
            if (
                len(selected) != 1
                or selected[0].get("passed") is not True
                or selected[0].get("tolerance") != tolerance
            ):
                raise ValueError(f"Missing, duplicate, failed or altered contract check: {name}")
    elif diagnostics:
        raise ValueError("Strict historical widths cannot demote trajectory checks")
    return len(diagnostics), len(failed)


def validate_regression_evidence(manifest, summary, report, preflight_dir, repo):
    """V5 retains all six current binaries and policy gates, including untimed widths."""
    require(
        manifest.get("regression_widths") == list(REGRESSION_WIDTHS)
        and summary.get("regression_widths") == list(REGRESSION_WIDTHS),
        "Wrong regression width ladder",
    )
    regression = manifest.get("regression_preflights", {})
    require(
        set(regression) == {str(width) for width in REGRESSION_WIDTHS},
        "Incomplete regression preflight set",
    )
    counts = {"auto": 0, "forced": 0, "all": 0}
    diagnostic_total = diagnostic_failed = 0
    payload = {}
    for width in REGRESSION_WIDTHS:
        tag = f"width-{width}"
        modes = regression[str(width)]
        main = modes["auto"]
        expected_modes = (
            {"auto", "neon", "sme"} if main["backend"]["sme_supported"] else {"auto", "neon"}
        )
        require(set(modes) == expected_modes, "Incomplete regression backend policy set")
        if width in manifest["widths"]:
            require(
                main == manifest["preflights"][str(width)]
                and {mode: gate for mode, gate in modes.items() if mode != "auto"}
                == manifest["backend_preflights"][str(width)],
                "Regression/measured gate mismatch",
            )
        for mode, gate in modes.items():
            suffix = "" if mode == "auto" else "-" + mode
            name = tag + suffix + ".json"
            gate_path = file_in(preflight_dir, name)
            require(
                gate
                == read_json(gate_path)
                == read_json(file_in(Path(report) / "regression-preflights", name)),
                "Retained regression gate identity mismatch",
            )
            check_audit(gate, tag + " " + mode + " regression")
            total, failed = check_validation_contract(gate, width)
            diagnostic_total += total
            diagnostic_failed += failed
            parameters = 8 * width**2 + 301 * width + 128
            require(
                gate.get("runtime_included") is True
                and gate.get("model_width") == width
                and gate.get("parameter_count") == parameters
                and gate.get("model_identity")
                == {"width": width, "parameter_count": parameters, "verified": True},
                "Regression model identity mismatch",
            )
            for field in (
                "source_sha256",
                "library_sha256",
                "executable_sha256",
                "generated_layout_sha256",
                "tolerances",
            ):
                require(
                    gate.get(field) == main.get(field), "Regression policy source/binary mismatch"
                )
            hashes = gate.get("source_sha256", {})
            require(
                bool(hashes)
                and all(
                    manifest["source_sha256"].get(key) == value for key, value in hashes.items()
                ),
                "Regression source mismatch",
            )
            backend = gate["backend"]
            require(
                backend["version"] == 1
                and backend["init_status"] == 0
                and backend["requested_policy"] == {"auto": 0, "neon": 1, "sme": 2}[mode]
                and backend["sme_supported"] == main["backend"]["sme_supported"]
                and backend["streaming_vector_bytes"] == main["backend"]["streaming_vector_bytes"],
                "Regression backend identity mismatch",
            )
            expected_kind = 0 if mode == "neon" else backend["sme_supported"]
            require(
                backend["selected_kind"] == expected_kind
                and (not expected_kind or backend["streaming_vector_bytes"] == 64),
                "Regression backend selection mismatch",
            )
            counts["auto" if mode == "auto" else "forced"] += gate["check_count"]
            payload["preflights/" + name] = gate_path
        for stem, filename in (
            ("library", "libtransformer.dylib"),
            ("executable", "transformer-arm64"),
            ("generated_layout", "layout.inc"),
        ):
            native = Path(repo) / f"native/build/{tag}/{filename}"
            recorded = Path(main[stem + "_path"])
            recorded = recorded if recorded.is_absolute() else Path(repo) / recorded
            require(recorded.resolve() == native.resolve(), "Regression binary path mismatch")
            check_hash(native, main[stem + "_sha256"])
            payload[f"bin/{tag}/{filename}"] = native
    require(
        summary.get("legacy_diagnostic_count") == diagnostic_total
        and summary.get("legacy_failed_diagnostic_count") == diagnostic_failed,
        "Global legacy diagnostic counts mismatch",
    )
    counts["all"] = counts["auto"] + counts["forced"]
    require(manifest.get("regression_gate_counts") == counts, "Regression manifest count mismatch")
    for key, field in (
        ("auto", "regression_preflight_check_count"),
        ("forced", "regression_forced_backend_preflight_check_count"),
        ("all", "regression_all_backend_preflight_check_count"),
    ):
        require(summary.get(field) == counts[key], "Regression summary count mismatch")
    return payload


def validate_evidence(inputs, results, report, preflight_dir, repo):
    """Authenticate all reported raw bytes and the complete cross-file identity chain."""
    manifest_path = file_in(results, "manifest.json")
    manifest = read_json(manifest_path)
    summary = read_json(file_in(report, "summary.json"))
    audit = read_json(file_in(report, "audit.json"))
    check_audit(audit, "global report")
    protocol = manifest.get("protocol_version")
    require(protocol in PROTOCOL_WIDTHS, "Unsupported study protocol")
    widths = PROTOCOL_WIDTHS[protocol]
    if protocol == "v5":
        require(
            manifest.get("validation_contract")
            == summary.get("validation_contract")
            == WIDE_VALIDATION_CONTRACT,
            "Missing prospective wide-study validation contract",
        )
    for value in (manifest, summary):
        require(
            value.get("status") == "completed" and value.get("protocol_version") == protocol,
            "Only completed matching-protocol studies and reports can be packaged",
        )
        require(value.get("widths") == list(widths), "Wrong width ladder")
    require(bool(manifest.get("source_sha256")), "Missing measured source hashes")
    for field in ("revision", "source_sha256", "models"):
        require(summary[field] == manifest[field], f"Global report {field} mismatch")
    require(summary["raw_manifest_sha256"] == digest(manifest_path), "Global raw manifest changed")
    require(summary["audit_check_count"] == audit["check_count"], "Global audit count mismatch")
    order = expected_order(protocol)
    require(
        [identity(row) for row in manifest["runs"]] == order,
        f"Incomplete or reordered {len(order)}-run sweep",
    )
    require([tuple(row) for row in manifest["order"]] == order, "Declared sweep order mismatch")
    global_runs = {identity(row): row for row in summary["runs"]}
    require(
        len(summary["runs"]) == len(order) and set(global_runs) == set(order),
        "Global report run identities mismatch",
    )
    require(
        summary["retained_test_prediction_rows"] == len(order) * 256 * 32,
        "Global prediction count mismatch",
    )
    natives, preflights = {}, {}
    total_preflight = total_child_audit = total_forced = 0
    for width in widths:
        key, tag = str(width), f"width-{width}"
        model = manifest["models"][key]
        expected_model = {
            "width": width,
            "heads": 4,
            "head_width": width // 4,
            "hidden": 2 * width,
            "context": 32,
            "vocab": 128,
            "parameters": 8 * width**2 + 301 * width + 128,
        }
        require(
            all(model.get(name) == value for name, value in expected_model.items()),
            f"Wrong model: {tag}",
        )
        selected_path = file_in(results, tag + "/manifest.json")
        selected = read_json(selected_path)
        width_report = Path(report) / tag
        child_summary = read_json(file_in(width_report, "summary.json"))
        child_audit = read_json(file_in(width_report, "audit.json"))
        check_audit(child_audit, tag)
        for value in (selected, child_summary):
            require(
                value.get("status") == "completed" and value.get("protocol_version") == protocol,
                f"Incomplete {tag}",
            )
            for field in ("revision", "source_sha256"):
                require(value[field] == manifest[field], f"{tag} {field} mismatch")
            require(value["model"] == model, f"{tag} model identity mismatch")
        require(
            selected["width"] == width
            and selected["model_width"] == width
            and selected["parameter_count"] == model["parameters"],
            f"{tag} compiled model mismatch",
        )
        selected_entries = [row for row in manifest["runs"] if row["width"] == width]
        require(selected["runs"] == selected_entries, f"{tag} runner entries mismatch")
        require(
            selected["order"] == [list(row[1:]) for row in order if row[0] == width],
            f"{tag} order mismatch",
        )
        require(
            selected["fixtures"] == manifest["fixtures"][key], f"{tag} fixture manifest mismatch"
        )
        if protocol == "v5":
            require(
                selected.get("validation_contract")
                == child_summary.get("validation_contract")
                == WIDE_VALIDATION_CONTRACT,
                "Child validation contract mismatch",
            )
        require(selected["preflight"] == manifest["preflights"][key], f"{tag} preflight mismatch")
        require(
            child_summary["raw_manifest_sha256"] == digest(selected_path),
            f"{tag} raw manifest changed",
        )
        require(
            child_summary["audit_check_count"] == child_audit["check_count"],
            f"{tag} audit count mismatch",
        )
        require(
            child_summary["retained_test_prediction_rows"] == 12 * 256 * 32,
            f"{tag} prediction count mismatch",
        )
        aggregate = summary["by_width"][key]
        for field in (
            "model",
            "aggregate",
            "audit_check_count",
            "preflight_check_count",
            "retained_test_prediction_rows",
            "raw_manifest_sha256",
        ):
            require(
                aggregate[field] == child_summary[field], f"{tag} aggregate field mismatch: {field}"
            )
        check_hash(width_report / "summary.json", aggregate["report_summary_sha256"])
        check_hash(width_report / "audit.json", aggregate["report_audit_sha256"])
        preflight_path = file_in(preflight_dir, tag + ".json")
        preflight = read_json(preflight_path)
        require(
            preflight
            == selected["preflight"]
            == read_json(file_in(width_report, "preflight.json")),
            f"{tag} retained preflight mismatch",
        )
        check_audit(preflight, tag + " preflight")
        require(
            preflight.get("runtime_included") is True
            and preflight.get("model_width") == width
            and preflight.get("parameter_count") == model["parameters"]
            and preflight.get("model_identity")
            == {"width": width, "parameter_count": model["parameters"], "verified": True},
            f"{tag} preflight identity mismatch",
        )
        require(
            child_summary["preflight_check_count"] == preflight["check_count"],
            f"{tag} preflight count mismatch",
        )
        require(
            bool(preflight.get("source_sha256"))
            and all(
                manifest["source_sha256"].get(name) == value
                for name, value in preflight["source_sha256"].items()
            ),
            f"{tag} preflight/source mismatch",
        )
        for filename, stem in (
            ("transformer-arm64", "executable"),
            ("libtransformer.dylib", "library"),
            ("layout.inc", "generated_layout"),
        ):
            relative = f"native/build/{tag}/{filename}"
            native = file_in(repo, relative)
            recorded = Path(preflight[stem + "_path"])
            recorded = recorded if recorded.is_absolute() else Path(repo) / recorded
            require(recorded.resolve() == native.resolve(), f"{tag} native path mismatch")
            check_hash(native, preflight[stem + "_sha256"])
            natives[f"bin/{tag}/{filename}"] = native
        preflights[f"preflights/{tag}.json"] = preflight_path
        total_preflight += preflight["check_count"]
        if protocol in ("v4", "v5"):
            modes = ("neon", "sme") if preflight["backend"]["sme_supported"] else ("neon",)
            supplements = manifest["backend_preflights"][key]
            require(
                set(supplements) == set(modes) and selected["backend_preflights"] == supplements,
                f"{tag} forced backend gate set mismatch",
            )
            forced_count = 0
            for mode in modes:
                mode_path = file_in(preflight_dir, f"{tag}-{mode}.json")
                gate = read_json(mode_path)
                require(
                    gate
                    == supplements[mode]
                    == read_json(file_in(width_report, f"preflight-{mode}.json")),
                    f"{tag} forced backend gate identity mismatch",
                )
                check_audit(gate, f"{tag} {mode} preflight")
                require(
                    gate.get("runtime_included") is True
                    and gate["backend"]["requested_policy"] == {"neon": 1, "sme": 2}[mode],
                    "Wrong forced backend policy",
                )
                for field in (
                    "model_width",
                    "parameter_count",
                    "model_identity",
                    "source_sha256",
                    "library_sha256",
                    "executable_sha256",
                    "generated_layout_sha256",
                    "tolerances",
                ):
                    require(gate.get(field) == preflight.get(field), f"Forced gate {field} differs")
                forced_count += gate["check_count"]
                preflights[f"preflights/{tag}-{mode}.json"] = mode_path
            require(
                child_summary["forced_backend_preflight_check_count"] == forced_count
                and child_summary["all_backend_preflight_check_count"]
                == preflight["check_count"] + forced_count,
                f"{tag} backend gate count mismatch",
            )
            total_forced += forced_count
        total_child_audit += child_audit["check_count"]
        require(
            len(child_summary["runs"]) == 12
            and {identity(row, width) for row in child_summary["runs"]}
            == {identity(row) for row in selected_entries},
            f"{tag} report identities mismatch",
        )
        for seed in SEEDS:
            directory = Path(inputs) / tag / f"seed-{seed}"
            fixture = read_json(file_in(directory, "manifest.json"))
            require(
                fixture == selected["fixtures"][str(seed)],
                f"{tag}/seed-{seed} fixture identity mismatch",
            )
            require(set(fixture["sha256"]) == FIXTURE_FILES, "Fixture payload set incomplete")
            for name, expected in fixture["sha256"].items():
                check_hash(file_in(directory, name), expected)
        for run in child_summary["runs"]:
            run_key = identity(run, width)
            name = f"seed-{run['seed']}-repeat-{run['repeat']}-{run['implementation']}"
            require(
                run["name"] == run["run"] == name and run["fixture"] == f"seed-{run['seed']}",
                "Unsafe or mismatched run identity",
            )
            require(
                global_runs[run_key] == run | {"width": width, "relative_run": tag + "/" + name},
                f"Global/child report run mismatch: {tag}/{name}",
            )
            require(
                run["fixture_sha256"] == selected["fixtures"][str(run["seed"])]["sha256"],
                "Run fixture identity mismatch",
            )
            require(
                run["checkpoint_predictions_recomputed"] is True
                and run["quality"]["tokens"] == 256 * 32,
                "Missing independent checkpoint prediction audit",
            )
            directory = Path(results) / tag / name
            hashes = run["artifact_sha256"]
            actual = {path.name for path in directory.iterdir() if path.is_file()}
            require(
                RUN_FILES <= set(hashes) and actual == set(hashes),
                f"Run artifact set changed: {tag}/{name}",
            )
            for filename, expected in hashes.items():
                check_hash(file_in(directory, filename), expected)
        for row in selected_entries:
            if protocol in ("v4", "v5") and row["implementation"] == "arm64":
                backend_path = file_in(Path(results) / tag / row["name"], "backend.u64")
                check_hash(backend_path, row["backend_sha256"])
                words = struct.unpack("<10Q", backend_path.read_bytes())
                names = (
                    "version",
                    "init_status",
                    "requested_policy",
                    "selected_kind",
                    "sme_supported",
                    "streaming_vector_bytes",
                    "sme_forward_calls",
                    "sme_backward_calls",
                    "neon_forward_calls",
                    "neon_backward_calls",
                )
                metadata = dict(zip(names, words, strict=True))
                require(
                    metadata == row["backend"] == global_runs[identity(row)]["backend"]
                    and row.get("backend_valid") is True
                    and row.get("backend_environment") == "auto",
                    "Native backend identity changed after audit",
                )
            require(
                row.get("status") == "completed"
                and row.get("returncode") == 0
                and all(
                    row.get(flag) is True
                    for flag in (
                        "conditions_valid",
                        "source_unchanged",
                        "binary_unchanged",
                        "fixtures_unchanged",
                    )
                ),
                f"Invalid measured child: {row['name']}",
            )
    if protocol in ("v4", "v5"):
        require(
            summary["forced_backend_preflight_check_count"] == total_forced
            and summary["all_backend_preflight_check_count"] == total_preflight + total_forced,
            "Global backend preflight count mismatch",
        )
    require(summary["preflight_check_count"] == total_preflight, "Total preflight count mismatch")
    require(
        summary["audit_check_count"] == summary["sweep_audit_check_count"] + total_child_audit,
        "Total report audit count mismatch",
    )
    regression = {}
    if protocol == "v5":
        regression = validate_regression_evidence(manifest, summary, report, preflight_dir, repo)
    return manifest, summary, natives | preflights | regression


def archive_source(repo, revision, destination, expected):
    require(
        len(revision) == 40 and all(char in "0123456789abcdef" for char in revision),
        "Measured revision must be a full Git commit hash",
    )
    with Path(destination).open("xb") as handle:
        subprocess.run(
            ["git", "archive", "--format=tar", revision], cwd=repo, stdout=handle, check=True
        )
    seen = set()
    with tarfile.open(destination, "r|") as archive:
        for member in archive:
            if member.isdir():
                continue
            require(
                member.isfile() and member.name not in seen,
                "Unsupported or duplicate measured-source member",
            )
            safe_name(member.name)
            require(member.name in expected, f"Unrecorded source member: {member.name}")
            handle = archive.extractfile(member)
            require(
                stream_digest(handle) == expected[member.name],
                f"Measured source hash mismatch: {member.name}",
            )
            seen.add(member.name)
    require(seen == set(expected), "Measured source archive is incomplete")


def add_tree(payload, directory, prefix):
    directory = Path(directory)
    require(
        directory.is_dir() and not directory.is_symlink(),
        f"Invalid evidence directory: {directory}",
    )
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), f"Symlink in release inputs: {path}")
        if path.is_file():
            name = safe_name(prefix + "/" + path.relative_to(directory).as_posix())
            require(name not in payload, f"Duplicate archive name: {name}")
            payload[name] = path


def write_archive(output, payload):
    hashes = {safe_name(name): digest(path) for name, path in sorted(payload.items())}
    checksums = "".join(f"{value}  {name}\n" for name, value in hashes.items()).encode()
    with Path(output).open("xb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=6, fileobj=raw, mtime=0
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.GNU_FORMAT) as archive:
                for name in sorted([*payload, "SHA256SUMS"]):
                    info = tarfile.TarInfo(name)
                    info.mode = (
                        0o755
                        if name.startswith("bin/") and name.endswith("/transformer-arm64")
                        else 0o644
                    )
                    info.mtime = info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    if name == "SHA256SUMS":
                        info.size = len(checksums)
                        archive.addfile(info, io.BytesIO(checksums))
                    else:
                        info.size = payload[name].stat().st_size
                        with payload[name].open("rb") as handle:
                            archive.addfile(info, handle)
    return hashes


def verify_archive(path, expected_payload=None):
    expected, seen = None, set()
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            safe_name(member.name)
            require(
                member.isfile() and member.name not in seen,
                "Unsupported or duplicate archive member",
            )
            seen.add(member.name)
            handle = archive.extractfile(member)
            if member.name == "SHA256SUMS":
                require(expected is None and len(seen) == 1, "Checksum manifest must be first")
                require(member.size <= 16 * CHUNK, "Unreasonably large checksum manifest")
                expected = {}
                for line in handle.read().decode().splitlines():
                    value, name = line.split("  ", 1)
                    safe_name(name)
                    require(
                        name not in expected and len(value) == 64, "Malformed checksum manifest"
                    )
                    expected[name] = value
                if expected_payload is not None:
                    require(
                        expected == expected_payload,
                        "Archive checksum manifest differs from the input inventory",
                    )
            else:
                require(
                    expected is not None and member.name in expected, "Unmanifested archive payload"
                )
                require(
                    stream_digest(handle) == expected[member.name],
                    f"Archive checksum failed: {member.name}",
                )
    require(
        expected is not None and seen == set(expected) | {"SHA256SUMS"},
        "Archive payload set is incomplete",
    )
    return expected


def package(args):
    output = Path(args.output).resolve()
    require(output.name.endswith(".tar.gz"), "Output must end in .tar.gz")
    base = output.name.removesuffix(".tar.gz")
    checksum = output.with_name(base + "-SHA256SUMS.txt")
    verification = output.with_name(base + "-verification.json")
    partial = output.with_name(output.name + ".partial")
    for path in (output, checksum, verification, partial):
        if path.exists():
            raise FileExistsError(path)
    repo = Path(args.repo).resolve()
    inputs, results, report, preflight_dir = (
        Path(value).resolve()
        for value in (args.inputs, args.results, args.report, args.preflight_dir)
    )
    extras = {}
    for item in getattr(args, "extra_evidence", []):
        name, separator, value = item.partition("=")
        require(
            separator and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name),
            "Extra evidence must be NAME=DIRECTORY with a simple unique name",
        )
        require(name not in extras, "Duplicate extra-evidence name")
        directory = Path(value).resolve()
        require(directory.is_dir(), "Extra-evidence directory is missing")
        extras[name] = directory
    for directory in (inputs, results, report, preflight_dir, *extras.values()):
        require(
            not output.is_relative_to(directory),
            "Archive output must be outside all input evidence directories",
        )
    manifest, summary, supplemental = validate_evidence(
        inputs, results, report, preflight_dir, repo
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sweep-source-", dir=output.parent) as temporary:
        source = Path(temporary) / "measured-source.tar"
        archive_source(repo, manifest["revision"], source, manifest["source_sha256"])
        payload = {"measured-source.tar": source, **supplemental}
        add_tree(payload, inputs, "inputs")
        add_tree(payload, results, "results")
        add_tree(payload, report, "report")
        extra_records = []
        for name, directory in extras.items():
            before = set(payload)
            add_tree(payload, directory, "development/" + name)
            count = len(set(payload) - before)
            require(count > 0, "Extra-evidence directory contains no files")
            extra_records.append({"name": name, "directory": str(directory), "files": count})
        tool = Path(__file__).resolve()
        payload["packaging/package_sweep_evidence.py"] = tool
        raw_bytes = sum(path.stat().st_size for path in payload.values())
        require(
            shutil.disk_usage(output.parent).free > raw_bytes + 32 * CHUNK,
            "Insufficient free space for bounded archive creation",
        )
        # Capture all inputs before writing, then verify streamed bytes and reread inputs.
        expected = write_archive(partial, payload)
        verified = verify_archive(partial, expected)
        for name, path in payload.items():
            check_hash(path, verified[name])
        rechecked_manifest, rechecked_summary, _ = validate_evidence(
            inputs, results, report, preflight_dir, repo
        )
        require(
            rechecked_manifest == manifest and rechecked_summary == summary,
            "Evidence identities changed during packaging",
        )
        archive_hash = digest(partial)
        evidence = {
            "verified": True,
            "schema_version": 1,
            "protocol_version": manifest["protocol_version"],
            "archive": output.name,
            "sha256": archive_hash,
            "archive_bytes": partial.stat().st_size,
            "payload_bytes": raw_bytes,
            "verified_payload_files": len(verified),
            "measured_revision": manifest["revision"],
            "widths": manifest["widths"],
            "measured_fits": len(manifest["runs"]),
            "models": manifest["models"],
            "extra_evidence": extra_records,
            "extra_evidence_scope": (
                "Separately labeled development/profiling bytes are authenticated and preserved; "
                "they are not additional formal matched fits or independent quality seeds."
            ),
            "preflight_checks": summary["preflight_check_count"],
            "forced_backend_preflight_checks": summary.get(
                "forced_backend_preflight_check_count", 0
            ),
            "all_backend_preflight_checks": summary.get(
                "all_backend_preflight_check_count", summary["preflight_check_count"]
            ),
            "validation_contract": summary.get("validation_contract"),
            "legacy_diagnostic_count": summary.get("legacy_diagnostic_count", 0),
            "legacy_failed_diagnostic_count": summary.get("legacy_failed_diagnostic_count", 0),
            "regression_widths": summary.get("regression_widths", manifest["widths"]),
            "regression_all_backend_preflight_checks": summary.get(
                "regression_all_backend_preflight_check_count",
                summary.get("all_backend_preflight_check_count", summary["preflight_check_count"]),
            ),
            "postrun_checks": summary["audit_check_count"],
            "retained_test_prediction_rows": summary["retained_test_prediction_rows"],
            "raw_manifest_sha256": digest(results / "manifest.json"),
            "global_report_sha256": digest(report / "summary.json"),
            "packaging_tool_sha256": digest(tool),
            "packaging_tool_is_measured_source": tool_is_measured_source(
                tool, repo, manifest["source_sha256"]
            ),
            "verification_scope": (
                "All payload bytes rehashed from the archive and inputs; measured source, "
                "binary/layout, fixture, raw-run and report identity chains checked. "
                "Saved numerical audits are authenticated, not rerun."
            ),
            "determinism": (
                "Sorted members, fixed modes/owners/mtime, empty gzip filename and mtime zero; "
                "identical inputs and Python/zlib produce identical archive bytes."
            ),
        }
        # Linking publishes the verified archive without ever replacing an existing path.
        os.link(partial, output)
        partial.unlink()
        with checksum.open("x") as handle:
            handle.write(f"{archive_hash}  {output.name}\n")
        with verification.open("x") as handle:
            handle.write(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--extra-evidence",
        action="append",
        default=[],
        metavar="NAME=DIRECTORY",
        help="Retain a named development/profiling directory; repeat as needed",
    )
    args = parser.parse_args()
    print(json.dumps(package(args), indent=2))


if __name__ == "__main__":
    main()
