"""Small saved-evidence tests only; no training, native calls or study timings."""

import copy
import gzip
import hashlib
import importlib.util
import io
import json
import struct
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from contract_fixtures import attach_contract

# The same relative location works in ignored development and after installation.
MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/package_sweep_evidence.py"
SPEC = importlib.util.spec_from_file_location("package_sweep_evidence", MODULE_PATH)
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def audit(name="passed"):
    return {
        "passed": True,
        "checks": [{"name": name, "passed": True}],
        "check_count": 1,
        "failed_checks": [],
        "error": None,
    }


def synthetic_backend(width, policy=0):
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
    values = [1, 0, policy, int(policy != 1), 1, 64, 0, 0, 0, 0]
    active = 6 if width >= 64 and policy != 1 else 8
    values[active : active + 2] = [94040, 81920]
    return dict(zip(names, values, strict=True))


def evidence_tree(tmp_path, protocol="v3"):
    """Build tiny fake payload bytes with real, internally consistent identity hashes."""
    widths = packager.PROTOCOL_WIDTHS[protocol]
    order = packager.expected_order(protocol)
    repo = tmp_path / "repository"
    repo.mkdir()
    (repo / "source.txt").write_text("measured source\n")
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    subprocess.run(["git", "add", "source.txt"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Evidence Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "frozen",
        ],
        cwd=repo,
        check=True,
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    inputs, results, reports, preflights = (
        tmp_path / name for name in ("inputs", "results", "reports", "preflights")
    )
    manifest = {
        "status": "completed",
        "protocol_version": protocol,
        "widths": list(widths),
        "revision": revision,
        "source_sha256": {"source.txt": packager.digest(repo / "source.txt")},
        "models": {},
        "fixtures": {},
        "preflights": {},
        "backend_preflights": {},
        "runs": [],
        "order": order,
    }
    for width in widths:
        key, tag = str(width), f"width-{width}"
        model = {
            "width": width,
            "heads": 4,
            "head_width": width // 4,
            "hidden": 2 * width,
            "context": 32,
            "vocab": 128,
            "parameters": 8 * width**2 + 301 * width + 128,
        }
        manifest["models"][key] = model
        preflight = audit() | {
            "runtime_included": True,
            "model_width": width,
            "parameter_count": model["parameters"],
            "model_identity": {
                "width": width,
                "parameter_count": model["parameters"],
                "verified": True,
            },
            "source_sha256": manifest["source_sha256"],
        }
        for filename, stem in (
            ("transformer-arm64", "executable"),
            ("libtransformer.dylib", "library"),
            ("layout.inc", "generated_layout"),
        ):
            relative = f"native/build/{tag}/{filename}"
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{width} {filename}\n")
            preflight[stem + "_path"] = relative
            preflight[stem + "_sha256"] = packager.digest(path)
        if protocol in ("v4", "v5"):
            preflight["backend"] = synthetic_backend(width)
            modes = {}
            for mode, number in (("neon", 1), ("sme", 2)):
                modes[mode] = copy.deepcopy(preflight)
                modes[mode]["backend"] = synthetic_backend(width, number)
                write_json(preflights / f"{tag}-{mode}.json", modes[mode])
                write_json(reports / tag / f"preflight-{mode}.json", modes[mode])
            manifest["backend_preflights"][key] = modes
        manifest["preflights"][key] = preflight
        write_json(preflights / f"{tag}.json", preflight)
        manifest["fixtures"][key] = {}
        for seed in packager.SEEDS:
            directory = inputs / tag / f"seed-{seed}"
            directory.mkdir(parents=True)
            hashes = {}
            for name in packager.FIXTURE_FILES:
                (directory / name).write_bytes(f"fixture {width} {seed} {name}\n".encode())
                hashes[name] = packager.digest(directory / name)
            fixture = {"sha256": hashes, "config": {"width": width, "seed": seed}}
            manifest["fixtures"][key][str(seed)] = fixture
            write_json(directory / "manifest.json", fixture)
    for width, seed, repeat, impl in order:
        name = f"seed-{seed}-repeat-{repeat}-{impl}"
        manifest["runs"].append(
            {
                "width": width,
                "seed": seed,
                "repeat": repeat,
                "implementation": impl,
                "name": name,
                "status": "completed",
                "returncode": 0,
                "conditions_valid": True,
                "source_unchanged": True,
                "binary_unchanged": True,
                "fixtures_unchanged": True,
            }
        )
        directory = results / f"width-{width}" / name
        directory.mkdir(parents=True)
        if protocol in ("v4", "v5") and impl == "arm64":
            backend = synthetic_backend(width)
            backend_path = directory / "backend.u64"
            backend_path.write_bytes(struct.pack("<10Q", *backend.values()))
            manifest["runs"][-1].update(
                backend=backend,
                backend_sha256=packager.digest(backend_path),
                backend_valid=True,
                backend_environment="auto",
            )
        for artifact in packager.RUN_FILES:
            (directory / artifact).write_bytes(f"{width} {name} {artifact}\n".encode())
    if protocol == "v5":
        regression = {}
        for width in packager.REGRESSION_WIDTHS:
            key, tag = str(width), f"width-{width}"
            if width in widths:
                main = manifest["preflights"][key]
                modes = manifest["backend_preflights"][key]
            else:
                main = copy.deepcopy(manifest["preflights"][str(widths[0])])
                parameters = 8 * width**2 + 301 * width + 128
                main.update(
                    model_width=width,
                    parameter_count=parameters,
                    model_identity={
                        "width": width,
                        "parameter_count": parameters,
                        "verified": True,
                    },
                    backend=synthetic_backend(width),
                )
                for stem, filename in (
                    ("library", "libtransformer.dylib"),
                    ("executable", "transformer-arm64"),
                    ("generated_layout", "layout.inc"),
                ):
                    relative = f"native/build/{tag}/{filename}"
                    path = repo / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"{width} {filename}\n")
                    main[stem + "_path"] = relative
                    main[stem + "_sha256"] = packager.digest(path)
                modes = {mode: copy.deepcopy(main) for mode in ("neon", "sme")}
                for mode, number in (("neon", 1), ("sme", 2)):
                    modes[mode]["backend"] = synthetic_backend(width, number)
            for gate in (main, *modes.values()):
                attach_contract(gate, width)
            regression[key] = {"auto": main, **modes}
            for mode, gate in regression[key].items():
                suffix = "" if mode == "auto" else "-" + mode
                name = tag + suffix + ".json"
                write_json(preflights / name, gate)
                write_json(reports / "regression-preflights" / name, gate)
                if width in widths and mode != "auto":
                    write_json(reports / tag / f"preflight-{mode}.json", gate)
        manifest.update(
            regression_widths=list(packager.REGRESSION_WIDTHS),
            regression_preflights=regression,
            regression_gate_counts={"auto": 1018, "forced": 2036, "all": 3054},
            validation_contract="v5-common-state-replay-v1",
        )
    write_json(results / "manifest.json", manifest)
    summary = {
        "status": "completed",
        "protocol_version": protocol,
        "widths": list(widths),
        "revision": revision,
        "source_sha256": manifest["source_sha256"],
        "models": manifest["models"],
        "raw_manifest_sha256": packager.digest(results / "manifest.json"),
        "audit_check_count": 1 + len(widths),
        "sweep_audit_check_count": 1,
        "preflight_check_count": sum(
            gate["check_count"] for gate in manifest["preflights"].values()
        ),
        "retained_test_prediction_rows": len(order) * 256 * 32,
        "by_width": {},
        "runs": [],
    }
    if protocol in ("v4", "v5"):
        summary.update(
            forced_backend_preflight_check_count=2 * summary["preflight_check_count"],
            all_backend_preflight_check_count=3 * summary["preflight_check_count"],
        )
    if protocol == "v5":
        summary.update(
            regression_widths=list(packager.REGRESSION_WIDTHS),
            regression_preflight_check_count=1018,
            regression_forced_backend_preflight_check_count=2036,
            regression_all_backend_preflight_check_count=3054,
            validation_contract="v5-common-state-replay-v1",
            legacy_diagnostic_count=438,
            legacy_failed_diagnostic_count=6,
        )
    for width in widths:
        key, tag = str(width), f"width-{width}"
        selected = {
            field: manifest[field]
            for field in ("status", "protocol_version", "revision", "source_sha256")
        }
        selected.update(
            width=width,
            model_width=width,
            parameter_count=manifest["models"][key]["parameters"],
            model=manifest["models"][key],
            fixtures=manifest["fixtures"][key],
            preflight=manifest["preflights"][key],
            runs=[row for row in manifest["runs"] if row["width"] == width],
            order=[list(row[1:]) for row in order if row[0] == width],
        )
        if protocol in ("v4", "v5"):
            selected["backend_preflights"] = manifest["backend_preflights"][key]
        if protocol == "v5":
            selected["validation_contract"] = "v5-common-state-replay-v1"
        write_json(results / tag / "manifest.json", selected)
        child = {
            field: selected[field]
            for field in ("status", "protocol_version", "revision", "source_sha256", "model")
        }
        child.update(
            raw_manifest_sha256=packager.digest(results / tag / "manifest.json"),
            audit_check_count=1,
            preflight_check_count=selected["preflight"]["check_count"],
            retained_test_prediction_rows=12 * 256 * 32,
            aggregate={"synthetic": width},
            runs=[],
        )
        if protocol in ("v4", "v5"):
            child.update(
                forced_backend_preflight_check_count=2 * child["preflight_check_count"],
                all_backend_preflight_check_count=3 * child["preflight_check_count"],
            )
        if protocol == "v5":
            child["validation_contract"] = "v5-common-state-replay-v1"
        for row in selected["runs"]:
            name = row["name"]
            directory = results / tag / name
            run = {
                "name": name,
                "run": name,
                "fixture": f"seed-{row['seed']}",
                "seed": row["seed"],
                "repeat": row["repeat"],
                "implementation": row["implementation"],
                "fixture_sha256": selected["fixtures"][str(row["seed"])]["sha256"],
                "artifact_sha256": {
                    path.name: packager.digest(path) for path in directory.iterdir()
                },
                "checkpoint_predictions_recomputed": True,
                "quality": {"tokens": 256 * 32},
            }
            if protocol in ("v4", "v5") and row["implementation"] == "arm64":
                run["backend"] = row["backend"]
            child["runs"].append(run)
            summary["runs"].append(run | {"width": width, "relative_run": tag + "/" + name})
        write_json(reports / tag / "summary.json", child)
        write_json(reports / tag / "audit.json", audit())
        write_json(reports / tag / "preflight.json", selected["preflight"])
        summary["by_width"][key] = {
            field: child[field]
            for field in (
                "model",
                "aggregate",
                "audit_check_count",
                "preflight_check_count",
                "retained_test_prediction_rows",
                "raw_manifest_sha256",
            )
        } | {
            "report_summary_sha256": packager.digest(reports / tag / "summary.json"),
            "report_audit_sha256": packager.digest(reports / tag / "audit.json"),
        }
    global_audit = audit()
    global_audit["checks"] = [
        {"name": f"check-{index}", "passed": True} for index in range(1 + len(widths))
    ]
    global_audit["check_count"] = 1 + len(widths)
    write_json(reports / "summary.json", summary)
    write_json(reports / "audit.json", global_audit)
    return SimpleNamespace(
        repo=repo,
        inputs=inputs,
        results=results,
        report=reports,
        preflight_dir=preflights,
        output=tmp_path / "release/evidence.tar.gz",
    )


def test_full_tiny_campaign_packages_all_48_runs_and_is_deterministic(tmp_path):
    args = evidence_tree(tmp_path)
    # Source after measurement may change; only the archived measured revision is relevant.
    (args.repo / "source.txt").write_text("post-measurement packaging work\n")
    first = packager.package(args)
    original = args.output.read_bytes()
    args.output = tmp_path / "release/second.tar.gz"
    second = packager.package(args)
    assert args.output.read_bytes() == original
    assert first["sha256"] == second["sha256"]
    assert first["measured_fits"] == 48
    assert first["retained_test_prediction_rows"] == 393216
    verified = packager.verify_archive(args.output)
    assert sum(name.endswith("/test_logits.f32") for name in verified) == 48
    assert sum(name.endswith("/config.bin") for name in verified) == 12
    assert sum(name.startswith("bin/") for name in verified) == 12
    assert "measured-source.tar" in verified
    assert "packaging/package_sweep_evidence.py" in verified
    assert args.output.with_name("second-SHA256SUMS.txt").read_text().startswith(second["sha256"])


@pytest.mark.parametrize(
    "target",
    [
        "raw",
        "fixture",
        "binary",
        "layout",
        "child_report",
        "child_audit",
        "preflight",
        "source_identity",
    ],
)
def test_each_identity_chain_rejects_tampering_before_archive_creation(tmp_path, target):
    args = evidence_tree(tmp_path)
    candidates = {
        "raw": args.results / "width-256/seed-44-repeat-2-arm64/test_logits.f32",
        "fixture": args.inputs / "width-64/seed-43/train.i32",
        "binary": args.repo / "native/build/width-128/libtransformer.dylib",
        "layout": args.repo / "native/build/width-32/layout.inc",
        "child_report": args.report / "width-128/summary.json",
        "child_audit": args.report / "width-64/audit.json",
        "preflight": args.preflight_dir / "width-32.json",
    }
    if target == "source_identity":
        # Keep source maps mutually consistent but give a digest the Git snapshot cannot satisfy.
        for directory in (args.results, args.report, args.preflight_dir):
            for path in directory.rglob("*.json"):
                text = path.read_text().replace(packager.digest(args.repo / "source.txt"), "0" * 64)
                path.write_text(text)
    else:
        path = candidates[target]
        if path.suffix == ".json":
            value = json.loads(path.read_text())
            value["tampered"] = True
            write_json(path, value)
        else:
            path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises((ValueError, KeyError)):
        packager.package(args)
    assert not args.output.exists()


def test_measured_source_archive_checks_exact_file_set_and_content(tmp_path):
    args = evidence_tree(tmp_path)
    manifest = json.loads((args.results / "manifest.json").read_text())
    with pytest.raises(ValueError, match="source hash mismatch"):
        packager.archive_source(
            args.repo, manifest["revision"], tmp_path / "wrong.tar", {"source.txt": "0" * 64}
        )
    with pytest.raises(ValueError, match="incomplete"):
        packager.archive_source(
            args.repo,
            manifest["revision"],
            tmp_path / "missing.tar",
            manifest["source_sha256"] | {"missing": "0" * 64},
        )


def test_missing_or_duplicate_fit_rejected(tmp_path):
    args = evidence_tree(tmp_path)
    path = args.results / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["runs"][-1] = copy.deepcopy(manifest["runs"][0])
    write_json(path, manifest)
    summary_path = args.report / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["raw_manifest_sha256"] = packager.digest(path)
    write_json(summary_path, summary)
    with pytest.raises(ValueError, match="48-run sweep"):
        packager.package(args)


def test_refuses_existing_outputs_and_symlink_payloads(tmp_path):
    args = evidence_tree(tmp_path)
    args.output.parent.mkdir()
    args.output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        packager.package(args)
    assert args.output.read_bytes() == b"keep"
    directory = tmp_path / "links"
    directory.mkdir()
    (directory / "alias").symlink_to(args.repo / "source.txt")
    with pytest.raises(ValueError, match="Symlink"):
        packager.add_tree({}, directory, "linked")


def test_streaming_hash_never_requests_an_unbounded_read():
    class BoundedReader(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= packager.CHUNK
            return super().read(size)

    data = b"small diagnostic payload"
    assert packager.stream_digest(BoundedReader(data)) == hashlib.sha256(data).hexdigest()


def test_archive_verifier_rejects_corrupted_payload(tmp_path):
    path = tmp_path / "bad.tar.gz"
    expected = hashlib.sha256(b"original").hexdigest()
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for name, data in (
                ("SHA256SUMS", f"{expected}  payload\n".encode()),
                ("payload", b"changed"),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match="Archive checksum failed"):
        packager.verify_archive(path)


def test_v4_packaging_requires_forced_gates_and_raw_backend_bytes(tmp_path):
    args = evidence_tree(tmp_path, protocol="v4")
    manifest, summary, extra = packager.validate_evidence(
        args.inputs, args.results, args.report, args.preflight_dir, args.repo
    )
    assert manifest["protocol_version"] == "v4"
    assert summary["all_backend_preflight_check_count"] == 12
    assert "preflights/width-256-sme.json" in extra
    path = args.results / "width-64/seed-42-repeat-1-arm64/backend.u64"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError):
        packager.validate_evidence(
            args.inputs, args.results, args.report, args.preflight_dir, args.repo
        )


def test_extra_evidence_is_labeled_retained_and_hashed(tmp_path):
    args = evidence_tree(tmp_path)
    extra = tmp_path / "profile"
    extra.mkdir()
    (extra / "samples.txt").write_text("synthetic CPU profile")
    args.extra_evidence = [f"baseline-profile={extra}"]
    result = packager.package(args)
    assert result["extra_evidence"] == [
        {"name": "baseline-profile", "directory": str(extra), "files": 1}
    ]
    members = packager.verify_archive(args.output)
    assert members["development/baseline-profile/samples.txt"] == packager.digest(
        extra / "samples.txt"
    )


@pytest.mark.parametrize("name", ["../escape", "nested/name", "", "a.b"])
def test_extra_evidence_rejects_unsafe_archive_names(tmp_path, name):
    args = evidence_tree(tmp_path)
    args.extra_evidence = [f"{name}={args.results}"]
    with pytest.raises(ValueError, match="simple unique name"):
        packager.package(args)
    assert not args.output.exists()


@pytest.mark.parametrize("case", ["matching", "changed", "outside"])
def test_packaging_tool_source_identity(tmp_path, case):
    repo = tmp_path / "repo"
    repo.mkdir()
    tool = (tmp_path if case == "outside" else repo) / "package.py"
    tool.write_text("# packaging tool\n")
    expected = packager.digest(tool) if case != "changed" else "0" * 64
    assert packager.tool_is_measured_source(tool, repo, {"package.py": expected}) == (
        case == "matching"
    )


def test_v5_packages_24_fits_and_all_six_current_regression_binaries(tmp_path):
    args = evidence_tree(tmp_path, protocol="v5")
    result = packager.package(args)
    assert result["widths"] == [512, 1024]
    assert result["measured_fits"] == 24
    assert result["retained_test_prediction_rows"] == 196608
    assert result["all_backend_preflight_checks"] == 3042
    assert result["regression_all_backend_preflight_checks"] == 3054
    verified = packager.verify_archive(args.output)
    assert sum(name.endswith("/test_logits.f32") for name in verified) == 24
    assert sum(name.startswith("bin/") for name in verified) == 18
    assert "preflights/width-32-neon.json" in verified
    assert "preflights/width-1024-sme.json" in verified


def test_v5_rejects_changed_untimed_regression_binary(tmp_path):
    args = evidence_tree(tmp_path, protocol="v5")
    (args.repo / "native/build/width-32/libtransformer.dylib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash|checksum|Checksum"):
        packager.validate_evidence(
            args.inputs, args.results, args.report, args.preflight_dir, args.repo
        )
