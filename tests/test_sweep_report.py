"""Independent synthetic evidence tests; no training, native calls or timing."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from transformer import spec, sweep_report
from transformer.report import aggregate_runs
from transformer.sweep import study_order
from transformer.validate import Checks


def synthetic_campaign():
    manifest = {
        "status": "completed",
        "protocol_version": "v3",
        "widths": list(spec.WIDTHS),
        "revision": "frozen-revision",
        "source_sha256": {"source": "frozen"},
        "models": {},
        "preflights": {},
        "fixtures": {},
        "order": study_order(),
        "runs": [],
    }
    for width in spec.WIDTHS:
        parameters = 8 * width**2 + 301 * width + 128
        manifest["models"][str(width)] = {
            "width": width,
            "parameters": parameters,
            "heads": 4,
            "head_width": width // 4,
            "hidden": 2 * width,
            "context": 32,
            "vocab": 128,
        }
        manifest["preflights"][str(width)] = {
            "passed": True,
            "runtime_included": True,
            "model_width": width,
            "parameter_count": parameters,
            "model_identity": {"width": width, "parameter_count": parameters, "verified": True},
            "checks": [{"name": "synthetic", "passed": True}],
            "check_count": 1,
            "failed_checks": [],
            "error": None,
        }
        manifest["fixtures"][str(width)] = {}
        for seed in spec.SEEDS:
            manifest["fixtures"][str(width)][str(seed)] = {
                "config": {
                    "version": 2,
                    "width": width,
                    "seed": seed,
                    "epochs": spec.EPOCHS,
                    "batch": spec.BATCH,
                    "train_sequences": spec.TRAIN_SEQUENCES,
                    "validation_sequences": spec.VALID_SEQUENCES,
                    "test_sequences": spec.TEST_SEQUENCES,
                },
                "source": {"sha256": sweep_report.SOURCE_SHA256},
                "splits": {"train": [0, 10]},
                "vocabulary": "fixed",
                "sequence_rule": "shared",
                "sha256": {
                    "train.i32": "train",
                    "validation.i32": "validation",
                    "test.i32": "test",
                    "schedule.i32": f"schedule-{seed}",
                    "initial.f32": f"init-{width}-{seed}",
                },
            }
    for width, seed, repeat, impl in study_order():
        name = f"seed-{seed}-repeat-{repeat}-{impl}"
        power = {"power": {"returncode": 0, "stdout": "AC Power"}}
        manifest["runs"].append(
            {
                "width": width,
                "seed": seed,
                "repeat": repeat,
                "implementation": impl,
                "name": name,
                "relative_run": f"width-{width}/{name}",
                "width_environment": str(width),
                "status": "completed",
                "returncode": 0,
                "conditions_valid": True,
                "source_unchanged": True,
                "binary_unchanged": True,
                "fixtures_unchanged": True,
                "before": power,
                "after": copy.deepcopy(power),
                "source_before": {"unchanged": True},
                "source_after": {"unchanged": True},
                "changed_files_before": [],
                "changed_authenticated_files": [],
                "changed_fixture_files": [],
                "process_wall_seconds": 1000.0,
            }
        )
    children = {}
    for width in spec.WIDTHS:
        children[width] = {
            "status": "completed",
            "protocol_version": "v3",
            "width": width,
            "model_width": width,
            "parameter_count": manifest["models"][str(width)]["parameters"],
            "model": copy.deepcopy(manifest["models"][str(width)]),
            "revision": manifest["revision"],
            "source_sha256": copy.deepcopy(manifest["source_sha256"]),
            "preflight": copy.deepcopy(manifest["preflights"][str(width)]),
            "fixtures": copy.deepcopy(manifest["fixtures"][str(width)]),
            "runs": copy.deepcopy([entry for entry in manifest["runs"] if entry["width"] == width]),
            "order": [list(row[1:]) for row in study_order() if row[0] == width],
        }
    return manifest, children


def synthetic_summaries(manifest):
    summaries = {}
    for width in spec.WIDTHS:
        runs = []
        for entry in manifest["runs"]:
            if entry["width"] != width:
                continue
            seed_index = spec.SEEDS.index(entry["seed"])
            seconds = (10 + seed_index) * entry["repeat"] * width / 32
            if entry["implementation"] == "arm64":
                seconds *= 0.5 if width < 128 else 2
            runs.append(
                {
                    "seed": entry["seed"],
                    "repeat": entry["repeat"],
                    "implementation": entry["implementation"],
                    "name": entry["name"],
                    "fixture_sha256": manifest["fixtures"][str(width)][str(entry["seed"])][
                        "sha256"
                    ],
                    "quality": {
                        "nll": 2 + seed_index * 0.01,
                        "bits_per_character": 3,
                        "perplexity": 8,
                        "accuracy": 0.3,
                        "tokens": sweep_report.TEST_ROWS_PER_FIT,
                    },
                    "training_seconds": seconds,
                    "training_cpu_seconds": seconds,
                    "process_wall_seconds": entry["process_wall_seconds"],
                    "peak_rss_after_bytes": 2**20,
                    "peak_rss_final_bytes": 2**20,
                    "forward_latency_seconds": {
                        "p50": seconds * 1e-4,
                        "p95": seconds * 2e-4,
                        "p99": seconds * 3e-4,
                    },
                    "selected_checkpoint_validation_nll": 2.2,
                    "optimizer_steps": sweep_report.OPTIMIZER_STEPS,
                    "checkpoint_predictions_recomputed": True,
                    "learning_history": [[2, 2]] * spec.EPOCHS,
                }
            )
        summaries[width] = {
            "status": "completed",
            "protocol_version": "v3",
            "revision": manifest["revision"],
            "source_sha256": manifest["source_sha256"],
            "model": manifest["models"][str(width)],
            "runs": runs,
            "aggregate": aggregate_runs(runs),
            "baselines": {"uniform": {"nll": 4.85}},
            "paired_predictions": [],
            "audit_check_count": width,
            "preflight_check_count": 1,
            "raw_manifest_sha256": f"manifest-{width}",
            "retained_test_prediction_rows": len(runs) * sweep_report.TEST_ROWS_PER_FIT,
        }
    return summaries


def test_global_and_per_width_manifests_must_agree_exactly():
    manifest, children = synthetic_campaign()
    checks = Checks()
    sweep_report.validate_manifest(manifest, children, checks)
    assert checks.passed
    children[128]["runs"][0]["process_wall_seconds"] += 1
    with pytest.raises(ValueError, match="manifest_matches_global/runs"):
        sweep_report.validate_manifest(manifest, children, Checks())


@pytest.mark.parametrize("change", ["duplicate", "width", "order", "failed", "fixtures"])
def test_incomplete_wrong_width_duplicate_and_failed_runs_are_rejected(change):
    manifest, children = synthetic_campaign()
    if change == "duplicate":
        manifest["runs"][-1] = copy.deepcopy(manifest["runs"][0])
    elif change == "width":
        manifest["runs"][0]["width"] = 512
    elif change == "order":
        manifest["runs"][:2] = manifest["runs"][1::-1]
    elif change == "failed":
        manifest["runs"][0]["returncode"] = 1
    else:
        manifest["runs"][0]["fixtures_unchanged"] = False
    with pytest.raises(ValueError):
        sweep_report.validate_manifest(manifest, children, Checks())


@pytest.mark.parametrize("kind", ["tokens", "schedule", "source", "initial"])
def test_shared_fixture_identity_is_required_but_initializations_differ(kind):
    manifest, _ = synthetic_campaign()
    fixture = manifest["fixtures"]["256"]["43"]
    if kind == "tokens":
        fixture["sha256"]["test.i32"] = "different"
    elif kind == "schedule":
        fixture["sha256"]["schedule.i32"] = "different"
    elif kind == "source":
        fixture["source"]["sha256"] = "different"
    else:
        fixture["sha256"]["initial.f32"] = manifest["fixtures"]["32"]["43"]["sha256"]["initial.f32"]
    with pytest.raises(ValueError):
        sweep_report.validate_shared_fixtures(manifest["fixtures"], Checks())


def test_aggregation_retains_slow_widths_and_averages_throughput_before_seeds():
    manifest, _ = synthetic_campaign()
    summaries = synthetic_summaries(manifest)
    checks = Checks()
    result = sweep_report.aggregate_summaries(manifest, summaries, checks)
    assert checks.passed
    assert len(result["runs"]) == 48
    assert result["retained_test_prediction_rows"] == 393216
    assert set(result["by_width"]) == {"32", "64", "128", "256"}
    for width, expected in ((32, 2), (64, 2), (128, 0.5), (256, 0.5)):
        aggregate = result["by_width"][str(width)]["aggregate"]
        assert aggregate["geometric_mean_training_ratio_python_over_arm64"] == pytest.approx(
            expected
        )
    throughput = result["by_width"]["32"]["training_target_tokens_per_second"]["python"]
    expected = [sweep_report.TRAINING_TOKENS_PER_FIT * 0.75 / (10 + index) for index in range(3)]
    assert throughput["seed_values"] == pytest.approx(expected)
    assert throughput["mean"] == pytest.approx(np.mean(expected))
    assert throughput["sample_sd"] == pytest.approx(np.std(expected, ddof=1))


@pytest.mark.parametrize("kind", ["aggregate", "model", "timer", "rows", "budget", "recompute"])
def test_summary_misalignment_is_rejected(kind):
    manifest, _ = synthetic_campaign()
    summaries = synthetic_summaries(manifest)
    summary = summaries[64]
    if kind == "aggregate":
        summary["aggregate"]["by_implementation"]["python"]["training_seconds"]["mean"] += 1
    elif kind == "model":
        summary["model"] = copy.deepcopy(summary["model"])
        summary["model"]["width"] = 32
    elif kind == "timer":
        summary["runs"][0]["process_wall_seconds"] += 1
    elif kind == "rows":
        summary["retained_test_prediction_rows"] -= 1
    elif kind == "budget":
        summary["runs"][0]["optimizer_steps"] -= 1
    else:
        summary["runs"][0]["checkpoint_predictions_recomputed"] = False
    with pytest.raises(ValueError):
        sweep_report.aggregate_summaries(manifest, summaries, Checks())


def test_failed_report_preserves_error_evidence_and_refuses_overwrite(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    manifest, children = synthetic_campaign()
    manifest["source_sha256"] = {
        name: sweep_report.sha256(sweep_report.ROOT / name) for name in sweep_report.REPORT_SOURCES
    }
    for width in spec.WIDTHS:
        children[width]["source_sha256"] = manifest["source_sha256"]
        path = results / f"width-{width}"
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps(children[width]))
    (results / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        sweep_report.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="partial child output", stderr="failure"
        ),
    )
    output = tmp_path / "report"
    audit = sweep_report.create_report(results, tmp_path / "inputs", output)
    assert not audit["passed"]
    assert "report_process" in audit["error"]
    assert (
        json.loads((output / "report-processes.json").read_text())[0]["stdout"]
        == "partial child output"
    )
    assert json.loads((output / "summary.json").read_text())["status"] == "audit_failed"
    assert not (output / "scaling.png").exists()
    before = (output / "audit.json").read_bytes()
    with pytest.raises(FileExistsError):
        sweep_report.create_report(results, tmp_path / "inputs", output)
    assert (output / "audit.json").read_bytes() == before


def v4_campaign():
    from transformer.backend import FIELDS

    manifest, children = synthetic_campaign()
    manifest["protocol_version"] = "v4"
    manifest["backend_preflights"] = {}
    for width in spec.WIDTHS:
        key = str(width)
        values = [1, 0, 0, 1, 1, 64, 0, 0, 0, 0]
        values[6 if width >= 64 else 8] = 94040
        values[7 if width >= 64 else 9] = 81920
        backend = dict(zip(FIELDS, values, strict=True))
        main = manifest["preflights"][key]
        main["backend"] = backend
        modes = {}
        for policy, number in (("neon", 1), ("sme", 2)):
            selected = copy.deepcopy(main)
            selected["backend"]["requested_policy"] = number
            if policy == "neon":
                selected["backend"].update(
                    selected_kind=0,
                    sme_forward_calls=0,
                    sme_backward_calls=0,
                    neon_forward_calls=94040,
                    neon_backward_calls=81920,
                )
            modes[policy] = selected
        manifest["backend_preflights"][key] = modes
        for entry in manifest["runs"]:
            if entry["width"] == width and entry["implementation"] == "arm64":
                entry.update(
                    backend=copy.deepcopy(backend),
                    backend_environment="auto",
                    backend_valid=True,
                    backend_sha256="authenticated",
                )
        children[width].update(
            protocol_version="v4",
            preflight=copy.deepcopy(main),
            backend_preflights=copy.deepcopy(modes),
            runs=copy.deepcopy([row for row in manifest["runs"] if row["width"] == width]),
        )
    return manifest, children


def test_v4_requires_complete_forced_mode_gates_and_actual_dispatch():
    manifest, children = v4_campaign()
    checks = Checks()
    sweep_report.validate_manifest(manifest, children, checks)
    assert checks.passed
    manifest["backend_preflights"]["64"].pop("sme")
    with pytest.raises(ValueError, match="forced-backend"):
        sweep_report.validate_manifest(manifest, children, Checks())


def test_v4_rejects_backend_counter_and_nested_protocol_mismatches():
    manifest, children = v4_campaign()
    children[128]["protocol_version"] = "v3"
    with pytest.raises(ValueError, match="manifest_status"):
        sweep_report.validate_manifest(manifest, children, Checks())
    manifest, children = v4_campaign()
    entry = next(row for row in manifest["runs"] if row["implementation"] == "arm64")
    entry["backend"]["neon_forward_calls"] -= 1
    children[entry["width"]]["runs"] = copy.deepcopy(
        [row for row in manifest["runs"] if row["width"] == entry["width"]]
    )
    with pytest.raises(ValueError, match="counters"):
        sweep_report.validate_manifest(manifest, children, Checks())


def test_v4_aggregation_preserves_separate_auto_forced_and_combined_counts():
    manifest, _ = v4_campaign()
    summaries = synthetic_summaries(manifest)
    for width, summary in summaries.items():
        summary.update(
            protocol_version="v4",
            forced_backend_preflight_check_count=2,
            all_backend_preflight_check_count=3,
            backend=manifest["preflights"][str(width)]["backend"],
        )
    aggregate = sweep_report.aggregate_summaries(manifest, summaries, Checks())
    assert all(
        item["forced_backend_preflight_check_count"] == 2 for item in aggregate["by_width"].values()
    )
    summaries[64]["forced_backend_preflight_check_count"] = 99
    with pytest.raises(ValueError, match="forced_backend_counts"):
        sweep_report.aggregate_summaries(manifest, summaries, Checks())
