"""Audit all twelve matched runs and write a compact, reproducible result report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from transformer import spec
from transformer.audit import audit_run, compare_repeats, paired_predictions, read_exact
from transformer.backend import (
    WIDE_VALIDATION_CONTRACT,
    read_backend,
    validate_backend_preflights,
    validate_runtime_backend,
    validate_validation_contract,
)
from transformer.validate import ROOT, SOURCE_FILES, Checks, sha256

IMPLEMENTATIONS = ("python", "arm64")
LABELS = {"python": "PyTorch", "arm64": "ARM64 assembly"}
COLORS = {"python": "#3568ad", "arm64": "#d66b34"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def seed_statistics(values):
    """Sample SD is over independent seeds, never over repeated timing runs."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("Statistics require at least two finite seed values")
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "n_seeds": len(array),
        "seed_values": array.tolist(),
    }


def measurements(run):
    quality = run["quality"]
    return {
        "test_nll": quality["nll"],
        "test_bits_per_character": quality["bits_per_character"],
        "test_perplexity": quality["perplexity"],
        "test_accuracy_percent": quality["accuracy"] * 100,
        "training_seconds": run["training_seconds"],
        "training_cpu_seconds": run["training_cpu_seconds"],
        "process_wall_seconds": run["process_wall_seconds"],
        "peak_rss_training_mib": run["peak_rss_after_bytes"] / 2**20,
        "peak_rss_final_mib": run["peak_rss_final_bytes"] / 2**20,
        "forward_p50_ms": run["forward_latency_seconds"]["p50"] * 1000,
        "forward_p95_ms": run["forward_latency_seconds"]["p95"] * 1000,
        "forward_p99_ms": run["forward_latency_seconds"]["p99"] * 1000,
        "selected_validation_nll": run["selected_checkpoint_validation_nll"],
    }


def aggregate_runs(runs):
    by_key = {(run["seed"], run["repeat"], run["implementation"]): run for run in runs}
    expected = {
        (seed, repeat, impl) for seed in spec.SEEDS for repeat in (1, 2) for impl in IMPLEMENTATIONS
    }
    if len(runs) != 12 or set(by_key) != expected:
        raise ValueError("Aggregation requires exactly the prespecified twelve run identities")
    per_seed, aggregate = {}, {}
    for impl in IMPLEMENTATIONS:
        per_seed[impl] = {}
        for seed in spec.SEEDS:
            repeats = [measurements(by_key[seed, repeat, impl]) for repeat in (1, 2)]
            per_seed[impl][str(seed)] = {
                metric: float(np.mean([repeat[metric] for repeat in repeats]))
                for metric in repeats[0]
            }
        aggregate[impl] = {
            metric: seed_statistics([per_seed[impl][str(seed)][metric] for seed in spec.SEEDS])
            for metric in repeats[0]
        }
    ratios = []
    for seed in spec.SEEDS:
        for repeat in (1, 2):
            ratio = (
                by_key[seed, repeat, "python"]["training_seconds"]
                / by_key[seed, repeat, "arm64"]["training_seconds"]
            )
            ratios.append({"seed": seed, "repeat": repeat, "python_over_arm64": ratio})
    geometric_ratio = math.exp(
        float(np.mean([math.log(row["python_over_arm64"]) for row in ratios]))
    )
    differences = {
        metric: seed_statistics(
            [
                per_seed["arm64"][str(seed)][metric] - per_seed["python"][str(seed)][metric]
                for seed in spec.SEEDS
            ]
        )
        for metric in ("test_nll", "test_bits_per_character", "test_accuracy_percent")
    }
    return {
        "per_seed": per_seed,
        "by_implementation": aggregate,
        "paired_training_ratios": ratios,
        "geometric_mean_training_ratio_python_over_arm64": geometric_ratio,
        "paired_quality_differences_arm64_minus_python": differences,
        "statistics_scope": (
            "Two repeats averaged within each of three seeds; sample SD across seed means. "
            "Exact within-implementation repeat outputs required before quality aggregation. "
            "Primary ratio is geometric mean of six paired training wall-time ratios."
        ),
    }


def count_baselines(fixture, checks):
    fixture = Path(fixture)
    train = read_exact(fixture / "train.i32", "<i4", (spec.TRAIN_SEQUENCES, spec.CONTEXT + 1))[
        :, 1:
    ]
    test = read_exact(fixture / "test.i32", "<i4", (spec.TEST_SEQUENCES, spec.CONTEXT + 1))[:, 1:]
    if np.any((train < 0) | (train >= spec.VOCAB)) or np.any((test < 0) | (test >= spec.VOCAB)):
        raise ValueError("Baseline received invalid token IDs")
    counts = np.bincount(train.reshape(-1), minlength=spec.VOCAB).astype(np.float64)
    checks.require("baselines/training_target_count", counts.sum() == train.size)
    probabilities = (counts + 1) / (counts.sum() + spec.VOCAB)
    labels = test.reshape(-1)
    nll = float(-np.log(probabilities[labels]).mean())
    most_frequent = int(counts.argmax())
    return {
        "uniform": {
            "nll": math.log(spec.VOCAB),
            "bits_per_character": math.log2(spec.VOCAB),
            "perplexity": float(spec.VOCAB),
            "accuracy": float(np.mean(labels == 0)),
            "argmax_tie_rule": "First vocabulary index, ASCII ID 0",
        },
        "train_only_unigram": {
            "nll": nll,
            "bits_per_character": nll / math.log(2),
            "perplexity": math.exp(nll),
            "accuracy": float(np.mean(labels == most_frequent)),
            "training_target_tokens": int(counts.sum()),
            "laplace_pseudocount_per_ascii_id": 1,
            "most_frequent_token_id": most_frequent,
            "training_counts": counts.astype(np.int64).tolist(),
            "test_target_tokens": int(labels.size),
        },
    }


def charts(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    aggregate = summary["aggregate"]["by_implementation"]

    def comparison_figure(filename, fields, titles, unit):
        fig, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 4), squeeze=False)
        for axis, field, title in zip(axes[0], fields, titles, strict=True):
            means = [aggregate[impl][field]["mean"] for impl in IMPLEMENTATIONS]
            sds = [aggregate[impl][field]["sample_sd"] for impl in IMPLEMENTATIONS]
            axis.bar(
                [0, 1],
                means,
                yerr=sds,
                capsize=5,
                color=[COLORS[impl] for impl in IMPLEMENTATIONS],
                alpha=0.9,
            )
            for index, impl in enumerate(IMPLEMENTATIONS):
                values = aggregate[impl][field]["seed_values"]
                axis.scatter(
                    np.full(3, index) + np.linspace(-0.08, 0.08, 3),
                    values,
                    color="#222222",
                    s=20,
                    zorder=3,
                )
            axis.set_xticks([0, 1], [LABELS[impl] for impl in IMPLEMENTATIONS])
            axis.set_ylabel(unit)
            axis.set_title(title)
            axis.set_ylim(bottom=0)
            axis.grid(axis="y", alpha=0.18)
            axis.set_axisbelow(True)
        fig.text(
            0.5,
            0.01,
            "Bars: mean of 3 seed means; error bars: sample SD; dots: seeds",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(output / filename)
        plt.close(fig)

    comparison_figure(
        "training.png",
        ["training_seconds", "process_wall_seconds"],
        ["Training + validation", "Whole child process"],
        "Seconds",
    )
    comparison_figure(
        "latency.png",
        ["forward_p50_ms", "forward_p95_ms", "forward_p99_ms"],
        ["Forward p50", "Forward p95", "Forward p99"],
        "Milliseconds / 32-character context",
    )
    comparison_figure(
        "memory.png",
        ["peak_rss_training_mib", "peak_rss_final_mib"],
        ["Peak RSS through training", "Lifetime peak RSS"],
        "MiB",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    epochs = np.arange(1, spec.EPOCHS + 1)
    for impl in IMPLEMENTATIONS:
        histories = []
        for seed in spec.SEEDS:
            rows = [
                np.asarray(run["learning_history"])
                for run in summary["runs"]
                if run["seed"] == seed and run["implementation"] == impl
            ]
            histories.append(np.mean(rows, axis=0))
        histories = np.stack(histories)
        for index, title in enumerate(("Training NLL", "Validation NLL")):
            means = histories[:, :, index].mean(axis=0)
            sds = histories[:, :, index].std(axis=0, ddof=1)
            axes[index].plot(epochs, means, label=LABELS[impl], color=COLORS[impl], marker="o")
            axes[index].fill_between(
                epochs, means - sds, means + sds, color=COLORS[impl], alpha=0.12
            )
            axes[index].set_title(title)
            axes[index].set_xlabel("Epoch")
            axes[index].set_ylabel("Mean negative log-likelihood / character")
            axes[index].grid(alpha=0.2)
    axes[0].legend()
    fig.text(
        0.5,
        0.01,
        "Lines: mean across 3 seeds; bands: sample SD; repeats averaged within seed",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output / "learning.png")
    plt.close(fig)


def report_markdown(summary):
    version = summary.get("protocol_version", "v1")
    model = summary.get("model", {"width": 32, "parameters": 17952})
    reuse_note = (
        "This optimization study reuses v1's fixed train/validation/test fixtures. "
        "It is an engineering regression benchmark, not a fresh untouched test set. "
        "Kernel development used synthetic correctness and timing inputs; the model, "
        "optimizer, seeds and stopping rule were unchanged.\n\n"
        if version in ("v2", "v3", "v4", "v5")
        else ""
    )
    backend_note = ""
    if version in ("v4", "v5"):
        actual = next(run["backend"] for run in summary["runs"] if run["implementation"] == "arm64")
        backend_note = (
            f"Native processes requested automatic dispatch. Hardware SME capability: "
            f"{bool(actual['sme_supported'])}; streaming vector length: "
            f"{actual['streaming_vector_bytes']} bytes. Actual dense calls per native fit were "
            f"SME {actual['sme_forward_calls']:,} forward / "
            f"{actual['sme_backward_calls']:,} backward, "
            f"and NEON {actual['neon_forward_calls']:,} forward / "
            f"{actual['neon_backward_calls']:,} backward. These authenticated counters cover the "
            "whole process, including evaluation and inference samples.\n\n"
            f"The auto-policy preflight contains {summary['preflight_check_count']:,} checks; "
            f"forced-policy gates add {summary['forced_backend_preflight_check_count']:,}, "
            f"for {summary['all_backend_preflight_check_count']:,} admission checks.\n\n"
        )
    contract_note = ""
    if version == "v5":
        total = sum(value["count"] for value in summary["legacy_diagnostic_counts"].values())
        failed = sum(value["failed"] for value in summary["legacy_diagnostic_counts"].values())
        contract_note = (
            f"Validation contract: `{summary['validation_contract']}`. "
            "The prospective wide-model criterion compares both engines at identical "
            "evolving states "
            "and replays each gradient through both optimizers with unchanged tolerances. "
            f"The original free-running comparisons remain {total:,} explicit diagnostic records, "
            f"including {failed:,} failures, and are excluded from the passing admission count. "
            "This does not claim the original cross-implementation trajectory contract passed. "
            "Independent four-update control replays also require exact own saved states/history "
            "and check actual partial-batch normalization plus identical-input Adam updates. "
            "The failed initial admission and its source/binaries remain preserved.\n\n"
        )
    aggregate = summary["aggregate"]["by_implementation"]
    ratio = summary["aggregate"]["geometric_mean_training_ratio_python_over_arm64"]
    speed = (
        f"ARM64 training was **{ratio:.3f}× faster** than PyTorch"
        if ratio >= 1
        else f"ARM64 training took **{1 / ratio:.3f}× as long** as PyTorch"
    )
    rows = [
        ("Test NLL (lower is better)", "test_nll", 4),
        ("Test bits / character", "test_bits_per_character", 4),
        ("Test perplexity", "test_perplexity", 3),
        ("Next-character accuracy (%)", "test_accuracy_percent", 3),
        ("Training + validation (s)", "training_seconds", 3),
        ("Training CPU time (s)", "training_cpu_seconds", 3),
        ("Whole child process (s)", "process_wall_seconds", 3),
        ("Forward p50, full context (ms)", "forward_p50_ms", 4),
        ("Forward p95, full context (ms)", "forward_p95_ms", 4),
        ("Forward p99, full context (ms)", "forward_p99_ms", 4),
        ("Peak RSS through training (MiB)", "peak_rss_training_mib", 2),
    ]
    table = ["| Measurement | PyTorch | ARM64 assembly |", "|---|---:|---:|"]
    for label, field, digits in rows:
        values = []
        for impl in IMPLEMENTATIONS:
            stats = aggregate[impl][field]
            values.append(f"{stats['mean']:.{digits}f} ± {stats['sample_sd']:.{digits}f}")
        table.append(f"| {label} | {' | '.join(values)} |")
    difference = summary["aggregate"]["paired_quality_differences_arm64_minus_python"]
    unigram = summary["baselines"]["train_only_unigram"]
    uniform = summary["baselines"]["uniform"]
    paired = summary["paired_predictions"]
    disagreement = np.mean([pair["prediction_disagreement"] for pair in paired]) * 100
    per_run = [
        "| Seed | Repeat | Implementation | Train (s) | Test NLL | Accuracy (%) | Best epoch |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        per_run.append(
            f"| {run['seed']} | {run['repeat']} | {LABELS[run['implementation']]} | "
            f"{run['training_seconds']:.3f} | {run['quality']['nll']:.5f} | "
            f"{run['quality']['accuracy'] * 100:.3f} | {run['best_epoch']} |"
        )
    table_text = "\n".join(table)
    per_run_text = "\n".join(per_run)
    return f"""# Matched transformer CPU comparison, {version}, width {model["width"]}

{reuse_note}\
{speed}, using the geometric mean of six paired training-time ratios.
This is the same {model["parameters"]:,}-parameter causal transformer implemented with ordinary
batched PyTorch (built-in scaled-dot-product attention and autograd) and handwritten
Apple ARM64 assembly. The experiment contains **12 complete fits**: three
initialization seeds, two timing repeats, and both implementations.

Measured source revision: `{summary["revision"]}`. Hardware: {summary["hardware"]}.
Platform: `{summary["platform"]}`. PyTorch: `{summary["torch"]}`.

{backend_note}{contract_note}{table_text}

Entries are mean ± sample SD of three seed means. Timing repeats are averaged
within seed; they are not six independent quality seeds. The primary ratio is
Python/ARM64 training time, **{ratio:.6f}**, where values above one favor assembly.
All within-implementation repeat checkpoints, optimizer states, held-out logits,
and non-timing learning histories matched exactly.

## Prediction quality and baselines

The mean assembly-minus-PyTorch test NLL difference was
**{difference["test_nll"]["mean"]:+.6f}**, and the accuracy difference was
**{difference["test_accuracy_percent"]["mean"]:+.6f} percentage points**.
Mean paired next-character prediction disagreement was **{disagreement:.3f}%**.
These are descriptive observations, not a statistical equivalence claim.

A uniform ASCII-128 predictor has NLL **{uniform["nll"]:.4f}** and perplexity
**{uniform["perplexity"]:.3f}**. An add-one-smoothed unigram predictor fitted only
to the 65,536 training target characters has held-out NLL **{unigram["nll"]:.4f}**,
perplexity **{unigram["perplexity"]:.3f}**, and accuracy **{unigram["accuracy"] * 100:.3f}%**.
Its training counts are retained in [summary.json](summary.json).

The source corpus is Tiny Shakespeare. Contiguous 80/10/10 partitions were made
before selecting the first 65,537/8,193/8,193 bytes for this bounded experiment.
There are 2,048 training and 256 validation/test sequences, each predicting 32
characters. Eight epochs and batch 16 give 1,024 optimizer updates per run.
Validation NLL selects the checkpoint; test data are used only after selection.

## Timing and memory scope

![Training and whole-process times](training.png)

Training includes forward/backward, Adam, finite checks, epoch validation,
checkpoint selection/writes and history bookkeeping. Setup, final evaluation,
last-state export and inference timing are outside this timer. Whole-process
time is independently measured by the parent from child launch through exit.

![Full-context forward latency](latency.png)

Latency is a **batch-one forward pass over a complete 32-character context**,
with 20 warmups and 100 measured samples per run. It excludes token loading,
softmax, sampling, loss, and I/O. It is not cached single-token generation or
end-to-end application latency. The p99 values describe these 100-sample
microbenchmarks and do not establish rare-event deployment guarantees.

![Process memory](memory.png)

Memory is process lifetime peak RSS through each boundary, including runtime,
framework and loaded data. It is not model-only memory; subtracting these peaks
does not isolate individual allocations.

![Training and validation learning curves](learning.png)

## Every measured run

{per_run_text}

## Controls and retained evidence

Both implementations used the same float32 initial parameters, token rows,
per-epoch schedules, Adam settings, one-thread CPU budget and stopping rule.
Run order was reversed between repeats, one child ran at a time, and fixed
15-second pauses separated children. Power, thermal and load snapshots are
retained in the raw runner manifest. Operating-system scheduling, temperature
and clock frequencies were not experimentally fixed.

The independent post-run audit passed **{summary["audit_check_count"]} checks**.
All **{summary["retained_test_prediction_rows"]:,} held-out prediction rows**
were recomputed from their selected checkpoints with a separate functional
PyTorch reference. Selected-checkpoint validation loss was independently
recomputed and matched the selection history. NumPy float64 calculations
produced the reported quality metrics. See [audit.json](audit.json) for every
check and [summary.json](summary.json) for per-run metrics and hashes.

The result applies to this small, short-context CPU workload on one machine.
It does not establish a general language-speed advantage, large-language-model
quality, GPU performance, energy use, cached decoding speed, or production
serving behavior. Eager PyTorch is the baseline; graph-compiled PyTorch is not
evaluated. Float32 operation grouping differs, so long training trajectories
need not be bitwise identical between implementations.

The compact report contains metrics, checks, hashes, and figures. Full inputs,
schedules, raw logits, checkpoints, optimizer states, latency samples, the source
snapshot and executable belong to the accompanying raw evidence bundle.
"""


def create_report(results, inputs, output):
    results, inputs, output = Path(results), Path(inputs), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    checks = Checks()
    summary = {"schema_version": 1, "status": "audit_failed", "runs": []}
    error = None
    try:
        manifest = json.loads((results / "manifest.json").read_text())
        if manifest.get("model", {"width": 32})["width"] != spec.WIDTH:
            raise ValueError("Report process width does not match the measured model")
        hardware = manifest["hardware"]["stdout"].strip()
        hardware_supplement = results / "hardware-verification.json"
        hardware_evidence = None
        if hardware_supplement.exists():
            hardware_evidence = json.loads(hardware_supplement.read_text())
            if hardware_evidence.get("phase") != "post-measurement":
                raise ValueError("Supplementary hardware verification must identify its timing")
            hardware = (
                hardware_evidence["hardware"][0]["chip_type"]
                + " (verified after measurement; the original sandboxed sysctl query failed)"
            )
            write_json(output / "hardware-verification.json", hardware_evidence)
        if not hardware:
            hardware = "unavailable from the measurement subprocess"
        if manifest.get("status") != "completed":
            raise ValueError(
                "Runner manifest is not completed; incomplete studies cannot be reported"
            )
        expected = {
            (seed, repeat, impl)
            for seed in spec.SEEDS
            for repeat in (1, 2)
            for impl in IMPLEMENTATIONS
        }
        identities = [
            (entry["seed"], entry["repeat"], entry["implementation"]) for entry in manifest["runs"]
        ]
        if len(identities) != 12 or set(identities) != expected:
            raise ValueError("Manifest must contain exactly the twelve prespecified run identities")
        checks.require(
            "manifest/preflight_passed",
            manifest["preflight"].get("passed") is True
            and manifest["preflight"].get("runtime_included") is True,
        )
        diagnostic_counts = {}
        if manifest.get("protocol_version") == "v5":
            if manifest.get("validation_contract") != WIDE_VALIDATION_CONTRACT:
                raise ValueError("Missing prospective wide-model validation contract")
            for mode, gate in {
                "auto": manifest["preflight"],
                **manifest["backend_preflights"],
            }.items():
                total, failed = validate_validation_contract(gate, spec.WIDTH)
                diagnostic_counts[mode] = {
                    "count": total,
                    "failed": failed,
                    "failed_checks": gate["legacy_failed_checks"],
                }
        forced_count = 0
        if manifest.get("protocol_version") in ("v4", "v5"):
            forced_count = validate_backend_preflights(
                manifest["preflight"], manifest["backend_preflights"]
            )
            checks.require("manifest/forced_backend_gates", forced_count > 0)
        for name in SOURCE_FILES:
            checks.require(
                "source/" + name, sha256(ROOT / name) == manifest["source_sha256"].get(name)
            )
        for entry in manifest["runs"]:
            expected_name = (
                f"seed-{entry['seed']}-repeat-{entry['repeat']}-{entry['implementation']}"
            )
            if entry["name"] != expected_name or entry.get("returncode") != 0:
                raise ValueError("Invalid child name or failed child in completed manifest")
            for condition in ("conditions_valid", "source_unchanged", "binary_unchanged"):
                checks.require(entry["name"] + "/" + condition, entry.get(condition) is True)
            fixture = inputs / f"seed-{entry['seed']}"
            print(json.dumps({"auditing": entry["name"]}), flush=True)
            run = audit_run(fixture, results / entry["name"], checks, recompute=True)
            checks.require(
                entry["name"] + "/runner_fixture_identity",
                run["fixture_sha256"] == manifest["fixtures"][str(entry["seed"])]["sha256"],
            )
            if (
                manifest.get("protocol_version") in ("v4", "v5")
                and entry["implementation"] == "arm64"
            ):
                backend_path = results / entry["name"] / "backend.u64"
                backend = read_backend(backend_path, policy="auto")
                validate_runtime_backend(
                    backend,
                    manifest["fixtures"][str(entry["seed"])]["config"],
                    admitted=manifest["preflight"]["backend"],
                )
                checks.require(
                    entry["name"] + "/backend_authenticated",
                    backend == entry.get("backend")
                    and entry.get("backend_valid") is True
                    and sha256(backend_path) == entry.get("backend_sha256")
                    and entry.get("backend_environment") == "auto",
                )
                run["backend"] = backend
            run.update(
                name=entry["name"],
                implementation=entry["implementation"],
                repeat=entry["repeat"],
                process_wall_seconds=entry["process_wall_seconds"],
                fixture=f"seed-{entry['seed']}",
                run=entry["name"],
            )
            checks.require(
                entry["name"] + "/parent_wall_includes_training",
                run["process_wall_seconds"] >= run["training_seconds"],
            )
            summary["runs"].append(run)
        for seed in spec.SEEDS:
            for impl in IMPLEMENTATIONS:
                compare_repeats(
                    results / f"seed-{seed}-repeat-1-{impl}",
                    results / f"seed-{seed}-repeat-2-{impl}",
                    checks,
                )
        base_hashes = manifest["fixtures"][str(spec.SEEDS[0])]["sha256"]
        for seed in spec.SEEDS[1:]:
            hashes = manifest["fixtures"][str(seed)]["sha256"]
            for split in ("train", "validation", "test"):
                checks.require(
                    f"fixtures/seed-{seed}/same_{split}",
                    base_hashes[f"{split}.i32"] == hashes[f"{split}.i32"],
                )
        baselines = count_baselines(inputs / f"seed-{spec.SEEDS[0]}", checks)
        pairs = []
        for seed in spec.SEEDS:
            for repeat in (1, 2):
                pair = paired_predictions(
                    results / f"seed-{seed}-repeat-{repeat}-python",
                    results / f"seed-{seed}-repeat-{repeat}-arm64",
                )
                pair.update(seed=seed, repeat=repeat)
                pairs.append(pair)
        if checks.passed:
            summary.update(
                status="completed",
                revision=manifest["revision"],
                protocol_version=manifest.get("protocol_version", "v1"),
                model=spec.model_metadata(),
                source_sha256=manifest["source_sha256"],
                preflight_check_count=manifest["preflight"]["check_count"],
                forced_backend_preflight_check_count=forced_count,
                all_backend_preflight_check_count=manifest["preflight"]["check_count"]
                + forced_count,
                backend=manifest["preflight"].get("backend"),
                validation_contract=manifest.get("validation_contract"),
                legacy_diagnostic_counts=diagnostic_counts,
                platform=manifest["platform"],
                hardware=hardware,
                supplementary_hardware_evidence=hardware_evidence,
                torch=manifest["torch"],
                numpy=manifest["numpy"],
                aggregate=aggregate_runs(summary["runs"]),
                paired_predictions=pairs,
                baselines=baselines,
                audit_check_count=len(checks.records),
                retained_test_prediction_rows=sum(
                    run["quality"]["tokens"] for run in summary["runs"]
                ),
                raw_manifest_sha256=sha256(results / "manifest.json"),
            )
            charts(summary, output)
            (output / "README.md").write_text(report_markdown(summary))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        checks.require("report/exception", False, error)
    audit = {
        "schema_version": 1,
        "passed": checks.passed,
        "check_count": len(checks.records),
        "checks": checks.records,
        "error": error,
        "failed_checks": [item["name"] for item in checks.records if not item["passed"]],
    }
    if not checks.passed:
        summary["status"] = "audit_failed"
        (output / "README.md").write_text(
            "# Matched transformer report: audit failed\n\n"
            "The evidence did not pass the frozen audit. "
            "No validated aggregate result is claimed.\n\n"
            "See [audit.json](audit.json) for failures and "
            "[summary.json](summary.json) for retained "
            "diagnostic measurements. All raw run artifacts remain untouched.\n"
        )
    write_json(output / "audit.json", audit)
    write_json(output / "summary.json", summary)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = create_report(args.results, args.inputs, args.output)
    print(
        json.dumps(
            {key: result[key] for key in ("passed", "check_count", "failed_checks", "error")}
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
