"""Aggregation tests distinguish timing repetitions from independent seeds."""

import math

import pytest

from transformer import spec
from transformer.report import aggregate_runs


def synthetic_runs():
    runs = []
    for index, seed in enumerate(spec.SEEDS):
        for repeat in (1, 2):
            for impl in ("python", "arm64"):
                seconds = (10 + index * 2) * repeat
                if impl == "arm64":
                    seconds /= 2
                runs.append(
                    {
                        "seed": seed,
                        "repeat": repeat,
                        "implementation": impl,
                        "quality": {
                            "nll": 2 + index * 0.1,
                            "bits_per_character": 3,
                            "perplexity": 8,
                            "accuracy": 0.3,
                        },
                        "training_seconds": seconds,
                        "training_cpu_seconds": seconds,
                        "process_wall_seconds": seconds + 1,
                        "peak_rss_after_bytes": 2**20,
                        "peak_rss_final_bytes": 2**20,
                        "forward_latency_seconds": {"p50": 0.001, "p95": 0.002, "p99": 0.003},
                        "selected_checkpoint_validation_nll": 2.2,
                    }
                )
    return runs


def test_timing_repeats_averaged_before_seed_statistics():
    result = aggregate_runs(synthetic_runs())
    stats = result["by_implementation"]["python"]["training_seconds"]
    assert stats["n_seeds"] == 3
    assert stats["seed_values"] == [15, 18, 21]
    assert stats["mean"] == 18
    assert stats["sample_sd"] == 3
    assert result["geometric_mean_training_ratio_python_over_arm64"] == pytest.approx(2)
    assert len(result["paired_training_ratios"]) == 6


def test_geometric_pair_ratio_is_not_ratio_of_overall_means():
    runs = synthetic_runs()
    runs[1]["training_seconds"] *= 4
    result = aggregate_runs(runs)
    expected = math.exp((5 * math.log(2) + math.log(0.5)) / 6)
    assert result["geometric_mean_training_ratio_python_over_arm64"] == pytest.approx(expected)


def test_duplicate_or_missing_identity_is_rejected():
    runs = synthetic_runs()
    runs[-1] = runs[0].copy()
    with pytest.raises(ValueError, match="twelve run identities"):
        aggregate_runs(runs)
