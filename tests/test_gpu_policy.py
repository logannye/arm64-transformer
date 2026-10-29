"""CPU-only tests of policy identity, selection rules, and timing boundaries."""

import copy
import json

import numpy as np
import pytest

from transformer.gpu_optimize import (
    family_costs,
    measure,
    measure_paired,
    select_winner,
    state_equal,
    structural_exclusion,
    write_fresh,
)
from transformer.gpu_policy import (
    apply_candidate,
    bundle,
    candidate_inventory,
    controls,
    defines,
    identity,
    reference,
    validate,
)


def test_inventory_complete_unique_and_valid():
    inventory = candidate_inventory()
    assert len(inventory) == 75
    assert len({row["id"] for row in inventory}) == 75
    expected = {
        "dense_forward": 16,
        "dense_dx": 16,
        "dense_dw": 16,
        "norm": 6,
        "attention": 16,
        "embedding": 2,
        "fusion": 3,
    }
    for family, count in expected.items():
        assert sum(row["family"] == family for row in inventory) == count
    original = reference()
    for candidate in inventory:
        policy = apply_candidate(original, candidate)
        assert validate(policy) == policy
        assert len(controls(policy)) == 21
        assert identity(policy) == identity(json.loads(json.dumps(policy)))
        assert defines(policy).startswith(b"#define")
    assert original == reference()


def test_structural_filters_are_declared_without_timing():
    inventory = candidate_inventory()
    assert all(structural_exclusion(row, "training") is None for row in inventory)
    assert sum(structural_exclusion(row, "inference") is None for row in inventory) == 31
    assert all(
        structural_exclusion(row, "inference")
        for row in inventory
        if row["family"] in ("dense_dx", "dense_dw", "embedding")
    )


@pytest.mark.parametrize("field,value", [("sg", True), ("m", 8), ("n", 16), ("k", 64)])
def test_invalid_dense_candidate_rejected(field, value):
    policy = reference()
    policy["dense"]["forward"] = {"m": 16, "n": 32, "k": 16, "sg": 2, field: value}
    with pytest.raises(ValueError):
        validate(policy)


def test_policy_copy_and_bundle_preserve_independent_modes():
    policy = reference()
    result = bundle(policy)
    result["training"]["fusion"]["epilogue"] = True
    assert not result["inference"]["fusion"]["epilogue"]
    assert not policy["fusion"]["epilogue"]


def test_family_costs_exclude_compensated_qkv_and_include_qkv_dw():
    values = {
        "dense_forward/ff1": 3,
        "qkv_forward": 100,
        "dense_dx/ff1": 4,
        "qkv_dx/qkv": 200,
        "dense_dw/qkv": 5,
        "norm_forward/1": 2,
        "norm_backward/1": 1,
        "attention_forward": 7,
        "attention_backward": 8,
        "embedding": 99,
        "embedding_backward": 6,
        "elementwise/relu": 1,
        "adam": 2,
    }
    costs = family_costs({"profiles": {"training": {"median_gpu_seconds": values}}}, "training")
    assert costs == {
        "dense_forward": 3,
        "dense_dx": 4,
        "dense_dw": 5,
        "norm": 3,
        "attention": 15,
        "embedding": 6,
        "fusion": 3,
    }


def test_evidence_output_is_never_overwritten(tmp_path):
    path = tmp_path / "record.json"
    write_fresh(path, {"first": True})
    with pytest.raises(FileExistsError):
        write_fresh(path, {"first": False})
    assert json.loads(path.read_text()) == {"first": True}


class FakeBackend:
    def __init__(self):
        self.state = {
            "parameters": np.zeros(3, "f4"),
            "m": np.zeros(3, "f4"),
            "v": np.zeros(3, "f4"),
            "step": 0,
        }
        self.profile = True
        self.training_calls = 0
        self.forward_calls = 0
        self.restore_steps = []

    def enable_profiling(self, enabled):
        self.profile = enabled

    def memory_stats(self):
        return {"profiling_enabled": self.profile}

    def export_state(self):
        return copy.deepcopy(self.state)

    def load_state(self, state):
        self.restore_steps.append(state["step"])
        self.state = copy.deepcopy(state)

    def train_step(self, batch):
        assert not self.profile
        self.training_calls += 1
        self.state["parameters"] += np.float32(1)
        self.state["step"] += 1
        return np.array(1, "f4")

    def forward(self, batch):
        assert not self.profile
        self.forward_calls += 1
        return self.state["parameters"].copy()

    def synchronize(self):
        pass

    def check_finite(self):
        return True

    def check_guards(self):
        return True

    def to_numpy(self, value):
        return value


def test_training_timing_restores_warmup_and_each_block():
    backend = FakeBackend()
    original = backend.export_state()
    result = measure(backend, object(), "training", warmups=2, observations=6, blocks=3)
    assert backend.training_calls == 8
    assert len(result["seconds"]) == 6
    assert len(result["block_median_seconds"]) == 3
    assert state_equal(backend.export_state(), original)
    assert backend.restore_steps == [0, 0, 0, 0, 0]
    assert result["exact_block_replay"]


def test_inference_timing_excludes_validation_readbacks_from_samples():
    backend = FakeBackend()
    result = measure(backend, object(), "inference", warmups=2, observations=6, blocks=3)
    assert backend.forward_calls == 11  # 2 warmup, 6 timed, 3 block replay captures.
    assert len(result["seconds"]) == 6
    assert backend.state["step"] == 0


def test_timing_refuses_enabled_intrusive_profiling():
    backend = FakeBackend()
    backend.enable_profiling = lambda enabled: None
    with pytest.raises(RuntimeError, match="profiling must be disabled"):
        measure(backend, object(), "training", warmups=1, observations=1, blocks=1)
    assert backend.training_calls == 0


def test_paired_timing_alternates_order_and_restores_each_backend():
    baseline, candidate = FakeBackend(), FakeBackend()
    result = measure_paired(
        baseline,
        candidate,
        object(),
        object(),
        "training",
        order_offset=1,
        warmups=2,
        observations=6,
        blocks=3,
    )
    assert baseline.training_calls == candidate.training_calls == 8
    assert result["block_order"] == [
        ["candidate", "reference"],
        ["reference", "candidate"],
        ["candidate", "reference"],
    ]
    assert len(result["paired_improvement_fractions"]) == 3
    assert baseline.state["step"] == candidate.state["step"] == 0
    assert baseline.restore_steps == candidate.restore_steps == [0, 0, 0, 0, 0]


def test_selection_requires_qualification_three_percent_gain_and_lexical_tie():
    def row(name, ratio, passed=True):
        return {
            "id": name,
            "status": "qualified_measured" if passed else "numerically_rejected_unranked",
            "qualification": {"passed": passed},
            "paired_timing": {"median_paired_latency_ratio": ratio},
        }

    rejected, slow, b, a = (
        row("rejected", 0.1, False),
        row("too_small_gain", 0.97001),
        row("b", 0.96),
        row("a", 0.96),
    )
    assert select_winner([rejected, slow]) is None
    assert select_winner([rejected, slow, b, a])["id"] == "a"
    assert select_winner([row("boundary", 0.97)])["id"] == "boundary"
