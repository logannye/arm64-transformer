"""CPU-only tests of policy identity, selection rules, and timing boundaries."""

import copy
import json

import numpy as np
import pytest

from transformer.gpu_optimize import (
    family_costs,
    measure,
    measure_paired,
    pipeline_resource_checks,
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
        "bias_gradient/head": 9,
    }
    costs = family_costs({"profiles": {"training": {"median_gpu_seconds": values}}}, "training")
    assert costs == {
        "dense_forward": 3,
        "dense_dx": 4,
        "dense_dw": 5,
        "norm": 12,
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


def fake_pipeline_capabilities():
    names = (
        "qkv_forward",
        "qkv_dx",
        "attention_dqkv",
        "linear_forward",
        "linear_dx",
        "linear_dw",
        "norm_forward",
        "norm_dx",
        "norm_dparam",
        "linear_db",
        "attention_scores",
        "attention_softmax",
        "attention_values",
        "attention_da",
        "attention_ds",
        "embedding_grad",
        "adam",
        "v2_linear_forward",
        "v2_linear_dx",
        "v2_linear_dw",
        "v2_linear_forward_epilogue",
        "v2_norm_forward",
        "v2_norm_dx",
        "v2_norm_dparam",
        "v2_linear_db",
        "v2_attention_forward",
        "v2_attention_dads",
        "v2_embedding_index",
        "v2_embedding_grad",
        "v2_reference_forward_epilogue",
        "v2_adam_finite",
    )
    return {
        "families": ["Apple9", "Metal4"],
        "max_threadgroup_memory_bytes": 32768,
        "pipelines": {
            prefix + name: {
                "thread_execution_width": 32,
                "max_threads_per_group": 256,
                "static_threadgroup_memory_bytes": 4096,
            }
            for prefix in ("", "inference:")
            for name in names
        },
    }


def test_pipeline_preflight_covers_all_75_candidates_without_dispatch():
    capabilities = fake_pipeline_capabilities()
    for candidate in candidate_inventory():
        policy = apply_candidate(reference(), candidate)
        assert pipeline_resource_checks(policy, capabilities)["passed"]


@pytest.mark.parametrize(
    "failure", ("threads", "memory", "simd_width", "matrix_feature", "missing_inference")
)
def test_pipeline_preflight_rejects_unsupported_resources(failure):
    capabilities = fake_pipeline_capabilities()
    policy = reference()
    policy["dense"]["forward"] = {"m": 32, "n": 64, "k": 32, "sg": 4}
    if failure == "threads":
        capabilities["pipelines"]["v2_linear_forward"]["max_threads_per_group"] = 64
    elif failure == "memory":
        capabilities["pipelines"]["v2_linear_forward"]["static_threadgroup_memory_bytes"] = 32769
    elif failure == "simd_width":
        capabilities["pipelines"]["v2_linear_forward"]["thread_execution_width"] = 16
    elif failure == "matrix_feature":
        capabilities["families"] = ["Apple6"]
    else:
        del capabilities["pipelines"]["inference:v2_linear_forward"]
    assert not pipeline_resource_checks(policy, capabilities)["passed"]


def test_pipeline_preflight_checks_future_fusion_and_bias_reduction():
    capabilities = fake_pipeline_capabilities()
    policy = reference()
    policy["dense"]["forward"] = {"m": 32, "n": 64, "k": 32, "sg": 4}
    capabilities["pipelines"]["v2_linear_forward_epilogue"]["max_threads_per_group"] = 64
    assert pipeline_resource_checks(policy, capabilities)["passed"]
    policy["fusion"]["epilogue"] = True
    assert not pipeline_resource_checks(policy, capabilities)["passed"]
    capabilities = fake_pipeline_capabilities()
    policy = reference()
    policy["norm"] = {"sg": 4, "vec": 4}
    capabilities["pipelines"]["v2_linear_db"]["max_threads_per_group"] = 64
    assert not pipeline_resource_checks(policy, capabilities)["passed"]


def test_profile_identity_covers_all_sources_runtime_environment_and_capabilities():
    from transformer import gpu_optimize as opt

    sources = {
        "native/metal/build/libtransformer_metal.dylib": "binary",
        "native/metal/backend.mm": "host",
        "transformer/gpu_precision.py": "oracle",
    }
    runtime = {"inventory_sha256": "runtime", "interpreter_sha256": "python"}
    fixture = {
        "input_sha256": {"train.i32": "train"},
        "input_directory": "inputs",
        "rows_used": list(range(16)),
    }
    capability = {
        "binary_sha256": "binary",
        "host_source_sha256": "host",
        "device": "M4 Max",
        "host_compiler": "compiler",
        "pipelines": {"one": {"max_threads_per_group": 256}},
        "initialization_seconds_including_shader_compilation": 1.0,
    }
    profile = {
        "source_sha256": sources,
        "runtime_identity": runtime,
        "environment": opt.environment_identity(),
        "python": opt.platform.python_version(),
        "platform": opt.platform.platform(),
        **fixture,
        "warmups": opt.WARMUPS,
        "observations": opt.OBSERVATIONS,
        "blocks": opt.BLOCKS,
        "profile_repeats": opt.PROFILE_REPEATS,
        "min_improvement": opt.MIN_IMPROVEMENT,
        "capabilities": capability,
    }
    changed_duration = copy.deepcopy(capability)
    changed_duration["initialization_seconds_including_shader_compilation"] = 9.0
    opt.validate_profile_identity(profile, sources, runtime, fixture, changed_duration)
    for field in ("source_sha256", "runtime_identity", "environment"):
        bad = copy.deepcopy(profile)
        bad[field]["changed"] = "different"
        with pytest.raises(ValueError, match="identity mismatch"):
            opt.validate_profile_identity(bad, sources, runtime, fixture, capability)
    changed_source = dict(sources, **{"transformer/gpu_precision.py": "different_oracle"})
    with pytest.raises(ValueError, match="source_sha256"):
        opt.validate_profile_identity(profile, changed_source, runtime, fixture, capability)
    changed_pipeline = copy.deepcopy(capability)
    changed_pipeline["pipelines"]["one"]["max_threads_per_group"] = 128
    with pytest.raises(ValueError, match="full device/compiler/pipeline"):
        opt.validate_profile_identity(profile, sources, runtime, fixture, changed_pipeline)


def normal_conditions():
    return {
        "power": {"returncode": 0, "stdout": "Now drawing from 'AC Power'"},
        "thermal": {
            "returncode": 0,
            "stdout": "No thermal warning level has been recorded",
            "stderr": "",
        },
        "load_average": [99, 99, 99],
    }


def test_conditions_fail_for_off_ac_or_known_throttle_but_allow_explicit_unavailable():
    from transformer.gpu_optimize import condition_assessment

    normal = normal_conditions()
    assert condition_assessment(normal)["valid"]
    unsupported = copy.deepcopy(normal)
    unsupported["thermal"] = {
        "returncode": 0,
        "stdout": "",
        "stderr": "Error: 0xe00002bc unsupported status",
    }
    assert condition_assessment(unsupported)["valid"]
    assert condition_assessment(unsupported)["thermal_status"] == "unavailable"
    for power in ({"returncode": 0, "stdout": "Battery Power"}, {"unavailable": "command missing"}):
        bad = copy.deepcopy(normal)
        bad["power"] = power
        assert not condition_assessment(bad)["valid"]
    throttle = copy.deepcopy(normal)
    throttle["thermal"]["stdout"] = "CPU_Speed_Limit = 80\nCPU_Scheduler_Limit = 100"
    assert not condition_assessment(throttle)["valid"]


def test_original_failure_and_cleanup_failure_both_persist(tmp_path, monkeypatch):
    from transformer import gpu_optimize as opt

    monkeypatch.setattr(opt, "conditions", normal_conditions)

    class BrokenClose:
        def close(self):
            raise RuntimeError("secondary close failure")

    record = {"status": "started"}
    path = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="original numeric failure"):
        try:
            raise ValueError("original numeric failure")
        except ValueError as error:
            opt.record_failure(record, error, "failed")
            raise
        finally:
            opt.finish_record(record, path, BrokenClose())
    saved = json.loads(path.read_text())
    assert saved["error"] == "ValueError: original numeric failure"
    assert saved["cleanup_errors"][0]["error"] == "RuntimeError: secondary close failure"
    assert saved["status"] == "failed"


def test_conditions_after_failure_is_saved_and_raised(tmp_path, monkeypatch):
    from transformer import gpu_optimize as opt

    bad = normal_conditions()
    bad["power"]["stdout"] = "Battery Power"
    monkeypatch.setattr(opt, "conditions", lambda: bad)
    path = tmp_path / "receipt.json"
    with pytest.raises(opt.MeasurementConditionsError):
        opt.finish_record({"status": "completed"}, path)
    saved = json.loads(path.read_text())
    assert saved["status"] == "failed"
    assert not saved["conditions_after_assessment"]["valid"]


def test_state_cleanup_runs_all_actions_without_replacing_original_failure():
    from transformer.gpu_optimize import cleanup_actions

    called = []

    def fail(label):
        called.append(label)
        raise RuntimeError(label)

    with pytest.raises(ValueError, match="primary") as failure:
        try:
            raise ValueError("primary")
        finally:
            cleanup_actions([("first", lambda: fail("first")), ("second", lambda: fail("second"))])
    assert called == ["first", "second"]
    assert len(failure.value._gpu_cleanup_errors) == 2


def test_cleanup_failure_disqualifies_an_already_measured_candidate():
    from transformer.gpu_optimize import cleanup

    record = {
        "id": "fast",
        "status": "qualified_measured",
        "qualification": {"passed": True},
        "paired_timing": {"median_paired_latency_ratio": 0.5},
    }

    def fail():
        raise RuntimeError("close failed")

    cleanup(record, "backend.close", fail, "cleanup_rejected_unranked")
    assert select_winner([record]) is None


def test_preflight_persists_primary_and_close_errors(tmp_path, monkeypatch):
    from transformer import gpu_metal
    from transformer import gpu_optimize as opt

    class BrokenBackend:
        def __init__(self, *args, **kwargs):
            pass

        def capabilities(self):
            raise ValueError("primary capability failure")

        def close(self):
            raise RuntimeError("secondary close failure")

    monkeypatch.setattr(gpu_metal, "Backend", BrokenBackend)
    monkeypatch.setattr(opt, "source_identity", lambda: {"source": "hash"})
    destination = tmp_path / "preflight.json"
    result = opt.preflight_policies(
        {"initial": np.zeros(1, "f4")}, [{"id": "reference", "policy": reference()}], destination
    )
    assert result["records"][0]["status"] == "cleanup_rejected_unranked"
    receipt = json.loads((tmp_path / "preflight.json.entries/reference.json").read_text())
    assert receipt["error"] == "ValueError: primary capability failure"
    assert receipt["cleanup_errors"][0]["error"] == "RuntimeError: secondary close failure"
