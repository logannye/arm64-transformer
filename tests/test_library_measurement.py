"""Selection mutations and timing-boundary checks without a device or a build."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_library_measurement", ROOT / "integration/library-study-v1/measurement.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


@pytest.mark.parametrize("engine", ["mps", "mlx"])
def test_selection_rejects_stock_or_changed_framework_models(engine):
    cap = dict(
        implementation="maintained-library-training-v1",
        standard_release_claim=False,
        library_api=dict(precise_stages=M.C.comparison()["model_selection"][engine]["stages"]),
        storage_dtype="float32",
        cpu_tensor_fallback=False,
        compiled=True,
        MLX_DISABLE_COMPILE=None,
    )
    assert M.E.validate_capabilities(engine, cap) is cap
    for key, value in (
        ("implementation", "stock"),
        ("standard_release_claim", True),
        ("library_api", dict(precise_stages=[])),
        ("storage_dtype", "float64"),
        ("cpu_tensor_fallback", True),
    ):
        with pytest.raises(ValueError, match="differs"):
            M.E.validate_capabilities(engine, dict(cap, **{key: value}))
    if engine == "mlx":
        with pytest.raises(ValueError, match="Compiled"):
            M.E.validate_capabilities(engine, dict(cap, compiled=False))


def test_metal_selection_checks_combined_shader_defines_and_host():
    from transformer.gpu_metal import OPTIMIZED_SHADERS
    from transformer.gpu_policy import bundle, defines, identity, reference
    from transformer.library_metal import SHADER, SHADER_SHA256
    from transformer.spec import WIDTH

    source = (ROOT / SHADER).read_bytes() + b"\n" + OPTIMIZED_SHADERS.read_bytes()
    policies = bundle(reference(), WIDTH)
    host = hashlib.sha256((ROOT / "native/metal/backend.mm").read_bytes()).hexdigest()
    cap = dict(
        implementation="precision-normalization-reference-metal-v1",
        selected_reference_shader_sha256=SHADER_SHA256,
        shader_sha256=hashlib.sha256(source).hexdigest(),
        precision_policy=1,
        optimized_kernel_claim=False,
        kernel_policy_sha256=identity(policies),
        binary_sha256=M.E.A.METAL_SHA256,
        compiled_host_source_sha256=host,
        host_source_sha256=host,
        framework_numerical_calls=0,
        storage_dtype="float32",
    )
    for phase in ("training", "inference"):
        cap[phase + "_shader_sha256"] = hashlib.sha256(
            defines(policies[phase]) + source
        ).hexdigest()
    assert M.E.validate_capabilities("metal", cap) is cap
    for key in (
        "shader_sha256",
        "training_shader_sha256",
        "inference_shader_sha256",
        "kernel_policy_sha256",
        "binary_sha256",
        "compiled_host_source_sha256",
    ):
        with pytest.raises(ValueError, match="differs"):
            M.E.validate_capabilities("metal", dict(cap, **{key: "0" * 64}))


class Backend:
    """Tiny recorder: supplies no model arithmetic and allocates no device data."""

    def __init__(self, *, broken_restore=False, inference_drift=False):
        self.state = dict(
            parameters=np.zeros(7, np.float32),
            m=np.zeros(7, np.float32),
            v=np.zeros(7, np.float32),
            step=0,
        )
        self.broken_restore, self.inference_drift = broken_restore, inference_drift
        self.updates, self.forwards, self.evaluations, self.closed = 0, 0, 0, False
        self.restores = []

    def upload_dataset(self, rows):
        return rows

    def upload_indices(self, indices):
        return indices

    def gather_batch(self, dataset, indices, offset, count):
        return dataset[indices[offset : offset + count]]

    def upload_inputs(self, rows):
        return rows

    def export_state(self):
        return copy.deepcopy(self.state)

    def load_state(self, state):
        self.restores.append(copy.deepcopy(state))
        if not self.broken_restore:
            self.state = copy.deepcopy(state)

    def export_parameters(self):
        return self.state["parameters"].copy()

    def train_step(self, batch):
        self.updates += 1
        self.state["step"] += 1
        self.state["parameters"] += np.float32(1)
        return np.array(2, np.float32)

    def evaluate(self, batch):
        self.evaluations += 1
        return np.array(2, np.float32), np.zeros((len(batch), 32, 128), np.float32)

    def forward(self, batch):
        self.forwards += 1
        value = self.forwards if self.inference_drift else 0
        return np.full((len(batch), 32, 128), value, np.float32)

    def to_numpy(self, value):
        return value

    def check_finite(self, value=None):
        return True

    def synchronize(self):
        pass

    def memory_stats(self):
        return {}

    def capabilities(self):
        return {}

    def close(self):
        self.closed = True


def measurement(monkeypatch, tmp_path, backend):
    from transformer import spec

    monkeypatch.setattr(spec, "PARAM_COUNT", 7)
    monkeypatch.setattr(M.E, "create_backend", lambda *a, **k: backend)
    monkeypatch.setattr(M.E, "validate_capabilities", lambda engine, cap: cap)
    fixture = dict(
        initial=np.zeros(7, np.float32),
        train=np.zeros((2048, 33), np.int32),
        validation=np.zeros((256, 33), np.int32),
        test=np.zeros((256, 33), np.int32),
        schedule=np.tile(np.arange(2048, dtype=np.int32), (8, 1)),
    )
    runtime = SimpleNamespace(origins=lambda engine: {}, recheck=lambda: None)
    M.gpu_fit(fixture, tmp_path, "mps", runtime=runtime, scope="development_only")


def test_fit_retains_workload_checkpoint_ties_warmup_and_inference_boundary(monkeypatch, tmp_path):
    backend = Backend()
    measurement(monkeypatch, tmp_path, backend)
    receipt = json.loads((tmp_path / "runtime.json").read_text())
    assert backend.closed and backend.updates == 1025
    assert backend.forwards == 120 and backend.evaluations == 161
    assert backend.restores[0]["step"] == 0 and receipt["steps"] == 1024
    assert receipt["best_epoch"] == 1 and receipt["warmup_restored_exactly"]
    assert np.fromfile(tmp_path / "best.f32", "<f4").tolist() == [128.0] * 7
    assert np.fromfile(tmp_path / "last.f32", "<f4").tolist() == [1024.0] * 7
    assert (tmp_path / "test_logits.f32").stat().st_size == 256 * 32 * 128 * 4
    assert (tmp_path / "timings.f64").stat().st_size == 100 * 8
    assert receipt["scope"] == "development_only" and receipt["status"] == "completed"


@pytest.mark.parametrize("failure", ["broken_restore", "inference_drift"])
def test_fit_rejects_state_or_output_drift_and_still_closes(monkeypatch, tmp_path, failure):
    backend = Backend(**{failure: True})
    with pytest.raises(ValueError, match="Warmup|replay"):
        measurement(monkeypatch, tmp_path, backend)
    assert backend.closed
    assert json.loads((tmp_path / "runtime.json").read_text())["status"] == "failed"
