"""Exercise runner decisions with a deterministic CPU mock, never a GPU backend."""

import hashlib
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from transformer import gpu_inputs, gpu_run, prepare, spec
from transformer.gpu_contract import GPU_ENGINES


class MockBackend:
    """Small state machine with observable uploads, updates, checkpoints and replay."""

    def __init__(self, initial, max_batch, *, fault=None):
        self.state = {
            "parameters": initial.copy(),
            "m": np.zeros_like(initial),
            "v": np.zeros_like(initial),
            "step": 0,
        }
        self.max_batch, self.fault = max_batch, fault
        self.datasets, self.gathers, self.updates, self.loads = [], [], [], []
        self.parameter_exports, self.state_exports = [], 0
        self.forward_count, self.synchronizations, self.closed = 0, 0, False

    def upload_dataset(self, rows):
        value = {"kind": ("train", "validation", "test")[len(self.datasets)], "rows": rows.copy()}
        self.datasets.append(value)
        return value

    def upload_indices(self, indices):
        return indices.copy()

    def upload_inputs(self, tokens, targets=None):
        return {"kind": "inference", "rows": tokens.copy()}

    def gather_batch(self, dataset, indices, offset, count):
        assert 1 <= count <= self.max_batch
        assert 0 <= offset and offset + count <= len(indices)
        selection = indices[offset : offset + count]
        assert np.all((0 <= selection) & (selection < len(dataset["rows"])))
        self.gathers.append((dataset["kind"], offset, count, self.state["step"]))
        return {"kind": dataset["kind"], "rows": dataset["rows"][selection]}

    def train_step(self, batch):
        if self.fault in {"training_exception", "training_and_close"} and self.loads:
            raise RuntimeError("Injected training failure")
        if self.fault == "after_checkpoint" and self.loads and self.state["step"] >= 3:
            raise RuntimeError("Injected failure after checkpoint")
        self.updates.append({"warmup": not self.loads, "rows": batch["rows"][:, 0].tolist()})
        self.state["step"] += 1
        for name, multiplier in (("parameters", 1), ("m", 10), ("v", 100)):
            self.state[name].fill(self.state["step"] * multiplier)
        return np.array(batch["rows"][:, 0].mean(), dtype=np.float32)

    def evaluate(self, batch):
        parameter = int(self.state["parameters"][0])
        # The second and third epochs tie when the standard fixture has three
        # updates per epoch. Selection must use validation, and retain epoch 2.
        loss = {3: 4.0, 6: 2.0, 9: 2.0}.get(parameter, 8.0)
        if (
            self.fault == "replay_mismatch"
            and len(self.loads) >= 2
            and batch["kind"] == "validation"
        ):
            loss = np.nextafter(np.float32(loss), np.float32(np.inf))
        return np.array(loss, dtype=np.float32), self._logits(batch)

    def _logits(self, batch):
        return np.full((len(batch["rows"]), 32, 128), self.state["parameters"][0], dtype=np.float32)

    def forward(self, batch):
        self.forward_count += 1
        values = self._logits(batch)
        if self.fault == "b1_nan" or (self.fault == "b1_timed_nan" and self.forward_count > 20):
            values[0, 0, 0] = np.nan
        elif self.fault == "b1_shape":
            values = values[:, :-1]
        elif self.fault == "b1_dtype":
            values = values.astype(np.float64)
        elif self.fault == "b1_timed_shape" and self.forward_count > 20:
            values = values.reshape(1, 16, 256)
        elif self.fault == "b1_timed_dtype" and self.forward_count > 20:
            values = values.astype(np.float64)
        elif self.fault == "b1_replay" and self.forward_count > 20:
            values[0, 0, 0] = np.nextafter(values[0, 0, 0], np.float32(np.inf))
        return values

    def check_finite(self, value=None):
        if value is not None:
            return bool(np.isfinite(value).all())
        return all(np.isfinite(self.state[key]).all() for key in ("parameters", "m", "v"))

    def to_numpy(self, value):
        return np.array(value, copy=True)

    def synchronize(self):
        self.synchronizations += 1

    def export_state(self):
        self.state_exports += 1
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self.state.items()
        }

    def export_parameters(self):
        self.parameter_exports.append(self.state["step"])
        return self.state["parameters"].copy()

    def load_state(self, state):
        self.state = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in state.items()
        }
        self.loads.append(
            {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self.state.items()
            }
        )
        if len(self.loads) == 1 and self.fault in {"parameters", "m", "v", "step"}:
            if self.fault == "step":
                self.state["step"] += 1
            else:
                # A one-ULP change would pass ordinary allclose tolerances.
                self.state[self.fault][0] = np.nextafter(
                    self.state[self.fault][0], np.float32(np.inf)
                )

    def memory_stats(self):
        return {"mock_allocated_bytes": 96, "scope": "Synthetic test state only"}

    def capabilities(self):
        return {"device": "cpu-mock", "gpu_execution": False}

    def close(self):
        self.closed = True
        if self.fault in {"close_exception", "training_and_close"}:
            raise RuntimeError("Injected cleanup failure")


@pytest.fixture
def mock_runtime(monkeypatch):
    monkeypatch.setattr(prepare, "PARAM_COUNT", 8)
    monkeypatch.setattr(gpu_inputs, "PARAM_COUNT", 8)
    monkeypatch.setattr(spec, "PARAM_COUNT", 8)
    for key, value in gpu_run.ENVIRONMENT.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    instances, imports = [], []
    settings = {"fault": None}

    def construct(initial, max_batch):
        backend = MockBackend(initial, max_batch, fault=settings["fault"])
        instances.append(backend)
        return backend

    def import_backend(name):
        assert name in {f"transformer.gpu_{engine}" for engine in GPU_ENGINES}
        imports.append(name)
        return SimpleNamespace(Backend=construct)

    monkeypatch.setattr(gpu_run, "importlib", SimpleNamespace(import_module=import_backend))
    monkeypatch.setattr(
        gpu_run,
        "subprocess",
        SimpleNamespace(
            check_output=lambda args, **kwargs: "mock-revision\n" if args[1] == "rev-parse" else ""
        ),
    )
    return SimpleNamespace(instances=instances, imports=imports, settings=settings)


@pytest.fixture
def fixture_factory(tmp_path, mock_runtime):
    def create(*, development=True, train=8, validation=4, test=5, epochs=3, batch=3):
        source = tmp_path / "input"
        source.mkdir()
        config = dict(
            zip(
                prepare.CONFIG_NAMES,
                (
                    prepare.MAGIC,
                    prepare.VERSION,
                    epochs,
                    batch,
                    train,
                    validation,
                    test,
                    42,
                    spec.WIDTH,
                ),
                strict=True,
            )
        )
        (source / "config.bin").write_bytes(
            struct.pack("<9I", *(config[key] for key in prepare.CONFIG_NAMES))
        )
        np.arange(8, dtype="<f4").tofile(source / "initial.f32")
        for name, count, first in (
            ("train", train, 10),
            ("validation", validation, 80),
            ("test", test, 100),
        ):
            np.repeat(np.arange(first, first + count, dtype="<i4")[:, None], 33, axis=1).tofile(
                source / f"{name}.i32"
            )
        schedule = np.stack(
            [np.arange(train)[:: -1 if epoch % 2 else 1] for epoch in range(epochs)]
        )
        schedule.astype("<i4").tofile(source / "schedule.i32")
        manifest = {
            "schema": 2,
            "config": config,
            "gpu_development_fixture": development,
            "sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source.iterdir()
            },
        }
        (source / "manifest.json").write_text(json.dumps(manifest))
        prepare.load_fixture(source)
        return source

    return create


def test_complete_runner_selects_own_best_epoch_and_exports_parameters_only(
    fixture_factory, mock_runtime, tmp_path
):
    source = fixture_factory()
    output = tmp_path / "run"
    receipt = gpu_run.run_development(source, output, "metal")
    backend = mock_runtime.instances[0]
    assert receipt["scope"] == "development_only" and receipt["status"] == "completed"
    assert receipt["steps"] == 9 and receipt["best_epoch"] == 2
    assert receipt["selected_validation_replay_exact"] is True
    assert receipt["inference_replay_exact"] is True
    assert receipt["warmup"]["state_restored_exactly"] is True
    assert receipt["inputs_unchanged"] is True
    assert backend.closed
    assert backend.parameter_exports == [3, 3, 6]
    assert backend.state_exports == 3
    assert backend.forward_count == 120 and backend.synchronizations >= 100
    for name, expected in (
        ("best.f32", 6),
        ("last.f32", 9),
        ("adam_m.f32", 90),
        ("adam_v.f32", 900),
    ):
        np.testing.assert_array_equal(np.fromfile(output / name, dtype="<f4"), np.full(8, expected))
    selected_state = backend.loads[-1]
    assert selected_state["step"] == 9
    np.testing.assert_array_equal(selected_state["parameters"], np.full(8, 6))
    np.testing.assert_array_equal(selected_state["m"], np.full(8, 90))
    np.testing.assert_array_equal(selected_state["v"], np.full(8, 900))
    history = np.fromfile(output / "history.f64", dtype="<f8").reshape(3, 3)
    np.testing.assert_array_equal(history[:, 1], [4.0, 2.0, 2.0])
    np.testing.assert_array_equal(
        np.fromfile(output / "test_logits.f32", dtype="<f4"), np.full(5 * 32 * 128, 6)
    )
    assert np.fromfile(output / "timings.f64", dtype="<f8").size == 100
    np.testing.assert_array_equal(
        np.fromfile(output / "inference_logits.f32", dtype="<f4"), np.full(32 * 128, 6)
    )
    assert all(len(update["rows"]) in (2, 3) for update in backend.updates if not update["warmup"])
    assert [len(update["rows"]) for update in backend.updates if not update["warmup"]] == [
        3,
        3,
        2,
    ] * 3
    assert {count for name, _, count, _ in backend.gathers if name == "validation"} == {1, 3}
    assert {count for name, _, count, _ in backend.gathers if name == "test"} == {2, 3}
    assert receipt["artifact_sha256"] == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.iterdir()
        if path.name != "runtime.json"
    }


def test_warmup_handles_evaluation_batch_larger_than_training_partition(
    fixture_factory, mock_runtime, tmp_path
):
    source = fixture_factory(train=2, validation=7, test=5, epochs=1, batch=4)
    receipt = gpu_run.run_development(source, tmp_path / "small-train", "mps")
    assert receipt["status"] == "completed" and receipt["steps"] == 1
    assert receipt["warmup"]["state_restored_exactly"]
    assert mock_runtime.instances[0].closed
    warmup = [update for update in mock_runtime.instances[0].updates if update["warmup"]]
    assert any(update["rows"] == [10, 11, 10, 11] for update in warmup)


@pytest.mark.parametrize("field", ["parameters", "m", "v", "step"])
def test_warmup_requires_exact_restoration_of_every_optimizer_component(
    field, fixture_factory, mock_runtime, tmp_path
):
    source = fixture_factory()
    mock_runtime.settings["fault"] = field
    output = tmp_path / "bad-warmup"
    with pytest.raises(RuntimeError, match="Warmup changed"):
        gpu_run.run_development(source, output, "mlx")
    receipt = json.loads((output / "runtime.json").read_text())
    assert receipt["status"] == "failed" and "Warmup changed" in receipt["error"]
    assert not (output / "best.f32").exists()
    assert mock_runtime.instances[0].closed


def test_formal_fixture_is_rejected_before_backend_construction(
    fixture_factory, mock_runtime, tmp_path
):
    source = fixture_factory(development=False)
    output = tmp_path / "formal"
    with pytest.raises(ValueError, match="Formal GPU measurement is not admitted"):
        gpu_run.run_development(source, output, "mps")
    assert not output.exists() and not mock_runtime.instances


@pytest.mark.parametrize("location", ["existing", "source", "inside", "parent"])
def test_runner_rejects_output_collisions_without_backend_execution(
    location, fixture_factory, mock_runtime, tmp_path
):
    source = fixture_factory()
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep"
    sentinel.write_bytes(b"existing bytes")
    output = {
        "existing": existing,
        "source": source,
        "inside": source / "nested",
        "parent": source.parent,
    }[location]
    with pytest.raises(ValueError, match="fresh and separate"):
        gpu_run.run_development(source, output, "metal")
    assert not mock_runtime.instances and sentinel.read_bytes() == b"existing bytes"
    assert not (source / "nested").exists()


def test_training_exception_keeps_failure_receipt_and_closes_backend(
    fixture_factory, mock_runtime, tmp_path
):
    source = fixture_factory()
    mock_runtime.settings["fault"] = "training_exception"
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="Injected training failure"):
        gpu_run.run_development(source, output, "metal")
    receipt = json.loads((output / "runtime.json").read_text())
    assert receipt["status"] == "failed" and receipt["scope"] == "development_only"
    assert "Injected training failure" in receipt["traceback"]
    assert receipt["warmup"]["state_restored_exactly"]
    assert receipt["input_sha256"] and mock_runtime.instances[0].closed


def test_failure_receipt_authenticates_partial_checkpoint_artifacts(
    fixture_factory, mock_runtime, tmp_path
):
    source = fixture_factory()
    mock_runtime.settings["fault"] = "after_checkpoint"
    output = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="failure after checkpoint"):
        gpu_run.run_development(source, output, "metal")
    receipt = json.loads((output / "runtime.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["artifact_sha256"] == {
        "best.f32": hashlib.sha256((output / "best.f32").read_bytes()).hexdigest()
    }
    np.testing.assert_array_equal(np.fromfile(output / "best.f32", dtype="<f4"), np.full(8, 3))
    assert mock_runtime.instances[0].closed


@pytest.mark.parametrize("fault", ["close_exception", "training_and_close"])
def test_cleanup_failure_is_retained_without_masking_original_training_error(
    fault, fixture_factory, mock_runtime, tmp_path
):
    source = fixture_factory()
    mock_runtime.settings["fault"] = fault
    output = tmp_path / "cleanup-failure"
    expected = "Injected training failure" if fault == "training_and_close" else "cleanup failed"
    with pytest.raises(RuntimeError, match=expected):
        gpu_run.run_development(source, output, "metal")
    receipt = json.loads((output / "runtime.json").read_text())
    assert receipt["status"] == "failed"
    assert "Injected cleanup failure" in receipt["cleanup_error"]
    assert "Injected cleanup failure" in receipt["cleanup_traceback"]
    if fault == "training_and_close":
        assert "Injected training failure" in receipt["error"]
    else:
        assert "timings.f64" in receipt["artifact_sha256"]
    assert mock_runtime.instances[0].closed


def test_existing_dangling_output_symlink_is_not_a_fresh_destination(
    fixture_factory, mock_runtime, tmp_path
):
    source = fixture_factory()
    target = tmp_path / "absent-target"
    output = tmp_path / "existing-link"
    output.symlink_to(target)
    with pytest.raises(ValueError, match="fresh"):
        gpu_run.run_development(source, output, "metal")
    assert output.is_symlink() and not target.exists()
    assert not mock_runtime.instances


def test_one_ulp_validation_replay_difference_remains_explicit_failure(
    fixture_factory, mock_runtime, tmp_path
):
    source = fixture_factory()
    mock_runtime.settings["fault"] = "replay_mismatch"
    output = tmp_path / "replay"
    receipt = gpu_run.run_development(source, output, "mps")
    assert receipt["status"] == "development_replay_failed"
    assert receipt["selected_validation_replay_exact"] is False
    assert receipt["artifact_sha256"] and mock_runtime.instances[0].closed


@pytest.mark.parametrize(
    "fault", ["b1_nan", "b1_timed_nan", "b1_shape", "b1_dtype", "b1_timed_shape", "b1_timed_dtype"]
)
def test_batch_one_inference_is_validated_separately_from_successful_batched_test(
    fault, fixture_factory, mock_runtime, tmp_path
):
    source = fixture_factory()
    mock_runtime.settings["fault"] = fault
    output = tmp_path / "bad-b1"
    expected = FloatingPointError if "nan" in fault else ValueError
    with pytest.raises(expected):
        gpu_run.run_development(source, output, "metal")
    receipt = json.loads((output / "runtime.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["selected_validation_replay_exact"] is True
    assert (output / "test_logits.f32").is_file()
    assert not (output / "timings.f64").exists()
    assert mock_runtime.instances[0].closed


def test_one_ulp_batch_one_replay_difference_is_not_replaced_with_allclose(
    fixture_factory, mock_runtime, tmp_path
):
    source = fixture_factory()
    mock_runtime.settings["fault"] = "b1_replay"
    receipt = gpu_run.run_development(source, tmp_path / "b1-replay", "metal")
    assert receipt["status"] == "development_replay_failed"
    assert receipt["selected_validation_replay_exact"] is True
    assert receipt["inference_replay_exact"] is False
    assert mock_runtime.instances[0].closed


def test_evaluation_weights_partial_batch_loss_and_preserves_prediction_order(tmp_path):
    class EvaluationBackend(MockBackend):
        def evaluate(self, batch):
            values = batch["rows"][:, 0]
            loss = np.array(values.mean(), dtype=np.float32)
            logits = np.broadcast_to(values[:, None, None], (len(values), 32, 128)).astype(
                np.float32
            )
            return loss, logits

    backend = EvaluationBackend(np.zeros(8, np.float32), max_batch=3)
    dataset = {
        "kind": "validation",
        "rows": np.repeat(np.arange(5, dtype=np.int32)[:, None], 33, axis=1),
    }
    order = np.array([4, 0, 2, 3, 1], dtype=np.int32)
    output = tmp_path / "logits.f32"
    with output.open("wb") as handle:
        loss = gpu_run.evaluate(backend, dataset, order, 5, 3, logits_file=handle)
    assert loss == 2.0
    logits = np.fromfile(output, dtype="<f4").reshape(5, 32, 128)
    np.testing.assert_array_equal(logits[:, 0, 0], order)
    assert [count for _, _, count, _ in backend.gathers] == [3, 2]


def test_environment_rejects_cpu_fallback_before_backend_import(
    fixture_factory, mock_runtime, tmp_path, monkeypatch
):
    source = fixture_factory()
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(ValueError, match="PYTORCH_ENABLE_MPS_FALLBACK"):
        gpu_run.run_development(source, tmp_path / "fallback", "mps")
    assert not mock_runtime.imports and not mock_runtime.instances
