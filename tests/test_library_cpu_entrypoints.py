"""Explicit CPU library selection must survive fit, admission and saved replay.

Hosted all-width checks use the source prototype. They cannot qualify installed
packages, grant formal eligibility or open a reserved assessment stream.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from transformer import gpu_study_cpu_admission, gpu_study_replay, prepare, python_train
from transformer.gpu_environment import ENVIRONMENT
from transformer.library_training import torch_model_type
from transformer.reference import Transformer

ROOT = Path(__file__).resolve().parents[1]


def module_at(name, relative, package=False):
    path = ROOT / relative
    specification = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def explicit_api(monkeypatch):
    import torch

    # Match both production CPU entrypoints before constructing the expected
    # trajectory; FP64 FF1 reductions can differ across thread counts.
    torch.set_num_threads(1)
    for name, value in ENVIRONMENT.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    api = module_at("entrypoint_api", "integration/precision-library-v1/torch_api.py", True)
    monkeypatch.setitem(sys.modules, "torch.nn.precise_fp32", api)
    return api


def fixture_data():
    rows = np.arange(4 * 33, dtype="i4").reshape(4, 33) % 128
    return dict(
        config=dict(
            seed=20261103,
            epochs=2,
            batch=2,
            train_sequences=4,
            validation_sequences=2,
            test_sequences=2,
        ),
        initial=Transformer(seed=20261103).flat_parameters(),
        train=rows,
        validation=rows[:2].copy(),
        test=rows[2:].copy(),
        schedule=np.array([[2, 0, 1, 3], [1, 3, 0, 2]], "i4"),
    )


def test_explicit_cpu_fit_saves_real_library_optimizer_state_and_replays_selected_model(
    tmp_path, monkeypatch, explicit_api
):
    worker = module_at("entrypoint_optimizer", "integration/trained-precision/library_optimizer.py")
    data = fixture_data()
    expected = worker.CPUTraining(data["initial"])
    for order in data["schedule"]:
        for offset in (0, 2):
            batch = data["train"][order[offset : offset + 2]]
            expected.train_step(batch[:, :-1], batch[:, 1:])
    expected_state = expected.export_state()
    monkeypatch.setattr(python_train, "load_fixture", lambda _: data)
    monkeypatch.setattr(prepare, "load_fixture", lambda _: data)
    output = tmp_path / "fit"
    result = python_train.run(tmp_path / "synthetic-inputs", output, library_api=True)
    assert result[6] == expected_state["step"] == 4
    for field, name in (("parameters", "last.f32"), ("m", "adam_m.f32"), ("v", "adam_v.f32")):
        assert (output / name).read_bytes() == expected_state[field].tobytes(), field
    runtime = json.loads((output / "runtime.json").read_text())
    assert runtime["explicit_model"]["identity"] == "maintained-library-training-v1"
    assert runtime["explicit_model"]["stock_control"] is False
    assert runtime["explicit_model"]["module_path"] == explicit_api.__file__
    assert "maintained-library" in runtime["implementation"]
    replay_path = tmp_path / "replay.npz"
    gpu_study_replay.replay(
        "cpu_torch",
        tmp_path / "synthetic-inputs",
        output / "best.f32",
        replay_path,
        library_api=True,
    )
    import torch

    best = torch_model_type("cpu_torch")().load_flat(np.fromfile(output / "best.f32", "f4"))
    with np.load(replay_path, allow_pickle=False) as replay, torch.no_grad():
        for split in ("validation", "test"):
            logits = best(torch.tensor(data[split][:, :-1].astype("i8"))).numpy()
            assert replay[split].tobytes() == logits.tobytes()


def test_explicit_cpu_admission_evaluates_library_forward_and_every_gradient(explicit_api):
    adapter = module_at(
        "entrypoint_model_adapter", "integration/trained-precision/library_candidate.py"
    )
    flat = Transformer(seed=20261103).flat_parameters()
    tokens = np.arange(64, dtype="i4").reshape(2, 32)
    targets = (tokens + 1) % 128
    wanted = adapter.Evaluator("cpu_torch", gradients=True).evaluate(flat, tokens, targets)
    actual = gpu_study_cpu_admission._compute(
        "cpu_torch",
        dict(mode=np.array("forward_gradient"), parameters=flat, tokens=tokens, targets=targets),
        library_api=True,
    )
    for name in ("logits", "gradient"):
        assert actual[name].dtype == np.float32
        assert actual[name].tobytes() == wanted[name].tobytes()
    assert float(actual["loss"]) == float(wanted["loss"])


@pytest.mark.parametrize("engine", ["cpu_arm64", "oracle", "mps"])
def test_library_cpu_flags_cannot_change_another_engine_or_fp64_oracle(engine):
    with pytest.raises(ValueError, match="requires cpu_torch"):
        gpu_study_replay.replay(engine, "absent", "absent", "absent", library_api=True)
    with pytest.raises(ValueError, match="requires cpu_torch"):
        gpu_study_cpu_admission.execute(engine, "absent", "absent", library_api=True)


def test_cpu_fit_rejects_ambiguous_library_selection_before_reading_inputs():
    with pytest.raises(ValueError, match="boolean"):
        python_train.run("absent", "absent", library_api="yes")
