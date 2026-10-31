"""Both remaining arms must execute the same nine model and six Adam cases."""

import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_native_common_controls", ROOT / "integration/library-study-v1/common_native_metal.py"
)
K = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(K)


@pytest.mark.parametrize("engine", ["cpu_arm64", "metal"])
def test_native_and_metal_use_identical_common_state_matrix(monkeypatch, tmp_path, engine):
    from transformer import gpu_admission_checks as math
    from transformer import gpu_repeatability as repeated
    from transformer import gpu_study_cpu_admission as cpu
    from transformer import spec

    monkeypatch.setattr(spec, "PARAM_COUNT", 7)
    monkeypatch.setattr(K.C, "MIN_FREE_BYTES", 0)
    row_path = tmp_path / "training.i32"
    np.zeros((2048, 33), np.int32).tofile(row_path)
    raw, output = tmp_path / "raw", tmp_path / "worker"
    raw.mkdir()
    output.mkdir()
    names = [f"model/step-{s}/b{b}-t32" for s in (0, 8, 64) for b in (1, 3, 16)] + [
        f"adam/step-{s}/count-{n}" for s in (0, 64) for n in (1, 2, 10)
    ]

    def state(step):
        return dict(
            parameters=np.zeros(7, np.float32),
            m=np.zeros(7, np.float32),
            v=np.zeros(7, np.float32),
            step=step,
        )

    def gold(parameters, x, y, **kw):
        return dict(
            logits=np.zeros((len(x), 32, 128), np.float64),
            loss=np.array(1.0),
            gradient=np.zeros(7, np.float64),
        )

    def adam(point, gradient, count):
        return dict(copy.deepcopy(point), step=np.array(int(point["step"]) + count, np.int64))

    def decision(np, a, b, kind):
        return dict(passed=bool(np.array_equal(a, b)), atol=0, rtol=0)

    helper = SimpleNamespace(
        cases=lambda: names,
        checkpoint=lambda np, src, step, count: state(step),
        compare=decision,
        exact_state=lambda np, a, b: all(np.array_equal(a[k], b[k]) for k in a),
    )
    monkeypatch.setattr(K, "helpers", lambda runtime: (helper, None))
    monkeypatch.setattr(
        K,
        "trajectory",
        lambda job, runtime: (
            {},
            dict(
                training=dict(path=str(row_path)),
                initial=dict(sha256=hashlib.sha256(np.zeros(7, np.float32).tobytes()).hexdigest()),
            ),
        ),
    )
    monkeypatch.setattr(math, "numpy_model", gold)
    monkeypatch.setattr(math, "adam_reference", adam)
    monkeypatch.setattr(repeated, "_torch_reference", gold)
    calls = dict(model=[], adam=[])

    def compute(actual_engine, values, *, native):
        assert actual_engine == "cpu_arm64" and native == "verified native"
        if values["mode"].item() == "forward_gradient":
            calls["model"].append((int(values["step"]), len(values["tokens"])))
            return {
                k: v.astype(np.float32)
                for k, v in gold(values["parameters"], values["tokens"], values["targets"]).items()
            }
        calls["adam"].append(int(values["step"]))
        return {k: np.asarray(values[k]).copy() for k in state(0)}

    class Metal:
        _selected_native_origin = {}
        closed = False

        def load_state(self, point):
            self.point = copy.deepcopy(point)

        def upload_inputs(self, x, y):
            return x, y

        def forward(self, batch):
            calls["model"].append((int(self.point["step"]), len(batch[0])))
            return np.zeros((len(batch[0]), 32, 128), np.float32)

        def loss_and_grad(self, batch):
            return np.array(1, np.float32), np.zeros(7, np.float32)

        def to_numpy(self, value):
            return value

        def export_state(self):
            return copy.deepcopy(self.point)

        def apply_gradient(self, gradient):
            self.point["step"] += 1
            calls["adam"].append(self.point["step"])

        def capabilities(self):
            return {}

        def close(self):
            self.closed = True

    backend = Metal()
    monkeypatch.setattr(cpu, "_compute", compute)
    monkeypatch.setattr(K.N, "load", lambda *a: "verified native")
    monkeypatch.setattr(K.E, "create_backend", lambda *a: backend)
    monkeypatch.setattr(K.E, "validate_capabilities", lambda engine, cap: cap)
    job = dict(engine=engine, width=spec.WIDTH, native_assets={}, trajectory={}, raw_store=str(raw))
    result = K.run(job, SimpleNamespace(roles={}), output)
    assert result["passed"] and [r["name"] for r in result["cases"]] == names
    assert calls["model"] == [(s, b) for s in (0, 8, 64) for b in (1, 3, 16) for _ in range(2)]
    assert calls["adam"] == [s + i for s in (0, 64) for n in (1, 2, 10) for i in range(1, n + 1)]
    assert backend.closed is (engine == "metal")
