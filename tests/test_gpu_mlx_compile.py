"""Pure-Python negative controls for the real-device compilation probe."""

from types import SimpleNamespace

import pytest

from transformer.gpu_mlx_compile import verify_compilation


class Array:
    def __init__(self, values, dtype="float32"):
        self.values = list(values)
        self.shape = (len(self.values),)
        self.dtype = dtype

    def tolist(self):
        return self.values.copy()


class FakeMLX:
    gpu = "gpu"
    float32 = "float32"

    def __init__(self, *, enabled=True, graph=True, corrupt=False, device="gpu", available=True):
        self.enabled, self.graph, self.corrupt = enabled, graph, corrupt
        self.device = device
        self.metal = SimpleNamespace(is_available=lambda: available)

    def default_device(self):
        return self.device

    def compile(self, callback):
        if not self.enabled:
            return callback
        seen = False

        def execute(x, y):
            nonlocal seen
            if not seen:
                seen = True
                return callback(x, y)
            values = [max(2 * a + b, 0) for a, b in zip(x.values, y.values)]
            if self.corrupt:
                values[-1] += 1 / 65536
            return Array(values)

        return execute

    def export_to_dot(self, buffer, actual):
        buffer.write('label="CompiledMultiplyAddMaximum"' if self.graph else 'label="Maximum"')

    @staticmethod
    def eval(*args):
        pass

    @staticmethod
    def synchronize(stream):
        assert stream == "gpu"

    @staticmethod
    def arange(count, *, dtype, stream):
        assert dtype == "float32" and stream == "gpu"
        return Array(range(count))

    @staticmethod
    def full(shape, value, *, dtype, stream):
        assert shape == (257,) and dtype == "float32" and stream == "gpu"
        return Array([value] * shape[0])

    @staticmethod
    def operation(left, right, function, stream):
        assert stream == "gpu"
        right = right.values if isinstance(right, Array) else [right] * len(left.values)
        return Array(function(a, b) for a, b in zip(left.values, right))

    def subtract(self, left, right, *, stream):
        return self.operation(left, right, lambda a, b: a - b, stream)

    def divide(self, left, right, *, stream):
        return self.operation(left, right, lambda a, b: a / b, stream)

    def multiply(self, left, right, *, stream):
        return self.operation(left, right, lambda a, b: a * b, stream)

    def add(self, left, right, *, stream):
        return self.operation(left, right, lambda a, b: a + b, stream)

    def maximum(self, left, right, *, stream):
        return self.operation(left, right, max, stream)


@pytest.fixture(autouse=True)
def absent_control(monkeypatch):
    monkeypatch.delenv("MLX_DISABLE_COMPILE", raising=False)


def test_probe_requires_trace_cache_graph_and_changed_exact_outputs():
    result = verify_compilation(FakeMLX())
    assert result["status"] == "passed"
    assert result["mode_override_used"] is False
    assert result["python_trace_count"] == 1 and result["executions"] == 4
    assert len({row["fp32_sha256"] for row in result["observations"]}) == 4


@pytest.mark.parametrize("value", ("0", "1", ""))
def test_probe_rejects_present_disable_control(monkeypatch, value):
    monkeypatch.setenv("MLX_DISABLE_COMPILE", value)
    with pytest.raises(RuntimeError, match="must be absent"):
        verify_compilation(FakeMLX())


def test_probe_rejects_cached_disabled_mode_even_with_a_forged_graph_marker():
    with pytest.raises(RuntimeError, match="re-entered Python"):
        verify_compilation(FakeMLX(enabled=False, graph=True))


def test_probe_rejects_graph_without_fused_primitive():
    with pytest.raises(RuntimeError, match="no Compiled primitive"):
        verify_compilation(FakeMLX(graph=False))


def test_probe_rejects_incorrect_replayed_arithmetic():
    with pytest.raises(RuntimeError, match="incorrect outputs"):
        verify_compilation(FakeMLX(corrupt=True))


@pytest.mark.parametrize("settings", ({"device": "cpu"}, {"available": False}))
def test_probe_rejects_cpu_or_unavailable_gpu(settings):
    with pytest.raises(RuntimeError, match="Metal GPU"):
        verify_compilation(FakeMLX(**settings))
