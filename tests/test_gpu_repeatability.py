"""CPU-only tests for the diagnostic ledger and exact-repeat classification."""

import json

import numpy as np
import pytest

from transformer import gpu_repeatability as diagnostic


@pytest.fixture(autouse=True)
def configured_environment(monkeypatch):
    for key, value in diagnostic.PINS.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


def test_inventory_has_complete_bounded_counts_and_explicit_unique_exclusions():
    cases = list(diagnostic.embedding_cases(1024))
    assert len(cases) == 72
    assert sum(case["supported"] for case in cases) == 69
    excluded = [case for case in cases if not case["supported"]]
    assert all(
        case["batch"] == 16 and case["length"] == 32 and case["pattern"] == "unique"
        for case in excluded
    )
    names = diagnostic.expected_case_names(32, diagnostic.STAGES)
    assert len(names) == len(set(names)) == 81
    assert len(diagnostic.expected_case_names(32, ["training"])) == 6


def test_globally_unique_control_is_never_silently_repeated():
    values = diagnostic.tokens_for(3, 32, "unique")
    assert np.unique(values).size == values.size
    with pytest.raises(ValueError, match="exceeds vocabulary"):
        diagnostic.tokens_for(16, 32, "unique")


def test_exact_check_detects_signed_zero_and_dtype_changes():
    positive = np.array([0.0, 1], np.float32)
    negative = np.array([-0.0, 1], np.float32)
    assert diagnostic.exact_comparison(negative, positive)["different_coordinates"] == 1
    assert not diagnostic.exact_comparison(positive.astype(np.float64), positive)["passed"]
    assert diagnostic.exact_comparison(positive, positive.copy())["passed"]


def test_accuracy_is_separate_from_exact_replay():
    a = np.array([1], np.float32)
    b = np.nextafter(a, np.float32(2))
    assert not diagnostic.exact_comparison(a, b)["passed"]
    assert diagnostic.accuracy(a, b, "forward")["passed"]
    assert not diagnostic.accuracy(np.array([np.nan]), a, "forward")["passed"]


def test_embedding_fixtures_are_reproducible_and_oracle_retains_cancellation():
    case = next(
        case for case in diagnostic.embedding_cases(32) if case["cotangent"] == "cancellation"
    )
    tokens, table, dy = diagnostic.embedding_operands(32, case)
    again = diagnostic.embedding_operands(32, case)
    assert all(a.tobytes() == b.tobytes() for a, b in zip((tokens, table, dy), again))
    oracle = diagnostic.embedding_oracle(tokens, table, dy)
    assert oracle["gradient"].dtype == np.float64
    np.testing.assert_array_equal(oracle["gradient"][65], dy.sum(axis=(0, 1), dtype=np.float64))


def test_repeat_case_keeps_first_divergence_and_all_hashes(tmp_path):
    counter = 0

    def operation():
        nonlocal counter
        counter += 1
        return {"gradient": np.array([counter], np.float32)}

    result = diagnostic.repeat_case("changing", operation, tmp_path)
    assert result["complete"] and not result["exact_passed"]
    assert len(result["executions"]) == 10
    assert len(list((tmp_path / "changing").glob("first-divergence-*.npz"))) == 1
    assert (
        len({entry["exact"]["gradient"]["actual"]["sha256"] for entry in result["executions"]})
        == 10
    )


def test_repeat_case_reports_runtime_failure_without_implicit_pass(tmp_path):
    def operation():
        raise RuntimeError("unsupported VJP shape")

    result = diagnostic.repeat_case("failure", operation, tmp_path)
    assert not result["complete"] and not result["exact_passed"]
    assert result["first_arrays"] is None
    assert result["unexecuted_repetitions"] == list(range(1, 10))
    assert result["executions"][0]["error"] == "unsupported VJP shape"


def test_repeat_case_does_not_promote_accuracy_failure_to_repeat_failure(tmp_path):
    result = diagnostic.repeat_case(
        "accurate-separate",
        lambda: {"a": np.ones(1, np.float32)},
        tmp_path,
        lambda actual: {"oracle": {"passed": False}},
    )
    assert result["exact_passed"] and not result["accuracy_passed"]


def test_repeat_case_refuses_existing_evidence(tmp_path):
    diagnostic.repeat_case("one", lambda: {"a": np.ones(1)}, tmp_path)
    with pytest.raises(FileExistsError):
        diagnostic.repeat_case("one", lambda: {"a": np.ones(1)}, tmp_path)


def test_cross_process_compares_actual_saved_bytes(tmp_path):
    for index, value in enumerate((1, 2)):
        directory = tmp_path / str(index)
        directory.mkdir()
        case = diagnostic.repeat_case(
            "case", lambda value=value: {"a": np.array([value], np.float32)}, directory
        )
        (directory / "worker.json").write_text(
            json.dumps({"source_sha256": {"source": "same"}, "cases": [case]})
        )
    result = diagnostic.compare_processes(
        tmp_path / "0", tmp_path / "1", tmp_path / "comparison.json"
    )
    assert not result["passed"]
    assert result["cases"][0]["exact"]["a"]["different_coordinates"] == 1


def test_inventory_command_never_imports_a_gpu_runtime(capsys):
    assert diagnostic.main(["--inventory", "--widths", "32"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["widths"] == [32]
    assert result["processes"] == 2


def test_metal_is_explicitly_full_model_only(tmp_path):
    with pytest.raises(SystemExit):
        diagnostic.main(["--engines", "metal", "--output", str(tmp_path / "new")])
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("length", [3, 32])
def test_numpy_forward_oracle_matches_cpu_fp64_for_partial_and_full_context(length):
    import torch

    from transformer.reference import Transformer

    torch.set_num_threads(1)
    initial = Transformer(seed=diagnostic.SEED).flat_parameters()
    tokens = diagnostic.tokens_for(3, length, "skewed")
    targets = np.roll(tokens, -1, axis=1).copy()
    expected = diagnostic._torch_reference(initial, tokens, targets, backward=False)["logits"]
    actual = diagnostic._numpy_forward(initial, tokens)
    assert diagnostic.accuracy(actual, expected, "oracle")["passed"]


def test_worker_fatal_error_cannot_pass_a_partial_inventory(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from transformer import spec

    class Backend:
        def __init__(self, initial):
            self.initial = initial

        def capabilities(self):
            return {"test_double": True}

        def export_state(self):
            return {
                "parameters": self.initial,
                "m": np.zeros_like(self.initial),
                "v": np.zeros_like(self.initial),
                "step": 0,
            }

        def upload_inputs(self, tokens, targets):
            return tokens, targets

        def close(self):
            pass

    monkeypatch.setattr(
        diagnostic.importlib, "import_module", lambda name: SimpleNamespace(Backend=Backend)
    )

    def failure(*args, **kwargs):
        raise RuntimeError("oracle setup failed")

    monkeypatch.setattr(diagnostic, "_torch_reference", failure)
    result = diagnostic.run_worker("mps", spec.WIDTH, tmp_path, ["full_model"])
    assert not result["complete"]
    assert not result["exact_passed"]
    assert not result["inventory_complete"]
    assert result["fatal_error"]["message"] == "oracle setup failed"


def test_poisoned_case_cleanup_is_retained_and_next_shape_uses_fresh_backend(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from transformer import spec

    instances = []

    class Backend:
        def __init__(self, initial):
            self.initial = initial
            self.poisoned = False
            self.executed_lengths = []
            instances.append(self)

        def capabilities(self):
            return {"test_double": True}

        def export_state(self):
            if self.poisoned:
                raise RuntimeError("poisoned pending graph")
            return {
                "parameters": self.initial,
                "m": np.zeros_like(self.initial),
                "v": np.zeros_like(self.initial),
                "step": 0,
            }

        def load_state(self, state):
            if self.poisoned:
                raise RuntimeError("poisoned pending graph")

        def upload_inputs(self, tokens, targets):
            return SimpleNamespace(owner=self, tokens=tokens)

        def forward(self, batch):
            assert batch.owner is self
            self.executed_lengths.append(batch.tokens.shape[1])
            return np.zeros((*batch.tokens.shape, 128), np.float32)

        def loss_and_grad(self, batch):
            assert batch.owner is self
            if batch.tokens.shape[1] == 3:
                self.poisoned = True
                raise RuntimeError("unsupported T3 VJP")
            return np.asarray(0, np.float32), np.zeros_like(self.initial)

        def to_numpy(self, value):
            return value

        def close(self):
            if self.poisoned:
                raise RuntimeError("cleanup replays poisoned graph")

    monkeypatch.setattr(
        diagnostic.importlib, "import_module", lambda name: SimpleNamespace(Backend=Backend)
    )
    monkeypatch.setattr(diagnostic, "BATCHES", (1,))
    monkeypatch.setattr(
        diagnostic,
        "_torch_reference",
        lambda initial, tokens, targets, **kw: {
            "logits": np.zeros((*tokens.shape, 128), np.float64),
            "loss": np.asarray(0, np.float64),
            "gradient": np.zeros_like(initial, dtype=np.float64),
        },
    )
    monkeypatch.setattr(
        diagnostic,
        "_numpy_forward",
        lambda initial, tokens: np.zeros((*tokens.shape, 128), np.float64),
    )
    result = diagnostic.run_worker("mlx", spec.WIDTH, tmp_path, ["full_model"])
    assert result["inventory_complete"]
    assert not result["complete"] and not result["exact_passed"]
    assert len(instances) == 2
    assert result["cases"][0]["executions"][0]["error"] == "unsupported T3 VJP"
    assert result["cases"][0]["recovery"]["fresh_backend"]
    assert result["cleanup_errors"][0]["message"] == "cleanup replays poisoned graph"
    assert result["cases"][1]["complete"] and result["cases"][1]["exact_passed"]
    assert instances[1].executed_lengths == [32] * diagnostic.REPETITIONS
    assert json.loads((tmp_path / "worker.json").read_text())["complete"] is False


def test_launcher_continues_after_worker_receipt_has_no_final_status(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    launched = []

    def run(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        (output / "worker.json").write_text(json.dumps({"cases": []}))
        launched.append(output)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(diagnostic.subprocess, "run", run)
    output = tmp_path / "study"
    result = diagnostic.main(
        [
            "--engines",
            "mlx",
            "--widths",
            "32",
            "64",
            "--stages",
            "training",
            "--output",
            str(output),
        ]
    )
    assert result == 1
    assert len(launched) == 4
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete"] is False
    assert len(summary["workers"]) == 4
    assert all(worker["incomplete_worker_receipt"] for worker in summary["workers"])
    assert len(summary["cross_process"]) == 2
