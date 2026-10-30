"""Development inventory authenticity and production-independent FP64 checks."""

import copy
import json

import numpy as np
import pytest
import torch

from transformer import spec
from transformer.gpu_admission_checks import numpy_model
from transformer.reference import Transformer
from transformer.trained_precision import checked_file, digest, inventory, torch_evaluate, write_new


@pytest.fixture
def retained(tmp_path):
    for width in (512, 1024):
        files = {
            f"results/width-sweep-v6/width-{width}/seed-42-repeat-1-arm64/best.f32": b"checkpoint",
            f"results/inputs-v6/width-{width}/seed-42/test.i32": b"rows",
        }
        for name, data in files.items():
            p = tmp_path / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        checkpoint, rows = files
        base = {
            "run": "seed-42-repeat-1-arm64",
            "fixture": "seed-42",
            "implementation": "arm64",
            "artifact_sha256": {"best.f32": digest(tmp_path / checkpoint)},
            "fixture_sha256": {"test.i32": digest(tmp_path / rows)},
            "numerical_audit": {
                "categories": {
                    "native_fp64": {
                        "failed_checks": ["seed-42-repeat-1-arm64/v6/test/sequence_3/native_fp64"]
                    },
                    "python_fp64": {
                        "failed_checks": ["seed-42-repeat-1-arm64/v6/test/sequence_3/python_fp64"]
                    },
                }
            },
        }
        repeat = copy.deepcopy(base)
        repeat["run"] = "seed-42-repeat-2-arm64"
        for v in repeat["numerical_audit"]["categories"].values():
            v["failed_checks"] = [s.replace("repeat-1", "repeat-2") for s in v["failed_checks"]]
        report = tmp_path / f"experiments/width-sweep-v6/width-{width}/summary.json"
        report.parent.mkdir(parents=True)
        write_new(report, {"runs": [base, repeat]})
    return tmp_path


def test_inventory_unions_both_engines_and_deduplicates_repeats(retained):
    result = inventory(retained)
    assert len(result["cases"]) == 2
    assert all(len(c["historical_failures"]) == 4 for c in result["cases"])
    assert result["formal_eligible"] is False


def test_inventory_rejects_changed_checkpoint(retained):
    path = retained / "results/width-sweep-v6/width-512/seed-42-repeat-1-arm64/best.f32"
    path.write_bytes(b"different")
    with pytest.raises(ValueError, match="Retained artifact changed"):
        inventory(retained)


def test_inventory_does_not_drop_unrecognized_failures(retained):
    report = retained / "experiments/width-sweep-v6/width-512/summary.json"
    value = json.loads(report.read_text())
    value["runs"][0]["numerical_audit"]["categories"]["native_fp64"]["failed_checks"] += ["NLL"]
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Unexpected non-sequence"):
        inventory(retained)


def test_evidence_escape_and_output_collision_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"evidence")
    with pytest.raises(ValueError, match="escaped"):
        checked_file(root, "../outside", digest(outside))
    write_new(root / "receipt.json", {"passed": False})
    with pytest.raises(FileExistsError):
        write_new(root / "receipt.json", {"passed": True})


def test_replay_dual_fp64_forward_and_every_gradient():
    # Small context bounds the test cost at every CI width; the actual replay
    # inventory always uses full context and trained parameters.
    torch.set_num_threads(1)
    flat = Transformer(seed=20261103).flat_parameters()
    tokens = np.array([[65, 65, 66]], "i4")
    targets = np.array([[65, 66, 67]], "i4")
    expected = numpy_model(flat, tokens, targets)
    actual = torch_evaluate(flat, tokens, targets, dtype=torch.float64, gradients=True)
    assert actual["gradient"].shape == (spec.PARAM_COUNT,)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-10, rtol=1e-10)
