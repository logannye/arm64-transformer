"""Prospective provenance inventory tests; no operands or reserved data are generated."""

import copy

import pytest

from transformer import gpu_owned_admission as owned
from transformer import gpu_study as study
from transformer import gpu_study_admission as admission


def records():
    manifest = study.planned_manifest("owned")
    inventory = owned.cpu_operation_inventory(manifest)
    gates = []
    for gate, operations in inventory.items():
        gates.append(
            {
                "id": gate,
                "owned_cpu_operations": [
                    {**row, "proof": {"path": f"{gate}/{i}.json", "sha256": "0" * 64}}
                    for i, row in enumerate(operations)
                ],
            }
        )
    return manifest, gates


def test_inventory_covers_cpu_reference_work_in_all_five_engine_gates():
    manifest, gates = records()
    expected = owned.cpu_operation_inventory(manifest)
    for width in study.WIDTHS:
        for engine in study.ENGINES:
            assert len(expected[f"{engine}/{width}/trajectory"]) == (
                12 if engine == "cpu_torch" else 6
            )
        rejected = [
            row["expected_rejection"]
            for gate, rows in expected.items()
            if gate.split("/")[1] == str(width)
            for row in rows
            if row["expected_rejection"]
        ]
        assert rejected == [
            "Out-of-range CPU tokens",
            "Malformed CPU token batch",
            "Malformed CPU parameters",
        ]
        assert len(expected[f"cpu_torch/{width}/control/repeat_training_fresh_process"]) == 14
        assert len(expected[f"cpu_arm64/{width}/common_state/evolved"]) == 3
    assert len(owned.verify_cpu_operation_records(manifest, gates)) > 6 * 100


@pytest.mark.parametrize(
    "mutation", ["only_ties", "missing", "reorder", "mode", "rejection", "duplicate"]
)
def test_removed_cpu_operations_cannot_be_hidden_by_updated_summary(mutation):
    manifest, gates = records()
    original = copy.deepcopy(gates)
    if mutation == "only_ties":
        for gate in gates:
            gate["owned_cpu_operations"] = [
                row for row in gate["owned_cpu_operations"] if row["mode"] == "checkpoint_first_tie"
            ]
    elif mutation == "missing":
        del gates[0]["owned_cpu_operations"]
    else:
        gate = next(
            g for g in gates if g["id"] == "cpu_torch/32/control/repeat_training_fresh_process"
        )
        rows = gate["owned_cpu_operations"]
        if mutation == "reorder":
            rows.reverse()
        elif mutation == "mode":
            rows[0]["mode"] = "adam"
        elif mutation == "rejection":
            rows[0]["expected_rejection"] = "Malformed CPU parameters"
        else:
            rows[1]["proof"] = rows[0]["proof"]
    assert gates != original
    with pytest.raises(ValueError):
        owned.verify_cpu_operation_records(manifest, gates)


def test_empty_proof_inventory_fails_even_if_summary_matches_files(tmp_path):
    manifest, gates = records()
    with pytest.raises(ValueError, match="origin proof"):
        admission.owned_cpu_proofs(tmp_path, manifest, records=[], gates=gates)
