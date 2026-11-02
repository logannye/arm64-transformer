"""Successor selection, denominator and closed historical routes, without admission."""

import ast
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from transformer import gpu_library_study as library
from transformer import gpu_study, gpu_study_context, library_metal


def test_prospective_selection_preserves_workload_and_full_denominator():
    old, new = gpu_study.planned_manifest(), library.plan()
    for field in (
        "workload",
        "timing",
        "environment",
        "cells",
        "planned_fits",
        "planned_test_token_predictions",
        "cpu_evidence_revision",
    ):
        assert old[field] == new[field]
    assert len(new["cells"]) == 180
    assert sum(c["engine"].startswith("cpu_") for c in new["cells"]) == 72
    assert {g["id"] for g in old["gate_inventory"]} == {g["id"] for g in new["gate_inventory"]}
    assert all(g["blocking"] for g in new["gate_inventory"] if "/full/" in g["id"])
    assert not new["formal_execution_supported"] and not new["formal_eligible"]
    assert new["formal_fits"] == 0 and not new["admission_stream_opened"]
    assert new["admission_seed"] in gpu_study_context.RESERVED_SEEDS
    assert library.validate_plan(new) is new
    for field in ("protocol", "admission_seed", "cpu_revision", "model_selection"):
        broken = copy.deepcopy(new)
        broken[field] = None
        with pytest.raises(ValueError, match="plan changed"):
            library.validate_plan(broken)
    with pytest.raises(ValueError, match="Unknown immutable study context"):
        gpu_study.validate_manifest(new)


def test_new_reserved_stream_is_closed_before_fixture_generation():
    from transformer.gpu_admission_checks import precision_fixtures

    with pytest.raises(ValueError, match="Reserved admission stream"):
        precision_fixtures(32, "attention", 20261112)


def test_metal_selection_fixes_reference_policy_without_changing_stock_default(monkeypatch):
    from transformer.gpu_policy import reference

    class FakeBackend:
        def __init__(self, initial, **options):
            self.initial, self.options = initial, options

        def capabilities(self):
            return {}

    monkeypatch.setitem(sys.modules, "transformer.gpu_metal", SimpleNamespace(Backend=FakeBackend))
    monkeypatch.setenv("ARM64_TRANSFORMER_METAL_POLICY", "/untrusted-ambient-policy")
    backend = library_metal.create_backend("initial", max_batch=16)
    assert backend.options == dict(kernel_policy=reference(), max_batch=16)
    assert backend.shader_source.name == "precision-normalization-v1.metal"
    assert backend.capabilities()["optimized_kernel_claim"] is False
    with pytest.raises(ValueError, match="fixes the reference"):
        library_metal.create_backend("initial", kernel_policy=reference())
    monkeypatch.setattr(library_metal, "SHADER_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="shader changed"):
        library_metal.create_backend("initial")
    tree = ast.parse((Path(__file__).resolve().parents[1] / "transformer/gpu_metal.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Backend")
    assignment = next(
        n
        for n in cls.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "shader_source" for t in n.targets)
    )
    assert isinstance(assignment.value, ast.Constant) and assignment.value.value is None
