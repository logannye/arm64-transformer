"""Prospective maintained-library successor selection; no execution authority.

Printing this immutable plan imports no tensor framework and opens no operands.
The historical command routes reject this protocol. Formal dispatch stays closed
until a dedicated authenticated successor implementation and all gates exist.
"""

import argparse
import copy
import json
from pathlib import Path

from transformer.gpu_study import canonical, digest, identity, planned_manifest, require, write_new
from transformer.gpu_study_context import LIBRARY_ADMISSION_SEED

PROTOCOL = "gpu-v5-library-five-engine-v1"
SELECTION = "experiments/gpu-v5-library/framework-selection.json"
SELECTION_SHA256 = "18d987b4418778945ae48b0e19baa9f72782ea8c27ecd94fdbfe7dcf80070a4f"
CPU_REVISION = "bcf0d5deb25c468ac1751d61f05caf75d23fd07e"
REQUIRED_GATES = (
    "exact_runtime_source_binary_dependencies_and_api_derivations",
    "current_source_hosted_python_and_native_checks_all_widths",
    "complete_numpy2_mps_and_mlx_cpp_python_suites",
    "all_18_installed_api_primitive_cells_and_independent_raw_audits",
    "complete_exposed_trained_state_library_forward_every_gradient_replays",
    "complete_embedding_and_full_model_repeatability_two_processes_ten_executions",
    "fresh_same_distribution_cpu_anchors_all_widths",
    "all_width_common_state_and_identical_gradient_optimizer_accuracy",
    "all_18_optimizer_same_fresh_process_and_step32_restart_cells",
    "selected_native_policy_and_reference_metal_all_width_qualification",
    "all_retained_admission_groups_and_all_width_full_model_accuracy",
    "authenticated_successor_dispatch_and_checkpoint_auditing",
)


def framework_selection():
    path = Path(__file__).resolve().parents[1] / SELECTION
    require(digest(path) == SELECTION_SHA256, "Maintained-library framework selection changed")
    return json.loads(path.read_text())


def plan():
    result = planned_manifest()
    for gate in result["gate_inventory"]:
        if "/full/" in gate["id"]:
            gate["blocking"] = True
    result.update(
        protocol=PROTOCOL,
        admission_seed=LIBRARY_ADMISSION_SEED,
        cpu_revision=CPU_REVISION,
        gate_inventory_sha256=identity(result["gate_inventory"]),
        baseline_definition="explicit_maintained_library_precision_models",
        standard_release_claim=False,
        framework_selection=framework_selection(),
        model_selection={
            "cpu_torch": {"module": "torch.nn.precise_fp32", "stages": ["ln1", "qkv", "attention"]},
            "mps": {"module": "torch.nn.precise_fp32", "stages": ["ln1", "qkv", "attention"]},
            "mlx": {"module": "mlx.nn.precise_fp32", "stages": ["ln1", "qkv", "attention", "ff1"]},
            "cpu_arm64": {
                "width_policies": {
                    str(w): 15 if w >= 512 else 0 for w in (32, 64, 128, 256, 512, 1024)
                }
            },
            "metal": {
                "precision_policy": 1,
                "kernel_policy": "reference",
                "optimized_kernel_claim": False,
            },
        },
        required_gates=list(REQUIRED_GATES),
        retained_admission_group_count=len(result["gate_inventory"]),
        older_reserved_streams_opened=False,
        formal_execution_supported=False,
        formal_eligible=False,
        formal_fits=0,
        admission_stream_opened=False,
    )
    return copy.deepcopy(result)


def validate_plan(value):
    require(canonical(value) == canonical(plan()), "Maintained-library prospective plan changed")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    write_new(parser.parse_args().output, plan())
