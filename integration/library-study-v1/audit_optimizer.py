"""Audit explicit optimizer control adapters with the unchanged frozen raw code.

Only the legacy auditor's provenance dependency is supplied with merged-role
validation for adapter workers. Its numeric/checkpoint/schedule checks and pair
comparison bytecode are unchanged. Legacy worker records use their old auditor.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import FunctionType

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


A = load("optimizer_audit_authority", HERE / "authority.py")
C = A.C
ORIGINS = load("optimizer_audit_origins", HERE / "audit.py")


def validate_exports(value):
    expected = {r["path"]: r for s in value["saved"] for r in s["files"].values()}
    proofs = value["control_adapter"]["writer_proofs"]
    C.require(
        len(proofs) == len(expected) and {p["binding"]["path"] for p in proofs} == set(expected),
        "Optimizer export proof inventory differs",
    )
    for proof in proofs:
        binding = proof["binding"]
        C.require(
            binding == expected[binding["path"]]
            and proof["policy"] == "fsync_then_two_stable_exact_content_bindings"
            and proof["fsync"] is True
            and type(proof["stable_reads"]) is int
            and proof["stable_reads"] == 2
            and type(proof["metadata_retries"]) is int
            and 0 <= proof["metadata_retries"] < 3
            and len(proof["metadata_failures"]) == proof["metadata_retries"]
            and all(
                e
                in {
                    "Bound file changed during read",
                    "Metadata changed between independently stable reads",
                }
                for e in proof["metadata_failures"]
            )
            and proof["expected_sha256"] == binding["sha256"]
            and proof["expected_bytes"] == binding["bytes"],
            "Optimizer export lacks exact durable bytes",
        )
        C.bound(binding)


def execute(args):
    C.require(
        sys.flags.isolated and sys.flags.dont_write_bytecode,
        "Use independent optimizer auditor -I -B",
    )
    C.require(
        not any(n.split(".", 1)[0] in {"transformer", "torch", "mlx"} for n in sys.modules),
        "Optimizer auditor requires fresh project/framework imports",
    )
    roles = dict(
        control=A.source_role(HERE.parents[1], args.control_revision, control=True),
        math=A.source_role(args.math_root, C.MATH_REVISION),
    )
    capture = C.read(args.capture, args.capture_sha256)
    inventory = C.read(C.bound(capture["runtime_inventory"]))
    sys.path.insert(0, roles["math"]["root"])
    source = (
        Path(roles["math"]["root"]) / "integration/trained-precision/audit_library_optimizer.py"
    )
    C.require(
        C.binding(source)["sha256"]
        == roles["math"]["files"][source.relative_to(roles["math"]["root"]).as_posix()],
        "Unbound frozen optimizer auditor",
    )
    legacy = load("frozen_optimizer_raw_auditor", source)

    def worker(path, expected, capture_path, capture_sha):
        value = C.read(path, expected)
        adapter = value.get("control_adapter")
        if adapter is None:
            return legacy.audit_worker(path, expected, capture_path, capture_sha)
        C.require(
            adapter["schema"] == "frozen-optimizer-control-adapter-v1"
            and adapter["roles"] == roles
            and adapter["overridden_dependencies"] == ["REPLAY", "SUITE", "save_state"]
            and adapter["entrypoint"] == C.binding(HERE / "optimizer_adapter.py")
            and adapter["mathematical_implementation"]
            == C.binding(
                Path(roles["math"]["root"]) / "integration/trained-precision/library_optimizer.py"
            ),
            "Optimizer control adapter source differs",
        )
        validate_exports(value)
        ORIGINS.validate_validation(
            adapter["runtime_validation_final"],
            roles,
            capture,
            inventory,
            pid=value["pid"],
            final=True,
        )

        class Provenance:
            def __getattr__(self, name):
                return getattr(legacy.PROVENANCE, name)

            def audit_origins(self, proof, engine, retained_capture, runtime, receipt):
                C.require(
                    retained_capture == capture and runtime == inventory,
                    "Optimizer audit capture differs",
                )
                ORIGINS.validate_origins(proof, roles, capture, inventory, engine, pid=value["pid"])

        namespace = dict(legacy.audit_worker.__globals__, PROVENANCE=Provenance())
        function = FunctionType(legacy.audit_worker.__code__, namespace)
        return function(path, expected, capture_path, capture_sha)

    function = FunctionType(
        legacy.audit_pair.__code__, dict(legacy.audit_pair.__globals__, audit_worker=worker)
    )
    result = function(
        (args.first, args.first_sha256),
        (args.second, args.second_sha256),
        (args.restart, args.restart_sha256),
        args.capture,
        args.capture_sha256,
    )
    C.require(
        "torch.nn.precise_fp32" not in sys.modules and "mlx.core" not in sys.modules,
        "Candidate APIs replaced independent optimizer audit",
    )
    A.verify_role(roles["control"], control=True)
    A.verify_role(roles["math"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-revision", required=True)
    parser.add_argument("--math-root", type=Path, required=True)
    for name in ("first", "second", "restart", "capture"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    print(json.dumps(execute(parser.parse_args()), sort_keys=True, allow_nan=False))
