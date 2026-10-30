"""Pure prospective inventory of frozen-CPU calls made by owned admission.

This schedules provenance checks only. It neither creates operands nor opens
either admission stream. Every numerical call still uses the unchanged bridge.
"""

from __future__ import annotations


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cpu_operation_inventory(manifest):
    """Declare every CPU Torch invocation, including cross-engine reference work."""
    require(manifest.get("protocol") == "gpu-v4-owned-five-engine-v1", "Wrong CPU proof contract")
    caches, result = {}, {}
    for gate in manifest["gate_inventory"]:
        engine, width, family, *parts = gate["id"].split("/")
        suffix = "/".join(parts)
        cache = caches.setdefault(width, set())
        operations = []

        def add(mode, count=1, rejection=None):
            operations.extend({"mode": mode, "expected_rejection": rejection} for _ in range(count))

        cpu = engine == "cpu_torch"
        if family == "full" and cpu:
            add("forward_gradient")
        elif family == "common_state":
            if suffix == "evolved" and "evolved" not in cache:
                add("advance", 3)
                cache.add("evolved")
            if cpu:
                add("forward_gradient")
        elif family == "adam":
            source, count = suffix.split("/")
            if source not in cache:
                if source == "from-cpu_torch":
                    add("forward_gradient")
                cache.add(source)
            if cpu:
                add("adam", int(count.removeprefix("step-")))
        elif family == "trajectory":
            for _ in range(3):
                for _ in range(2 if cpu else 1):
                    add("forward_gradient")
                    add("advance")
        elif family == "control" and cpu:
            if suffix in (
                "causal_prefix",
                "sequence_isolation",
                "repeat_embedding_gradient",
                "repeat_forward",
            ):
                add("forward_gradient", 2)
            elif suffix == "repeat_training_same_process":
                add("repeat_training")
            elif suffix == "repeat_training_fresh_process":
                for _ in range(2):
                    add("advance", 6)
                    add("forward_gradient")
            elif suffix == "checkpoint_first_tie":
                add("checkpoint_first_tie")
            elif suffix in ("state_restore", "checkpoint_replay", "warmup_restore"):
                if suffix == "checkpoint_replay":
                    add("advance", 3)
                add("forward_gradient")
                add("advance", 2)
                add("forward_gradient")
            elif suffix in ("partial_batch", "tail_bounds"):
                add("forward_gradient", 2)
            elif suffix == "finite_state":
                add("advance", 2)
            elif suffix in ("input_immutability", "device_execution", "pending_work_cleanup"):
                add("forward_gradient")
            elif suffix == "malformed_input":
                add("forward_gradient", rejection="Out-of-range CPU tokens")
                add("forward_gradient", rejection="Malformed CPU token batch")
            elif suffix == "runtime_failure":
                add("forward_gradient", rejection="Malformed CPU parameters")
            else:
                require(suffix == "output_collision", "Unscheduled CPU control: " + suffix)
        result[gate["id"]] = operations
    require(len(result) == len(manifest["gate_inventory"]), "Duplicate CPU gate inventory")
    return result


def verify_cpu_operation_records(manifest, gates):
    expected = cpu_operation_inventory(manifest)
    require([g["id"] for g in gates] == list(expected), "CPU proof gate inventory changed")
    proofs = []
    for gate in gates:
        records = gate.get("owned_cpu_operations")
        require(isinstance(records, list), "Gate omits its owned CPU operation inventory")
        require(
            [{k: row.get(k) for k in ("mode", "expected_rejection")} for row in records]
            == expected[gate["id"]],
            "Missing, reordered, or altered owned CPU operation: " + gate["id"],
        )
        for row in records:
            require(
                set(row) == {"mode", "expected_rejection", "proof"}, "Unexpected CPU proof fields"
            )
            proof = row["proof"]
            require(
                isinstance(proof, dict) and set(proof) == {"path", "sha256"},
                "Malformed CPU proof binding",
            )
            proofs.append(proof)
    require(
        len({row["path"] for row in proofs}) == len(proofs), "Duplicate owned CPU operation proof"
    )
    return sorted(proofs, key=lambda row: row["path"])
