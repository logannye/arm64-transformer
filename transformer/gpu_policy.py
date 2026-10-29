"""Immutable Metal dispatch policies; v2 identities remain unchanged."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path


def reference():
    return {
        "schema": 2,
        "dense": dict.fromkeys(("forward", "dx", "dw")),
        "norm": None,
        "attention": None,
        "embedding": None,
        "fusion": {"epilogue": False, "adam_finite": False},
    }


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def with_qkv(value):
    """Upgrade explicitly without changing the identity of an existing v2 policy."""
    result = validate(value)
    if result["schema"] == 2:
        result.update(schema=3, qkv={"forward": None, "dx": None})
    return result


def validate(value):
    if (
        not isinstance(value, dict)
        or set(value) != (set(reference()) | ({"qkv"} if value.get("schema") == 3 else set()))
        or type(value["schema"]) is not int
        or value["schema"] not in (2, 3)
    ):
        raise ValueError("Invalid Metal policy schema/keys")
    if not isinstance(value["dense"], dict) or set(value["dense"]) != {"forward", "dx", "dw"}:
        raise ValueError("Invalid dense policy")

    def fields(item, choices):
        if item is None:
            return
        if not isinstance(item, dict) or set(item) != set(choices):
            raise ValueError("Invalid policy fields")
        for name, allowed in choices.items():
            if (
                type(item[name]) not in (int, str)
                or isinstance(item[name], bool)
                or item[name] not in allowed
            ):
                raise ValueError(f"Invalid policy value: {name}")

    for item in value["dense"].values():
        fields(item, {"m": (16, 32), "n": (32, 64), "k": (16, 32), "sg": (2, 4)})
    if value["schema"] == 3:
        if not isinstance(value["qkv"], dict) or set(value["qkv"]) != {"forward", "dx"}:
            raise ValueError("Invalid protected QKV policy")
        for item in value["qkv"].values():
            fields(item, {"m": (16, 32), "n": (32, 64), "k": (16, 32)})
    fields(value["norm"], {"sg": (1, 2, 4), "vec": (1, 4)})
    fields(
        value["attention"],
        {
            "rows": (1, 2),
            "sg": (1, 2),
            "stage": ("register", "threadgroup"),
            "backward": ("saved", "recompute"),
        },
    )
    fields(value["embedding"], {"block": (64, 128)})
    if (
        not isinstance(value["fusion"], dict)
        or set(value["fusion"]) != {"epilogue", "adam_finite"}
        or any(type(v) is not bool for v in value["fusion"].values())
    ):
        raise ValueError("Invalid fusion policy")
    return json.loads(canonical(value))


def bundle(value=None, width=None):
    if value is None and os.environ.get("ARM64_TRANSFORMER_METAL_POLICY"):
        value = json.loads(Path(os.environ["ARM64_TRANSFORMER_METAL_POLICY"]).read_text())
    if value is None:
        value = reference()
    if "widths" in value:
        value = value["widths"][str(width)]
    if "training" not in value:
        value = {"training": value, "inference": value}
    if set(value) != {"training", "inference"}:
        raise ValueError("Invalid Metal policy bundle")
    return {name: validate(policy) for name, policy in value.items()}


def controls(policy, *, version=None):
    p = validate(policy)
    result = []
    for direction in ("forward", "dx", "dw"):
        d = p["dense"][direction]
        result.extend([d[k] for k in ("m", "n", "k", "sg")] if d else [0, 0, 0, 0])
    n, a, e, f = p["norm"], p["attention"], p["embedding"], p["fusion"]
    result.extend([n["sg"], n["vec"]] if n else [0, 0])
    result.extend(
        [a["rows"], a["sg"], int(a["stage"] == "threadgroup"), int(a["backward"] == "recompute")]
        if a
        else [0, 0, 0, 0]
    )
    result.extend([e["block"] if e else 0, int(f["epilogue"]), int(f["adam_finite"])])
    if version == 3 or p["schema"] == 3:
        for direction in ("forward", "dx"):
            q = p.get("qkv", {}).get(direction)
            result.extend([q[k] for k in ("m", "n", "k")] if q else [0, 0, 0])
    return result


def defines(policy):
    p = validate(policy)
    definitions = {}
    for direction, prefix in (("forward", "FWD"), ("dx", "DX"), ("dw", "DW")):
        d = p["dense"][direction] or {"m": 16, "n": 32, "k": 16, "sg": 2}
        definitions.update({f"V2_{prefix}_{k.upper()}": v for k, v in d.items()})
    n = p["norm"] or {"sg": 1, "vec": 1}
    definitions.update({f"V2_NORM_{k.upper()}": v for k, v in n.items()})
    a = p["attention"] or {"rows": 1, "sg": 1, "stage": "register", "backward": "saved"}
    definitions.update(
        V2_ATT_ROWS=a["rows"],
        V2_ATT_SG=a["sg"],
        V2_ATT_STAGE=int(a["stage"] == "threadgroup"),
        V2_ATT_RECOMPUTE=int(a["backward"] == "recompute"),
    )
    definitions["V2_EMBED_BLOCK"] = (p["embedding"] or {"block": 64})["block"]
    if p["schema"] == 3:
        for direction, prefix in (("forward", "FWD"), ("dx", "DX")):
            q = p["qkv"][direction] or {"m": 16, "n": 32, "k": 16}
            definitions.update({f"V3_QKV_{prefix}_{k.upper()}": v for k, v in q.items()})
    return "".join(
        f"#define {name} {value}\n" for name, value in sorted(definitions.items())
    ).encode()


def candidate_inventory():
    """Complete prospective grid; implementation/feature filtering precedes scoring."""
    rows = []
    for direction in ("forward", "dx", "dw"):
        for m, n, k, sg in itertools.product((16, 32), (32, 64), (16, 32), (2, 4)):
            rows.append(
                {
                    "id": f"dense-{direction}-{m}-{n}-{k}-{sg}",
                    "family": f"dense_{direction}",
                    "value": {"m": m, "n": n, "k": k, "sg": sg},
                }
            )
    for sg, vec in itertools.product((1, 2, 4), (1, 4)):
        rows.append({"id": f"norm-{sg}-{vec}", "family": "norm", "value": {"sg": sg, "vec": vec}})
    for q, sg, stage, back in itertools.product(
        (1, 2), (1, 2), ("register", "threadgroup"), ("saved", "recompute")
    ):
        rows.append(
            {
                "id": f"attention-{q}-{sg}-{stage}-{back}",
                "family": "attention",
                "value": {"rows": q, "sg": sg, "stage": stage, "backward": back},
            }
        )
    for block in (64, 128):
        rows.append({"id": f"embedding-{block}", "family": "embedding", "value": {"block": block}})
    for epilogue, finite in ((True, False), (False, True), (True, True)):
        rows.append(
            {
                "id": f"fusion-{int(epilogue)}-{int(finite)}",
                "family": "fusion",
                "value": {"epilogue": epilogue, "adam_finite": finite},
            }
        )
    return rows


def apply_candidate(policy, candidate):
    value = validate(policy)
    family = candidate["family"]
    if family.startswith("dense_"):
        value["dense"][family.removeprefix("dense_")] = candidate["value"]
    else:
        value[family] = candidate["value"]
    return validate(value)
