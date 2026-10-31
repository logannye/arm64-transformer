"""Bind twice-reproduced development trajectories into a replay inventory.

No framework imports, training, timing, assessment stream or formal admission.
Require externally frozen receipt hashes and all parameter/moment state hashes.
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def checked(root, relative, expected):
    root = root.resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root) and sha(path) == expected, "Changed or escaped fixture")
    return path


def inventory(root, manifest_path, manifest_sha):
    require(sha(manifest_path) == manifest_sha, "Trajectory authority changed")
    manifest = json.loads(Path(manifest_path).read_text())
    policy = manifest.get("precision_policy", 1)
    require(
        type(policy) is int
        and (manifest["schema"], policy)
        in {("trained-precision-fresh-pairs-v1", 1), ("trained-precision-fresh-pairs-v2", 15)},
        "Unknown development trajectory policy",
    )
    require(
        [p["width"] for p in manifest["pairs"]] == [512, 1024],
        "Wrong development trajectory inventory",
    )
    cases, files, proofs = [], {}, []
    for pair in manifest["pairs"]:
        width, repeats = pair["width"], []
        for name in ("first", "second"):
            bound = pair[name]
            path = checked(root, bound["path"], bound["sha256"])
            result = json.loads(path.read_text())
            require(
                result["schema"]
                == f"trained-precision-fresh-development-v{1 if policy == 1 else 2}"
                and result.get("precision_policy", 1) == policy
                and result["complete"] is True
                and result["formal_eligible"] is False
                and result["admission_stream_opened"] is False
                and result["width"] == width
                and result["steps"] == 64
                and result["seed"] == 20261103
                and result["checkpoints"] == [0, 1, 8, 32, 64],
                "Wrong or incomplete development trajectory",
            )
            states = result["states"]
            require([s["step"] for s in states] == list(range(65)), "Missing optimizer states")
            require(
                all(set(s) == {"step", "parameters", "m", "v"} for s in states),
                "Incomplete optimizer state identity",
            )
            require(
                states[0]["parameters"] == result["training_inputs"]["initial.f32"]["sha256"],
                "Initial checkpoint changed",
            )
            relative = Path(bound["path"]).parent
            saved = result["saved"]
            require(set(saved) == {f"step-{n:03d}.f32" for n in (0, 1, 8, 32, 64)}, "Missing save")
            for step in (0, 1, 8, 32, 64):
                file = f"step-{step:03d}.f32"
                require(saved[file] == states[step]["parameters"], "Saved state differs")
                require(
                    (path.parent / file).stat().st_size == (8 * width**2 + 301 * width + 128) * 4,
                    "Wrong checkpoint size",
                )
            moments = {}
            if policy == 15:
                moments = result["saved_optimizer"]
                require(
                    set(moments)
                    == {f"step-{n:03d}-{key}.f32" for n in (0, 1, 8, 32, 64) for key in ("m", "v")},
                    "Missing saved optimizer state",
                )
                for step in (0, 1, 8, 32, 64):
                    for key in ("m", "v"):
                        filename = f"step-{step:03d}-{key}.f32"
                        require(moments[filename] == states[step][key], "Saved moment differs")
                        require(
                            (path.parent / filename).stat().st_size
                            == (8 * width**2 + 301 * width + 128) * 4,
                            "Wrong saved moment size",
                        )
            for file, digest in {
                **saved,
                **moments,
                "rows.i32": result["rows_sha256"],
                "schedule.i32": result["schedule_sha256"],
            }.items():
                key = (relative / file).as_posix()
                checked(root, key, digest)
                files[key] = digest
            require((path.parent / "rows.i32").stat().st_size == 32 * 33 * 4, "Wrong probe size")
            require(
                (path.parent / "schedule.i32").stat().st_size == 64 * 16 * 4,
                "Wrong training schedule size",
            )
            before, after = result["backend_before"], result["backend_after"]
            selected = "sme" if before["selected_kind"] == 1 else "neon"
            require(after["selected_kind"] == before["selected_kind"], "Backend changed")
            for backend in ("sme", "neon"):
                for direction in ("forward", "backward"):
                    key = f"{backend}_{direction}_calls"
                    dense_calls = 5 - bool(policy & 2) - bool(policy & 8)
                    expected = 64 * 16 * dense_calls if backend == selected else 0
                    require(after[key] - before[key] == expected, "Missing dense dispatch proof")
            files[bound["path"]] = bound["sha256"]
            repeats.append(result)
        require(repeats[0]["pid"] != repeats[1]["pid"], "Not separate processes")
        require(
            {k: v for k, v in repeats[0].items() if k != "pid"}
            == {k: v for k, v in repeats[1].items() if k != "pid"},
            "Fresh-process trajectory replay differs",
        )
        proofs.append({"width": width, "states_compared": 65, "receipts": pair})
        first = repeats[0]
        relative = Path(pair["first"]["path"]).parent
        prefix = "fresh-p15" if policy == 15 else "fresh"
        for step in (0, 1, 8, 32, 64):
            checkpoint = f"step-{step:03d}.f32"
            for sequence in (0, 16):
                cases.append(
                    {
                        "id": f"{prefix}-w{width}-step{step:03d}-sequence{sequence:02d}",
                        "width": width,
                        "checkpoint": (relative / checkpoint).as_posix(),
                        "checkpoint_sha256": first["saved"][checkpoint],
                        "rows": (relative / "rows.i32").as_posix(),
                        "rows_sha256": first["rows_sha256"],
                        "sequence": sequence,
                        "split": "development-training"
                        if sequence == 0
                        else "development-synthetic",
                        "checkpoint_origin": f"native-precision-policy-{policy}",
                        "historical_failures": [],
                    }
                )
    return {
        "schema": "trained-precision-development-v1",
        "scope": "fresh development trajectories; training/synthetic inputs only",
        "formal_eligible": False,
        "admission_stream_opened": False,
        "trajectory_manifest_sha256": manifest_sha,
        "trajectory_proofs": proofs,
        "files": files,
        "cases": sorted(cases, key=lambda c: c["id"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.root, args.manifest, args.manifest_sha256)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
