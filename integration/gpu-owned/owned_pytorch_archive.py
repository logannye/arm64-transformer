"""Retain and audit portable MPS conformance evidence without importing a framework.

Original absolute paths are identity labels only during audit. Every file read by
that audit is an authenticated regular member of the supplied evidence directory.
This does not select a new framework or suite inventory, or admit formal fits.
"""

import hashlib
import re
import shlex
import shutil
from pathlib import Path

SCHEMA = "gpu-owned-mps-portable-evidence-v1"
SELF = "integration/gpu-owned/owned_pytorch_archive.py"
MAX_METADATA_BYTES = 256 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def absolute(value):
    require(isinstance(value, str), "Missing original absolute identity")
    path = Path(value)
    require(
        path.is_absolute() and str(path) == value and ".." not in path.parts,
        "Noncanonical original absolute identity",
    )
    return path


def relative(value):
    require(isinstance(value, str) and value, "Missing archive member")
    path = Path(value)
    require(
        not path.is_absolute()
        and path.parts
        and path.as_posix() == value
        and all(part not in (".", "..") for part in path.parts),
        "Unsafe archive member",
    )
    return path


def raw(root, row):
    require(set(row) == {"path", "sha256", "bytes"}, "Malformed portable binding")
    path = Path(root) / relative(row["path"])
    require(
        type(row["bytes"]) is int
        and row["bytes"] >= 0
        and isinstance(row["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]),
        "Malformed portable file identity",
    )
    require(
        not path.is_symlink()
        and path.resolve(strict=True) == path
        and path.is_file()
        and path.stat().st_size == row["bytes"]
        and digest(path) == row["sha256"],
        "Portable evidence bytes changed",
    )
    return path


def retain(plan, plan_path, root, record, bind, limits):
    """Copy bounded task-owned evidence after successful live postflight checks."""
    root = Path(root).resolve(strict=True)
    target = root / "portable"
    target.mkdir(exist_ok=False)
    evidence = dict(
        schema=SCHEMA,
        original_plan=bind(plan_path),
        original_output=str(root),
        authority={},
        helpers={},
        source_inputs={},
        headers={},
        cache={},
        descendants=[],
    )
    copied = 0

    def keep(path, name, expected=None):
        nonlocal copied
        before = bind(path, expected)
        destination = root / relative(name)
        require(destination.is_relative_to(target), "Copy escaped portable directory")
        copied += before["bytes"]
        require(copied <= limits["output_bytes"], "Portable evidence exceeds output bound")
        destination.parent.mkdir(parents=True, exist_ok=True)
        require(not destination.exists(), "Duplicate portable output")
        shutil.copyfile(path, destination)
        after = bind(destination)
        bind(path, before)
        require(
            after["sha256"] == before["sha256"] and after["bytes"] == before["bytes"],
            "Portable copy changed",
        )
        return dict(path=name, sha256=after["sha256"], bytes=after["bytes"])

    evidence["plan"] = keep(plan_path, "portable/plan.json", evidence["original_plan"])
    evidence["source_manifest"] = keep(
        plan["source_manifest"]["path"], "portable/source-manifest.json", plan["source_manifest"]
    )
    for name, row in plan["authority_inputs"].items():
        evidence["authority"][name] = keep(row["path"], f"portable/authority/{name}.json", row)
    for name, row in plan["retained_helpers"].items():
        evidence["helpers"][name] = keep(
            row["path"], f"portable/helpers/{Path(row['path']).name}", row
        )
    for name, sha in plan["fixture_preparation"]["source_inputs"].items():
        row = keep(Path(plan["source_root"]) / name, "portable/source/" + name)
        require(row["sha256"] == sha, "Fixture source changed before retention")
        evidence["source_inputs"][name] = row
    require(copied <= MAX_METADATA_BYTES, "Portable authority/source metadata exceeds bound")
    for index, (path, sha) in enumerate(plan["fixture_preparation"]["include_inputs"].items()):
        row = keep(path, f"portable/headers/{index:02d}-{Path(path).name}")
        require(row["sha256"] == sha, "Fixture header changed before retention")
        evidence["headers"][path] = row
    for name, item in record["fixture_after"]["files"].items():
        row = keep(Path(plan["cache"]) / relative(name), "portable/cache/" + name)
        require({k: row[k] for k in ("sha256", "bytes")} == item, "Fixture output changed")
        evidence["cache"][name] = row
    evidence["launcher"] = keep(root / "owned-suite-python", "portable/owned-suite-python")
    for row in record["descendants"]:
        name = Path(row["path"]).relative_to(root).as_posix()
        require(
            name.startswith("descendant-") and name.endswith("/receipt.json"),
            "Foreign child receipt",
        )
        evidence["descendants"].append(keep(row["path"], "portable/" + name, row))
    require(
        sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) <= limits["output_bytes"],
        "Retained suite output exceeds bound",
    )
    return evidence


def source_authority(value, selected):
    combined = selected.get("combined_patch_sha256")
    if combined is None:
        require(len(selected["patches"]) == 1, "Missing combined source patch authority")
        combined = selected["patches"][0]
    require(
        value.get("source_revision", value.get("base")) == selected["base"]
        and value.get("patch_sha256") == combined,
        "Source manifest selects another build",
    )
    files = value.get("files")
    require(isinstance(files, dict) and files, "Missing complete source manifest")
    for name, item in files.items():
        relative(name)
        require(
            isinstance(item, dict)
            and isinstance(item.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
            and type(item.get("bytes")) is int
            and item["bytes"] >= 0,
            "Invalid complete source member",
        )
    return {name: item["sha256"] for name, item in files.items()}


def snapshot(value, cache, policy):
    require(
        value["path"] == cache
        and value["max_cache_bytes"] == policy.MAX_CACHE_BYTES
        and value["max_cache_files"] == policy.MAX_CACHE_FILES
        and value["min_free_bytes"] == policy.MIN_FREE_BYTES,
        "Fixture limits changed",
    )
    for key in ("bytes", "file_count", "free_bytes"):
        require(type(value[key]) is int and value[key] >= 0, "Invalid fixture resource observation")
    files = value["files"]
    require(
        isinstance(files, dict) and len(files) == value["file_count"], "Fixture file count differs"
    )
    for name, item in files.items():
        relative(name)
        require(
            set(item) == {"bytes", "sha256"}
            and type(item["bytes"]) is int
            and item["bytes"] >= 0
            and isinstance(item["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]),
            "Invalid fixture member",
        )
    require(sum(item["bytes"] for item in files.values()) == value["bytes"], "Fixture bytes differ")
    policy.require_bounds(value)
    return files


def audit(root, record, manifest, runtime_receipt, producer, policy):
    """Require every portable input and recheck identities, launch and descendants."""
    root = Path(root).resolve(strict=True)
    evidence = record.get("portable_evidence", {})
    require(
        set(evidence)
        == {
            "schema",
            "original_plan",
            "original_output",
            "plan",
            "authority",
            "source_manifest",
            "helpers",
            "source_inputs",
            "headers",
            "cache",
            "descendants",
            "launcher",
        }
        and evidence["schema"] == SCHEMA,
        "Missing complete portable MPS evidence",
    )
    seen, retained = set(), []

    def member(row):
        require(
            row["path"].startswith("portable/") and row["path"] not in seen,
            "Duplicate or foreign portable member",
        )
        seen.add(row["path"])
        retained.append(row)
        return raw(root, row)

    def read(row):
        require(row["bytes"] <= MAX_METADATA_BYTES, "Portable metadata exceeds bound")
        return producer.read(member(row))

    plan = read(evidence["plan"])
    original_plan = evidence["original_plan"]
    require(
        evidence["plan"]["sha256"] == original_plan["sha256"] == record["plan_sha256"]
        and evidence["plan"]["bytes"] == original_plan["bytes"],
        "Plan identity changed",
    )
    plan_path, original_output = (
        absolute(original_plan["path"]),
        absolute(evidence["original_output"]),
    )
    binding = manifest["bindings"]
    project = runtime_receipt["project"]
    require(project["files"] == binding["source"], "Project authority differs")
    for name in (producer.PRODUCER, producer.POLICY, SELF):
        require(
            binding["source"].get(name) == digest(Path(__file__).resolve().parents[2] / name),
            "Portable auditor or producer is outside selected project",
        )
    require(
        plan["schema"] == producer.PLAN_SCHEMA
        and plan["protocol"] == producer.PROTOCOL
        and plan["limits"] == producer.LIMITS
        and plan["formal_fits"] == 0
        and plan["formal_admission_opened"] is False,
        "Plan contract differs",
    )
    require(
        absolute(plan["producer"]["path"]) == absolute(project["root"]) / producer.PRODUCER
        and plan["producer"]["sha256"] == binding["source"][producer.PRODUCER],
        "Producer authority differs",
    )
    inventory, inventory_sha = producer.inventory_spec(plan)
    require(
        plan["inventory"]["sha256"] == inventory_sha == binding["source"].get(inventory)
        and absolute(plan["inventory"]["path"]) == absolute(project["root"]) / inventory,
        "Plan suite inventory differs from the selected contract",
    )
    require(
        set(evidence["authority"]) == {"inventory", "selection", "receipt"},
        "Incomplete runtime authority",
    )
    authorities = {}
    for name, row in evidence["authority"].items():
        authorities[name] = read(row)
        expected = plan["authority_inputs"][name]
        require(all(row[k] == expected[k] for k in ("sha256", "bytes")), "Plan authority differs")
    runtime = authorities["inventory"]
    selected = authorities["selection"]
    # Authenticate the explicit successor selection without reading any original
    # producer path. The source-pinned contract lives beside this current auditor.
    producer.suite_inventory(plan, selected)
    require(
        runtime["files"] == binding["runtime"]
        and runtime["layout"] == binding["runtime_layout"]
        and authorities["receipt"] == runtime_receipt
        and selected == manifest["owned_frameworks"]["frameworks"],
        "Archived runtime differs",
    )
    require(
        evidence["authority"]["receipt"]["sha256"] == record["runtime_receipt_sha256"],
        "Runtime receipt hash differs",
    )
    sources_record = read(evidence["source_manifest"])
    require(
        all(
            evidence["source_manifest"][k] == plan["source_manifest"][k]
            for k in ("sha256", "bytes")
        )
        and record["source_manifest"] == plan["source_manifest"],
        "Complete source manifest differs",
    )
    sources = source_authority(sources_record, selected["pytorch"])
    environment = plan["environment"]
    require(
        all(environment.get(k) == v for k, v in producer.PINS.items())
        and not any(k in environment for k in producer.ABSENT)
        and all(environment.get(k) == v for k, v in policy.CONTROLS.items())
        and not any(k in environment for k in policy.ABSENT_FLAGS)
        and environment.get("TORCH_EXTENSIONS_DIR") == plan["cache"],
        "Frozen execution environment differs",
    )
    require(
        not any(k.startswith("PYTHON") and k not in producer.PINS for k in environment),
        "Unrecognized archived Python override",
    )
    require(set(evidence["helpers"]) == set(producer.HELPERS), "Missing retained result helper")
    extras = {}
    for name, (filename, sha) in producer.HELPERS.items():
        row = evidence["helpers"][name]
        member(row)
        original = plan["retained_helpers"][name]
        require(
            Path(original["path"]).name == filename
            and row["sha256"] == original["sha256"] == sha
            and row["bytes"] == original["bytes"],
            "Retained result helper differs",
        )
        extras[original["path"]] = sha
    require(
        Path(plan["retained_helpers"]["driver_policy"]["path"]).parent
        == Path(plan["retained_helpers"]["driver"]["path"]).parent,
        "Driver policy was not adjacent to its retained helper",
    )
    require(
        set(evidence["source_inputs"]) == set(policy.SOURCE_INPUTS), "Incomplete fixture source"
    )
    for name, row in evidence["source_inputs"].items():
        path = member(row)
        require(
            row["sha256"] == sources[name]
            and row["bytes"] == sources_record["files"][name]["bytes"],
            "Fixture source differs",
        )
        if name == "test/test_mps.py":
            require(
                record["test_source_sha256"] == {name: row["sha256"]},
                "Selected test source differs",
            )
            policy.extension_ast(path)
            require(
                producer.subprocess_contract(path) == plan["subprocess_contract"],
                "Conformance child scripts differ",
            )
    fixture = plan["fixture_preparation"]
    require(
        sum(row["bytes"] for row in retained) <= MAX_METADATA_BYTES,
        "Portable authority/source metadata exceeds bound",
    )
    require(
        fixture["source_inputs"] == {n: r["sha256"] for n, r in evidence["source_inputs"].items()}
        and fixture["runtime_manifest_sha256"] == selected["pytorch"]["runtime_manifest_sha256"]
        and fixture["controls"] == policy.CONTROLS
        and fixture["conformance_ast_sha256"] == policy.CONFORMANCE_AST_SHA,
        "Fixture preparation differs",
    )
    require(set(evidence["headers"]) == set(fixture["include_inputs"]), "Missing fixture headers")
    for original, row in evidence["headers"].items():
        absolute(original)
        member(row)
        require(row["sha256"] == fixture["include_inputs"][original], "Fixture header differs")
    cache = str(absolute(plan["cache"]))
    before = snapshot(record["fixture_before"], cache, policy)
    after = snapshot(record["fixture_after"], cache, policy)
    require(before == snapshot(fixture["before"], cache, policy), "Preparation cache differs")
    require(set(evidence["cache"]) == set(after), "Incomplete retained fixture cache")
    for name, row in evidence["cache"].items():
        member(row)
        require({k: row[k] for k in ("sha256", "bytes")} == after[name], "Fixture cache differs")
    marker = producer.read(root / evidence["cache"]["cache-identity.json"]["path"])
    require(
        set(marker)
        == {
            "schema",
            "runtime_manifest_sha256",
            "source_inputs",
            "conformance_ast_sha256",
            "controls",
            "ninja",
            "compiler",
            "include_inputs",
            "toolchain_environment",
        }
        and marker == {key: fixture[key] for key in marker}
        and marker["schema"] == policy.SCHEMA
        and evidence["cache"]["cache-identity.json"]["sha256"] == plan["cache_identity"]["sha256"],
        "Fixture toolchain/cache identity differs",
    )
    for key in ("compiler", "ninja"):
        tool = marker[key]
        absolute(tool["path"])
        require(
            set(tool) == {"path", "sha256"}
            and isinstance(tool["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", tool["sha256"]),
            "Invalid fixture tool identity",
        )
    require(marker["compiler"]["path"] == policy.CONTROLS["CXX"], "Fixture compiler differs")
    loaded = record["stock_fixture_loaded"]
    native_name = "mps_iter_conformance/mps_iter_conformance.so"
    ninja_name = "mps_iter_conformance/build.ninja"
    require(
        loaded["path"] == str(Path(cache) / native_name)
        and loaded["sha256"] == after[native_name]["sha256"]
        and loaded["build_ninja_sha256"] == after[ninja_name]["sha256"],
        "Loaded fixture differs",
    )
    ninja = (root / evidence["cache"][ninja_name]["path"]).read_text()
    require(ninja == loaded["build_ninja"], "Generated build commands differ")
    for line in ninja.splitlines():
        if line.startswith(("cflags =", "post_cflags =")):
            require(
                not any(v.startswith("-g") and v != "-g0" for v in line.split()),
                "Fixture enabled debug information",
            )
    extras[loaded["path"]] = loaded["sha256"]
    process = record["process"]
    require(
        type(record["worker_pid"]) is int
        and type(record["started_unix"]) in (int, float)
        and process["started_unix"] <= record["started_unix"] <= process["ended_unix"],
        "Worker interval differs from supervisor",
    )
    require(
        record["worker_pid"] == process["pid"]
        and process["command"]
        == producer.worker_command(
            plan, plan_path, record["plan_sha256"], original_output, runtime
        ),
        "Actual parent launch differs from frozen isolated interpreter",
    )
    launcher = member(evidence["launcher"]).read_text()
    command = producer.child_command(
        plan, plan_path, record["plan_sha256"], original_output, process["pid"], runtime=runtime
    )
    require(
        launcher == "#!/bin/sh\nexec " + shlex.join(command) + ' "$@"\n',
        "Conformance child launcher differs",
    )
    require(
        record["bytecode_prefix"] == str(original_output / "unused-bytecode-cache"),
        "Parent bytecode routing differs",
    )
    children = []
    invocations = plan["subprocess_contract"]["invocations"]
    require(
        len(evidence["descendants"]) == len(invocations) == 2 and len(record["descendants"]) == 2,
        "Missing or duplicate descendants",
    )
    for row, original, invocation in zip(
        evidence["descendants"], record["descendants"], invocations, strict=True
    ):
        child = read(row)
        name = "descendant-" + invocation["name"] + "/receipt.json"
        require(
            row["path"] == "portable/" + name
            and original["path"] == str(original_output / name)
            and all(row[k] == original[k] for k in ("sha256", "bytes")),
            "Child binding differs",
        )
        require(child["invocation"] == invocation, "Reordered child invocation")
        producer.validate_descendant_proof(
            child,
            plan,
            record["plan_sha256"],
            process["pid"],
            str((original_output / name).parent / "unused-bytecode-cache"),
            runtime,
            runtime_receipt,
            sources,
        )
        children.append(child)
    producer.validate_descendant_intervals(
        children, process["pid"], (process["started_unix"], process["ended_unix"])
    )
    for phase in ("before", "after"):
        native = record["native_origin_verification"][phase]
        producer.validate_native_proof(native, runtime, plan["runtime_root"])
        producer.validate_loaded_proof(
            record["loaded_python_files_" + phase],
            native,
            runtime,
            plan["runtime_root"],
            project,
            plan["source_root"],
            sources,
            extras,
        )
    actual = set()
    for path in (root / "portable").rglob("*"):
        require(not path.is_symlink(), "Symlink in portable evidence")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        else:
            require(path.is_dir(), "Special portable member")
    require(actual == seen, "Unbound or missing portable evidence member")
    require(
        sum(row["bytes"] for row in retained) <= producer.LIMITS["output_bytes"],
        "Portable evidence exceeds the full-suite output bound",
    )
    for row in retained:
        raw(root, row)
    return dict(
        schema=SCHEMA,
        files=len(seen),
        descendants=len(children),
        archive_valid=True,
        formal_eligible=False,
    )
