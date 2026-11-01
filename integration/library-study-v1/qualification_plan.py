"""Build an exact, bounded raw re-audit schedule from completed development gates.

No numerical execution or reserved operands occur here. Collector completion
flags are preconditions, not acceptance: every scheduled raw audit must run.
"""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "library_qualification_contract", HERE / "contract.py"
)
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)
WIDTHS = (32, 64, 128, 256, 512, 1024)
FRAMEWORKS = ("cpu_torch", "mps", "mlx")
INVENTORIES = {
    "historical": "021f8e232863b437198c65e35c06dc6f95162cc07294a3f30714b60e6aeef18e",
    "fresh": "fdcc95dd1eb193b2c0115c50d0faca301f5487f75aa84418a8f84c000cc2fb48",
    "fresh-policy15": "5e0d799aa53b9ae8c065da520675c6161269b275e366408cb294c84c18c630ad",
}


def cell_inventory(rows, engines, widths=WIDTHS):
    expected = [(e, w) for e in engines for w in widths]
    C.require(
        all(type(r["width"]) is int for r in rows)
        and [(r["engine"], r["width"]) for r in rows] == expected,
        "Missing, duplicate, reordered or foreign qualification cell",
    )
    return rows


def load(binding):
    C.bound(binding)
    return C.read(binding["path"], binding["sha256"])


def reuse_context(binding, capture):
    """Authenticate the one pre-FF1 context; never transfer its CPU model gates."""
    if binding is None:
        return None
    value = load(binding)
    roles = value["roles"]
    C.require(
        set(value) == {"roles", "capture", "source_ci"}
        and roles["math"]["revision"] == "9f57a0d58011dbb27cf0cd8f7a4862fc538df65b"
        and roles["control"]["revision"] == "233822ee628c22a519dfbc212fc72192e3fc0962"
        and value["capture"]["sha256"]
        == "284a8f95e041558e017d3ccbf72244ce85d798d8615ada527f35932ecd2a5b54",
        "Unselected pre-FF1 evidence context",
    )
    spec = importlib.util.spec_from_file_location("ff1_current_authority", HERE / "authority.py")
    current = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(current)
    current.verify_role(roles["control"], control=True)
    source = Path(roles["control"]["root"]) / "integration/library-study-v1/authority.py"
    spec = importlib.util.spec_from_file_location("ff1_previous_authority", source)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    previous.verify_role(roles["math"])
    old, new = load(value["capture"]), load(capture)
    old_receipt, new_receipt = (load(c["runtime_receipt"]) for c in (old, new))
    math = current.source_role(new_receipt["project"]["root"], C.MATH_REVISION)
    C.require(
        old_receipt["project"] == {k: roles["math"][k] for k in ("root", "files")}
        and new_receipt["project"] == {k: math[k] for k in ("root", "files")}
        and old["runtime_identity_sha256"] == new["runtime_identity_sha256"]
        and load(old["runtime_inventory"]) == load(new["runtime_inventory"])
        and load(old["framework_selection"]) == load(new["framework_selection"]),
        "Pre-FF1 source/runtime bytes differ",
    )
    old_comparison = previous.C.comparison()
    old_comparison["model_selection"]["cpu_torch"]["stages"].append("ff1")
    C.require(
        old_comparison == C.original_comparison(),
        "Changes exceed the approved CPU FF1 selection",
    )
    replacements = {
        "integration/trained-precision/audit_replay.py": (
            (
                '(["ff1"] if engine == "mlx" else [])',
                '(["ff1"] if engine in {"cpu_torch", "mlx"} else [])',
            ),
        ),
        "integration/trained-precision/library_candidate.py": (
            (
                '"cpu_torch": ["ln1", "qkv", "attention"],',
                '"cpu_torch": ["ln1", "qkv", "attention", "ff1"],',
            ),
            ("precise_ff1=False)", 'precise_ff1=self.engine == "cpu_torch")'),
        ),
        "transformer/library_training.py": (
            ("precise_ff1=False,", 'precise_ff1=engine == "cpu_torch",'),
        ),
    }
    prefixes = (
        "transformer/",
        "integration/trained-precision/",
        "integration/gpu-owned/",
        "integration/precision-library-v1/",
        "native/",
    )
    old_files = {n: h for n, h in roles["math"]["files"].items() if n.startswith(prefixes)}
    new_files = {n: h for n, h in math["files"].items() if n.startswith(prefixes)}
    C.require(old_files.keys() == new_files.keys(), "Numerical source inventory changed")
    for name, digest in old_files.items():
        if name in replacements:
            text = (Path(roles["math"]["root"]) / name).read_text()
            for before, after in replacements[name]:
                C.require(text.count(before) == 1, "Ambiguous CPU FF1 correction")
                text = text.replace(before, after)
            C.require(text == (Path(math["root"]) / name).read_text(), "Unapproved model change")
        elif name == "native/metal/backend.mm":
            # Both campaigns select this already-qualified host from control.
            C.require(new_files[name] == roles["control"]["files"][name], "Metal host changed")
        else:
            C.require(new_files[name] == digest, "Unchanged-arm numerical source changed")
    return value


def previous_scope(selected, capture, reuse, kind, engine=None, library_api=False):
    if selected == capture:
        return False
    C.require(reuse is not None and selected == reuse["capture"], "Foreign audit capture")
    C.require(
        not (engine == "cpu_torch" and (library_api or kind in {"optimizer", "common_state"})),
        "Pre-FF1 CPU model evidence cannot qualify the revised CPU arm",
    )
    return True


def collector(binding, capture, *, native=False, reuse=None):
    value = load(binding)
    selected = value["capture"]
    previous = previous_scope(selected, capture, reuse, "collector")
    C.require(
        value["math_revision" if native else "project_revision"]
        == (reuse["roles"]["math"]["revision"] if previous else C.MATH_REVISION)
        and value["formal_eligible"] is False
        and type(value["formal_fits"]) is int
        and value["formal_fits"] == 0
        and value["admission_stream_opened"] is False,
        "Foreign development collector",
    )
    return value


def build(inputs, capture, inventories, raw_store, *, reuse=None):
    reuse_binding = reuse
    reuse = reuse_context(reuse, capture)
    C.require(
        set(inputs)
        == {
            "device",
            "optimizer",
            "repeatability",
            "common_state",
            "native_metal",
            "native_common_state",
        },
        "Complete development collector set required",
    )
    C.require(set(inventories) == set(INVENTORIES), "Complete exposed inventory set required")
    for name, row in inventories.items():
        C.require(row["inventory"]["sha256"] == INVENTORIES[name], "Exposed inventory changed")
        load(row["inventory"])
    jobs = []

    def add(
        name,
        kind,
        width,
        arguments,
        *,
        expected_numerics=True,
        selected_capture=capture,
        engine=None,
        library_api=False,
        **metadata,
    ):
        C.require(not any(j["name"] == name for j in jobs), "Duplicate raw audit job")
        jobs.append(
            dict(
                name=name,
                kind=kind,
                width=width,
                arguments=list(map(str, arguments)),
                expected_numerics=expected_numerics,
                previous_context=previous_scope(
                    selected_capture, capture, reuse, kind, engine, library_api
                ),
                **metadata,
            )
        )

    def capture_arguments(selected):
        return ["--capture", selected["path"], "--capture-sha256", selected["sha256"]]

    device = collector(inputs["device"], capture, reuse=reuse)
    C.require(
        device["library_exposed_model_gates_passed"] is True
        and device["ordinary_outcomes_retained"] is True,
        "Complete matched replay required",
    )
    foundation = load(device["foundation_gates"])
    for row in cell_inventory(foundation["primitive_cells"], FRAMEWORKS):
        selected = load(row["receipt"])["capture"]
        base = capture_arguments(selected)
        add(
            f"primitive-{row['engine']}-w{row['width']}",
            "primitive",
            row["width"],
            [
                "--receipt",
                row["receipt"]["path"],
                "--receipt-sha256",
                row["receipt"]["sha256"],
                *base,
                "--raw-store",
                raw_store,
            ],
            selected_capture=selected,
            engine=row["engine"],
        )
    expected_groups = [
        (inv, e, w, gradients, library)
        for inv in INVENTORIES
        for e in FRAMEWORKS
        for w in (512, 1024)
        for gradients in (False, True)
        for library in (False, True)
    ]
    rows = device["model_groups"]
    C.require(
        [(r["inventory"], r["engine"], r["width"], r["gradients"], r["library_api"]) for r in rows]
        == expected_groups
        and len(rows) == 72,
        "Matched exposed collection denominator changed",
    )
    for row in rows:
        inv, width = inventories[row["inventory"]], row["width"]
        mode = "gradient" if row["gradients"] else "forward"
        variant = "library" if row["library_api"] else "ordinary"
        name = f"{row['inventory']}-{row['engine']}-w{width}-{variant}-{mode}"
        C.require(row["name"] == name, "Foreign replay collection name")
        manifest = C.binding(Path(inputs["device"]["path"]).parent / (name + "-collection.json"))
        manifest_value = load(manifest)
        selected = manifest_value["capture"]
        C.require(len(manifest_value["groups"]) == 1, "Foreign replay group count")
        implementation = manifest_value["groups"][0]["implementation"]
        C.require(
            implementation["engine"] == row["engine"]
            and implementation["width"] == width
            and (implementation.get("library_api") is not None) == row["library_api"],
            "Replay implementation differs from selected arm",
        )
        base = capture_arguments(selected)
        for index, proof in enumerate(row["worker_provenance"]):
            load(proof["sidecar"])
            add(
                name + f"-provenance-{index}",
                "provenance",
                width,
                [
                    "--sidecar",
                    proof["sidecar"]["path"],
                    "--sidecar-sha256",
                    proof["sidecar"]["sha256"],
                    *base,
                ],
                expected_numerics=None,
                selected_capture=selected,
                engine=row["engine"],
                library_api=row["library_api"],
            )
        add(
            name + "-raw",
            "collection",
            width,
            [
                "--manifest",
                manifest["path"],
                "--manifest-sha256",
                manifest["sha256"],
                "--inventory",
                inv["inventory"]["path"],
                "--inventory-sha256",
                inv["inventory"]["sha256"],
                "--evidence-root",
                inv["evidence_root"],
                "--workers-root",
                inv["workers_root"],
                "--raw-store",
                raw_store,
            ],
            expected_numerics=True if row["library_api"] else None,
            selected_capture=selected,
            engine=row["engine"],
            library_api=row["library_api"],
        )
    optimizer = collector(inputs["optimizer"], capture, reuse=reuse)
    C.require(
        optimizer["exact_optimizer_repeat_and_restart_passed"] is True, "Optimizer gate incomplete"
    )
    for row in cell_inventory(optimizer["optimizer_cells"], FRAMEWORKS):
        C.require(
            [w["operation"] for w in row["workers"]] == ["first", "second", "restart"],
            "Optimizer fresh/restart denominator changed",
        )
        first = load(row["workers"][0]["receipt"])
        C.require(first["engine"] == row["engine"], "Optimizer implementation differs")
        selected = first["capture"]
        arguments = capture_arguments(selected)
        for worker in row["workers"]:
            load(worker["receipt"])
            arguments += [
                "--" + worker["operation"],
                worker["receipt"]["path"],
                "--" + worker["operation"] + "-sha256",
                worker["receipt"]["sha256"],
            ]
        add(
            f"optimizer-{row['engine']}-w{row['width']}",
            "optimizer",
            row["width"],
            arguments,
            selected_capture=selected,
            engine=row["engine"],
        )
    repeated = collector(inputs["repeatability"], capture, reuse=reuse)
    C.require(repeated["complete_repeatability_passed"] is True, "Complete repeatability required")
    for row in cell_inventory(repeated["cells"], ("mps", "mlx")):
        C.require(len(row["workers"]) == 2, "Two fresh repeatability processes required")
        C.require(
            all(load(w["receipt"])["engine"] == row["engine"] for w in row["workers"]),
            "Repeatability implementation differs",
        )
        selected = repeated["capture"]
        arguments = capture_arguments(selected)
        for label, worker in zip(("first", "second"), row["workers"], strict=True):
            load(worker["receipt"])
            arguments += [
                "--" + label,
                worker["receipt"]["path"],
                "--" + label + "-sha256",
                worker["receipt"]["sha256"],
            ]
        add(
            f"repeatability-{row['engine']}-w{row['width']}",
            "repeatability",
            row["width"],
            arguments,
            selected_capture=selected,
            engine=row["engine"],
        )
    common = collector(inputs["common_state"], capture, reuse=reuse)
    C.require(
        common["all_18_accuracy_cells_passed"] is True, "All-width common-state accuracy required"
    )
    for row in cell_inventory(common["cells"], FRAMEWORKS):
        record = load(row["receipt"])
        C.require(record["engine"] == row["engine"], "Common-state implementation differs")
        selected = record["capture"]
        base = capture_arguments(selected)
        add(
            f"common-state-{row['engine']}-w{row['width']}",
            "common_state",
            row["width"],
            [
                "--receipt",
                row["receipt"]["path"],
                "--receipt-sha256",
                row["receipt"]["sha256"],
                *base,
            ],
            selected_capture=selected,
            engine=row["engine"],
        )
    native = collector(inputs["native_metal"], capture, native=True, reuse=reuse)
    base = capture_arguments(native["capture"])
    C.require(
        native["all_6_selected_native_metal_widths_passed"] is True
        and [r["width"] for r in native["cells"]] == list(WIDTHS),
        "Selected native/Metal gate incomplete",
    )
    for row in native["cells"]:
        record = load(row["receipt"])
        producer = load(record["job"])["roles"]["control"]
        C.require(
            producer["revision"] == native["control_revision"], "Foreign native/Metal producer"
        )
        add(
            f"native-metal-w{row['width']}",
            "native_metal",
            row["width"],
            [
                "--receipt",
                row["receipt"]["path"],
                "--receipt-sha256",
                row["receipt"]["sha256"],
                *base,
            ],
            producer_control_revision=native["control_revision"],
            producer_control_root=producer["root"],
            selected_capture=native["capture"],
        )
    native_common = collector(inputs["native_common_state"], capture, native=True, reuse=reuse)
    base = capture_arguments(native_common["capture"])
    C.require(
        native_common["all_12_native_metal_accuracy_cells_passed"] is True,
        "Native/Metal common-state and identical-gradient accuracy required",
    )
    for row in cell_inventory(native_common["cells"], ("cpu_arm64", "metal")):
        record = load(row["receipt"])
        producer = load(record["job"])["roles"]["control"]
        C.require(
            producer["revision"] == native_common["control_revision"],
            "Foreign native common producer",
        )
        add(
            f"native-common-{row['engine']}-w{row['width']}",
            "native_common_state",
            row["width"],
            [
                "--receipt",
                row["receipt"]["path"],
                "--receipt-sha256",
                row["receipt"]["sha256"],
                *base,
            ],
            producer_control_revision=producer["revision"],
            producer_control_root=producer["root"],
            selected_capture=native_common["capture"],
        )
    return dict(
        schema="frozen-library-raw-qualification-plan-v1",
        inputs=inputs,
        capture=capture,
        inventories=inventories,
        raw_store=str(raw_store),
        reuse=reuse_binding,
        jobs=[j for j in jobs if j["width"] in C.WIDTHS],
        scope_amendment=C.comparison()["scope_amendment"],
        retained_deferred_job_count=sum(j["width"] in C.DEFERRED_WIDTHS for j in jobs),
        acceptance="every scheduled raw audit must execute and pass its declared requirement",
        formal_eligible=False,
        formal_fits=0,
        admission_stream_opened=False,
    )
