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


def collector(binding, capture, *, native=False):
    value = load(binding)
    C.require(
        value["math_revision" if native else "project_revision"] == C.MATH_REVISION
        and value["capture"] == capture
        and value["formal_eligible"] is False
        and type(value["formal_fits"]) is int
        and value["formal_fits"] == 0
        and value["admission_stream_opened"] is False,
        "Foreign development collector",
    )
    return value


def build(inputs, capture, inventories, raw_store):
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

    def add(name, kind, width, arguments, *, expected_numerics=True, **metadata):
        C.require(not any(j["name"] == name for j in jobs), "Duplicate raw audit job")
        jobs.append(
            dict(
                name=name,
                kind=kind,
                width=width,
                arguments=list(map(str, arguments)),
                expected_numerics=expected_numerics,
                **metadata,
            )
        )

    base = ["--capture", capture["path"], "--capture-sha256", capture["sha256"]]
    device = collector(inputs["device"], capture)
    C.require(
        device["library_exposed_model_gates_passed"] is True
        and device["ordinary_outcomes_retained"] is True,
        "Complete matched replay required",
    )
    foundation = load(device["foundation_gates"])
    for row in cell_inventory(foundation["primitive_cells"], FRAMEWORKS):
        load(row["receipt"])
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
        )
    optimizer = collector(inputs["optimizer"], capture)
    C.require(
        optimizer["exact_optimizer_repeat_and_restart_passed"] is True, "Optimizer gate incomplete"
    )
    for row in cell_inventory(optimizer["optimizer_cells"], FRAMEWORKS):
        C.require(
            [w["operation"] for w in row["workers"]] == ["first", "second", "restart"],
            "Optimizer fresh/restart denominator changed",
        )
        arguments = list(base)
        for worker in row["workers"]:
            load(worker["receipt"])
            arguments += [
                "--" + worker["operation"],
                worker["receipt"]["path"],
                "--" + worker["operation"] + "-sha256",
                worker["receipt"]["sha256"],
            ]
        add(f"optimizer-{row['engine']}-w{row['width']}", "optimizer", row["width"], arguments)
    repeated = collector(inputs["repeatability"], capture)
    C.require(repeated["complete_repeatability_passed"] is True, "Complete repeatability required")
    for row in cell_inventory(repeated["cells"], ("mps", "mlx")):
        C.require(len(row["workers"]) == 2, "Two fresh repeatability processes required")
        arguments = list(base)
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
        )
    common = collector(inputs["common_state"], capture)
    C.require(
        common["all_18_accuracy_cells_passed"] is True, "All-width common-state accuracy required"
    )
    for row in cell_inventory(common["cells"], FRAMEWORKS):
        load(row["receipt"])
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
        )
    native = collector(inputs["native_metal"], capture, native=True)
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
        )
    native_common = collector(inputs["native_common_state"], capture, native=True)
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
        )
    return dict(
        schema="frozen-library-raw-qualification-plan-v1",
        inputs=inputs,
        capture=capture,
        inventories=inventories,
        raw_store=str(raw_store),
        jobs=jobs,
        acceptance="every scheduled raw audit must execute and pass its declared requirement",
        formal_eligible=False,
        formal_fits=0,
        admission_stream_opened=False,
    )
