"""Frozen successor authority. Reserved streams remain gated by complete support."""

import copy
import importlib.util
import platform
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("library_frozen_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A, QUALIFY, PLAN, FIXTURES, N = (
    load(n) for n in ("authority", "qualify", "qualification_plan", "fixtures", "native_assets")
)
C = A.C
KEYS = {"roles", "capture", "raw_prerequisites", "native_assets", "fixtures", "hardware"}


def hardware_identity():
    C.require(
        platform.system() == "Darwin" and platform.machine() == "arm64",
        "Frozen comparison requires macOS ARM64",
    )

    def query(name):
        return subprocess.check_output(["sysctl", "-n", name], text=True, timeout=30).strip()

    return dict(
        schema="frozen-library-host-v1",
        system=platform.system(),
        machine=platform.machine(),
        macos=platform.mac_ver()[0],
        release=platform.release(),
        version=platform.version(),
        cpu_brand=query("machdep.cpu.brand_string"),
        memory_bytes=int(query("hw.memsize")),
        physical_cpus=int(query("hw.physicalcpu")),
        logical_cpus=int(query("hw.logicalcpu")),
    )


def validate_shape(value):
    C.require(
        type(value) is dict
        and type(value.get("bindings")) is dict
        and set(value["bindings"]) == KEYS,
        "Frozen successor bindings are incomplete",
    )
    expected = C.plan()
    expected.update(
        phase="frozen", bindings=value.get("bindings"), frozen_utc=value.get("frozen_utc")
    )
    C.require(
        C.canonical(value) == C.canonical(expected)
        and set(value["bindings"]) == KEYS
        and isinstance(value["frozen_utc"], str)
        and len(value["frozen_utc"]) == 20,
        "Frozen successor shape or comparison changed",
    )
    return value


def prerequisites(binding, roles, capture, fixtures, assets):
    report = PLAN.load(binding)
    C.require(
        report["schema"] == "frozen-library-raw-prerequisites-v1"
        and report["raw_prerequisites_passed"] is True
        and report["roles"] == roles
        and report["capture"] == capture
        and report["formal_execution_supported"] is True
        and report["formal_eligible"] is False
        and type(report["formal_fits"]) is int
        and report["formal_fits"] == 0
        and report["admission_stream_opened"] is False,
        "Complete exact-source raw prerequisites required",
    )
    request = PLAN.load(report["request"])
    C.require(
        request["roles"] == roles
        and request["capture"] == capture
        and request["fixtures"] == fixtures
        and request["native_assets"] == assets,
        "Qualification request differs from the frozen study",
    )
    plan = PLAN.load(report["plan"])
    expected = PLAN.build(
        request["collectors"],
        capture,
        request["inventories"],
        request["raw_store"],
        reuse=request.get("reuse"),
    )
    C.require(C.canonical(plan) == C.canonical(expected), "Raw qualification schedule changed")
    completed = report["completed"]
    C.require(
        [r["name"] for r in completed] == ["full-suites", *[j["name"] for j in plan["jobs"]]],
        "Raw prerequisite audit denominator changed",
    )
    full = PLAN.load(completed[0]["result"])
    reuse = PLAN.reuse_context(request.get("reuse"), capture)
    previous_suites = PLAN.previous_scope(full["capture"], capture, reuse, "full_suites")
    suite_roles = reuse["roles"] if previous_suites else roles
    suite_capture = reuse["capture"] if previous_suites else capture
    C.require(
        full["full_suites_passed"] is True
        and full["evidence_valid"] is True
        and full["control"] == suite_roles["control"]
        and full["math"] == suite_roles["math"]
        and full["capture"] == suite_capture,
        "Full suite acceptance is stale or foreign",
    )
    for row, job in zip(completed[1:], plan["jobs"], strict=True):
        C.require(
            row["kind"] == job["kind"]
            and row["width"] == job["width"]
            and row["execution"]["exit_code"] == 0,
            "Raw audit identity/exit differs",
        )
        value = PLAN.load(row["execution"]["stdout"])
        C.bound(row["execution"]["stderr"])
        C.require(
            QUALIFY.require_result(job, value) == row["decision"],
            "Raw prerequisite decision changed",
        )
    for role in ("math", "control"):
        QUALIFY.ci_evidence(request["source_ci"][role], roles[role]["revision"])
        if reuse is not None:
            QUALIFY.ci_evidence(reuse["source_ci"][role], reuse["roles"][role]["revision"])
    return report


def verify(value):
    C.require_execution_support()
    validate_shape(value)
    bindings = value["bindings"]
    C.require(bindings["hardware"] == hardware_identity(), "Frozen hardware/OS identity changed")
    roles = bindings["roles"]
    C.require(Path(roles["control"]["root"]) == HERE.parents[1], "Frozen controller origin differs")
    A.verify_role(roles["control"], control=True)
    A.verify_role(roles["math"])
    capture = PLAN.load(bindings["capture"])
    C.bound(capture["runtime_receipt"])
    runtime = C.read(capture["runtime_receipt"]["path"], capture["runtime_receipt"]["sha256"])
    C.require(
        runtime["project"] == dict(root=roles["math"]["root"], files=roles["math"]["files"]),
        "Frozen numerical source differs from qualified runtime",
    )
    fixtures = PLAN.load(bindings["fixtures"])
    FIXTURES.verify(fixtures, roles["math"])
    N.verify(bindings["native_assets"], roles["math"])
    prerequisites(
        bindings["raw_prerequisites"],
        roles,
        bindings["capture"],
        bindings["fixtures"],
        bindings["native_assets"],
    )
    return value


def read(path, digest):
    return verify(C.read(path, digest))


def freeze(roles, capture, raw_prerequisites, native_assets, fixtures, hardware, output):
    C.require_execution_support()
    value = C.plan()
    value.update(
        phase="frozen",
        frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        bindings=dict(
            roles=roles,
            capture=capture,
            raw_prerequisites=raw_prerequisites,
            native_assets=native_assets,
            fixtures=fixtures,
            hardware=hardware,
        ),
    )
    verify(value)
    C.write(output, value)
    return C.binding(output)


class AdmissionPermission:
    """Created by verifying an external frozen-manifest trust anchor before RNG."""

    def __init__(self, manifest, digest):
        C.require_execution_support()
        self.binding = C.binding(manifest)
        C.require(self.binding["sha256"] == digest, "Frozen manifest trust anchor differs")
        self._value = C.canonical(read(manifest, digest))

    def checked(self):
        C.bound(self.binding)
        value = read(self.binding["path"], self.binding["sha256"])
        C.require(C.canonical(value) == self._value, "Frozen admission authority changed")
        return copy.deepcopy(value)

    @property
    def seed(self):
        self.checked()
        return C.ADMISSION_SEED
