"""Bind an audit-only repair to the unchanged, completed 30390c8 campaign."""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(root, name):
    spec = importlib.util.spec_from_file_location(
        "postrun_recovery_" + name, Path(root) / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A, F, AD = (load(HERE, name) for name in ("authority", "frozen", "admission"))
C = A.C
PRODUCER = "30390c80573baf687f9287f2b1921c79d5bc67c0"
MANIFEST = "d753078098ac3b740cb83171ebfbe14f2bbd08a4a0878bb40357fabeeacdb082"
CAMPAIGN = "3e90d4b61690ced18991cbbf3f4eec7237a60adcb1d195735e32e3093ce220ab"
ALLOWED = {
    A.CONTROL_PREFIX + name
    for name in (
        "postrun_recovery.py",
        "checkpoint_replay.py",
        "checkpoint_batch.py",
        "audit_checkpoint.py",
        "audit_campaign.py",
    )
}


def compatible(producer, auditor):
    C.require(
        producer["revision"] == PRODUCER
        and {k: v for k, v in producer["files"].items() if k not in ALLOWED}
        == {k: v for k, v in auditor["files"].items() if k not in ALLOWED},
        "Postrun repair changed a fit, model, gate, comparison or other frozen dependency",
    )


class Context:
    def __init__(self, job):
        self.binding = job.get("audit_recovery")
        self.F, self.AD = F, AD
        if self.binding is not None:
            recovery = F.PLAN.load(self.binding)
            C.require(
                set(recovery)
                == {
                    "schema",
                    "manifest",
                    "admission",
                    "campaign",
                    "diagnosis",
                    "auditor",
                    "source_ci",
                }
                and recovery["schema"] == "frozen-library-postrun-recovery-v1"
                and recovery["manifest"] == job["manifest"]
                and recovery["manifest"]["sha256"] == MANIFEST
                and recovery["admission"] == job["admission"]
                and recovery["campaign"]["sha256"] == CAMPAIGN,
                "Foreign postrun recovery authority",
            )
            manifest = F.PLAN.load(recovery["manifest"])
            producer = manifest["bindings"]["roles"]["control"]
            A.verify_role(producer, control=True)
            self.auditor = recovery["auditor"]
            C.require(Path(self.auditor["root"]) == HERE.parents[1], "Foreign audit source root")
            A.verify_role(self.auditor, control=True)
            compatible(producer, self.auditor)
            F.QUALIFY.ci_evidence(recovery["source_ci"], self.auditor["revision"])
            diagnosis = F.PLAN.load(recovery["diagnosis"])
            C.require(
                diagnosis["schema"] == "postrun-thread-failure-diagnosis-v1"
                and diagnosis["manifest"] == recovery["manifest"]
                and diagnosis["admission"] == recovery["admission"]
                and diagnosis["campaign"] == recovery["campaign"]
                and diagnosis["all_formal_artifact_bindings_verified"] is True
                and diagnosis["completed_fits"] == 120
                and diagnosis["repeat_pairs_passed"] == 60
                and diagnosis["accepted_replays"] == 0,
                "Recovery lacks original completed-fit evidence",
            )
            failure = C.bound(diagnosis["failure"]).read_text()
            C.require(
                "torch.set_num_interop_threads(1)" in failure
                and "cannot set number of interop threads" in failure,
                "Recovery failure differs",
            )
            C.bound(diagnosis["stopped"])
            C.bound(recovery["campaign"])
            original = Path(producer["root"]) / A.CONTROL_PREFIX
            self.F, self.AD = (load(original, name) for name in ("frozen", "admission"))
        self.manifest = self.F.read(job["manifest"]["path"], job["manifest"]["sha256"])
        self.AD.verify(self.manifest, job["manifest"], job["admission"])
        self.roles = dict(self.manifest["bindings"]["roles"])
        if self.binding is not None:
            self.roles["control"] = self.auditor

    def verify(self):
        self.F.verify(self.manifest)
        if self.binding is not None:
            C.bound(self.binding)
            A.verify_role(self.auditor, control=True)

    def protect(self, output):
        """The repaired audit also keeps the original producer source immutable."""
        root = Path(self.manifest["bindings"]["roles"]["control"]["root"])
        output = Path(output).absolute()
        C.require(
            not output.is_relative_to(root) and not root.is_relative_to(output),
            "Audit output overlaps original producer source",
        )
