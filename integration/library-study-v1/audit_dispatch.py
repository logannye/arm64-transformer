"""Run frozen raw auditors with explicit source imports and independent packages."""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "library_raw_dispatch_authority", HERE / "authority.py"
)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
C = A.C
AUDITORS = {
    "primitive": "integration/precision-library-v1/audit_runtime.py",
    "optimizer": "integration/trained-precision/audit_library_optimizer.py",
    "provenance": "integration/trained-precision/audit_captured_replay.py",
    "collection": "integration/trained-precision/audit_collection.py",
}


class BoundedOutput(io.StringIO):
    def write(self, text):
        C.require(self.tell() + len(text) <= C.MAX_JSON_BYTES, "Auditor metadata exceeds bound")
        return super().write(text)


def execute(args):
    C.require(
        sys.flags.isolated and sys.flags.dont_write_bytecode, "Use independent audit Python -I -B"
    )
    C.require(
        not any(n.split(".", 1)[0] in {"transformer", "torch", "mlx"} for n in sys.modules),
        "Auditor requires fresh project/framework imports",
    )
    C.require(
        args.kind in AUDITORS
        and type(args.width) is int
        and args.width in (32, 64, 128, 256, 512, 1024)
        and os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(args.width),
        "Select frozen auditor and width before imports",
    )
    control = A.source_role(HERE.parents[1], args.control_revision, control=True)
    math = A.source_role(args.math_root, C.MATH_REVISION)
    source = Path(math["root"]) / AUDITORS[args.kind]
    source_binding = C.binding(source)
    C.require(source_binding["sha256"] == math["files"][AUDITORS[args.kind]], "Unbound raw auditor")
    # The independently installed NumPy/Torch stay active. Only this exact
    # project's source precedes the environment's editable checkout.
    sys.path.insert(0, math["root"])
    arguments = list(args.audit_arguments)
    C.require(arguments and arguments.pop(0) == "--", "Separate frozen auditor arguments with --")
    original = sys.argv
    exit_code = 0
    output = BoundedOutput()
    try:
        sys.argv = [str(source), *arguments]
        with contextlib.redirect_stdout(output):
            try:
                runpy.run_path(str(source), run_name="__main__")
            except SystemExit as error:
                C.require(
                    error.code is None or type(error.code) is int,
                    "Auditor returned a nonnumerical exit status",
                )
                exit_code = 0 if error.code is None else error.code
    finally:
        sys.argv = original
    C.require(exit_code == 0, "Frozen raw auditor failed; no qualification granted")
    result = json.loads(
        output.getvalue(),
        object_pairs_hook=C._pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite auditor JSON")),
    )
    spec = importlib.util.spec_from_file_location("library_raw_dispatch_origins", HERE / "audit.py")
    origins = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(origins)
    imported = {
        n: m for n, m in sys.modules.items() if n == "transformer" or n.startswith("transformer.")
    }
    proof = origins.project_origins(math, imported) if imported else {}
    C.require(
        "torch.nn.precise_fp32" not in sys.modules and "mlx.core" not in sys.modules,
        "Candidate packages must not replace the independent oracle",
    )
    A.verify_role(control, control=True)
    A.verify_role(math)
    return dict(
        schema="frozen-library-raw-audit-dispatch-v1",
        kind=args.kind,
        width=args.width,
        result=result,
        auditor_source=source_binding,
        auditor_control_role=control,
        auditor_math_revision=math["revision"],
        auditor_project_origins=proof,
        formal_eligible=False,
        formal_fits=0,
        admission_stream_opened=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=AUDITORS, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--control-revision", required=True)
    parser.add_argument("--math-root", type=Path, required=True)
    parser.add_argument("audit_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    print(json.dumps(execute(args), allow_nan=False))
