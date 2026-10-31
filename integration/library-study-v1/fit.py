"""One fresh-process formal fit of the exact admitted successor implementation."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        "library_formal_fit_" + name, HERE / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A, F, ADMISSION, W, M = (
    load(n) for n in ("authority", "frozen", "admission", "worker", "measurement")
)
C = A.C
MAX_FIT_SECONDS = 48 * 3600


def execute(path, expected):
    C.require_execution_support()
    job = C.read(path, expected)
    C.require(
        job["schema"] == "frozen-library-fit-job-v1"
        and set(job) == {"schema", "manifest", "admission", "cell", "output", "parent_pid"}
        and type(job["parent_pid"]) is int
        and job["parent_pid"] == os.getppid(),
        "Unknown formal fit job",
    )
    manifest = F.read(job["manifest"]["path"], job["manifest"]["sha256"])
    C.bound(job["manifest"])
    ADMISSION.verify(manifest, job["manifest"], job["admission"])
    cells = [c for c in manifest["comparison"]["cells"] if c["id"] == job["cell"]["id"]]
    C.require(
        len(cells) == 1 and C.canonical(cells[0]) == C.canonical(job["cell"]), "Unplanned fit cell"
    )
    cell = cells[0]
    C.require(
        os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(cell["width"]),
        "Select fit width before imports",
    )
    bindings = manifest["bindings"]
    sys.pycache_prefix = str(Path(job["output"]) / "unused-bytecode-cache")
    runtime = A.activate(bindings["roles"], bindings["capture"])
    prerequisite = F.PLAN.load(bindings["raw_prerequisites"])
    request = F.PLAN.load(prerequisite["request"])
    protected = dict(job, raw_store=request["raw_store"])
    output = W.protected_output(protected, runtime)
    fixtures = F.PLAN.load(bindings["fixtures"])
    fixture = next(
        r for r in fixtures["fixtures"] if (r["width"], r["seed"]) == (cell["width"], cell["seed"])
    )
    for binding in fixture["files"].values():
        C.bound(binding)
    C.require(
        not output.is_relative_to(Path(fixtures["input_root"])), "Fit output overlaps original data"
    )
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    proof = dict(
        schema="frozen-library-fit-authority-v1",
        job=C.binding(path),
        manifest=job["manifest"],
        admission=job["admission"],
        cell=cell,
        pid=os.getpid(),
        parent_pid=os.getppid(),
        scope="formal_measurement",
        complete=False,
    )
    try:
        proof["origins_before"] = runtime.origins(cell["engine"])
        if cell["engine"] == "cpu_torch":
            from transformer.python_train import run

            run(fixture["directory"], output, library_api=True)
        elif cell["engine"] == "cpu_arm64":
            assets = W.N.verify(bindings["native_assets"], runtime.roles["math"])
            row = next(r for r in assets["widths"] if r["width"] == cell["width"])
            executable = row["selected"]["transformer-arm64"]
            C.bound(executable)
            subprocess.run(
                [executable["path"], fixture["directory"], str(output)],
                check=True,
                cwd=runtime.roles["math"]["root"],
                env=dict(os.environ),
                timeout=MAX_FIT_SECONDS,
            )
            C.bound(executable)
            from transformer.backend import read_backend, validate_runtime_backend

            metadata = read_backend(output / "backend.u64", policy="auto")
            validate_runtime_backend(
                metadata,
                dict(
                    width=cell["width"],
                    epochs=8,
                    train_sequences=2048,
                    validation_sequences=256,
                    test_sequences=256,
                ),
                policy="auto",
                precision_policy=row["policy"],
            )
            proof["native_executable"] = executable
            proof["native_backend"] = metadata
        else:
            from transformer.gpu_inputs import load_fixture

            value = load_fixture(fixture["directory"])
            output.mkdir()
            M.gpu_fit(value, output, cell["engine"], runtime=runtime, scope="formal_measurement")
        from transformer.gpu_study import required_artifacts

        C.require(
            all((output / n).is_file() for n in required_artifacts(cell["engine"])),
            "Missing measured artifacts",
        )
        proof["origins_after"] = runtime.origins(cell["engine"])
        runtime.recheck()
        F.verify(manifest)
        C.require(C.binding(path)["sha256"] == expected, "Formal fit job changed")
        proof["complete"] = True
        C.write(output / "fit-authority.json", proof)
        files = {
            p.relative_to(output).as_posix(): C.binding(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        }
        C.write(
            output / "cell.json",
            dict(
                schema="frozen-library-formal-cell-v1",
                job=C.binding(path),
                cell=cell,
                manifest=job["manifest"],
                admission=job["admission"],
                artifacts=files,
                scope="formal_measurement",
                complete=True,
            ),
        )
        print(json.dumps(dict(cell=cell["id"], complete=True)))
    except BaseException:
        proof["error"] = traceback.format_exc()
        C.write(output.parent / (output.name + "-failed-fit.json"), proof)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--job-sha256", required=True)
    args = parser.parse_args()
    execute(args.job, args.job_sha256)
