"""Untimed selected-engine or independent FP64 replay of an authenticated checkpoint."""

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        "library_checkpoint_" + name, HERE / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A, F, AD, W, E, ORIGINS = (
    load(n) for n in ("authority", "frozen", "admission", "worker", "execution", "audit")
)
C = A.C
ENGINES = ("cpu_arm64", "cpu_torch", "metal", "mps", "mlx")
NAMES = ("validation", "test", "validation_nll", "test_nll", "b1")


def cell_receipt(binding, manifest, manifest_binding, admission):
    receipt = F.PLAN.load(binding)
    cell = receipt["cell"]
    C.require(
        receipt["schema"] == "frozen-library-formal-cell-v1"
        and receipt["complete"] is True
        and receipt["scope"] == "formal_measurement"
        and receipt["manifest"] == manifest_binding
        and receipt["admission"] == admission
        and cell in manifest["comparison"]["cells"],
        "Foreign or incomplete measured cell",
    )
    directory = Path(binding["path"]).parent
    for name, artifact in receipt["artifacts"].items():
        A.relative(name)
        C.require(Path(artifact["path"]) == directory / name, "Cell artifact path differs")
        C.bound(artifact)
    C.require("fit-authority.json" in receipt["artifacts"], "Missing measured source authority")
    proof = F.PLAN.load(receipt["artifacts"]["fit-authority.json"])
    C.require(
        proof["schema"] == "frozen-library-fit-authority-v1"
        and proof["job"] == receipt["job"]
        and proof["manifest"] == manifest_binding
        and proof["admission"] == admission
        and proof["cell"] == cell
        and proof["scope"] == "formal_measurement"
        and proof["complete"] is True
        and "error" not in proof,
        "Measured source authority differs",
    )
    job = F.PLAN.load(proof["job"])
    C.require(
        job["cell"] == cell
        and job["manifest"] == manifest_binding
        and job["admission"] == admission
        and Path(job["output"]) == directory
        and type(proof["pid"]) is int
        and type(proof["parent_pid"]) is int
        and proof["pid"] > 0
        and proof["pid"] != proof["parent_pid"] == job["parent_pid"],
        "Measured job differs",
    )
    return receipt, proof


def replay_values(engine, parameters, fixture, *, runtime=None, assets=None):
    """Keep the original B16 NLL reduction and resident B1 forward boundaries."""
    import numpy as np
    import torch
    from torch.nn import functional as functional

    from transformer import spec

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    native, backend, model, oracle = None, None, None, None
    if engine == "cpu_arm64":
        from transformer.validate import workspace_tensor

        native = W.N.load(runtime, assets, spec.WIDTH)
        gradient = np.empty((32, 128), np.float32)
    elif engine == "cpu_torch":
        from transformer.library_training import torch_model_type

        model = torch_model_type("cpu_torch")().load_flat(parameters)
    elif engine == "oracle":
        from transformer.precision_audit import FP64Transformer
        from transformer.reference import Transformer

        oracle, model = FP64Transformer(parameters), Transformer().load_flat(parameters).double()
    else:
        backend = E.create_backend(runtime, engine, parameters, max_batch=16)
    result, evidence = {}, {}
    try:
        for split in ("validation", "test"):
            rows = fixture[split]
            C.require(rows.shape == (256, 33) and rows.dtype == np.int32, "Wrong replay rows")
            values = np.empty((256, 32, 128), np.float64 if oracle else np.float32)
            total = 0.0
            if native is not None:
                for i, row in enumerate(rows):
                    workspace = native.forward(parameters, np.ascontiguousarray(row[:-1]))
                    values[i] = workspace_tensor(workspace, "logits")
                    total += native.call(
                        "cross_entropy",
                        workspace_tensor(workspace, "logits"),
                        np.ascontiguousarray(row[1:]),
                        gradient,
                        32,
                        128,
                    )
            else:
                for start in range(0, 256, 16):
                    selected = rows[start : start + 16]
                    if oracle is not None:
                        logits, loss = oracle(selected[:, :-1]), 0.0
                    elif model is not None:
                        with torch.no_grad():
                            tensor = model(torch.tensor(selected[:, :-1].astype(np.int64)))
                            loss = functional.cross_entropy(
                                tensor.reshape(-1, 128),
                                torch.tensor(selected[:, 1:].astype(np.int64)).reshape(-1),
                            ).item()
                            logits = tensor.numpy()
                    else:
                        batch = backend.upload_inputs(selected[:, :-1], selected[:, 1:])
                        loss_value, device_logits = backend.evaluate(batch)
                        logits = backend.to_numpy(device_logits)
                        loss = float(backend.to_numpy(loss_value).reshape(-1)[0])
                    values[start : start + 16] = logits
                    total += loss * 16
            C.require(np.isfinite(values).all(), "Nonfinite replay logits")
            result[split] = values
            if oracle is not None:
                with torch.no_grad():
                    # The independent PyTorch double equations cross-check the
                    # first and last rows, matching the original audit contract.
                    result[split + "_torch_fp64"] = model(
                        torch.tensor(rows[[0, -1], :-1].astype(np.int64))
                    ).numpy()
            else:
                result[split + "_nll"] = np.array(total / 256, np.float64)
        tokens = fixture["test"][:1, :32]
        if native is not None:
            result["b1"] = workspace_tensor(
                native.forward(parameters, np.ascontiguousarray(tokens[0])), "logits"
            )[None].copy()
        elif model is not None:
            with torch.no_grad():
                result["b1"] = model(torch.tensor(tokens.astype(np.int64))).numpy()
        else:
            result["b1"] = backend.to_numpy(backend.forward(backend.upload_inputs(tokens))).copy()
        if backend is not None:
            evidence["capabilities"] = E.validate_capabilities(engine, backend.capabilities())
            if engine == "metal":
                evidence["metal_library_origin"] = E.loaded_metal_library(runtime)
        if native is not None:
            evidence["native_assets"] = assets
        return result, evidence
    finally:
        if backend is not None:
            backend.close()


def execute(path, digest):
    C.require_execution_support()
    job = C.read(path, digest)
    C.require(
        set(job)
        == {
            "schema",
            "manifest",
            "admission",
            "cell_receipt",
            "checkpoint",
            "engine",
            "output",
            "raw_store",
            "parent_pid",
        }
        and job["schema"] == "frozen-library-checkpoint-job-v1"
        and job["engine"] in (*ENGINES, "oracle")
        and job["checkpoint"] in ("best.f32", "last.f32")
        and type(job["parent_pid"]) is int
        and job["parent_pid"] == os.getppid(),
        "Unknown checkpoint replay job",
    )
    manifest = F.read(job["manifest"]["path"], job["manifest"]["sha256"])
    AD.verify(manifest, job["manifest"], job["admission"])
    receipt, _ = cell_receipt(job["cell_receipt"], manifest, job["manifest"], job["admission"])
    bindings, cell = manifest["bindings"], receipt["cell"]
    C.require(os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(cell["width"]), "Select width first")
    capture = F.PLAN.load(bindings["capture"])
    inventory = F.PLAN.load(capture["runtime_inventory"])
    output = Path(job["output"])
    if job["engine"] == "oracle":
        C.require(
            sys.flags.isolated and sys.flags.dont_write_bytecode, "Use independent -I -B oracle"
        )
        C.require(
            not any(
                n.split(".", 1)[0] in {"torch", "mlx", "numpy", "transformer"} for n in sys.modules
            ),
            "Fresh independent FP64 oracle required",
        )
        sys.path.insert(0, bindings["roles"]["math"]["root"])
        runtime = None
    else:
        sys.pycache_prefix = str(output / "unused-bytecode-cache")
        runtime = A.activate(bindings["roles"], bindings["capture"])
    # Apply identical source/runtime/raw-store output protections to the oracle.
    from types import SimpleNamespace

    W.protected_output(
        job, runtime or SimpleNamespace(roles=bindings["roles"], inventory=inventory)
    )
    request = F.PLAN.load(F.PLAN.load(bindings["raw_prerequisites"])["request"])
    C.require(job["raw_store"] == request["raw_store"], "Replay raw-store selection changed")
    import numpy as np

    from transformer import spec
    from transformer.prepare import load_fixture
    from transformer.trained_precision import save_arrays

    fixture = next(
        r
        for r in F.PLAN.load(bindings["fixtures"])["fixtures"]
        if (r["width"], r["seed"]) == (cell["width"], cell["seed"])
    )
    parameters = np.fromfile(C.bound(receipt["artifacts"][job["checkpoint"]]), "<f4")
    C.require(
        parameters.shape == (spec.PARAM_COUNT,) and np.isfinite(parameters).all(),
        "Invalid checkpoint",
    )
    output.mkdir()
    proof = dict(
        schema="frozen-library-checkpoint-replay-v1",
        job=C.binding(path),
        manifest=job["manifest"],
        cell_receipt=job["cell_receipt"],
        engine=job["engine"],
        checkpoint=job["checkpoint"],
        pid=os.getpid(),
        parent_pid=os.getppid(),
        complete=False,
    )
    if runtime:
        proof["origins_before"] = runtime.origins(job["engine"])
    values, evidence = replay_values(
        job["engine"],
        parameters,
        load_fixture(fixture["directory"]),
        runtime=runtime,
        assets=bindings["native_assets"],
    )
    C.require(
        shutil.disk_usage(output).free > C.MIN_FREE_BYTES + 128 * 1024**2,
        "Replay disk reserve reached",
    )
    raw = save_arrays(output / "raw.npz", values, job["raw_store"])
    proof.update(
        raw={k: raw[k] for k in ("sha256", "bytes")}, raw_store=job["raw_store"], **evidence
    )
    if runtime:
        proof["origins_after"] = runtime.origins(job["engine"])
        runtime.recheck()
    else:
        proof["auditor_project_origins"] = ORIGINS.project_origins(
            bindings["roles"]["math"], sys.modules
        )
        C.require(
            not any(n.startswith(("torch.nn.precise_fp32", "mlx")) for n in sys.modules),
            "FP64 oracle loaded selected candidate APIs",
        )
        proof["oracle_versions"] = dict(numpy=np.__version__, torch=__import__("torch").__version__)
    F.verify(manifest)
    cell_receipt(job["cell_receipt"], manifest, job["manifest"], job["admission"])
    C.require(C.binding(path)["sha256"] == digest, "Replay job changed")
    proof["complete"] = True
    C.write(output / "receipt.json", proof)
    print(json.dumps(dict(complete=True, receipt=C.binding(output / "receipt.json"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--job-sha256", required=True)
    args = parser.parse_args()
    execute(args.job, args.job_sha256)
