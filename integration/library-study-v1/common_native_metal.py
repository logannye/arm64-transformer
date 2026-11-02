"""Apply the identical exposed common-state/gradient matrix to native and Metal."""

import hashlib
import importlib.util
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E = load(HERE / "execution.py", "native_common_execution")
N = load(HERE / "native_assets.py", "native_common_assets")
C = E.C


def helpers(runtime):
    math = runtime.roles["math"]
    result = []
    for name in ("library_common_state", "audit_library_common_state"):
        path = Path(math["root"]) / f"integration/trained-precision/{name}.py"
        C.require(
            C.binding(path)["sha256"] == math["files"][path.relative_to(math["root"]).as_posix()],
            "Unbound common-state helper",
        )
        result.append(load(path, "native_common_" + name))
    return result


def trajectory(job, runtime):
    C.bound(job["trajectory"])
    value = C.read(job["trajectory"]["path"], job["trajectory"]["sha256"])
    C.require(
        value["schema"] == "maintained-library-optimizer-development-v1"
        and value["engine"] == "cpu_torch"
        and value["width"] == job["width"]
        and value["capture"] == job["capture"]
        and value["seed"] == 20261103
        and value["start_step"] == 0
        and value["steps"] == 64
        and value["complete"] is True
        and value["passed"] is True
        and value["integrity_after"] is True,
        "Wrong complete CPU common trajectory",
    )
    path = (
        Path(runtime.roles["math"]["root"]) / "integration/trained-precision/library_optimizer.py"
    )
    C.require(
        value["producer"] == C.binding(path), "Common trajectory used another numerical source"
    )
    C.bound(value["fixture"])
    fixture = C.read(value["fixture"]["path"], value["fixture"]["sha256"])
    C.require(
        fixture["schema"] == "maintained-library-optimizer-inputs-v1"
        and fixture["width"] == job["width"]
        and fixture["seed"] == 20261103
        and fixture["training_only"] is True,
        "Wrong common training-only inputs",
    )
    for key in ("initial", "training"):
        C.bound(fixture[key])
    return value, fixture


def fingerprint(value):
    import numpy as np

    value = np.asarray(value)
    return dict(
        dtype=value.dtype.str,
        shape=list(value.shape),
        sha256=hashlib.sha256(value.tobytes()).hexdigest(),
    )


def run(job, runtime, output):
    import numpy as np

    from transformer import spec
    from transformer.gpu_admission_checks import adam_reference, numpy_model
    from transformer.gpu_repeatability import _torch_reference
    from transformer.gpu_study_cpu_admission import _compute
    from transformer.trained_precision import save_arrays

    C.require(job["engine"] in ("cpu_arm64", "metal"), "Native/Metal common-state arm required")
    common, independent = helpers(runtime)
    source, fixture = trajectory(job, runtime)
    rows = np.fromfile(fixture["training"]["path"], "<i4").reshape(2048, 33)
    gradient = (
        np.random.default_rng(np.random.SeedSequence([20261103, job["width"], 91]))
        .normal(0, 0.01, spec.PARAM_COUNT)
        .astype(np.float32)
    )
    initial = common.checkpoint(np, source, 0, spec.PARAM_COUNT)
    C.require(
        fingerprint(initial["parameters"])["sha256"] == fixture["initial"]["sha256"]
        and (initial["m"] == 0).all()
        and (initial["v"] == 0).all(),
        "Wrong initial common state",
    )
    native, backend, records = None, None, []
    try:
        if job["engine"] == "cpu_arm64":
            native = N.load(runtime, job["native_assets"], job["width"])
        else:
            backend = E.create_backend(runtime, "metal", initial["parameters"])
        for name in common.cases():
            C.require(
                shutil.disk_usage(output).free > C.MIN_FREE_BYTES + 1024**3,
                "Preserve disk reserve and complete common-state output",
            )
            family, step_text, shape = name.split("/")
            state = common.checkpoint(np, source, int(step_text.split("-")[1]), spec.PARAM_COUNT)
            before = {k: fingerprint(v) for k, v in state.items()}
            oracle, repetitions = {}, None
            if family == "model":
                count = int(shape.split("-")[0][1:])
                x, y = (
                    np.ascontiguousarray(rows[:count, :-1]),
                    np.ascontiguousarray(rows[:count, 1:]),
                )
                gold = numpy_model(state["parameters"], x, y)
                double = _torch_reference(state["parameters"], x, y, backward=True)
                oracle = {k: common.compare(np, double[k], gold[k], "oracle") for k in gold}
                observed = []
                for _ in range(2):
                    if native is not None:
                        value = _compute(
                            "cpu_arm64",
                            dict(state, mode=np.asarray("forward_gradient"), tokens=x, targets=y),
                            native=native,
                        )
                        actual = {k: value[k] for k in gold}
                    else:
                        backend.load_state(state)
                        batch = backend.upload_inputs(x, y)
                        logits = backend.to_numpy(backend.forward(batch)).copy()
                        loss, grad = backend.loss_and_grad(batch)
                        actual = dict(
                            logits=logits,
                            loss=backend.to_numpy(loss).copy(),
                            gradient=backend.to_numpy(grad).copy(),
                        )
                        C.require(
                            common.exact_state(np, backend.export_state(), state),
                            "Forward/backward changed common state",
                        )
                    observed.append(actual)
                repetitions = {k: [fingerprint(v[k]) for v in observed] for k in gold}
                C.require(
                    all(pair[0] == pair[1] for pair in repetitions.values()),
                    "Common model repetition differs",
                )
                actual = observed[0]
                checks = {
                    k: common.compare(
                        np, actual[k], gold[k], "gradient" if k == "gradient" else "forward"
                    )
                    for k in gold
                }
            else:
                count = int(shape.split("-")[1])
                gold = adam_reference(state, gradient, count)
                if native is not None:
                    actual = {k: np.asarray(v).copy() for k, v in state.items()}
                    for _ in range(count):
                        value = _compute(
                            "cpu_arm64",
                            dict(
                                actual,
                                mode=np.asarray("adam"),
                                step=np.asarray(int(actual["step"]) + 1, np.int64),
                                gradient=gradient,
                            ),
                            native=native,
                        )
                        actual = {k: value[k] for k in state}
                else:
                    backend.load_state(state)
                    for _ in range(count):
                        backend.apply_gradient(gradient)
                    actual = backend.export_state()
                checks = {
                    k: common.compare(
                        np, actual[k], gold[k], "exact" if k == "step" else "optimizer"
                    )
                    for k in gold
                }
            C.require(
                before == {k: fingerprint(v) for k, v in state.items()}, "Common operands mutated"
            )
            actual = {k: np.asarray(v) for k, v in actual.items()}
            raw = save_arrays(output / (name.replace("/", "-") + ".npz"), actual, job["raw_store"])
            row = dict(
                name=name,
                actual=raw,
                comparisons=checks,
                oracle=oracle,
                repetitions=repetitions,
                passed=all(c["passed"] for c in [*checks.values(), *oracle.values()]),
            )
            C.write(output / "cases" / (name.replace("/", "-") + ".json"), row)
            records.append(row)
        result = dict(
            schema="frozen-library-native-metal-common-state-v1",
            engine=job["engine"],
            width=job["width"],
            trajectory=job["trajectory"],
            seed=20261103,
            expected_cases=common.cases(),
            cases=records,
            passed=all(r["passed"] for r in records),
            raw_store=job["raw_store"],
        )
        if backend is not None:
            result.update(
                capabilities=E.validate_capabilities("metal", backend.capabilities()),
                metal_library_origin=backend._selected_native_origin,
            )
        return result
    finally:
        if backend is not None:
            backend.close()
