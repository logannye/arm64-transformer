"""Selected-library development gate context using frozen mathematical handlers.

Development uses the exposed stream. A reserved stream requires complete
execution support and an independently verified frozen-manifest trust anchor.
"""

import importlib.util
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("library_context_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E, N = load("execution"), load("native_assets")
C = E.C
DEVELOPMENT_SEED = 20261103


def create(
    runtime, output, width, assets, raw_store, *, fresh_training, checkpoint_tie, admission=None
):
    import numpy as np

    from transformer import gpu_admission_checks as math
    from transformer import spec
    from transformer.gpu_study_cpu_admission import _compute
    from transformer.reference import Transformer
    from transformer.trained_precision import save_arrays

    C.require(
        type(width) is int
        and width == spec.WIDTH
        and os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(width),
        "Select width first",
    )
    C.require(
        Path(output).is_dir() and Path(output).resolve() == Path(output).absolute(),
        "Canonical development output required",
    )
    runtime.recheck()
    seed = DEVELOPMENT_SEED
    if admission is not None:
        frozen = load("frozen")
        C.bound(admission)
        grant = frozen.AdmissionPermission(admission["path"], admission["sha256"])
        manifest = grant.checked()
        C.require(
            manifest["bindings"]["roles"] == runtime.roles
            and manifest["bindings"]["capture"] == runtime.capture_binding,
            "Admission permission belongs to another runtime/source",
        )
        seed = C.ADMISSION_SEED

    class LibraryContext(math.Context):
        def __init__(self):
            self.runtime, self.width, self.output = runtime, width, Path(output)
            self.backends, self.cache, self.counter, self.cpu_operations = {}, {}, 0, []
            self.native = None
            self.study_context = SimpleNamespace(owned=False, admission_seed=seed)
            self.initial = dict(parameters=Transformer(seed=seed).flat_parameters(), step=0)
            self.initial.update(
                m=np.zeros_like(self.initial["parameters"]),
                v=np.zeros_like(self.initial["parameters"]),
            )
            self.rng = np.random.default_rng(np.random.SeedSequence([seed, width, 99]))
            self.tokens = self.rng.integers(0, 128, (16, 32), dtype=np.int32)
            self.tokens[:, ::3] = 65
            self.targets = self.rng.integers(0, 128, (16, 32), dtype=np.int32)
            math._npz(
                self.path("initial-operands"),
                {**self.initial, "tokens": self.tokens, "targets": self.targets},
            )

        def backend(self, engine):
            if engine not in self.backends:
                self.backends[engine] = E.create_backend(
                    runtime, engine, self.initial["parameters"]
                )
            return self.backends[engine]

        def cpu(self, engine, mode, state, *, expected_rejection=None, **values):
            C.require(engine in ("cpu_arm64", "cpu_torch"), "CPU engine required")
            if engine == "cpu_arm64" and self.native is None:
                self.native = N.load(runtime, assets, width)
            operands = {k: np.asarray(v) for k, v in {**state, **values, "mode": mode}.items()}
            C.require(
                shutil.disk_usage(raw_store).free
                > C.MIN_FREE_BYTES + sum(v.nbytes for v in operands.values()) + 8 * 1024**2,
                "Preserve complete CPU operands and disk reserve",
            )
            inputs = save_arrays(self.path("cpu-input"), operands, raw_store)
            row = dict(
                engine=engine,
                mode=mode,
                pid=os.getpid(),
                inputs=inputs,
                expected_rejection=expected_rejection,
            )
            try:
                if mode == "repeat_training":
                    result = {}
                    for prefix in ("first", "second"):
                        current = {k: np.asarray(state[k]).copy() for k in self.initial}
                        for _ in range(6):
                            updated = _compute(
                                engine,
                                {
                                    **current,
                                    "mode": np.asarray("advance"),
                                    "tokens": values["tokens"],
                                    "targets": values["targets"],
                                    "step": np.asarray(int(current["step"]) + 1, np.int64),
                                },
                                native=self.native,
                                library_api=engine == "cpu_torch",
                            )
                            current = {k: updated[k] for k in self.initial}
                        probe = _compute(
                            engine,
                            {
                                **current,
                                "mode": np.asarray("forward_gradient"),
                                "tokens": values["tokens"],
                                "targets": values["targets"],
                            },
                            native=self.native,
                            library_api=engine == "cpu_torch",
                        )
                        result.update(
                            {
                                prefix + "_" + k: v
                                for k, v in {**current, "logits": probe["logits"]}.items()
                            }
                        )
                else:
                    result = _compute(
                        engine, operands, native=self.native, library_api=engine == "cpu_torch"
                    )
            except Exception as error:
                if expected_rejection is None:
                    raise
                if type(error) is not ValueError or str(error) != expected_rejection:
                    raise AssertionError(
                        "CPU rejection differs from the declared malformed-input case"
                    ) from error
                row["rejected"] = str(error)
                self.cpu_operations.append(row)
                raise
            # A malformed-input probe cannot pass merely because its wrapper raises.
            if expected_rejection is not None:
                raise AssertionError("CPU malformed-input probe was unexpectedly accepted")
            C.require(
                shutil.disk_usage(raw_store).free
                > C.MIN_FREE_BYTES
                + sum(np.asarray(v).nbytes for v in result.values())
                + 8 * 1024**2,
                "Preserve complete CPU outputs and disk reserve",
            )
            row["outputs"] = save_arrays(self.path("cpu-output"), result, raw_store)
            self.cpu_operations.append(row)
            return result

        def produce(self, engine, family, suffix):
            if (engine, family, suffix) == ("mlx", "control", "device_execution"):
                self.model(engine)
                capabilities = self.backend(engine).capabilities()
                mx = importlib.import_module("mlx.core")
                # Device(gpu, 0) equals DeviceType.gpu, but their strings differ.
                return math._bools(
                    [
                        mx.default_device() == mx.gpu,
                        capabilities["cpu_tensor_fallback"] is False,
                        capabilities["compiled"] is True,
                        capabilities["MLX_DISABLE_COMPILE"] is None,
                        capabilities["compilation_probe"]["status"] == "passed",
                        capabilities["compilation_probe"]["python_trace_count"] == 1,
                        capabilities["compilation_probe"]["executions"] == 4,
                        capabilities["compilation_probe"]["compiled_primitive_present"] is True,
                    ]
                )
            if family == "control" and suffix == "repeat_training_fresh_process":
                first, second = fresh_training(self, engine), fresh_training(self, engine)
                return math._bools(
                    [math._exact(first[k], second[k]) for k in (*self.initial, "logits")]
                )
            if (
                family == "control"
                and suffix == "checkpoint_first_tie"
                and engine.startswith("cpu_")
            ):
                return {"observations": checkpoint_tie(self, engine)}
            return math._produce(self, engine, family, suffix)

        def close(self):
            errors = []
            for backend in self.backends.values():
                try:
                    backend.close()
                except Exception as error:
                    errors.append(repr(error))
            self.backends.clear()
            C.require(not errors, "Selected backend cleanup failed: " + repr(errors))

    return LibraryContext()
