import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from transformer import spec
from transformer.gpu_metal import Backend
from transformer.precision import attention_reference
from transformer.precision_audit import comparison
from transformer.reference import Transformer

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
args = parser.parse_args()
dest = Path("work/gpu-v1/metal-development") / args.label
dest.mkdir(parents=True, exist_ok=True)
out = dest / f"smoke-width-{spec.WIDTH}.json"
if out.exists():
    raise FileExistsError(out)
model = Transformer(42)
initial = model.flat_parameters()
backend = Backend(initial)
records = []


def check(name, actual, expected, tolerance="forward"):
    item = comparison(name, actual, expected, tolerance=tolerance)
    records.append(item)
    print(name, item["passed"], item["max_absolute_error"], item["failed_coordinates"], flush=True)
    return item


rng = np.random.default_rng(20261105)
for m, n, k in [(1, 1, 1), (5, 19, 17), (16, 32, 32), (31, 65, 33)]:
    x = rng.normal(0, 0.2, (m, k)).astype("f4")
    w = rng.normal(0, 0.2, (n, k)).astype("f4")
    bias = rng.normal(0, 0.1, n).astype("f4")
    dy = rng.normal(0, 0.2, (m, n)).astype("f4")
    vals = {key: backend.to_numpy(v) for key, v in backend.linear(x, w, bias, dy).items()}
    check(f"linear{m, n, k}/y", vals["output"], x.astype("f8") @ w.astype("f8").T + bias)
    check(f"linear{m, n, k}/dx", vals["dx"], dy.astype("f8") @ w.astype("f8"), "gradient")
    check(f"linear{m, n, k}/dw", vals["dweight"], dy.astype("f8").T @ x.astype("f8"), "gradient")
    check(f"linear{m, n, k}/db", vals["dbias"], dy.astype("f8").sum(0), "gradient")

for t in (1, 3, 5, 31, 32):
    qkv = rng.normal(0, 0.3, (t, 3 * spec.WIDTH)).astype("f4")
    dy = rng.normal(0, 0.2, (t, spec.WIDTH)).astype("f4")
    vals = {key: backend.to_numpy(v) for key, v in backend.attention(qkv, dy).items()}
    gold = attention_reference(qkv, 4, dy)
    check(f"attention{t}/output", vals["output"], gold["output"])
    check(f"attention{t}/prob", vals["probabilities"], gold["probabilities"], "math")
    check(f"attention{t}/grad", vals["dqkv"], gold["dqkv"], "gradient")

x = rng.normal(0, 0.2, (5, spec.WIDTH)).astype("f4")
gamma = rng.normal(1, 0.1, spec.WIDTH).astype("f4")
beta = rng.normal(0, 0.1, spec.WIDTH).astype("f4")
dy = rng.normal(0, 0.2, x.shape).astype("f4")
tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
tg = torch.tensor(gamma, dtype=torch.float64, requires_grad=True)
tb = torch.tensor(beta, dtype=torch.float64, requires_grad=True)
y = F.layer_norm(tx, (spec.WIDTH,), tg, tb, spec.EPS)
y.backward(torch.tensor(dy, dtype=torch.float64))
vals = {key: backend.to_numpy(v) for key, v in backend.layernorm(x, gamma, beta, dy).items()}
for key, gold in [("output", y.detach()), ("dx", tx.grad), ("dgamma", tg.grad), ("dbeta", tb.grad)]:
    check("norm/" + key, vals[key], gold.numpy(), "forward" if key == "output" else "gradient")

tokens = np.array([[1, 1, 2, 3, 1], [3, 1, 2, 2, 3]], dtype="i4")
targets = np.roll(tokens, -1, axis=1).copy()
batch = backend.upload_inputs(tokens, targets)
gold_model = Transformer(42).double()
gold_logits = gold_model(torch.from_numpy(tokens.astype("i8")))
gold_loss = F.cross_entropy(
    gold_logits.reshape(-1, 128), torch.from_numpy(targets.astype("i8")).reshape(-1)
)
gold_loss.backward()
gold_grad = torch.cat(
    [getattr(gold_model, name).grad.flatten() for name in spec.PARAM_SHAPES]
).numpy()
native_logits = backend.to_numpy(backend.forward(batch))
check("full/logits", native_logits, gold_logits.detach().numpy())
loss, grad = backend.loss_and_grad(batch)
native_loss = backend.to_numpy(loss)
native_grad = backend.to_numpy(grad)
check("full/loss", native_loss, gold_loss.detach().numpy())
check("full/grad", native_grad, gold_grad, "gradient")
state0 = backend.export_state()
native_loss_step = backend.to_numpy(backend.train_step(batch))
state1 = backend.export_state()
expected_m = (native_grad.astype("f8") * 0.1).astype("f4")
expected_v = (native_grad.astype("f8") ** 2 * 0.001).astype("f4")
expected_p = (
    initial.astype("f8")
    - 0.001 * (expected_m.astype("f8") / 0.1) / (np.sqrt(expected_v.astype("f8") / 0.001) + 1e-8)
).astype("f4")
check("adam/m", state1["m"], expected_m, "optimizer")
check("adam/v", state1["v"], expected_v, "optimizer")
check("adam/parameters", state1["parameters"], expected_p, "optimizer")
assert state1["step"] == 1
backend.load_state(state0)
backend.to_numpy(backend.train_step(batch))
state_repeat = backend.export_state()
exact_repeat = all(
    state1[k].tobytes() == state_repeat[k].tobytes() for k in ("parameters", "m", "v")
)
backend.load_state(state0)
backend.warmup(batch, 2)
restored = backend.export_state()
exact_restore = (
    all(state0[k].tobytes() == restored[k].tobytes() for k in ("parameters", "m", "v"))
    and restored["step"] == 0
)
dataset = backend.upload_dataset(np.concatenate((tokens, targets[:, -1:]), axis=1).astype("i4"))
schedule = backend.upload_indices(np.array([1, 0], dtype="i4"))
gathered = backend.gather_batch(dataset, schedule, 0, 2)
gather_logits = backend.to_numpy(backend.forward(gathered))
check("gather/logits", gather_logits, native_logits[::-1])
finite = backend.check_finite()
guards = backend.check_guards()
integrity = {}
saved = backend.export_state()
exported = backend.export_parameters()
exported[0] += np.float32(1)
integrity["parameter_export_is_independent"] = (
    backend.export_parameters().tobytes() == saved["parameters"].tobytes()
)
bad = dict(saved)
bad["v"] = saved["v"].copy()
bad["v"][0] = -1
try:
    backend.load_state(bad)
    integrity["invalid_state_rejected"] = False
except ValueError:
    integrity["invalid_state_rejected"] = True
unchanged = backend.export_state()
integrity["invalid_state_transactional"] = (
    all(saved[k].tobytes() == unchanged[k].tobytes() for k in ("parameters", "m", "v"))
    and saved["step"] == unchanged["step"]
)
for label, invalid in [
    ("token_dtype", tokens.astype("i8")),
    ("token_range", np.full(tokens.shape, 128, dtype="i4")),
    ("token_rank", tokens.ravel()),
    ("batch_limit", np.zeros((17, 2), dtype="i4")),
    ("context_limit", np.zeros((1, 33), dtype="i4")),
]:
    try:
        backend.upload_inputs(invalid)
        integrity[label + "_rejected"] = False
    except ValueError:
        integrity[label + "_rejected"] = True
v = backend.forward(batch)
backend.forward(batch)
try:
    backend.to_numpy(v)
    integrity["stale_workspace_rejected"] = False
except ValueError:
    integrity["stale_workspace_rejected"] = True
first = backend.gather_batch(dataset, schedule, 0, 1)
backend.gather_batch(dataset, schedule, 1, 1)
try:
    backend.forward(first)
    integrity["stale_gather_rejected"] = False
except ValueError:
    integrity["stale_gather_rejected"] = True
uploaded_copy = backend.to_numpy(backend.forward(batch))
tokens[:] = 0
targets[:] = 0
integrity["uploaded_inputs_owned"] = np.array_equal(
    backend.to_numpy(backend.forward(batch)), uploaded_copy
)
if spec.WIDTH == 32:
    poisoned = Backend(np.full(spec.PARAM_COUNT, 1e20, dtype="f4"))
    poisoned_batch = poisoned.upload_inputs(np.ones((1, 2), dtype="i4"))
    poisoned_value = poisoned.forward(poisoned_batch)
    integrity["gpu_finite_scan_detects_nonfinite"] = not poisoned.check_finite(poisoned_value)
    poisoned.close()
result = {
    "width": spec.WIDTH,
    "scope": "Development-only primitive/full-gradient/one-step smoke; no formal data or timing",
    "checks": records,
    "integrity": integrity,
    "passed": all(c["passed"] for c in records)
    and all(integrity.values())
    and exact_repeat
    and exact_restore
    and finite
    and guards,
    "exact_state_repeat": exact_repeat,
    "exact_warmup_restore": exact_restore,
    "finite": finite,
    "guards": guards,
    "capabilities": backend.capabilities(),
    "memory": backend.memory_stats(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
out.write_text(json.dumps(result, indent=2) + "\n")
backend.close()
print("PASSED", result["passed"], flush=True)
raise SystemExit(0 if result["passed"] else 1)
