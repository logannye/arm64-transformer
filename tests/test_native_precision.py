"""FP64 candidate primitive checks. Compilation is always explicit and external."""

import ctypes as ct
import platform

import numpy as np
import pytest

from transformer import spec
from transformer.validate import DEFAULT_LIBRARY, Native, workspace_tensor

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or not DEFAULT_LIBRARY.is_file(),
    reason="Requires an explicitly built native candidate",
)


@pytest.fixture
def native():
    value = Native()
    for name, count in (
        ("linear_forward_precise", 5),
        ("linear_backward_precise", 7),
        ("ln_backward_precise", 8),
    ):
        fn = getattr(value.lib, "tf_" + name)
        fn.argtypes, fn.restype = [ct.c_void_p] * count, None
    value.lib.tf_ln_forward_precise.argtypes = [ct.c_void_p] * 6 + [ct.c_uint64] * 2
    value.lib.tf_ln_forward_precise.restype = None
    value.lib.tf_precision_policy.argtypes = []
    value.lib.tf_precision_policy.restype = ct.c_uint64
    return value


@pytest.mark.parametrize("shape", [(3, 7, 5), (2, 512, 17), (2, 1024, 19)])
def test_precise_linear_forward_backward_accumulation(native, shape):
    rng = np.random.default_rng(20261103)
    r, i, o = shape
    x = rng.normal(size=(r, i)).astype("f4")
    w = rng.normal(size=(o, i)).astype("f4")
    b = rng.normal(size=o).astype("f4")
    dy = rng.normal(size=(r, o)).astype("f4")
    y, dx = np.full((r, o), np.nan, "f4"), np.full_like(x, np.nan)
    dw, db = rng.normal(size=w.shape).astype("f4"), rng.normal(size=o).astype("f4")
    old_dw, old_db = dw.copy(), db.copy()
    dims = np.array(shape, "i4")
    native.call("linear_forward_precise", x, w, b, y, dims)
    native.call("linear_backward_precise", x, w, dy, dx, dw, db, dims)
    for actual, expected in (
        (y, x.astype("f8") @ w.astype("f8").T + b),
        (dx, dy.astype("f8") @ w.astype("f8")),
        (dw, old_dw.astype("f8") + dy.astype("f8").T @ x.astype("f8")),
        (db, old_db.astype("f8") + dy.astype("f8").sum(0)),
    ):
        np.testing.assert_array_equal(actual, expected.astype("f4"))


@pytest.mark.parametrize("width", [7, 512, 1024])
def test_precise_norm_forward_backward_saved_state(native, width):
    rng = np.random.default_rng(20261103)
    x = (100 + rng.normal(size=(3, width)) * 0.1).astype("f4")
    gamma, beta = rng.normal(size=(2, width)).astype("f4")
    dy = rng.normal(size=x.shape).astype("f4")
    y, norm, dx = [np.full_like(x, np.nan) for _ in range(3)]
    inv = np.empty(3, "f4")
    dg, db = np.zeros(width, "f4"), np.zeros(width, "f4")
    native.call("ln_forward_precise", x, gamma, beta, y, norm, inv, 3, width)
    centered = x.astype("f8") - x.astype("f8").mean(-1, keepdims=True)
    scale = 1 / np.sqrt((centered * centered).mean(-1, keepdims=True) + spec.EPS)
    gold_norm = centered * scale
    np.testing.assert_array_equal(y, (gold_norm * gamma + beta).astype("f4"))
    np.testing.assert_array_equal(norm, gold_norm.astype("f4"))
    np.testing.assert_array_equal(inv, scale[:, 0].astype("f4"))
    native.call("ln_backward_precise", dy, norm, inv, gamma, dx, dg, db, np.array([3, width], "i4"))
    u = dy.astype("f8") * gamma
    n = norm.astype("f8")
    expected = u - u.mean(-1, keepdims=True) - n * (u * n).mean(-1, keepdims=True)
    np.testing.assert_allclose(dx, expected * inv[:, None], atol=1e-6, rtol=1e-6)
    # Parameter gradients accumulate one FP32 row contribution at a time.
    expected_dg, expected_db = np.zeros(width, "f4"), np.zeros(width, "f4")
    for row in range(3):
        expected_dg = (expected_dg.astype("f8") + dy[row].astype("f8") * n[row]).astype("f4")
        expected_db = (expected_db.astype("f8") + dy[row]).astype("f4")
    np.testing.assert_array_equal(dg, expected_dg)
    np.testing.assert_array_equal(db, expected_db)


def test_compiled_policy_is_explicit(native):
    header = (DEFAULT_LIBRARY.parent / "precision.h").read_text()
    policy = int(header.split()[-1]) if spec.WIDTH >= 512 else 0
    assert native.lib.tf_precision_policy() == policy


def test_second_normalization_candidate_is_wired_in_both_directions(native):
    if not native.lib.tf_precision_policy() & 4:
        pytest.skip("Second-normalization refinement is not selected")
    from transformer.reference import Transformer

    model = Transformer(1729)
    flat = model.flat_parameters()
    tokens = np.arange(spec.CONTEXT, dtype="i4")
    targets = (tokens + 1) % spec.VOCAB
    work = native.forward(flat, tokens)
    r1 = workspace_tensor(work, "r1").astype("f8")
    gamma, beta = (getattr(model, name).detach().numpy() for name in ("ln2_weight", "ln2_bias"))
    centered = r1 - r1.mean(-1, keepdims=True)
    normalized = centered / np.sqrt((centered * centered).mean(-1, keepdims=True) + spec.EPS)
    np.testing.assert_array_equal(
        workspace_tensor(work, "n2"), (normalized * gamma + beta).astype("f4")
    )
    native.backward(flat, tokens, targets, work)
    dx = np.empty((spec.CONTEXT, spec.WIDTH), "f4")
    dg, db = np.zeros(spec.WIDTH, "f4"), np.zeros(spec.WIDTH, "f4")
    native.call(
        "ln_backward_precise",
        workspace_tensor(work, "dn2"),
        workspace_tensor(work, "norm2"),
        workspace_tensor(work, "inv2"),
        gamma.copy(),
        dx,
        dg,
        db,
        np.array([spec.CONTEXT, spec.WIDTH], "i4"),
    )
    np.testing.assert_array_equal(workspace_tensor(work, "temp"), dx)
