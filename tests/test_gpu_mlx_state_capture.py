"""Guard the actual gradient adapter's capture contract without importing MLX."""

import ast
import copy
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(
    os.environ.get(
        "ARM64_MLX_CAPTURE_SOURCE",
        str(Path(__file__).resolve().parents[1] / "transformer/gpu_mlx.py"),
    )
)


def gradient_assignment():
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    backend = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Backend"
    )
    constructor = next(
        node
        for node in backend.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assignments = [
        node
        for node in constructor.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].value.id == "self"
        and node.targets[0].attr == "_loss_grad"
    ]
    if len(assignments) != 1:
        raise AssertionError("Expected one actual Backend._loss_grad compile declaration")
    return copy.deepcopy(assignments[0])


def exercise_assignment(assignment):
    """Execute only the compile declaration, with opaque state and token objects."""
    model = SimpleNamespace(state={"parameter": object()})
    instance = SimpleNamespace(model=model, optimizer=SimpleNamespace(state={"moment": object()}))
    captures = []
    calls = []

    def compile_stub(function, **kwargs):
        captures.append(kwargs)
        return function

    def value_grad_stub(actual_model, tokens, targets):
        calls.append((actual_model, tokens, targets))
        return "opaque-result"

    namespace = {
        "self": instance,
        "mx": SimpleNamespace(compile=compile_stub),
        "loss_grad": value_grad_stub,
    }
    module = ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    if len(captures) != 1:
        raise AssertionError("Expected exactly one compile call")
    if captures[0].get("inputs") is not model.state:
        raise AssertionError("Gradient capture inputs must be the exact model state container")
    if captures[0].get("outputs") is not model.state:
        raise AssertionError("Gradient capture outputs must be the same model state container")
    tokens, targets = object(), object()
    if instance._loss_grad(tokens, targets) != "opaque-result" or calls != [
        (model, tokens, targets)
    ]:
        raise AssertionError(
            "The compiled adapter no longer invokes the model value/gradient transform"
        )


class MLXGradientStateCaptureTests(unittest.TestCase):
    def test_actual_gradient_wrapper_captures_identical_input_output_state(self):
        exercise_assignment(gradient_assignment())

    def test_removing_output_capture_is_detected(self):
        assignment = gradient_assignment()
        assignment.value.keywords = [
            keyword for keyword in assignment.value.keywords if keyword.arg != "outputs"
        ]
        with self.assertRaisesRegex(AssertionError, "outputs must be the same model state"):
            exercise_assignment(assignment)

    def test_substituting_optimizer_state_as_output_is_detected(self):
        assignment = gradient_assignment()
        for keyword in assignment.value.keywords:
            if keyword.arg == "outputs":
                keyword.value = ast.parse("self.optimizer.state", mode="eval").body
        with self.assertRaisesRegex(AssertionError, "outputs must be the same model state"):
            exercise_assignment(assignment)

    def test_missing_input_capture_is_detected(self):
        assignment = gradient_assignment()
        assignment.value.keywords = [
            keyword for keyword in assignment.value.keywords if keyword.arg != "inputs"
        ]
        with self.assertRaisesRegex(AssertionError, "inputs must be the exact model state"):
            exercise_assignment(assignment)


if __name__ == "__main__":
    unittest.main(verbosity=2)
