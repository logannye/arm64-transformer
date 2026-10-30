"""The hosted formatting revision changes only declared identity constants.

No framework import, package sync, build, compiler or device execution.
"""

import ast
import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "validation/upstream-pytorch"
NEW = ROOT / "validation/upstream-pytorch-format-v2"
OLD_PATCH = "3dd6bf6567284d798074e7d6102ff7f8b2a051a3a4ff2824f86184c120eb0ce9"
NEW_PATCH = "3834dca844ea3de6b40467ce214b99bce83fa06dae2bc454623516dafc0fdd71"
OLD_METAL = "d436b98da2ccde806e07e33d6d46156da8aa141707ce74deca62746f814cec9f"
NEW_METAL = "e49bd60d9a396272af948ec4a482987916a9952118447b5c5bc01e41350a5e85"


class NormalizeIdentity(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = {NEW_PATCH: OLD_PATCH, NEW_METAL: OLD_METAL}.get(node.value, node.value)
        return node

    def visit_Dict(self, node):
        self.generic_visit(node)
        retained = [
            (k, v)
            for k, v in zip(node.keys, node.values, strict=True)
            if not isinstance(k, ast.Constant)
            or k.value not in {"patch_revision", "prior_patch_sha256"}
        ]
        node.keys = [k for k, _ in retained]
        node.values = [v for _, v in retained]
        return node


class FormatRevisionTests(unittest.TestCase):
    def test_patch_identity_and_only_exact_two_line_formatter_suggestion(self):
        old = (OLD / "changes.patch").read_bytes()
        new = (NEW / "changes.patch").read_bytes()
        self.assertEqual(hashlib.sha256(old).hexdigest(), OLD_PATCH)
        self.assertEqual(hashlib.sha256(new).hexdigest(), NEW_PATCH)
        left, right = old.splitlines(), new.splitlines()
        self.assertEqual(len(left), len(right))
        changed = [(a, b) for a, b in zip(left, right, strict=True) if a != b]
        self.assertEqual(len(changed), 3)
        self.assertTrue(changed[0][0].startswith(b"index "))
        self.assertTrue(changed[0][1].startswith(b"index "))
        for a, b in changed[1:]:
            self.assertTrue(a.startswith(b"+") and b.startswith(b"+"))
            self.assertTrue(a.endswith(b"\\") and b.endswith(b"\\"))
        self.assertEqual(
            b"".join(b"".join(row[1:-1].split()) for row in (a for a, _ in changed[1:])),
            b"".join(b"".join(row[1:-1].split()) for row in (b for _, b in changed[1:])),
        )

    def test_build_implementation_and_limits_are_unchanged(self):
        old = ast.parse((OLD / "hosted_build.py").read_text())
        new = ast.parse((NEW / "hosted_build.py").read_text())
        self.assertEqual(ast.dump(old), ast.dump(NormalizeIdentity().visit(new)))

    def test_verifier_and_dependency_constraints_are_byte_identical(self):
        for filename in ("verify_runtime.py", "hosted-constraints.txt"):
            with self.subTest(filename=filename):
                self.assertEqual((OLD / filename).read_bytes(), (NEW / filename).read_bytes())

    def test_new_workflow_preserves_base_platform_bounds_and_serial_group(self):
        old = (ROOT / ".github/workflows/upstream-pytorch.yml").read_text()
        new = (ROOT / ".github/workflows/upstream-pytorch-format-v2.yml").read_text()
        normalized = new.replace(
            "PyTorch formatted-v2 integration validation", "PyTorch upstream integration validation"
        ).replace("upstream-pytorch-format-v2", "upstream-pytorch")
        normalized = normalized.replace("integration-format-v2-result", "integration-result")
        normalized = normalized.replace(
            "pytorch-format-v2-integration-${{", "pytorch-integration-${{"
        )
        self.assertEqual(old, normalized)


if __name__ == "__main__":
    unittest.main()
