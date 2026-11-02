"""Source and prospective test-inventory checks without importing PyTorch."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

NEW_METHODS = [
    "TestMPS.test_index_put_half_ordered_layouts_and_chunks",
    "TestMPS.test_index_put_half_ordered_multiple_index_strides",
    "TestMPS.test_index_put_half_ordered_repeated_index_backward",
    "TestMPS.test_index_put_half_ordered_storage_words_and_tail",
]
ALLOWED_BODY = "TestAdvancedIndexing.test_index_put_accumulate_nondeterministic"
ORIGINAL_RESULT_SHA = "eab61f146499b51c30e9cf4ec4b422914f2c89accebd3ef8fbde4dedd68919e2"
ORIGINAL_AST_SHA = "9ba8de955cb5b0cb1c8101bde990daf8aba35b63f0fb73566ab88ae7812c672a"

INVENTORY_DIGESTS = {
    "original_ids": "2991758b54e7e8557c8d5d35ae064e0bd79fffcc3be93b0340badd98683dabc8",
    "retained_skip_ids": "d68c54da9a58c695d944295acc05f6ca2bcbde97389e9255e8f43419c9f3bf68",
    "retained_expected_failure_ids": (
        "9586d30b6fbc7c19890f012dfd5bc4185f4a90ab37288d9c99cdf0e1cc4e2912"
    ),
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inventory(inventory, policy):
    for key, expected in INVENTORY_DIGESTS.items():
        digest = hashlib.sha256(
            json.dumps(inventory[key], separators=(",", ":")).encode()
        ).hexdigest()
        require(digest == expected, "Retained IDs or eligibility changed")
    old, added, required = (
        inventory["original_ids"],
        inventory["added_ids"],
        inventory["required_ids"],
    )
    require(
        inventory["schema"] == "owned-half-prospective-test-inventory-v1"
        and inventory["actual_collection_executed"] is False
        and inventory["original_result_sha256"] == ORIGINAL_RESULT_SHA
        and inventory["original_failures_preserved"] is True,
        "Wrong prospective inventory identity or qualification",
    )
    require(
        len(old) == len(set(old)) == 10353
        and old == sorted(old)
        and added == ["integrated_upstream_test_mps." + name for name in NEW_METHODS]
        and required == sorted(old + added)
        and len(set(required)) == 10357,
        "Missing, duplicate, reordered or altered test inventory",
    )
    for key, count in (("retained_skip_ids", 744), ("retained_expected_failure_ids", 382)):
        values = inventory[key]
        require(
            len(values) == len(set(values)) == count and set(values) <= set(old),
            "Changed retained eligibility inventory",
        )
    require(
        policy["previous_exact_methods"] == 10353
        and policy["added_exact_methods"] == 4
        and policy["required_exact_methods"] == 10357
        and policy["existing_ids_eligibility_tolerances_unchanged"] is True
        and policy["new_methods"] == NEW_METHODS
        and policy["allowed_existing_method_body_change"] == ALLOWED_BODY
        and policy["unchanged_test_module_ast_sha256"] == ORIGINAL_AST_SHA,
        "Changed test policy",
    )


def normalized_test_ast(text, *, require_added=True):
    tree = ast.parse(text)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        retained = []
        for method in node.body:
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name + "." + method.name
                if name in NEW_METHODS:
                    require(
                        not method.decorator_list, "Added test cannot be skipped or parametrized"
                    )
                    found.append(name)
                    continue
                if name == ALLOWED_BODY:
                    method.body = [ast.Pass()]
            retained.append(method)
        node.body = retained
    if require_added:
        require(sorted(found) == NEW_METHODS, "Missing or duplicate added source methods")
    return hashlib.sha256(
        json.dumps(portable_ast(tree), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def portable_ast(value):
    """Canonical structural fields, independent of ast.dump's Python 3.13 change.

    Empty optional fields carry no test behavior and are omitted consistently.
    Nonempty bodies, decorators, arguments, constants and type parameters remain
    in the digest. Source bytes are independently pinned by validate_source.
    """
    if isinstance(value, ast.AST):
        return {
            "node": type(value).__name__,
            "fields": {
                key: portable_ast(item)
                for key, item in ast.iter_fields(value)
                if item is not None and item != []
            },
        }
    if isinstance(value, list):
        return [portable_ast(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    if value is Ellipsis:
        return {"ellipsis": True}
    return value


def validate_source(source, contract, folder=None):
    folder = Path(folder) if folder else Path(__file__).parent
    policy = contract["test_inventory_policy"]
    inventory_path = folder / "test-inventory.json"
    require(sha256(inventory_path) == policy["inventory_sha256"], "Test inventory bytes changed")
    inventory = json.loads(inventory_path.read_text())
    validate_inventory(inventory, policy)
    for name, expected in contract["source_sha256"].items():
        require(sha256(source / name) == expected, "Native or Python source changed: " + name)
    require(
        normalized_test_ast((source / "test/test_mps.py").read_text()) == ORIGINAL_AST_SHA,
        "Existing test definitions, decorators, eligibility or tolerances changed",
    )
    return {
        "schema": "owned-half-source-static-check-v1",
        "passed": True,
        "source_files": len(contract["source_sha256"]),
        "prospective_test_count": len(inventory["required_ids"]),
        "original_ids_preserved": True,
        "original_eligibility_preserved": True,
        "actual_test_collection_executed": False,
        "native_compiled": False,
        "device_numerics_validated": False,
        "formal_eligible": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(Path(__file__).with_name("contract.json").read_text())
    print(json.dumps(validate_source(args.source, contract), indent=2))
