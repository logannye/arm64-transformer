"""No-network regressions for owned-release authentication and byte verification."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import verify_release_assets as verifier


class FakeClient:
    def __init__(self, contract, workspace):
        self.contract, self.workspace = contract, workspace
        self.user_calls = self.inventory_calls = self.download_calls = 0
        self.asset_calls = {}
        self.change = None
        self.release_metadata = {
            "id": contract["release_id"],
            "node_id": "TEST_NODE",
            "tag_name": verifier.TAG,
            "name": "Test release",
            "body": "A synthetic fixture",
            "target_commitish": "testcommit",
            "draft": False,
            "prerelease": False,
            "created_at": "test_time",
            "published_at": "test_time",
            "url": f"https://api.github.com/repos/{verifier.REPOSITORY}/releases/{contract['release_id']}",
        }
        self.metadata = {
            row["asset_id"]: {
                "id": row["asset_id"],
                "node_id": "TEST_ASSET",
                "name": row["name"],
                "label": None,
                "state": "uploaded",
                "size": row["bytes"],
                "digest": "sha256:" + row["sha256"],
                "content_type": "application/octet-stream",
                "created_at": "test_time",
                "updated_at": "test_time",
                "url": f"https://api.github.com/repos/{verifier.REPOSITORY}/releases/assets/{row['asset_id']}",
            }
            for row in contract["assets"]
        }

    def altered(self, operation, value):
        result = copy.deepcopy(value)
        return self.change(operation, result) if self.change else result

    def user(self):
        self.user_calls += 1
        return self.altered("user", {"login": verifier.LOGIN, "id": 7})

    def release(self):
        return self.altered("release", self.release_metadata)

    def assets(self):
        self.inventory_calls += 1
        return self.altered("assets", list(self.metadata.values()))

    def asset(self, identifier):
        self.asset_calls[identifier] = self.asset_calls.get(identifier, 0) + 1
        return self.altered("asset", self.metadata[identifier])

    def download(self, identifier, expected_bytes):
        self.download_calls += 1
        row = next(row for row in self.contract["assets"] if row["asset_id"] == identifier)
        return self.altered(
            "download",
            {
                "stdout_bytes": expected_bytes,
                "stdout_sha256": row["sha256"],
                "returncode": 0,
                "problem": None,
                "payload_copied_to_disk": False,
            },
        )


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name).resolve()
        (self.workspace / "work").mkdir()
        assets = []
        for index, name in enumerate(("alpha.tar.gz", "beta.json")):
            contents = f"synthetic asset {name}".encode()
            path = self.workspace / "work" / name
            path.write_bytes(contents)
            assets.append(
                {
                    "name": name,
                    "asset_id": 10 + index,
                    "local_path": "work/" + name,
                    "bytes": len(contents),
                    "sha256": hashlib.sha256(contents).hexdigest(),
                }
            )
        self.contract = {
            "schema": "gpu-owned-release-assets-v1",
            "repository": verifier.REPOSITORY,
            "tag_name": verifier.TAG,
            "release_id": 123,
            "draft": False,
            "prerelease": False,
            "expected_login": verifier.LOGIN,
            "assets": assets,
        }
        self.path = self.workspace / "contract.json"
        self.path.write_text(json.dumps(self.contract, indent=3))
        self.output = self.workspace / "receipts"
        self.client = FakeClient(self.contract, self.workspace)
        self.factories = 0

    def factory(self, output, release_id):
        self.factories += 1
        self.assertEqual(release_id, 123)
        return self.client

    def verify(self, **overrides):
        arguments = {
            "expected_release_id": 123,
            "expected_contract_sha256": verifier.canonical_sha256(self.contract),
            "client_factory": self.factory,
        }
        arguments.update(overrides)
        return verifier.verify(self.path, self.workspace, self.output, **arguments)

    def test_valid_exact_inventory_streams_all_assets_without_payload_files_or_urls(self):
        for metadata in self.client.metadata.values():
            metadata["label"] = "https://private.invalid/?token=TEST_SECRET"
        result = self.verify()
        self.assertTrue(result["passed"])
        self.assertTrue(result["all_assets_streamed_and_exact_inventory_verified"])
        self.assertEqual(self.client.download_calls, 2)
        self.assertEqual(self.client.inventory_calls, 2)
        names = sorted(path.name for path in self.output.iterdir())
        self.assertEqual(
            names, ["asset-0000.json", "asset-0001.json", "contract.json", "summary.json"]
        )
        self.assertNotIn("https://", "".join(path.read_text() for path in self.output.iterdir()))
        self.assertNotIn("TEST_SECRET", "".join(path.read_text() for path in self.output.iterdir()))

    def test_external_hash_or_release_id_mismatch_fails_before_network(self):
        for overrides in (
            {"expected_release_id": 124},
            {"expected_contract_sha256": "0" * 64},
            {"expected_release_id": True},
            {"expected_release_id": 0},
        ):
            with self.assertRaises(ValueError):
                self.verify(**overrides)
        self.assertEqual(self.factories, 0)
        self.assertFalse(self.output.exists())

    def test_duplicate_json_keys_and_nonfinite_values_rejected(self):
        for text in ('{"id":1,"id":2}', '{"value":NaN}', '{"value":Infinity}'):
            with self.assertRaises(ValueError):
                verifier.parse(text)

    def test_wrong_repository_tag_or_login_rejected(self):
        for key, value in (
            ("repository", "someone/else"),
            ("tag_name", "another-tag"),
            ("expected_login", "someone"),
        ):
            changed = {**self.contract, key: value}
            with self.assertRaises(ValueError):
                verifier.validate_contract(changed, 123, verifier.canonical_sha256(changed))

    def test_contract_duplicate_or_reordered_assets_rejected(self):
        for assets in (self.contract["assets"] * 2, list(reversed(self.contract["assets"]))):
            changed = {**self.contract, "assets": assets}
            with self.assertRaises(ValueError):
                verifier.validate_contract(changed, 123, verifier.canonical_sha256(changed))

    def test_unsafe_local_paths_rejected(self):
        for path in ("/etc/passwd", "work/../secret", "work//asset", "work/./asset", "work/a\\b"):
            with self.assertRaises(ValueError):
                verifier.relative_path(path)

    def test_changed_local_bytes_or_symlink_rejected_before_network(self):
        asset = self.workspace / self.contract["assets"][0]["local_path"]
        asset.write_bytes(b"substituted payload")
        with self.assertRaisesRegex(ValueError, "Local asset differs"):
            self.verify()
        self.assertEqual(self.factories, 0)
        self.output = self.workspace / "second-attempt"
        asset.unlink()
        asset.symlink_to(self.path)
        with self.assertRaisesRegex(ValueError, "ordinary"):
            self.verify()
        self.assertEqual(self.factories, 0)

    def test_authentication_mismatch_prevents_download(self):
        self.client.change = lambda op, value: (
            {**value, "login": "other"} if op == "user" else value
        )
        with self.assertRaisesRegex(ValueError, "Authenticated"):
            self.verify()
        self.assertEqual(self.client.download_calls, 0)

    def test_missing_extra_duplicate_remote_assets_rejected(self):
        assets = list(self.client.metadata.values())
        for mutated in (
            assets[:1],
            assets + [assets[0]],
            [dict(assets[0], name="unexpected"), assets[1]],
        ):
            with self.assertRaises(ValueError):
                verifier.inventory_check(mutated, self.contract)

    def test_wrong_server_digest_asset_id_or_origin_rejected(self):
        row = self.contract["assets"][0]
        value = self.client.metadata[row["asset_id"]]
        for change in (
            {"digest": "sha256:" + "0" * 64},
            {"id": 99},
            {"state": "new"},
            {"url": "https://untrusted.example/asset"},
            {"digest": None},
        ):
            with self.assertRaises(ValueError):
                verifier.asset_identity({**value, **change}, row)

    def test_truncated_or_changed_download_rejected(self):
        self.client.change = lambda op, value: (
            {**value, "stdout_bytes": 1} if op == "download" else value
        )
        with self.assertRaisesRegex(ValueError, "Downloaded"):
            self.verify()
        self.assertFalse(json.loads((self.output / "summary.json").read_text())["passed"])

    def test_one_byte_hash_change_rejected(self):
        self.client.change = lambda op, value: (
            {**value, "stdout_sha256": "0" * 64} if op == "download" else value
        )
        with self.assertRaisesRegex(ValueError, "Downloaded"):
            self.verify()

    def test_remote_metadata_change_during_download_rejected(self):
        def change(op, value):
            if op == "asset" and self.client.download_calls:
                value["label"] = "changed label"
            return value

        self.client.change = change
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            self.verify()

    def test_final_inventory_change_rejected(self):
        def change(op, value):
            return value[:1] if op == "assets" and self.client.inventory_calls == 2 else value

        self.client.change = change
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            self.verify()

    def test_final_release_or_account_change_rejected(self):
        def change(op, value):
            if op == "release" and self.client.download_calls == 2:
                value["body"] = "changed body"
            return value

        self.client.change = change
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.verify()

    def test_authenticated_user_id_change_rejected_at_completion(self):
        def change(operation, value):
            if operation == "user" and self.client.user_calls == 2:
                value["id"] += 1
            return value

        self.client.change = change
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.verify()

    def test_wrong_remote_release_id_rejected(self):
        self.client.release_metadata["id"] = 124
        with self.assertRaisesRegex(ValueError, "Release identity"):
            self.verify()
        self.assertEqual(self.client.download_calls, 0)

    def test_contract_file_change_during_download_rejected(self):
        def change(operation, value):
            if operation == "download":
                self.path.write_text(json.dumps(self.contract, indent=4))
            return value

        self.client.change = change
        with self.assertRaisesRegex(ValueError, "Contract changed"):
            self.verify()

    def test_changed_local_file_during_download_rejected(self):
        def change(op, value):
            if op == "download":
                path = self.workspace / self.contract["assets"][0]["local_path"]
                path.write_bytes(b"unexpected changed file")
            return value

        self.client.change = change
        with self.assertRaisesRegex(ValueError, "Local asset changed"):
            self.verify()

    def test_failure_details_cannot_leak_remote_text(self):
        def change(op, value):
            raise RuntimeError("https://secret.invalid/token?signed=PRIVATE")

        self.client.change = change
        with self.assertRaises(RuntimeError):
            self.verify()
        saved = (self.output / "summary.json").read_text()
        self.assertNotIn("PRIVATE", saved)
        self.assertNotIn("https://", saved)
        self.assertIn('"error_type": "RuntimeError"', saved)


class StreamingTests(unittest.TestCase):
    def test_payload_is_hashed_not_captured(self):
        data, result = verifier.run_stream(
            [sys.executable, "-c", "print('payload', end='')"], max_stdout_bytes=7
        )
        self.assertIsNone(data)
        self.assertEqual(result["stdout_sha256"], hashlib.sha256(b"payload").hexdigest())

    def test_stderr_secret_url_never_retained(self):
        with self.assertRaises(verifier.CommandFailure) as caught:
            verifier.run_stream(
                [
                    sys.executable,
                    "-c",
                    "import sys;sys.stderr.write('HTTP 403 https://secret.invalid/?token=PRIVATE');sys.exit(1)",
                ]
            )
        receipt = json.dumps(caught.exception.receipt)
        self.assertNotIn("PRIVATE", receipt)
        self.assertNotIn("https://", receipt)
        self.assertIn("authentication_or_permission", receipt)

    def test_stdout_and_stderr_bounds_abort(self):
        for channel, limits, problem in (
            ("stdout", {"max_stdout_bytes": 1}, "payload_byte_limit"),
            ("stderr", {}, "stderr_byte_limit"),
        ):
            with (
                self.subTest(channel=channel),
                self.assertRaises(verifier.CommandFailure) as caught,
            ):
                verifier.run_stream(
                    [
                        sys.executable,
                        "-c",
                        f"import sys;sys.{channel}.write('x'*{verifier.STDERR_LIMIT + 10})",
                    ],
                    **limits,
                )
            self.assertEqual(caught.exception.receipt["problem"], problem)

    def test_timeout_kills_child(self):
        with self.assertRaises(verifier.CommandFailure) as caught:
            verifier.run_stream(
                [sys.executable, "-c", "import time;time.sleep(5)"],
                total_timeout=0.1,
                idle_timeout=0.1,
            )
        self.assertEqual(caught.exception.receipt["problem"], "command_timeout")

    def test_captured_metadata_has_independent_memory_bound(self):
        with self.assertRaises(verifier.CommandFailure) as caught:
            verifier.run_stream(
                [
                    sys.executable,
                    "-c",
                    f"import sys;sys.stdout.write('x'*{verifier.JSON_LIMIT + 1})",
                ],
                capture=True,
            )
        self.assertEqual(caught.exception.receipt["problem"], "metadata_output_limit")

    def test_pagination_reads_until_short_page(self):
        client = object.__new__(verifier.GitHubClient)
        client.release_id = 123
        calls = []

        def call(endpoint, operation):
            calls.append(endpoint)
            return list(range(100)) if len(calls) == 1 else [100]

        client.call = call
        self.assertEqual(len(client.assets()), 101)
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[-1].endswith("page=2"))

    def test_pagination_bound_prevents_endless_inventory(self):
        client = object.__new__(verifier.GitHubClient)
        client.release_id = 123
        client.call = lambda *args: [None] * 100
        with (
            patch.object(verifier, "MAX_PAGES", 2),
            self.assertRaisesRegex(ValueError, "pagination"),
        ):
            client.assets()


if __name__ == "__main__":
    unittest.main()
