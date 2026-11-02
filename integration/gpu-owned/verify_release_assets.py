"""Read-only, externally bound verification of the owned GPU engineering release.

No remote request occurs on import. Payloads are streamed to SHA-256, never saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

REPOSITORY = "logannye/arm64-transformer"
TAG = "transformer-gpu-owned-v4"
LOGIN = "logannye"
CHUNK = 1024 * 1024
JSON_LIMIT = 4 * CHUNK
STDERR_LIMIT = CHUNK
MAX_PAGES = 100
ASSET_FIELDS = (
    "id",
    "node_id",
    "name",
    "label",
    "state",
    "size",
    "digest",
    "content_type",
    "url",
    "created_at",
    "updated_at",
)
RELEASE_FIELDS = (
    "id",
    "node_id",
    "tag_name",
    "name",
    "body",
    "target_commitish",
    "draft",
    "prerelease",
    "created_at",
    "published_at",
    "url",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(UTC).isoformat()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def parse(data):
    def invalid_constant(value):
        raise ValueError("Nonfinite JSON value")

    return json.loads(data, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def canonical_sha256(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def hexsha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def relative_path(value):
    require(isinstance(value, str) and bool(value), "Missing relative asset path")
    parts = value.split("/")
    require(
        parts[0] in ("work", "outputs")
        and len(parts) > 1
        and all(
            part not in ("", ".", "..") and re.fullmatch(r"[A-Za-z0-9._-]+", part) for part in parts
        )
        and str(PurePosixPath(value)) == value,
        "Unsafe local asset path",
    )
    return PurePosixPath(value)


def validate_contract(contract, expected_release_id, expected_contract_sha256):
    require(
        type(expected_release_id) is int and expected_release_id > 0,
        "A positive externally supplied release ID is required",
    )
    require(hexsha(expected_contract_sha256), "An external canonical contract SHA-256 is required")
    require(
        isinstance(contract, dict)
        and set(contract)
        == {
            "schema",
            "repository",
            "release_id",
            "tag_name",
            "draft",
            "prerelease",
            "expected_login",
            "assets",
        },
        "Unexpected contract fields",
    )
    require(contract["schema"] == "gpu-owned-release-assets-v1", "Wrong asset contract schema")
    require(
        contract["repository"] == REPOSITORY
        and contract["tag_name"] == TAG
        and contract["expected_login"] == LOGIN,
        "Unexpected release scope",
    )
    require(
        type(contract["release_id"]) is int and contract["release_id"] == expected_release_id,
        "Release ID differs from external anchor",
    )
    require(
        type(contract["draft"]) is bool and type(contract["prerelease"]) is bool,
        "Invalid release flags",
    )
    require(
        isinstance(contract["assets"], list) and 0 < len(contract["assets"]) <= 10_000,
        "Invalid expected asset inventory",
    )
    names, ids, paths = [], set(), set()
    for row in contract["assets"]:
        require(
            isinstance(row, dict)
            and set(row) == {"name", "asset_id", "local_path", "bytes", "sha256"},
            "Unexpected asset fields",
        )
        require(
            isinstance(row["name"], str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", row["name"]),
            "Unsafe asset name",
        )
        require(
            type(row["asset_id"]) is int and row["asset_id"] > 0 and row["asset_id"] not in ids,
            "Duplicate or invalid asset ID",
        )
        require(
            type(row["bytes"]) is int
            and 0 <= row["bytes"] < 2_000_000_000
            and hexsha(row["sha256"]),
            "Invalid expected asset bytes or SHA-256",
        )
        relative_path(row["local_path"])
        require(row["local_path"] not in paths, "Duplicate local asset path")
        names.append(row["name"])
        ids.add(row["asset_id"])
        paths.add(row["local_path"])
    require(names == sorted(set(names)), "Assets must have unique canonical name order")
    require(
        canonical_sha256(contract) == expected_contract_sha256, "Canonical contract hash mismatch"
    )
    return contract


def ordinary(path):
    require(path.is_file() and not path.is_symlink(), "Expected an ordinary local file")
    require(all(not parent.is_symlink() for parent in path.parents), "Symlink parent forbidden")
    return path


def stat_key(path):
    info = ordinary(path).stat()
    return {
        name: getattr(info, name)
        for name in (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mode",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_nlink",
        )
    }


def file_identity(path):
    before = stat_key(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        require(
            {key: getattr(os.fstat(stream.fileno()), key) for key in before} == before,
            "Local file replaced before hashing",
        )
        value = hashlib.file_digest(stream, "sha256").hexdigest()
    require(stat_key(path) == before, "Local file changed during hashing")
    return {"bytes": before["st_size"], "sha256": value, "stat": before}


def write_new(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_summary(path, value):
    temporary = path.with_suffix(".writing")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


class CommandFailure(RuntimeError):
    def __init__(self, receipt):
        super().__init__("Read-only GitHub request failed")
        self.receipt = receipt


def run_stream(
    command, *, capture=False, total_timeout=3600, idle_timeout=120, max_stdout_bytes=None
):
    """Bounded stdout/stderr; no command arguments or raw stderr enter receipts."""
    environment = os.environ.copy()
    for key in ("GH_DEBUG", "GH_LOG", "GH_FORCE_TTY"):
        environment.pop(key, None)
    environment.update(GH_PROMPT_DISABLED="1", GH_PAGER="cat", PAGER="cat", NO_COLOR="1")
    start = last = time.monotonic()
    stdout_hash, stderr_hash = hashlib.sha256(), hashlib.sha256()
    stdout_bytes = stderr_bytes = 0
    data, stderr_prefix = bytearray(), bytearray()
    problem = None
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        start_new_session=True,
    )
    selector = selectors.DefaultSelector()
    for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
        os.set_blocking(pipe.fileno(), False)
        selector.register(pipe, selectors.EVENT_READ, name)
    try:
        while selector.get_map():
            now = time.monotonic()
            remaining = min(total_timeout - (now - start), idle_timeout - (now - last))
            if remaining <= 0:
                problem = "command_timeout"
                break
            for key, _ in selector.select(timeout=min(1.0, remaining)):
                block = os.read(key.fileobj.fileno(), CHUNK)
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                last = time.monotonic()
                if key.data == "stdout":
                    stdout_hash.update(block)
                    stdout_bytes += len(block)
                    if max_stdout_bytes is not None and stdout_bytes > max_stdout_bytes:
                        problem = "payload_byte_limit"
                        break
                    if capture:
                        if stdout_bytes > JSON_LIMIT:
                            problem = "metadata_output_limit"
                            break
                        data.extend(block)
                else:
                    stderr_hash.update(block)
                    stderr_bytes += len(block)
                    stderr_prefix.extend(block[: max(0, 16384 - len(stderr_prefix))])
                    if stderr_bytes > STDERR_LIMIT:
                        problem = "stderr_byte_limit"
                        break
            if problem:
                break
        if problem:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        else:
            try:
                process.wait(timeout=max(0.1, min(10, total_timeout - (time.monotonic() - start))))
            except subprocess.TimeoutExpired:
                problem = "command_exit_timeout"
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
    lower = bytes(stderr_prefix).lower()
    patterns = {
        "authentication_or_permission": (
            b"http 401",
            b"http 403",
            b"authentication",
            b"gh auth login",
        ),
        "not_found": (b"http 404",),
        "rate_limited": (b"rate limit",),
        "network": (b"timeout", b"connection", b"no such host"),
    }
    receipt = {
        "returncode": process.returncode,
        "seconds": time.monotonic() - start,
        "stdout_bytes": stdout_bytes,
        "stdout_sha256": stdout_hash.hexdigest(),
        "stderr_bytes": stderr_bytes,
        "stderr_sha256": stderr_hash.hexdigest(),
        "stderr_markers": [
            name for name, values in patterns.items() if any(p in lower for p in values)
        ],
        "raw_stderr_retained": False,
        "payload_copied_to_disk": False,
        "problem": problem,
    }
    if problem or process.returncode != 0:
        raise CommandFailure(receipt)
    return bytes(data) if capture else None, receipt


class GitHubClient:
    def __init__(self, output, release_id):
        self.gh = shutil.which("gh")
        require(self.gh, "GitHub CLI is unavailable")
        self.output, self.release_id, self.calls = output, release_id, []

    def call(self, endpoint, operation, *, binary=False, expected_bytes=None):
        command = [
            self.gh,
            "api",
            "--method",
            "GET",
            "--hostname",
            "github.com",
            "-H",
            "Accept:application/octet-stream" if binary else "Accept:application/vnd.github+json",
            endpoint,
        ]
        try:
            data, receipt = run_stream(
                command,
                capture=not binary,
                total_timeout=3600 if binary else 90,
                max_stdout_bytes=expected_bytes if binary else JSON_LIMIT,
            )
        except CommandFailure as error:
            receipt = {**error.receipt, "operation": operation, "http_method": "GET"}
            self.calls.append(receipt)
            write_new(self.output / f"api-{len(self.calls):04d}.json", receipt)
            raise
        receipt.update(operation=operation, http_method="GET")
        self.calls.append(receipt)
        write_new(self.output / f"api-{len(self.calls):04d}.json", receipt)
        return receipt if binary else parse(data)

    def user(self):
        return self.call("user", "authenticated_user")

    def release(self):
        return self.call(f"repos/{REPOSITORY}/releases/{self.release_id}", "release_metadata")

    def assets(self):
        result = []
        for page in range(1, MAX_PAGES + 1):
            rows = self.call(
                f"repos/{REPOSITORY}/releases/{self.release_id}/assets?per_page=100&page={page}",
                "paginated_asset_inventory",
            )
            require(isinstance(rows, list) and len(rows) <= 100, "Invalid paginated asset response")
            result.extend(rows)
            if len(rows) < 100:
                return result
        raise ValueError("Asset pagination exceeds its explicit bound")

    def asset(self, asset_id):
        return self.call(f"repos/{REPOSITORY}/releases/assets/{asset_id}", "asset_metadata")

    def download(self, asset_id, expected_bytes):
        return self.call(
            f"repos/{REPOSITORY}/releases/assets/{asset_id}",
            "asset_payload_sha256",
            binary=True,
            expected_bytes=expected_bytes,
        )


def user_identity(user):
    require(
        isinstance(user, dict)
        and user.get("login") == LOGIN
        and type(user.get("id")) is int
        and user["id"] > 0,
        "Authenticated account mismatch",
    )
    return {"login": LOGIN, "id": user["id"]}


def release_identity(release, contract):
    require(
        isinstance(release, dict) and all(key in release for key in RELEASE_FIELDS),
        "Incomplete release metadata",
    )
    require(
        type(release["id"]) is int
        and release["id"] == contract["release_id"]
        and release["tag_name"] == TAG
        and release["draft"] is contract["draft"]
        and release["prerelease"] is contract["prerelease"],
        "Release identity or state differs",
    )
    require(
        (release["published_at"] is None) == contract["draft"], "Release publication state differs"
    )
    require(
        release["url"]
        == f"https://api.github.com/repos/{REPOSITORY}/releases/{contract['release_id']}",
        "Unexpected release API origin",
    )
    return {
        "id": release["id"],
        "tag_name": TAG,
        "draft": release["draft"],
        "prerelease": release["prerelease"],
        "metadata_sha256": canonical_sha256({key: release[key] for key in RELEASE_FIELDS}),
    }


def asset_identity(asset, expected=None):
    require(
        isinstance(asset, dict) and all(key in asset for key in ASSET_FIELDS),
        "Incomplete asset metadata",
    )
    require(
        type(asset["id"]) is int
        and asset["id"] > 0
        and type(asset["size"]) is int
        and asset["size"] >= 0,
        "Invalid asset metadata identity",
    )
    require(
        isinstance(asset["name"], str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", asset["name"]),
        "Unsafe remote asset name",
    )
    require(
        asset["url"] == f"https://api.github.com/repos/{REPOSITORY}/releases/assets/{asset['id']}",
        "Unexpected asset API origin",
    )
    require(
        asset["state"] == "uploaded"
        and isinstance(asset["digest"], str)
        and asset["digest"].startswith("sha256:")
        and hexsha(asset["digest"][7:]),
        "Incomplete upload or missing server SHA-256",
    )
    if expected is not None:
        require(
            asset["id"] == expected["asset_id"] and asset["name"] == expected["name"],
            "Asset name or ID differs",
        )
        require(
            asset["size"] == expected["bytes"]
            and asset["digest"] == "sha256:" + expected["sha256"],
            "Asset size or server digest differs",
        )
    return {
        "id": asset["id"],
        "name": asset["name"],
        "size": asset["size"],
        "state": asset["state"],
        "digest": asset["digest"],
        "metadata_sha256": canonical_sha256({key: asset[key] for key in ASSET_FIELDS}),
    }


def inventory_check(assets, contract):
    names, ids = {}, set()
    for asset in assets:
        identity = asset_identity(asset)
        require(
            identity["name"] not in names and identity["id"] not in ids,
            "Duplicate remote asset name or ID",
        )
        names[identity["name"]] = asset
        ids.add(identity["id"])
    require(
        set(names) == {row["name"] for row in contract["assets"]},
        "Exact release inventory has missing or extra assets",
    )
    return [asset_identity(names[row["name"]], row) for row in contract["assets"]]


def verify(
    contract_path,
    workspace,
    output,
    expected_release_id,
    expected_contract_sha256,
    client_factory=GitHubClient,
):
    contract_path, workspace, output = Path(contract_path), Path(workspace), Path(output)
    require(workspace.is_absolute() and ".." not in workspace.parts, "Invalid workspace root")
    contract_before = file_identity(contract_path)
    require(contract_before["bytes"] <= JSON_LIMIT, "Contract exceeds metadata byte bound")
    contract_bytes = contract_path.read_bytes()
    require(
        hashlib.sha256(contract_bytes).hexdigest() == contract_before["sha256"], "Contract changed"
    )
    contract = validate_contract(
        parse(contract_bytes), expected_release_id, expected_contract_sha256
    )
    require(
        not output.exists() and not output.is_symlink(), "Use a fresh verification output directory"
    )
    require(
        output.parent.is_dir()
        and all(not p.is_symlink() for p in [output.parent, *output.parent.parents]),
        "Unsafe verification output parent",
    )
    output.mkdir()
    (output / "contract.json").write_bytes(contract_bytes)
    verifier_before = file_identity(Path(__file__))
    summary = {
        "schema": "gpu-owned-release-verification-v1",
        "repository": REPOSITORY,
        "tag_name": TAG,
        "expected_release_id": expected_release_id,
        "expected_contract_sha256": expected_contract_sha256,
        "contract_file_sha256": contract_before["sha256"],
        "verifier_sha256": verifier_before["sha256"],
        "started_utc": utc(),
        "passed": False,
        "complete": False,
        "status": "running",
        "formal_eligible": False,
        "assets": [{"expected": row, "status": "not_started"} for row in contract["assets"]],
        "payload_copied_to_disk": False,
        "credentials_or_download_urls_recorded": False,
        "scope": "Exact owned-release inventory and streamed byte identity; no study qualification",
    }
    write_summary(output / "summary.json", summary)
    try:
        local_before = {}
        for row in contract["assets"]:
            actual = file_identity(workspace / row["local_path"])
            require(
                actual["bytes"] == row["bytes"] and actual["sha256"] == row["sha256"],
                "Local asset differs from canonical contract",
            )
            local_before[row["name"]] = actual
        summary["local_preflight"] = local_before
        client = client_factory(output, expected_release_id)
        auth = user_identity(client.user())
        release = release_identity(client.release(), contract)
        initial = inventory_check(client.assets(), contract)
        summary.update(authenticated_before=auth, release_before=release, inventory_before=initial)
        write_summary(output / "summary.json", summary)
        for index, row in enumerate(contract["assets"]):
            item = summary["assets"][index]
            item["status"] = "running"
            before = asset_identity(client.asset(row["asset_id"]), row)
            require(before == initial[index], "Asset changed after initial inventory")
            item["before"] = before
            write_summary(output / "summary.json", summary)
            streamed = client.download(row["asset_id"], row["bytes"])
            item["stream"] = streamed
            require(
                streamed.get("returncode") == 0
                and streamed.get("problem") is None
                and streamed["stdout_bytes"] == row["bytes"]
                and streamed["stdout_sha256"] == row["sha256"],
                "Downloaded byte count, hash or execution differs",
            )
            after = asset_identity(client.asset(row["asset_id"]), row)
            require(after == before, "Asset metadata changed during download")
            require(
                file_identity(workspace / row["local_path"]) == local_before[row["name"]],
                "Local asset changed during remote verification",
            )
            item.update(after=after, status="passed")
            write_new(output / f"asset-{index:04d}.json", item)
            write_summary(output / "summary.json", summary)
        final = inventory_check(client.assets(), contract)
        require(final == initial, "Exact asset inventory changed during verification")
        summary["inventory_after"] = final
        summary["release_after"] = release_identity(client.release(), contract)
        summary["authenticated_after"] = user_identity(client.user())
        require(
            summary["release_after"] == release and summary["authenticated_after"] == auth,
            "Release or authenticated identity changed",
        )
        for row in contract["assets"]:
            require(
                file_identity(workspace / row["local_path"]) == local_before[row["name"]],
                "Local asset changed before completion",
            )
        require(
            file_identity(contract_path) == contract_before
            and hashlib.sha256((output / "contract.json").read_bytes()).hexdigest()
            == contract_before["sha256"],
            "Contract changed during verification",
        )
        require(
            file_identity(Path(__file__)) == verifier_before,
            "Verifier source changed during execution",
        )
        summary.update(
            passed=True,
            complete=True,
            status="passed",
            all_assets_streamed_and_exact_inventory_verified=True,
        )
    except BaseException as error:
        summary.update(status="failed", error_type=type(error).__name__)
        # Do not emit exception text: subprocess/JSON/filesystem errors may include remote text.
        if isinstance(error, CommandFailure):
            summary["failed_command"] = error.receipt
        for item in summary["assets"]:
            if item["status"] == "running":
                item["status"] = "failed"
        raise
    finally:
        summary["finished_utc"] = utc()
        write_summary(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-release-id", type=int, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    args = parser.parse_args()
    try:
        result = verify(
            args.contract,
            args.workspace,
            args.output,
            args.expected_release_id,
            args.expected_contract_sha256,
        )
    except BaseException as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        raise SystemExit(1) from None
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary_sha256": file_identity(args.output / "summary.json")["sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
