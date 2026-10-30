"""Bounded subprocess and file receipts for isolated MLX integration only."""

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in retained artifact: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {
                "sha256": digest(path),
                "bytes": path.stat().st_size,
                "executable": bool(path.stat().st_mode & 0o111),
            }
    return result


def run(command, *, cwd, output, name, env=None, timeout=3600, check=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"[{name}] {' '.join(map(str, command))}", flush=True)
    result = {"command": list(map(str, command)), "cwd": str(cwd), "name": name}
    with (output / f"{name}.log").open("xb") as log:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            result["returncode"] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            result["timed_out"] = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            result["returncode"] = process.returncode
    result["seconds"] = time.monotonic() - started
    write(output / f"{name}.json", result)
    if check and (result["returncode"] != 0 or result.get("timed_out")):
        raise RuntimeError(f"{name} failed; retained {output / (name + '.log')}")
    return result
