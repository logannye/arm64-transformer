"""Fetch and verify the immutable public text fixture source."""

import argparse
import hashlib
import urllib.request
from pathlib import Path

from transformer.prepare import SOURCE_SHA256, SOURCE_URL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        if hashlib.sha256(args.output.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise ValueError("Existing source differs from pinned SHA256; refusing overwrite")
        print("Existing pinned corpus verified")
        return
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        data = response.read(2_000_000)
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("Downloaded source does not match pinned SHA256")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(data)
    print(f"Verified {len(data)} bytes: {SOURCE_SHA256}")


if __name__ == "__main__":
    main()
