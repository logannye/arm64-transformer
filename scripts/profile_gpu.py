"""Serial development-only Metal profiling; never opens held-out/admission data."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, required=True, choices=(32, 64, 128, 256, 512, 1024))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("profile", "search"), default="profile")
    parser.add_argument(
        "--profile", type=Path, help="Completed reference-profile JSON required for search"
    )
    args = parser.parse_args()
    previous = os.environ.get("ARM64_TRANSFORMER_WIDTH")
    if previous is not None and previous != str(args.width):
        raise ValueError("ARM64_TRANSFORMER_WIDTH conflicts with --width")
    os.environ["ARM64_TRANSFORMER_WIDTH"] = str(args.width)
    from transformer.gpu_optimize import profile_reference, search_policies

    if args.mode == "search":
        if args.profile is None:
            parser.error("--mode search requires --profile")
        result = search_policies(args.width, args.output, args.profile)
    else:
        if args.profile is not None:
            parser.error("--profile is only accepted with --mode search")
        result = profile_reference(args.width, args.output)
    print(f"{result['status']}: {args.output}", flush=True)


if __name__ == "__main__":
    main()
