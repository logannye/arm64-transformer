"""Process-local model layout shared by binary interfaces, with no model math.

Select a supported width before Python starts with ARM64_TRANSFORMER_WIDTH.
Separate processes and native build directories keep configurations isolated.
"""

import argparse
import json
import os
from pathlib import Path

WIDTHS = (32, 64, 128, 256, 512, 1024)


def selected_width(value):
    try:
        width = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ARM64_TRANSFORMER_WIDTH must be one of {WIDTHS}") from exc
    if width not in WIDTHS:
        raise ValueError(f"ARM64_TRANSFORMER_WIDTH must be one of {WIDTHS}")
    return width


CONTEXT = 32
WIDTH = selected_width(os.environ.get("ARM64_TRANSFORMER_WIDTH", "32"))
HEADS = 4
HIDDEN = 2 * WIDTH
VOCAB = 128
EPS = 1e-5
LR = 0.001
BATCH = 16
EPOCHS = 8
TRAIN_SEQUENCES = 2048
VALID_SEQUENCES = 256
TEST_SEQUENCES = 256
SEEDS = (42, 43, 44)

PARAM_SHAPES = {
    "token": (VOCAB, WIDTH),
    "position": (CONTEXT, WIDTH),
    "ln1_weight": (WIDTH,),
    "ln1_bias": (WIDTH,),
    "qkv_weight": (3 * WIDTH, WIDTH),
    "qkv_bias": (3 * WIDTH,),
    "attention_weight": (WIDTH, WIDTH),
    "attention_bias": (WIDTH,),
    "ln2_weight": (WIDTH,),
    "ln2_bias": (WIDTH,),
    "ff1_weight": (HIDDEN, WIDTH),
    "ff1_bias": (HIDDEN,),
    "ff2_weight": (WIDTH, HIDDEN),
    "ff2_bias": (WIDTH,),
    "ln3_weight": (WIDTH,),
    "ln3_bias": (WIDTH,),
    "head_weight": (VOCAB, WIDTH),
    "head_bias": (VOCAB,),
}


def layouts(shapes):
    offsets = {}
    cursor = 0
    for name, shape in shapes.items():
        size = 1
        for dim in shape:
            size *= dim
        offsets[name] = (cursor, size)
        cursor += size
    return offsets, cursor


PARAMS, PARAM_COUNT = layouts(PARAM_SHAPES)

WORK_SHAPES = {
    "x": (CONTEXT, WIDTH),
    "n1": (CONTEXT, WIDTH),
    "qkv": (CONTEXT, WIDTH * 3),
    "attention": (CONTEXT, WIDTH),
    "projected": (CONTEXT, WIDTH),
    "r1": (CONTEXT, WIDTH),
    "n2": (CONTEXT, WIDTH),
    "ff1": (CONTEXT, HIDDEN),
    "relu": (CONTEXT, HIDDEN),
    "ff2": (CONTEXT, WIDTH),
    "r2": (CONTEXT, WIDTH),
    "n3": (CONTEXT, WIDTH),
    "logits": (CONTEXT, VOCAB),
    "norm1": (CONTEXT, WIDTH),
    "inv1": (CONTEXT,),
    "norm2": (CONTEXT, WIDTH),
    "inv2": (CONTEXT,),
    "norm3": (CONTEXT, WIDTH),
    "inv3": (CONTEXT,),
    "probs": (HEADS, CONTEXT, CONTEXT),
    "dlogits": (CONTEXT, VOCAB),
    "dn3": (CONTEXT, WIDTH),
    "dr2": (CONTEXT, WIDTH),
    "dr1": (CONTEXT, WIDTH),
    "drelu": (CONTEXT, HIDDEN),
    "dff1": (CONTEXT, HIDDEN),
    "dn2": (CONTEXT, WIDTH),
    "temp": (CONTEXT, WIDTH),
    "dattention": (CONTEXT, WIDTH),
    "dqkv": (CONTEXT, WIDTH * 3),
    "dn1": (CONTEXT, WIDTH),
    "dx": (CONTEXT, WIDTH),
}
WORK, WORK_COUNT = layouts(WORK_SHAPES)


def assembly_constants():
    lines = ["// Generated layout constants only; numerical code is handwritten."]
    for name, value in {
        "CONTEXT": CONTEXT,
        "WIDTH": WIDTH,
        "WIDTH_SHIFT": WIDTH.bit_length() - 1,
        "HEADS": HEADS,
        "HIDDEN": HIDDEN,
        "VOCAB": VOCAB,
        "PARAM_COUNT": PARAM_COUNT,
        "PARAM_BYTES": PARAM_COUNT * 4,
        "WORK_BYTES": WORK_COUNT * 4,
    }.items():
        lines.append(f".equ {name}, {value}")
    for prefix, values in (("P", PARAMS), ("W", WORK)):
        for name, (offset, _) in values.items():
            lines.append(f".equ {prefix}_{name.upper()}, {offset * 4}")
    return "\n".join(lines) + "\n"


def model_metadata():
    return {
        "width": WIDTH,
        "heads": HEADS,
        "head_width": WIDTH // HEADS,
        "hidden": HIDDEN,
        "context": CONTEXT,
        "vocab": VOCAB,
        "parameters": PARAM_COUNT,
        "workspace_floats": WORK_COUNT,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(assembly_constants())
    else:
        print(json.dumps(model_metadata()))
