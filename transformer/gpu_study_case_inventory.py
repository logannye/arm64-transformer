"""Pure metadata for the unchanged GPU stress inventory, without numerical imports.

Tests compare these complete tuples with the original fixture specifications at
every width. This module never constructs arrays, chooses seeds or runs a model.
"""

WIDTHS = (32, 64, 128, 256, 512, 1024)


def attention_case_specs(width):
    if width not in WIDTHS:
        raise ValueError("Unknown GPU width")
    cases = [(f"random_t{tokens}", tokens, "random", 0.0, 0.0, 0.0) for tokens in (1, 3, 32)]
    for magnitude in (0.0, 64.0, 512.0, 4096.0):
        for gap in (0.0, 0.125, 0.75):
            cases.append(
                (f"offset_{magnitude:g}_gap_{gap:g}", 32, "controlled", magnitude, gap, 0.0)
            )
    cases.extend(
        (f"prefix_t{tokens}", tokens, "controlled", 512.0, 0.125, 0.0) for tokens in (2, 3, 5, 31)
    )
    cases.append(("negative_offset", 32, "controlled", -4096.0, 0.125, 0.0))
    cases.extend(
        (f"cancellation_{scale:g}", 32, "controlled", 64.0, 0.125, scale)
        for scale in (16.0, 64.0, 256.0)
    )
    return cases


def layernorm_qkv_case_specs(width):
    if width not in WIDTHS:
        raise ValueError("Unknown GPU width")
    cases = [(f"random_rows{rows}", rows, "random", 0.0, 1.0) for rows in (1, 3, 32, 33, 511, 512)]
    cases.extend(
        (f"offset_{offset:g}_spread_{spread:g}", 3, "offset", offset, spread)
        for offset in (0.0, 64.0, 4096.0)
        for spread in (0.0, 2.0**-12, 0.01, 1.0)
    )
    cases.extend(
        [
            ("affine_signed_scale", 3, "affine", 0.0, 1.0),
            ("affine_zero_channels", 3, "zero_channels", 0.0, 1.0),
            ("qkv_common_bias", 32, "qkv_common_bias", 64.0, 0.01),
            ("qkv_reduction_cancellation", 3, "qkv_cancellation", 64.0, 0.01),
        ]
    )
    return cases
