"""Scalar design model for an uncompiled, unqualified MPS FP16 candidate.

This models CPU iterator ordinals, stable integer merge ranks, storage-word
ownership and per-add half rounding. It is never a framework backend.
"""

import math
import struct

CHUNK_LIMIT = 65536
MAX_SCRATCH_BYTES = 2 * CHUNK_LIMIT * 2 * 8
SENTINEL = (1 << 64) - 1


def half(value):
    try:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def half_bytes(value):
    return struct.pack("<e", value)


def iterator_permutation(shape, operand_byte_strides):
    """Transcribe TensorIterator's stride-priority insertion sort, without tensors."""
    if any(type(size) is not int or size < 0 for size in shape):
        raise ValueError("Invalid shape")
    if any(
        len(strides) != len(shape) or any(s < 0 for s in strides)
        for strides in operand_byte_strides
    ):
        raise ValueError("Invalid stride inventory")
    permutation = list(reversed(range(len(shape))))

    def compare(left, right):
        for strides in operand_byte_strides:
            a, b = strides[left], strides[right]
            if a == 0 or b == 0:
                continue
            if a < b:
                return -1
            if a > b or shape[left] > shape[right]:
                return 1
        return 0

    for index in range(1, len(shape)):
        right = index
        for left in range(index - 1, -1, -1):
            order = compare(permutation[left], permutation[right])
            if order > 0:
                permutation[left], permutation[right] = permutation[right], permutation[left]
                right = left
            elif order < 0:
                break
    return permutation


def coordinates(shape, permutation, ordinal):
    if sorted(permutation) != list(range(len(shape))) or not 0 <= ordinal < math.prod(shape):
        raise ValueError("Invalid iterator coordinate")
    result = [0] * len(shape)
    for dim in permutation:
        result[dim] = ordinal % shape[dim]
        ordinal //= shape[dim]
    return result


def physical_destination(output_base, outer_offset, indices, sizes, strides, storage_bytes):
    if not len(indices) == len(sizes) == len(strides):
        raise ValueError("Mismatched index metadata")
    if min(output_base, outer_offset, storage_bytes, *strides, 0) < 0:
        raise ValueError("Negative storage metadata")
    destination = output_base + outer_offset
    for index, size, stride in zip(indices, sizes, strides, strict=True):
        if not -size <= index < size:
            raise IndexError("Index outside dimension")
        destination += (index % size) * stride
    if destination % 2 or destination > storage_bytes - 2 or destination >= SENTINEL:
        raise IndexError("Destination outside FP16 storage")
    return destination


def merge_rank_sort(pairs):
    """Same rank equations as the proposed GPU passes, checked against Python sort."""
    current = list(pairs)
    width = 1
    while width < len(current):
        output = [None] * len(current)
        for tid, value in enumerate(current):
            start = (tid // (2 * width)) * (2 * width)
            middle = min(start + width, len(current))
            end = min(start + 2 * width, len(current))
            left = tid < middle
            own_start = start if left else middle
            other_start = middle if left else start
            lo, hi = other_start, end if left else middle
            while lo < hi:
                mid = lo + (hi - lo) // 2
                if current[mid] < value:
                    lo = mid + 1
                else:
                    hi = mid
            rank = start + tid - own_start + lo - other_start
            if output[rank] is not None:
                raise ValueError("Duplicate merge rank")
            output[rank] = value
        if any(value is None for value in output):
            raise ValueError("Missing merge rank")
        current = output
        width *= 2
    return current


def serial(initial, updates):
    result = dict(initial)
    for destination, value in updates:
        result[destination] = half(result[destination] + value)
    return result


def candidate(initial, updates, chunk_limit=CHUNK_LIMIT):
    if type(chunk_limit) is not int or not 1 <= chunk_limit <= CHUNK_LIMIT:
        raise ValueError("Invalid chunk bound")
    result = dict(initial)
    for begin in range(0, len(updates), chunk_limit):
        end = min(begin + chunk_limit, len(updates))
        ordered = merge_rank_sort([(updates[index][0], index) for index in range(begin, end)])
        cursor = 0
        owned_words = set()
        while cursor < len(ordered):
            word = ordered[cursor][0] // 4
            if word in owned_words:
                raise ValueError("Multiple packed-word writers")
            owned_words.add(word)
            while cursor < len(ordered) and ordered[cursor][0] // 4 == word:
                destination = ordered[cursor][0]
                if destination not in result or destination % 2:
                    raise IndexError("Unknown half destination")
                value = result[destination]
                while cursor < len(ordered) and ordered[cursor][0] == destination:
                    value = half(value + updates[ordered[cursor][1]][1])
                    cursor += 1
                result[destination] = value
    return result


def byte_state(state):
    return {key: half_bytes(value) for key, value in state.items()}
