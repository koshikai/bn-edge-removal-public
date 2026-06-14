"""Helpers for encoding boolean vectors to integers and back."""

from __future__ import annotations

from collections.abc import Iterable


def bits_to_int(bits: Iterable[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | (1 if bit else 0)
    return value


def int_to_bits(value: int, width: int) -> list[int]:
    if value < 0:
        raise ValueError("value must be non-negative")
    if width < 0:
        raise ValueError("width must be non-negative")
    if width == 0:
        if value != 0:
            raise ValueError("value does not fit in width")
        return []
    bits: list[int] = [0] * width
    for i in range(width - 1, -1, -1):
        bits[i] = value & 1
        value >>= 1
    if value != 0:
        raise ValueError("value does not fit in width")
    return bits


def all_states(width: int) -> list[list[int]]:
    if width < 0:
        raise ValueError("width must be non-negative")
    return [int_to_bits(i, width) for i in range(2**width)]
