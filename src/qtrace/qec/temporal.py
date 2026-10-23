"""Temporal MAP decoder for the three-qubit repetition memory.

Extracted without changing the algorithm from M33.

Marginalizes eight hidden X-error masks across repeated
noisy syndrome observations and a final three-bit measurement.

Assumptions: independent X noise, stationary parameters
within a block, and ideal quantum operations other than
the declared phenomenological noise.

The target is a Z-basis logical bit, not arbitrary-state QEC.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


SYNDROME = np.array([((m & 1) ^ ((m >> 1) & 1)) |
                     ((((m >> 1) & 1) ^ ((m >> 2) & 1)) << 1)
                     for m in range(8)], dtype=np.uint8)


WEIGHT = np.array([int(m).bit_count() for m in range(8)], dtype=np.int8)


def validate(p: float, label: str) -> None:
    if not (0.0 <= p < 0.5 and math.isfinite(p)):
        raise ValueError(f"{label} must be finite in [0, 0.5): {p}")


@lru_cache(maxsize=256)
def temporal_lookup(rounds: int, p_data: float, p_syndrome: float,
                    p_final: float) -> np.ndarray:
    """MAP logical-bit decoder, marginalizing ALL 8 hidden error masks.

    Index: base-4 packed syndrome history * 8 + 3-bit final measurement.
    For the declared independent, stationary-per-block noise model the MAP
    decision is exact (floating point); no use of injected-error labels.
    """
    if not 1 <= rounds <= 7:
        raise ValueError("rounds must be in [1,7] to bound lookup memory")
    for label, p in (("p_data", p_data), ("p_syndrome", p_syndrome),
                     ("p_final", p_final)):
        validate(p, label)
    states = np.arange(8)
    xor = np.bitwise_xor(states[:, None], states[None, :])
    dist = WEIGHT[xor]
    trans = p_data ** dist * (1 - p_data) ** (3 - dist)
    obs = np.arange(4)
    distance = ((SYNDROME[:, None] ^ obs[None, :]) & 1) + (
        ((SYNDROME[:, None] ^ obs[None, :]) >> 1) & 1)
    emission = (p_syndrome ** distance * (1 - p_syndrome) ** (2 - distance)).T
    alpha = np.zeros((1, 8), dtype=np.float64)
    alpha[0, 0] = 1.0
    for _ in range(rounds):
        alpha = ((alpha @ trans)[:, None, :] * emission[None, :, :]).reshape(-1, 8)

    data = np.arange(8)
    likelihoods = []
    for logical in (0, 1):
        bit_distance = WEIGHT[(states[:, None] ^ data[None, :] ^ (7 * logical))]
        likelihoods.append(p_final ** bit_distance *
                           (1 - p_final) ** (3 - bit_distance))
    score0 = alpha @ likelihoods[0]
    score1 = alpha @ likelihoods[1]
    # Deterministic tie break; benchmark uses randomized and balanced inputs.
    result = (score1 > score0).astype(np.uint8).reshape(-1)
    result.setflags(write=False)
    return result


def packed_history(syndromes: list[int]) -> int:
    h = 0
    for s in syndromes:
        if not 0 <= int(s) <= 3:
            raise ValueError("syndrome out of range")
        h = h * 4 + int(s)
    return h


def decode_one(syndromes: list[int], data: int, p_data: float,
               p_syndrome: float, p_final: float) -> int:
    if not 0 <= data < 8:
        raise ValueError("invalid final data mask")
    return int(temporal_lookup(len(syndromes), p_data, p_syndrome, p_final)[
        packed_history(syndromes) * 8 + data])
