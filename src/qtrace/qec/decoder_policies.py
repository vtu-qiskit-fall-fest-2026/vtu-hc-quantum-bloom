
"""Guarded and cost-sensitive recovery policies for Q-TRACE.

Extracted from Missions 28 and 29.

Calibration counts are indexed by the three-bit X-error mask.
The policies do not receive true noise probabilities or
held-out test outcomes.
"""

from functools import lru_cache
from math import comb

import numpy as np


STANDARD = np.array([0, 1, 4, 2], dtype=int)
ALTERNATIVE = STANDARD ^ 7


@lru_cache(maxsize=100000)
def alternative_confidence(n_alt, n_standard):
    """P(theta > 0.5 | counts), theta ~ Beta(1, 1)."""
    n_alt, n_standard = int(n_alt), int(n_standard)
    n = n_alt + n_standard + 1

    return sum(
        comb(n, k)
        for k in range(n_alt + 1)
    ) / (2 ** n)


def guarded_decoder(counts, threshold=0.95):
    """Select alternative recovery only above confidence threshold."""
    table = STANDARD.copy()
    confidence = []

    for s in range(4):
        base = int(STANDARD[s])
        alt = int(ALTERNATIVE[s])

        probability = alternative_confidence(
            int(counts[alt]),
            int(counts[base]),
        )

        confidence.append(probability)

        if probability >= threshold:
            table[s] = alt

    return table, confidence


@lru_cache(maxsize=100000)
def posterior_moments(n_alt, n_standard):
    """Expected positive gain and harm under Beta posterior."""
    mu = (n_alt + 1) / (n_alt + n_standard + 2)

    tail = alternative_confidence(
        n_alt, n_standard
    )

    shifted_tail = alternative_confidence(
        n_alt + 1, n_standard
    )

    gain = max(
        0.0,
        2 * mu * shifted_tail - tail,
    )

    harm = max(
        0.0,
        gain - (2 * mu - 1),
    )

    return gain, harm


def cost_decoder(counts, penalty):
    """Select recovery under asymmetric gain/harm penalty.

    penalty=1 preserves the original greedy majority-count rule.
    Other penalties use posterior expected gain and harm.
    """
    table = STANDARD.copy()

    for s in range(4):
        a = int(counts[ALTERNATIVE[s]])
        b = int(counts[STANDARD[s]])

        if penalty == 1:
            switch = a > b
        else:
            gain, harm = posterior_moments(a, b)
            switch = gain > penalty * harm + 1e-12

        if switch:
            table[s] = ALTERNATIVE[s]

    return table
