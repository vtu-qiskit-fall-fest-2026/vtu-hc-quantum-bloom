"""Statistical drift detection for Q-TRACE's syndrome decoder.

Extracted from Mission 32. The test compares historical and recent
calibration windows for the ambiguous recovery pair at syndrome 1.

The returned p-value is per-look, not an anytime-valid or
multiple-testing-corrected false-alarm guarantee.
"""

import math

import numpy as np

from qtrace.qec.decoder_policies import STANDARD, ALTERNATIVE


def fisher_p_two_sided(
    a: int,
    b: int,
    c: int,
    d: int
) -> float:
    """Two-sided Fisher exact test."""

    a, b, c, d = map(int, [a, b, c, d])

    if min(a, b, c, d) < 0:
        raise ValueError(
            "negative contingency entry"
        )

    n1 = a + b
    n2 = c + d
    col_alt = a + c
    total = n1 + n2

    if not n1 or not n2:
        return 1.0

    den = math.comb(total, n1)

    def mass(x):
        return (
            math.comb(col_alt, x)
            * math.comb(total - col_alt, n1 - x)
            / den
        )

    observed = mass(a)

    low = max(
        0,
        n1 - (total - col_alt)
    )

    high = min(n1, col_alt)

    return min(
        1.0,
        sum(
            mass(x)
            for x in range(low, high + 1)
            if mass(x) <= observed * (1 + 1e-10)
        )
    )


def detect_shift(
    history: list[np.ndarray],
    window: int,
    alpha: float
) -> tuple[bool, float]:
    """Detect distribution change in syndrome 1."""

    if len(history) < 2 * window:
        return False, 1.0

    old = np.sum(
        history[-2 * window:-window],
        axis=0
    )

    recent = np.sum(
        history[-window:],
        axis=0
    )

    base = int(STANDARD[1])
    alt = int(ALTERNATIVE[1])

    p_value = fisher_p_two_sided(
        old[alt],
        old[base],
        recent[alt],
        recent[base]
    )

    return p_value <= alpha, p_value
