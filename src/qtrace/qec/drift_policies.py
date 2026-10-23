"""Decoder policy selection for Q-TRACE Mission 32.

Uses the calibrated greedy, guarded, and cost-sensitive
decoders extracted from Missions 28 and 29.

All decisions use observed calibration counts.
No true noise parameters or test outcomes are supplied.
"""

import numpy as np

from qtrace.qec.decoder_policies import (
    STANDARD,
    guarded_decoder,
    cost_decoder,
)


def policy(
    counts: np.ndarray,
    method: str,
) -> np.ndarray:
    """Choose a syndrome recovery table from calibration counts."""

    counts = np.asarray(counts, dtype=int)

    if counts.shape != (8,) or np.any(counts < 0):
        raise ValueError(
            "expected eight non-negative counts"
        )

    if method == "standard":
        return STANDARD.copy()

    if method == "greedy":
        return cost_decoder(counts, 1)

    if method == "guarded95":
        table, _ = guarded_decoder(
            counts,
            threshold=0.95,
        )
        return table

    if method == "cost5":
        return cost_decoder(counts, 5)

    raise ValueError(
        f"unknown policy {method}"
    )


def initial_tables(
    counts: np.ndarray,
) -> dict[str, np.ndarray]:
    """Initialize the decoder variants used by Mission 32."""

    return {
        "standard": STANDARD.copy(),

        "fixed_greedy":
            policy(counts, "greedy"),

        "fixed_guarded95":
            policy(counts, "guarded95"),

        "fixed_cost5":
            policy(counts, "cost5"),

        "periodic_cost5":
            policy(counts, "cost5"),

        "drift_cost5":
            policy(counts, "cost5"),
    }
