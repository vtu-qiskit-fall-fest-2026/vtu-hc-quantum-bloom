
"""Bayesian inference utilities for Q-TRACE diagnosis.

These functions are independent of the experimental probe set,
shot budget, and quantum backend.
"""

import numpy as np
from scipy.special import xlogy


def entropy(probabilities):
    """Shannon entropy in nats."""
    return float(
        -np.sum(xlogy(probabilities, probabilities))
    )


def information_gain(prior, likelihood):
    """Expected entropy reduction from one measurement batch.

    Parameters
    ----------
    prior:
        Probability distribution over fault hypotheses, shape (H,).
    likelihood:
        P(outcome | hypothesis), shape (K, H).

    Returns
    -------
    float
        Expected information gain in nats.
    """
    predictive = likelihood @ prior
    valid = predictive > 0

    posterior = (
        likelihood[valid] * prior[None, :]
    ) / predictive[valid, None]

    conditional_entropies = -np.sum(
        xlogy(posterior, posterior),
        axis=1,
    )

    expected_future_entropy = float(
        predictive[valid] @ conditional_entropies
    )

    return entropy(prior) - expected_future_entropy


def update_posterior(prior, likelihood):
    """Update the fault posterior using an observed likelihood.

    likelihood has shape (H,) for one observed outcome.
    """
    weights = prior * likelihood
    normalizer = weights.sum()

    if normalizer <= 0:
        raise RuntimeError(
            "Observation impossible under the assumed model"
        )

    return weights / normalizer
