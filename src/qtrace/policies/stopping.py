
"""Dynamic-programming stopping policy for Q-TRACE.

Extracted from Mission 18. The policy uses diagnostic counts and a
prior over effective noise rates, not the true injected noise.
"""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom


BATCH = 16
M2 = 0.26497528958302274


class Plan:
    """Policy receives calibration counts, never true noise or ideal values."""

    def __init__(self, name, budget):
        self.name = name
        self.budget = budget
        grid = (np.arange(80) + 0.5) * 0.4 / 80

        if name == "old":
            # Marginal of the original fault prior for one XX/ZZ probe.
            self.p = np.r_[0.0, grid]
            self.prior = np.r_[0.6, np.full(80, 0.4 / 80)]
        elif name == "effective":
            self.p = grid
            self.prior = np.full(80, 1.0 / 80)
        else:
            raise ValueError(name)

        self.lam = 1.0 - 2.0 * self.p
        self.last = budget // BATCH - 1
        self.tables = {}
        next_values = None

        predictive_likelihood = binom.pmf(
            np.arange(BATCH + 1)[:, None], BATCH, self.p
        )
        fixed_prior_risks = []

        for t in range(self.last, -1, -1):
            n_diag = t * BATCH
            k = np.arange(n_diag + 1)

            log_joint = (
                np.log(self.prior)[None, :]
                + binom.logpmf(k[:, None], n_diag, self.p)
            )
            post = np.exp(
                log_joint - logsumexp(
                    log_joint, axis=1, keepdims=True
                )
            )

            mean = post @ self.lam
            second = post @ (self.lam ** 2)
            variance = np.maximum(second - mean ** 2, 0.0)
            n_app = budget - n_diag

            a = (1.0 - M2 * second) / n_app + M2 * second
            gain = np.maximum(1.0, M2 * mean / a)
            stop_risk = np.maximum(
                a * gain ** 2
                - 2.0 * M2 * mean * gain
                + M2,
                0.0,
            )

            if next_values is None:
                continuation = np.full(len(k), np.nan)
                keep_measuring = np.zeros(len(k), dtype=bool)
                value = stop_risk.copy()
            else:
                predictive = post @ predictive_likelihood.T
                destinations = k[:, None] + np.arange(BATCH + 1)
                continuation = np.sum(
                    predictive * next_values[destinations],
                    axis=1,
                )
                keep_measuring = continuation < stop_risk - 1e-12
                value = np.where(
                    keep_measuring, continuation, stop_risk
                )

            self.tables[t] = {
                "mean": mean,
                "variance": variance,
                "gain": gain,
                "risk": stop_risk,
                "continuation": continuation,
                "continue": keep_measuring,
                "value": value,
            }
            next_values = value

            count_probability = (
                binom.pmf(k[:, None], n_diag, self.p)
                @ self.prior
            )
            fixed_prior_risks.append(
                count_probability @ stop_risk
            )

        self.prior_risk = float(self.tables[0]["value"][0])

        assert self.prior_risk <= min(fixed_prior_risks) + 1e-10
        assert all(
            np.isfinite(table["risk"]).all()
            for table in self.tables.values()
        )

    def state(self, batches, minus_count):
        table = self.tables[batches]
        return {
            key: values[minus_count]
            for key, values in table.items()
        }
