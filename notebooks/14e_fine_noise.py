"""Fine noise-grid candidate for Q-TRACE.

Requires 14d_dynamic_stopping.py and its dependencies.
Both DP and target-fixed use the same new inference model.

Fault prior stays unchanged:
    None: 0.2
    X0, X1, Z0, Z1: 0.2 each

Conditional on a non-identity fault:
    80 equally weighted midpoint rates in (0, 0.4).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom


source = Path(__file__).with_name("14d_dynamic_stopping.py")
if not source.exists():
    raise FileNotFoundError(source)

spec = importlib.util.spec_from_file_location("mission14d", source)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)

m = d.m
runner = d.runner


def configure_model():
    # Midpoints: 0.0025, 0.0075, ..., 0.3975.
    # Chosen independently of individual test rates.
    rates = (np.arange(80) + 0.5) * (0.4 / 80)

    m.RATES = rates
    m.STATE_FAULT = np.r_[
        0,
        np.repeat(np.arange(1, 5), len(rates)),
    ].astype(int)

    m.STATE_RATE = np.r_[0.0, np.tile(rates, 4)]

    m.PRIOR = np.r_[
        0.2,
        np.full(4 * len(rates), 0.2 / len(rates)),
    ]

    minus_probability = (
        m.SENS[:, m.STATE_FAULT] * m.STATE_RATE
    )
    m.LIKELIHOOD = binom.pmf(
        np.arange(m.BATCH + 1)[None, :, None],
        m.BATCH,
        minus_probability[:, None, :],
    )

    assert np.isclose(m.PRIOR.sum(), 1.0)
    assert np.allclose(m.LIKELIHOOD.sum(axis=1), 1.0)

    # Cached policies depend on the inference model.
    d.build_plan.cache_clear()
    m.trial = d.trial


if __name__ == "__main__":
    configure_model()
    d.c.check_gain()

    for budget in (128, 512):
        for obs in ("ZZ", "XX"):
            plan = d.build_plan(
                budget, obs, runner.SECOND_MOMENT
            )
            print(
                f"Fine-grid DP: budget={budget}, obs={obs}, "
                f"prior risk={plan['prior_risk']:.8f}, "
                f"best fixed prior risk="
                f"{plan['best_fixed_prior_risk']:.8f}",
                flush=True,
            )

    d.add_default("--seed", 94153)
    d.add_default(
        "--output",
        "results/mitigation_14e_smoke"
        if "--smoke" in sys.argv
        else "results/mitigation_14e",
    )

    print(
        "14e: 80 midpoint rates; uniform conditional rate prior. "
        "DP and target-fixed share this model.",
        flush=True,
    )

    # Retunes fixed allocation on validation, then evaluates new targets.
    # Also executes existing physics and hidden-truth isolation checks.
    runner.main()