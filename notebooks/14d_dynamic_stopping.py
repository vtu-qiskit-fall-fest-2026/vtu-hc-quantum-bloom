"""Finite-horizon diagnostic stopping for the existing mitigation model.

Requires 14c_risk_gain.py and its dependencies beside this file.
Uses the same prior, noise grid, gain rule and fixed baseline as 14c.
Changes adaptive measurement/stopping to a target-probe dynamic program.
"""

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom


source = Path(__file__).with_name("14c_risk_gain.py")
if not source.exists():
    raise FileNotFoundError(source)

spec = importlib.util.spec_from_file_location("mission14c", source)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

runner = c.runner
m = runner.m
m.terminal = c.terminal_risk_gain
original_trial = m.trial


@lru_cache(maxsize=32)
def build_plan(budget, observable, second_moment):
    """Optimize stopping over all remaining batches.

    State: number of diagnostic batches and total minus outcomes.
    Terminal action: measure application using every remaining shot.
    """
    batch = m.BATCH
    if budget < 2 * batch or budget % batch:
        raise ValueError(
            f"Budget must be a multiple of {batch}, >= {2 * batch}."
        )

    probe = {"ZZ": 0, "XX": 1}[observable]

    # Effective minus probability of the matching Bell probe.
    # Derived from the inference model, never from the hidden test fault.
    probabilities = (
        m.SENS[probe, m.STATE_FAULT] * m.STATE_RATE
    )
    lam = m.attenuation(observable)
    assert np.allclose(lam, 1.0 - 2.0 * probabilities)

    outcomes = np.arange(batch + 1)
    likelihood = binom.pmf(
        outcomes[:, None], batch, probabilities[None, :]
    )

    last = (budget - batch) // batch
    values = {}
    gains = {}
    decisions = {}
    fixed_risks = []

    for step in range(last, -1, -1):
        diagnostic_shots = step * batch
        counts = np.arange(diagnostic_shots + 1)

        log_joint = (
            np.log(m.PRIOR)[None, :]
            + binom.logpmf(
                counts[:, None],
                diagnostic_shots,
                probabilities[None, :],
            )
        )
        log_evidence = logsumexp(log_joint, axis=1)
        posterior = np.exp(
            log_joint - log_evidence[:, None]
        )

        remaining = budget - diagnostic_shots
        stop_risk, gain = m.terminal(
            posterior, lam, remaining, second_moment
        )
        gains[step] = gain

        # Prior-averaged risk for this deterministic stopping time.
        fixed_risks.append(
            float(np.exp(log_evidence) @ stop_risk)
        )

        if step == last:
            values[step] = stop_risk
            decisions[step] = np.zeros(
                len(counts), dtype=bool
            )
            continue

        predictive = posterior @ likelihood.T
        next_indices = (
            counts[:, None] + outcomes[None, :]
        )
        continue_risk = np.sum(
            predictive * values[step + 1][next_indices],
            axis=1,
        )

        take_more = continue_risk < stop_risk - 1e-12
        decisions[step] = take_more
        values[step] = np.where(
            take_more, continue_risk, stop_risk
        )

    # Model-based guarantee, not a guarantee on empirical test MSE.
    assert values[0][0] <= min(fixed_risks) + 1e-10

    return {
        "probe": probe,
        "continue": decisions,
        "gain": gains,
        "prior_risk": float(values[0][0]),
        "best_fixed_prior_risk": min(fixed_risks),
    }


def trial(
    backend, qc, obs, fault, rate, budget,
    seed, strategy, config, m2
):
    if not strategy.startswith("adaptive"):
        return original_trial(
            backend, qc, obs, fault, rate, budget,
            seed, strategy, config, m2
        )

    plan = build_plan(int(budget), obs, float(m2))
    probe = plan["probe"]
    mode, measured = m.PROBES[probe]

    step = 0
    minus_total = 0

    while plan["continue"][step][minus_total]:
        # Hidden truth is used only to simulate observations.
        k = backend.minus(
            m.prepare(mode),
            measured,
            fault,
            rate,
            m.BATCH,
            seed + step * 101 + probe,
        )
        minus_total += int(k)
        step += 1

    diagnostic_shots = step * m.BATCH
    application_shots = budget - diagnostic_shots
    gain = float(plan["gain"][step][minus_total])

    k = backend.minus(
        qc, obs, fault, rate,
        application_shots, seed + 1000003
    )
    raw = 1.0 - 2.0 * k / application_shots

    assert diagnostic_shots + application_shots == budget

    return {
        "estimate": gain * raw,
        "raw_same_shots": raw,
        "diagnostic_shots": diagnostic_shots,
        "application_shots": application_shots,
        "mitigated": int(gain > 1.0 + 1e-10),
        "gain": gain,
    }


def add_default(flag, value):
    provided = any(
        arg == flag or arg.startswith(flag + "=")
        for arg in sys.argv[1:]
    )
    if not provided:
        sys.argv.extend([flag, str(value)])


if __name__ == "__main__":
    m.trial = trial
    c.check_gain()

    # Check the DP against all feasible fixed batch counts.
    for budget in (128, 512):
        for obs in ("ZZ", "XX"):
            plan = build_plan(
                budget, obs, runner.SECOND_MOMENT
            )
            print(
                f"DP check: budget={budget}, obs={obs}, "
                f"prior risk={plan['prior_risk']:.8f}, "
                f"best fixed prior risk="
                f"{plan['best_fixed_prior_risk']:.8f}",
                flush=True,
            )

    add_default("--seed", 73189)
    add_default(
        "--output",
        "results/mitigation_14d_smoke"
        if "--smoke" in sys.argv
        else "results/mitigation_14d",
    )

    # Includes existing physics, risk and hidden-truth isolation checks.
    runner.main()