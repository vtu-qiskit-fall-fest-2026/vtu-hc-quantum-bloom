"""Compare adaptive and target-fixed using risk-optimal gain >= 1.

Requires these sibling files:
    14_mitigation_pilot.py
    14b_target_fixed_validation.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).with_name("14b_target_fixed_validation.py")

if not SOURCE.exists():
    raise FileNotFoundError(f"Missing required file: {SOURCE}")

spec = importlib.util.spec_from_file_location("mission14b", SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def terminal_risk_gain(
    posterior, lam, n, second_moment, selective=True
):
    """Minimize posterior-predictive MSE over gains >= 1.

    Uses diagnostic posterior and validation second moment only.
    No test ideal expectation or true noise rate enters this decision.

    `selective` is retained for compatibility. Both settings now use
    the same continuous optimization, which includes raw gain=1.
    """
    if n <= 0:
        raise ValueError("Application shots must be positive.")

    post = np.atleast_2d(np.asarray(posterior, dtype=float))
    lam = np.asarray(lam, dtype=float)
    m2 = float(second_moment)

    if not 0 <= m2 <= 1:
        raise ValueError("second_moment must lie in [0, 1].")

    mu = post @ lam
    q = post @ (lam * lam)

    # R(g) = a*g^2 - 2*m2*mu*g + m2
    a = (1.0 - m2 * q) / n + m2 * q

    # Exact constrained minimizer for g >= 1.
    gain = np.maximum(1.0, m2 * mu / a)
    risk = a * gain**2 - 2.0 * m2 * mu * gain + m2

    return np.maximum(risk, 0.0), gain


def check_gain():
    """Check risk against raw and the previous inverse estimator."""
    rng = np.random.default_rng(7)
    lam = runner.m.attenuation("ZZ")
    m2 = runner.SECOND_MOMENT

    for n in (16, 128, 512):
        post = rng.dirichlet(
            np.ones(len(runner.m.PRIOR)), size=100
        )
        risk, gain = terminal_risk_gain(post, lam, n, m2)
        mu = post @ lam
        q = post @ (lam * lam)

        for baseline_gain in (np.ones_like(mu), 1.0 / mu):
            baseline_risk = (
                baseline_gain**2 * (1.0 - m2 * q) / n
                + m2 * (
                    baseline_gain**2 * q
                    - 2.0 * baseline_gain * mu
                    + 1.0
                )
            )
            assert np.all(risk <= baseline_risk + 1e-12)

        assert np.all(gain >= 1.0)

    print("Gain risk checks passed.", flush=True)


if __name__ == "__main__":
    # This changes both adaptive lookahead and final estimation,
    # including the target-fixed baseline.
    runner.m.terminal = terminal_risk_gain
    check_gain()

    # Separate output directories; fresh test circuits/sampling seed.
    if not any(
        arg == "--seed" or arg.startswith("--seed=")
        for arg in sys.argv[1:]
    ):
        sys.argv.extend(["--seed", "62039"])

    if not any(
        arg == "--output" or arg.startswith("--output=")
        for arg in sys.argv[1:]
    ):
        folder = (
            "results/mitigation_14c_smoke"
            if "--smoke" in sys.argv
            else "results/mitigation_14c"
        )
        sys.argv.extend(["--output", folder])

    print(
        "14c: constrained risk-optimal gain; "
        "adaptive_always and adaptive_selective are now identical.",
        flush=True,
    )
    runner.main()