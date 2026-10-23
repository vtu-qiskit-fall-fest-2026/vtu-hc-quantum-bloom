
"""
Q-TRACE | Mission 05
Uncertainty-Aware Bayesian Quantum Fault Diagnosis

Compare point-estimate Bayesian inference against
joint Bayesian inference over fault identity and
unknown Pauli error probability.

Measurement outcomes are synthetic binomial samples.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binom


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

FAULTS = ["None", "X0", "X1", "Z0", "Z1"]

PROBES = [
    "bell_ZZ", "bell_XX",
    "zero_Z0", "zero_Z1",
    "plus_X0", "plus_X1",
]

SENS = np.array([
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
])

BATCH = 16
BUDGET = 192
TRIALS = 80
THRESHOLD = 0.95
SEED = 2026

# Candidate rates used by the robust Bayesian model.
RATES_GRID = np.array([0.01, 0.03, 0.05, 0.10, 0.20])

# Includes out-of-grid rates to test generalization.
TRUE_RATES = [0.0, 0.02, 0.05, 0.10, 0.15]

POLICIES = [
    "robust_active",
    "point_active",
    "robust_fixed4",
    "robust_round6",
]


# ============================================================
# 2. JOINT BAYESIAN MODEL
# ============================================================

# First state = no fault.
# Remaining states = (fault identity, error probability).

state_fault = np.r_[
    0,
    np.repeat(np.arange(1, 5), len(RATES_GRID))
]

state_rate = np.r_[
    0.0,
    np.tile(RATES_GRID, 4)
]

# Equal prior mass for each of the five fault identities.
# Spread each fault's mass across candidate error rates.

robust_prior = np.r_[
    0.2,
    np.full(
        4 * len(RATES_GRID),
        0.2 / len(RATES_GRID)
    )
]

point_prior = np.full(5, 0.2)

# Marginalization matrices:
# joint posterior -> posterior over five fault identities.

MAP_ROBUST = np.eye(5)[state_fault]
MAP_POINT = np.eye(5)

COUNTS = np.arange(BATCH + 1)[None, :, None]

ROBUST_LIKE = binom.pmf(
    COUNTS,
    BATCH,
    (
        SENS[:, state_fault] * state_rate[None, :]
    )[:, None, :]
)

POINT_LIKE = binom.pmf(
    COUNTS,
    BATCH,
    (SENS * 0.10)[:, None, :]
)

assert np.isclose(robust_prior.sum(), 1.0)
assert ROBUST_LIKE.shape == (
    len(PROBES), BATCH + 1, len(state_fault)
)


# ============================================================
# 3. INFORMATION GAIN ABOUT FAULT IDENTITY
# ============================================================

def row_entropy(probabilities):
    return -np.sum(
        xlogy(probabilities, probabilities),
        axis=-1,
    )


def information_gain(prior, likelihood, mapping):
    """
    Expected entropy reduction in fault identity,
    marginalizing over unknown error probabilities.
    """

    predictive = likelihood @ prior
    valid = predictive > 1e-15

    posterior = (
        likelihood[valid] * prior[None, :]
    ) / predictive[valid, None]

    fault_posterior = posterior @ mapping

    current_entropy = float(
        row_entropy(prior @ mapping)
    )

    expected_entropy = float(
        predictive[valid] @ row_entropy(fault_posterior)
    )

    return current_entropy - expected_entropy


# ============================================================
# 4. ONE DIAGNOSIS TRIAL
# ============================================================

def simulate(policy, truth, actual_rate, uniforms):

    robust = policy != "point_active"

    prior = (
        robust_prior.copy()
        if robust else point_prior.copy()
    )

    likelihoods = (
        ROBUST_LIKE if robust else POINT_LIKE
    )

    mapping = (
        MAP_ROBUST if robust else MAP_POINT
    )

    first_stop_shots = None
    first_stop_correct = np.nan
    accuracy64 = 0

    for step in range(BUDGET // BATCH):

        # Select probe.

        if policy in ("robust_active", "point_active"):

            scores = [
                information_gain(
                    prior, likelihoods[i], mapping
                )
                for i in range(len(PROBES))
            ]

            probe = int(np.argmax(scores))

        elif policy == "robust_fixed4":

            probe = [0, 1, 2, 4][step % 4]

        else:

            probe = step % len(PROBES)

        # Generate quantum measurement outcomes from
        # the actual error probability.

        probability = (
            SENS[probe, truth] * actual_rate
        )

        negative_count = int(
            np.sum(uniforms[step] < probability)
        )

        # Bayesian update.

        prior *= likelihoods[probe, negative_count]
        prior /= prior.sum()

        # Marginalize over error strength.

        fault_posterior = prior @ mapping

        prediction = int(np.argmax(fault_posterior))
        shots = (step + 1) * BATCH

        if shots == 64:
            accuracy64 = int(prediction == truth)

        # Record the first confidence-threshold crossing.
        # The experiment continues for fixed-budget evaluation.

        if (
            first_stop_shots is None
            and fault_posterior.max() >= THRESHOLD
        ):
            first_stop_shots = shots
            first_stop_correct = int(
                prediction == truth
            )

    accuracy192 = int(
        np.argmax(fault_posterior) == truth
    )

    one_hot = np.eye(5)[truth]

    brier = float(
        np.sum((fault_posterior - one_hot) ** 2)
    )

    return {
        "accuracy64": accuracy64,
        "accuracy192": accuracy192,

        "reached_confidence": int(
            first_stop_shots is not None
        ),

        "accuracy_at_first_confidence":
            first_stop_correct,

        "shots_to_confidence_or_cap":
            first_stop_shots or BUDGET,

        "brier192": brier,
    }


# ============================================================
# 5. MONTE CARLO BENCHMARK
# ============================================================

def main():

    rng = np.random.default_rng(SEED)
    rows = []

    for rate in TRUE_RATES:

        # At p=0, all faults have zero physical effect.
        # Treat this as the "None" scenario rather than
        # five artificially distinct labels.

        scenarios = (
            [0] if rate == 0
            else list(range(len(FAULTS)))
        )

        for fault in scenarios:

            actual_rate = (
                0.0 if fault == 0 else rate
            )

            for repeat in range(TRIALS):

                # Shared random numbers across policies.

                uniforms = rng.random(
                    (BUDGET // BATCH, BATCH)
                )

                for policy in POLICIES:

                    metrics = simulate(
                        policy,
                        fault,
                        actual_rate,
                        uniforms,
                    )

                    rows.append({
                        "true_rate": rate,
                        "fault": FAULTS[fault],
                        "repeat": repeat,
                        "policy": policy,
                        **metrics,
                    })

    df = pd.DataFrame(rows)

    summary = df.groupby(
        ["true_rate", "policy"]
    ).agg(
        accuracy64=("accuracy64", "mean"),
        accuracy192=("accuracy192", "mean"),
        stop_coverage=("reached_confidence", "mean"),
        stop_accuracy=(
            "accuracy_at_first_confidence", "mean"
        ),
        stop_shots=(
            "shots_to_confidence_or_cap", "mean"
        ),
        brier192=("brier192", "mean"),
    ).reset_index()

    print("\n=== Q-TRACE MISSION 05 ===")
    print("Uncertainty-Aware Bayesian Diagnosis")
    print("Budget:", BUDGET)
    print("Trials per scenario:", TRIALS)

    print("\n=== BENCHMARK ===")
    print(summary.round(3).to_string(index=False))

    output = Path("results")
    output.mkdir(exist_ok=True)

    df.to_csv(
        output / "mission05_trials.csv",
        index=False,
    )

    summary.to_csv(
        output / "mission05_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Plot correctness at the first confidence crossing
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 5))

    for policy in POLICIES:

        subset = summary[
            summary["policy"] == policy
        ]

        ax.plot(
            subset["true_rate"],
            subset["stop_accuracy"],
            marker="o",
            label=policy,
        )

    ax.set(
        xlabel="Actual Pauli error probability",
        ylabel="Accuracy conditional on reaching 95% confidence",
        title="Q-TRACE: Reliability of Confident Diagnoses",
        ylim=(0, 1.05),
    )

    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output / "mission05_stopping_accuracy.png",
        dpi=180,
    )

    plt.close(fig)

    print("\nSaved results/mission05_*")
    print(
        "NOTE: Synthetic data, not Aer or hardware data."
    )
    print(
        "Missing stop_accuracy = no trials reached confidence."
    )


if __name__ == "__main__":
    main()
