
"""
Q-TRACE | Mission 04
Model Mismatch & Robustness Benchmark

Synthetic quantum measurement experiment.
The Bayesian model assumes p=0.10, while the actual
fault probability varies across experimental scenarios.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binom


FAULTS = ["None", "X0", "X1", "Z0", "Z1"]
PROBES = [
    "bell_ZZ", "bell_XX",
    "zero_Z0", "zero_Z1",
    "plus_X0", "plus_X1",
]

# 1 = probe sensitive to fault, 0 = insensitive.
SENSITIVITY = np.array([
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
], dtype=float)

ASSUMED_ERROR = 0.10
TRUE_ERROR_RATES = [0.0, 0.03, 0.05, 0.10, 0.20]

BATCH = 16
MAX_SHOTS = 192
TRIALS_PER_FAULT = 80
THRESHOLD = 0.95
SEED = 2026

POLICIES = [
    "adaptive_eig",
    "fixed_min4",
    "round_robin6",
    "random6",
]

FIXED_MIN4 = [0, 1, 2, 4]

# Under a sensitive Pauli probe, the probability
# of measuring eigenvalue -1 equals the error rate.
P_ASSUMED = SENSITIVITY * ASSUMED_ERROR

K = np.arange(BATCH + 1)[:, None]

LIKELIHOODS = np.stack([
    binom.pmf(K, BATCH, probs[None, :])
    for probs in P_ASSUMED
])


def entropy(p):
    return float(-np.sum(xlogy(p, p)))


def expected_information_gain(prior, probe):
    likelihood = LIKELIHOODS[probe]
    predictive = likelihood @ prior

    valid = predictive > 0

    posterior = (
        likelihood[valid] * prior[None, :]
    ) / predictive[valid, None]

    conditional_entropy = -np.sum(
        xlogy(posterior, posterior),
        axis=1,
    )

    return entropy(prior) - float(
        predictive[valid] @ conditional_entropy
    )


def bayesian_update(prior, probe, negative_count):
    likelihood = LIKELIHOODS[probe, negative_count]
    weights = prior * likelihood
    total = weights.sum()

    if total <= 0:
        raise RuntimeError(
            "Observation impossible under the assumed model."
        )

    return weights / total


def select_probe(policy, posterior, step, random_schedule):
    if policy == "adaptive_eig":
        scores = [
            expected_information_gain(posterior, i)
            for i in range(len(PROBES))
        ]
        return int(np.argmax(scores))

    if policy == "fixed_min4":
        return FIXED_MIN4[step % len(FIXED_MIN4)]

    if policy == "round_robin6":
        return step % len(PROBES)

    if policy == "random6":
        return int(random_schedule[step])

    raise ValueError(policy)


def run_trial(
    true_fault,
    true_error,
    policy,
    uniforms,
    random_schedule,
):
    posterior = np.full(len(FAULTS), 1.0 / len(FAULTS))

    first_confidence = None
    wrong_first_confidence = 0
    accuracy_64 = None
    accuracy_192 = None

    for step in range(MAX_SHOTS // BATCH):
        probe = select_probe(
            policy, posterior, step, random_schedule
        )

        actual_p = (
            SENSITIVITY[probe, true_fault] * true_error
        )

        # Common random numbers across policies:
        # fairer paired Monte Carlo comparison.
        negative_count = int(
            np.sum(uniforms[step] < actual_p)
        )

        posterior = bayesian_update(
            posterior, probe, negative_count
        )

        predicted_fault = int(np.argmax(posterior))
        correct = int(predicted_fault == true_fault)
        shots = (step + 1) * BATCH

        if shots == 64:
            accuracy_64 = correct

        if shots == MAX_SHOTS:
            accuracy_192 = correct

        if (
            first_confidence is None
            and posterior.max() >= THRESHOLD
        ):
            first_confidence = shots
            wrong_first_confidence = 1 - correct

    return {
        "correct_64": accuracy_64,
        "correct_192": accuracy_192,
        "confidence_reached": int(
            first_confidence is not None
        ),
        "wrong_first_confidence": wrong_first_confidence,
        "shots_to_confidence_or_cap": (
            first_confidence
            if first_confidence is not None
            else MAX_SHOTS
        ),
    }


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    for true_error in TRUE_ERROR_RATES:
        for fault_idx, fault_name in enumerate(FAULTS):
            for repeat in range(TRIALS_PER_FAULT):

                # The same underlying shot randomness
                # is reused for all policies in this trial.
                uniforms = rng.random(
                    (MAX_SHOTS // BATCH, BATCH)
                )

                random_schedule = rng.integers(
                    0, len(PROBES),
                    size=MAX_SHOTS // BATCH,
                )

                for policy in POLICIES:
                    metrics = run_trial(
                        fault_idx,
                        true_error,
                        policy,
                        uniforms,
                        random_schedule,
                    )

                    rows.append({
                        "true_error": true_error,
                        "fault": fault_name,
                        "repeat": repeat,
                        "policy": policy,
                        **metrics,
                    })

    df = pd.DataFrame(rows)

    summary = df.groupby(
        ["true_error", "policy"]
    ).agg(
        accuracy_64=("correct_64", "mean"),
        accuracy_192=("correct_192", "mean"),
        confidence_rate=("confidence_reached", "mean"),
        false_confidence_rate=(
            "wrong_first_confidence", "mean"
        ),
        mean_shots=(
            "shots_to_confidence_or_cap", "mean"
        ),
    ).reset_index()

    print("\n=== MISSION 04: MODEL MISMATCH ===")
    print("Assumed error:", ASSUMED_ERROR)
    print("\n=== ROBUSTNESS RESULTS ===")
    print(summary.round(3).to_string(index=False))

    output = Path("results")
    output.mkdir(exist_ok=True)

    df.to_csv(output / "mission04_trials.csv", index=False)
    summary.to_csv(
        output / "mission04_summary.csv", index=False
    )

    # Negative control:
    # With p_true=0 all five fault hypotheses
    # generate the same measurement distribution.
    zero_noise_accuracy = df.loc[
        df["true_error"] == 0.0, "correct_192"
    ].mean()

    assert np.isclose(zero_noise_accuracy, 0.20)
    print("\nPASS: Zero-noise identifiability control.")

    # Accuracy at a constrained 64-shot budget.
    fig, ax = plt.subplots(figsize=(9, 5))

    for policy in POLICIES:
        subset = summary[summary["policy"] == policy]
        ax.plot(
            subset["true_error"],
            subset["accuracy_64"],
            marker="o",
            label=policy,
        )

    ax.set(
        xlabel="Actual Pauli error probability",
        ylabel="Fault identification accuracy (64 shots)",
        title="Q-TRACE: Robustness to Noise Model Mismatch",
        ylim=(0, 1.05),
    )

    ax.axvline(
        ASSUMED_ERROR,
        linestyle="--",
        alpha=0.5,
        label="Assumed p = 0.10",
    )

    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output / "mission04_robustness.png",
        dpi=180,
    )
    plt.close(fig)

    print("\nSaved:")
    print("results/mission04_trials.csv")
    print("results/mission04_summary.csv")
    print("results/mission04_robustness.png")
    print("\nNOTE: Synthetic binomial measurements; "
          "not real-hardware or Aer measurements.")


if __name__ == "__main__":
    main()
