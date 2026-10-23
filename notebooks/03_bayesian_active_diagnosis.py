
"""
Q-TRACE | Mission 03
Budget-Aware Bayesian Active Quantum Noise Diagnosis

The measurement model comes from Mission 02.
This experiment samples measurement outcomes from the
analytical probabilities, previously checked with Aer.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qtrace.diagnosis.bayesian import (
    entropy,
    information_gain as core_information_gain,
    update_posterior as core_update_posterior,
)
from scipy.stats import binom


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

FAULTS = ["None", "X0", "X1", "Z0", "Z1"]

PROBES = [
    "bell_ZZ",
    "bell_XX",
    "zero_Z0",
    "zero_Z1",
    "plus_X0",
    "plus_X1",
]

# Theoretical expectation values from Mission 02.
# Rows = probes; columns = fault hypotheses.

EXPECTATIONS = np.array([
    [1.0, 0.8, 0.8, 1.0, 1.0],
    [1.0, 1.0, 1.0, 0.8, 0.8],
    [1.0, 0.8, 1.0, 1.0, 1.0],
    [1.0, 1.0, 0.8, 1.0, 1.0],
    [1.0, 1.0, 1.0, 0.8, 1.0],
    [1.0, 1.0, 1.0, 1.0, 0.8],
])

BATCH = 16
MAX_SHOTS = 192
TRIALS_PER_FAULT = 60

CONFIDENCE_THRESHOLD = 0.95
SEED = 2026

POLICIES = [
    "adaptive_eig",
    "fixed_min4",
    "round_robin6",
    "random6",
]

FIXED_MIN4 = [0, 1, 2, 4]

assert MAX_SHOTS % BATCH == 0
assert EXPECTATIONS.shape == (len(PROBES), len(FAULTS))


# ============================================================
# 2. MEASUREMENT LIKELIHOODS
# ============================================================

# Pauli measurement outcomes are +1 or -1.
#
# E[O] = P(+1) - P(-1)
#      = 1 - 2P(-1)
#
# Therefore:
# P(-1) = (1 - E[O]) / 2

P_MINUS = (1.0 - EXPECTATIONS) / 2.0

# Precompute binomial likelihoods for a batch of shots.
# Shape: (BATCH + 1, number_of_faults)

K = np.arange(BATCH + 1)[:, None]

LIKELIHOODS = [
    binom.pmf(K, BATCH, probs[None, :])
    for probs in P_MINUS
]


# ============================================================
# 3. BAYESIAN INFORMATION THEORY
# ============================================================



def information_gain(prior, probe_id):
    return core_information_gain(
        prior, LIKELIHOODS[probe_id]
    )


def update_posterior(prior, probe_id, count_minus):
    return core_update_posterior(
        prior, LIKELIHOODS[probe_id][count_minus]
    )


# ============================================================
# 4. PROBE SELECTION POLICIES
# ============================================================

def choose_probe(policy, posterior, batch_idx, rng):

    if policy == "adaptive_eig":

        scores = [
            information_gain(posterior, probe_id)
            for probe_id in range(len(PROBES))
        ]

        return int(np.argmax(scores))

    if policy == "fixed_min4":
        return FIXED_MIN4[
            batch_idx % len(FIXED_MIN4)
        ]

    if policy == "round_robin6":
        return batch_idx % len(PROBES)

    if policy == "random6":
        return int(rng.integers(len(PROBES)))

    raise ValueError(f"Unknown policy: {policy}")


# ============================================================
# 5. SINGLE DIAGNOSIS TRIAL
# ============================================================

def run_trial(true_fault_idx, policy, rng):

    posterior = np.full(
        len(FAULTS),
        1.0 / len(FAULTS),
    )

    trace = []

    first_confident_shots = None
    first_confident_correct = None

    total_batches = MAX_SHOTS // BATCH

    for batch_idx in range(total_batches):

        # Choose the next quantum measurement.

        probe_id = choose_probe(
            policy, posterior, batch_idx, rng
        )

        # Generate shot outcomes from the noise model.

        observed_minus = int(
            rng.binomial(
                BATCH,
                P_MINUS[probe_id, true_fault_idx],
            )
        )

        # Update beliefs about possible faults.

        posterior = update_posterior(
            posterior,
            probe_id,
            observed_minus,
        )

        prediction = int(np.argmax(posterior))
        shots_used = (batch_idx + 1) * BATCH

        correct = int(prediction == true_fault_idx)

        trace.append({
            "shots": shots_used,
            "correct": correct,
        })

        # Record first confidence-threshold crossing.
        # Continue until MAX_SHOTS for fair-budget comparison.

        if (
            first_confident_shots is None
            and posterior.max() >= CONFIDENCE_THRESHOLD
        ):
            first_confident_shots = shots_used
            first_confident_correct = correct

    return trace, {
        "correct_at_budget": trace[-1]["correct"],

        "shots_to_confidence_or_cap": (
            first_confident_shots
            if first_confident_shots is not None
            else MAX_SHOTS
        ),

        "reached_confidence": int(
            first_confident_shots is not None
        ),

        "correct_at_first_confidence":
            first_confident_correct,
    }


# ============================================================
# 6. MONTE CARLO BENCHMARK
# ============================================================

def main():

    rng = np.random.default_rng(SEED)

    final_rows = []
    curve_rows = []

    for policy in POLICIES:

        for true_idx, true_name in enumerate(FAULTS):

            for repeat in range(TRIALS_PER_FAULT):

                trace, metrics = run_trial(
                    true_idx,
                    policy,
                    rng,
                )

                final_rows.append({
                    "policy": policy,
                    "fault": true_name,
                    "repeat": repeat,
                    **metrics,
                })

                for step in trace:
                    curve_rows.append({
                        "policy": policy,
                        "fault": true_name,
                        "repeat": repeat,
                        **step,
                    })

    finals = pd.DataFrame(final_rows)
    curves = pd.DataFrame(curve_rows)

    summary = finals.groupby("policy").agg(
        accuracy_at_budget=(
            "correct_at_budget", "mean"
        ),
        mean_shots_to_confidence_or_cap=(
            "shots_to_confidence_or_cap", "mean"
        ),
        confidence_reached_rate=(
            "reached_confidence", "mean"
        ),
        accuracy_when_confident=(
            "correct_at_first_confidence", "mean"
        ),
    ).reindex(POLICIES)

    print("\n=== Q-TRACE MISSION 03 ===")
    print("Bayesian Active Fault Diagnosis")
    print(f"Batch size: {BATCH}")
    print(f"Maximum shots: {MAX_SHOTS}")
    print(f"Trials per fault: {TRIALS_PER_FAULT}")

    print("\n=== BENCHMARK RESULTS ===")
    print(summary.round(3).to_string())

    # --------------------------------------------------------
    # Save numerical results
    # --------------------------------------------------------

    output = Path("results")
    output.mkdir(exist_ok=True)

    finals.to_csv(
        output / "mission03_trials.csv",
        index=False,
    )

    summary.to_csv(
        output / "mission03_summary.csv"
    )

    # --------------------------------------------------------
    # Plot accuracy versus measurement budget
    # --------------------------------------------------------

    accuracy = (
        curves.groupby(["policy", "shots"])["correct"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for policy in POLICIES:

        subset = accuracy[
            accuracy["policy"] == policy
        ]

        ax.plot(
            subset["shots"],
            subset["correct"],
            marker="o",
            label=policy,
        )

    ax.set(
        xlabel="Total measurement shots",
        ylabel="Correct fault identification rate",
        title="Q-TRACE: Adaptive vs Fixed Probe Selection",
        ylim=(0, 1.05),
    )

    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output / "mission03_accuracy.png",
        dpi=180,
    )

    plt.close(fig)

    print("\nResults saved to results/")
    print("mission03_summary.csv")
    print("mission03_trials.csv")
    print("mission03_accuracy.png")
    print("\nNOTE: Synthetic measurements under the "
          "idealized Mission-02 noise model.")


if __name__ == "__main__":
    main()
