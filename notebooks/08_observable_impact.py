
"""
Q-TRACE | Mission 08
Observable-Impact-Aware Bayesian Fault Diagnosis

Quantum: Qiskit Statevector and Pauli observables.
Benchmark: synthetic binomial measurements.

Run:
    python notebooks/08_observable_impact.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from scipy.special import xlogy
from scipy.stats import binom


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 2026

BATCH = 16
BUDGET = 192
REPEATS = 80

EPSILON = 0.06
ABSTAIN_COST = 0.80

FAULTS = ["None", "X0", "X1", "Z0", "Z1"]

CLASSES = [
    "acceptable",
    "X_impact",
    "Z_impact",
]

PROBES = [
    "bell_ZZ",
    "bell_XX",
    "zero_Z0",
    "zero_Z1",
    "plus_X0",
    "plus_X1",
]

POLICIES = [
    "decision_voi",
    "label_eig",
    "latent_eig",
    "fixed4",
    "round6",
]

# Rows: quantum probes
# Columns: physical fault identities
#
# 1 means sensitive to the fault.
# 0 means insensitive.

SENS = np.array([
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
], dtype=float)


# ============================================================
# 2. QUANTUM OBSERVABLE IMPACT
# ============================================================

def quantum_impact_coefficients():
    """
    Calculate how strongly each Pauli fault affects
    the observable:

        O = 0.7 ZZ + 0.3 XX

    The Pauli fault is inserted immediately after
    Bell-state preparation.

    For stochastic fault probability p:

        observable deviation = p * impact_coefficient
    """

    circuit = QuantumCircuit(2)

    circuit.h(0)
    circuit.cx(0, 1)

    psi = Statevector.from_instruction(
        circuit
    ).data

    target = (
        0.7 * Pauli("ZZ").to_matrix()
        + 0.3 * Pauli("XX").to_matrix()
    )

    ideal = float(
        np.real(
            np.vdot(psi, target @ psi)
        )
    )

    coefficients = [0.0]

    # Qiskit ordering: q1 q0
    # IX = X on q0
    # XI = X on q1

    for operator in ["IX", "XI", "IZ", "ZI"]:

        error_matrix = Pauli(
            operator
        ).to_matrix()

        perturbed_state = error_matrix @ psi

        perturbed = float(
            np.real(
                np.vdot(
                    perturbed_state,
                    target @ perturbed_state,
                )
            )
        )

        impact = abs(
            ideal - perturbed
        )

        coefficients.append(impact)

    return np.array(coefficients), ideal


# ============================================================
# 3. DECISION COST
# ============================================================

# Rows = actions
# Columns = true operational classes
#
# Missing a significant fault: cost 8
# False alarm: cost 4
# Misidentifying the relevant error family: cost 3

LOSS = np.array([
    [0.0, 8.0, 8.0],
    [4.0, 0.0, 3.0],
    [4.0, 3.0, 0.0],
])

# Candidate noise strengths in the inference model.

RATES = np.array([
    0.01,
    0.02,
    0.035,
    0.05,
    0.075,
    0.11,
    0.16,
    0.20,
])


# ============================================================
# 4. BUILD LATENT BAYESIAN MODEL
# ============================================================

def build_model(coefficients):

    physical = [0]
    strength = [0.0]

    for fault in range(1, 5):

        for rate in RATES:

            physical.append(fault)
            strength.append(float(rate))

    physical = np.array(
        physical,
        dtype=int,
    )

    strength = np.array(
        strength,
        dtype=float,
    )

    # Task-specific observable impact.

    impact = (
        coefficients[physical]
        * strength
    )

    # Class 0: acceptable observable deviation
    # Class 1: significant X-type fault
    # Class 2: significant Z-type fault

    labels = np.zeros(
        len(physical),
        dtype=int,
    )

    labels[
        (physical <= 2)
        & (impact > EPSILON + 1e-12)
    ] = 1

    labels[
        (physical >= 3)
        & (impact > EPSILON + 1e-12)
    ] = 2

    # Equal prior probability for each operational class.
    # This is an experimental design assumption,
    # not an estimate of real hardware fault prevalence.

    prior = np.zeros(
        len(physical)
    )

    for cls in range(3):

        mask = labels == cls

        prior[mask] = (
            1.0 / (3 * np.sum(mask))
        )

    class_map = np.eye(3)[labels]

    latent_map = np.eye(
        len(physical)
    )

    # Binomial measurement likelihoods.

    outcomes = np.arange(
        BATCH + 1
    )[None, :, None]

    p_minus = (
        SENS[:, physical]
        * strength[None, :]
    )

    likelihoods = binom.pmf(
        outcomes,
        BATCH,
        p_minus[:, None, :],
    )

    assert np.allclose(
        likelihoods.sum(axis=1),
        1.0,
    )

    return (
        physical,
        strength,
        labels,
        prior,
        class_map,
        latent_map,
        likelihoods,
    )


# ============================================================
# 5. ENTROPY AND BAYESIAN PREDICTION
# ============================================================

def entropy(probabilities):

    return -np.sum(
        xlogy(
            probabilities,
            probabilities,
        ),
        axis=-1,
    )


def expected_posteriors(
    posterior,
    likelihood,
):
    """
    Compute every possible posterior after
    one measurement batch.
    """

    predictive = likelihood @ posterior

    valid = predictive > 1e-14

    conditional = (
        likelihood[valid, :]
        * posterior[None, :]
    ) / predictive[valid, None]

    return (
        predictive[valid],
        conditional,
    )


# ============================================================
# 6. INFORMATION GAIN AND VALUE OF INFORMATION
# ============================================================

def probe_score(
    policy,
    posterior,
    likelihood,
    grouping,
):

    predictive, conditionals = (
        expected_posteriors(
            posterior,
            likelihood,
        )
    )

    if policy == "decision_voi":

        # Current minimum expected decision loss.

        class_posterior = (
            posterior @ grouping
        )

        current_risk = float(
            np.min(
                LOSS @ class_posterior
            )
        )

        # Expected decision loss after measurement.

        future_labels = (
            conditionals @ grouping
        )

        future_risks = np.min(
            future_labels @ LOSS.T,
            axis=1,
        )

        expected_future_risk = float(
            predictive @ future_risks
        )

        return (
            current_risk
            - expected_future_risk
        )

    # Conventional expected information gain.

    current_entropy = float(
        entropy(
            posterior @ grouping
        )
    )

    future_entropies = entropy(
        conditionals @ grouping
    )

    expected_future_entropy = float(
        predictive @ future_entropies
    )

    return (
        current_entropy
        - expected_future_entropy
    )


# ============================================================
# 7. PROBE SELECTION
# ============================================================

def choose_probe(
    policy,
    posterior,
    step,
    likelihoods,
    class_map,
    latent_map,
):

    if policy in {
        "decision_voi",
        "label_eig",
        "latent_eig",
    }:

        grouping = (
            latent_map
            if policy == "latent_eig"
            else class_map
        )

        scores = [
            probe_score(
                policy,
                posterior,
                likelihoods[k],
                grouping,
            )
            for k in range(len(PROBES))
        ]

        return int(
            np.argmax(scores)
        )

    if policy == "fixed4":

        return [0, 1, 2, 4][step % 4]

    if policy == "round6":

        return step % 6

    raise ValueError(policy)


# ============================================================
# 8. MINIMUM-RISK DIAGNOSIS
# ============================================================

def classify(posterior, class_map):

    class_posterior = (
        posterior @ class_map
    )

    risks = LOSS @ class_posterior

    action = int(
        np.argmin(risks)
    )

    minimum_risk = float(
        risks[action]
    )

    return (
        action,
        minimum_risk,
        class_posterior,
    )


# ============================================================
# 9. ONE EXPERIMENTAL TRIAL
# ============================================================

def single_trial(
    policy,
    physical_fault,
    p_true,
    truth,
    uniforms,
    prior,
    class_map,
    latent_map,
    likelihoods,
):

    posterior = prior.copy()

    first_decision = -1
    first_shots = BUDGET
    first_correct = 0

    for step in range(
        BUDGET // BATCH
    ):

        probe = choose_probe(
            policy,
            posterior,
            step,
            likelihoods,
            class_map,
            latent_map,
        )

        # Synthetic shot generation.

        minus_probability = (
            SENS[probe, physical_fault]
            * p_true
        )

        negative_count = int(
            np.count_nonzero(
                uniforms[step] < minus_probability
            )
        )

        # Posterior update.

        posterior *= likelihoods[
            probe,
            negative_count,
        ]

        posterior /= posterior.sum()

        decision, risk, _ = classify(
            posterior,
            class_map,
        )

        if (
            first_decision < 0
            and risk <= ABSTAIN_COST
        ):

            first_decision = decision

            first_correct = int(
                decision == truth
            )

            first_shots = (
                step + 1
            ) * BATCH

    # The simulation continues to full budget
    # for offline fixed-budget comparison.
    # first_decision records the earlier stopping action.

    decision, risk, _ = classify(
        posterior,
        class_map,
    )

    accepted = int(
        first_decision >= 0
    )

    operational_loss = (
        float(
            LOSS[first_decision, truth]
        )
        if accepted
        else ABSTAIN_COST
    )

    return {
        "correct192": int(
            decision == truth
        ),
        "accepted": accepted,
        "correct_accepted": first_correct,
        "shots_or_cap": first_shots,
        "risk192": risk,
        "decision192": decision,
        "realized_loss192": float(
            LOSS[decision, truth]
        ),
        "realized_operational_loss":
            operational_loss,
    }


# ============================================================
# 10. MONTE CARLO BENCHMARK
# ============================================================

def main():

    coefficients, ideal = (
        quantum_impact_coefficients()
    )

    print(
        "\nTarget: O = 0.7 ZZ + 0.3 XX"
    )

    print(
        "Ideal expectation:",
        round(ideal, 6),
    )

    print(
        "Observable impact coefficients:"
    )

    print(
        dict(
            zip(
                FAULTS,
                coefficients.round(6),
            )
        )
    )

    assert np.allclose(
        coefficients,
        [0, 1.4, 1.4, 0.6, 0.6],
    )

    (
        physical,
        strength,
        labels,
        prior,
        class_map,
        latent_map,
        likelihoods,
    ) = build_model(coefficients)

    # Out-of-grid testing scenarios.
    # Include difficult cases immediately
    # below and above the impact threshold.

    scenarios = [
        ("safe_none", 0, 0.000),
        ("safe_X0", 1, 0.025),
        ("safe_X1_boundary", 2, 0.040),
        ("safe_Z0", 3, 0.080),
        ("safe_Z1_boundary", 4, 0.095),
        ("X0_boundary", 1, 0.046),
        ("X1_medium", 2, 0.065),
        ("X0_strong", 1, 0.120),
        ("X1_strong", 2, 0.170),
        ("Z0_boundary", 3, 0.105),
        ("Z1_medium", 4, 0.130),
        ("Z0_strong", 3, 0.160),
        ("Z1_strong", 4, 0.200),
    ]

    rng = np.random.default_rng(SEED)

    rows = []

    for name, fault, rate in scenarios:

        impact = (
            coefficients[fault] * rate
        )

        if impact <= EPSILON:
            truth = 0
        elif fault <= 2:
            truth = 1
        else:
            truth = 2

        for repeat in range(REPEATS):

            # Common random numbers across policies.

            uniforms = rng.random(
                (
                    BUDGET // BATCH,
                    BATCH,
                )
            )

            for policy in POLICIES:

                result = single_trial(
                    policy,
                    fault,
                    rate,
                    truth,
                    uniforms,
                    prior,
                    class_map,
                    latent_map,
                    likelihoods,
                )

                rows.append({
                    "scenario": name,
                    "true_class": CLASSES[truth],
                    "true_p": rate,
                    "impact": impact,
                    "repeat": repeat,
                    "policy": policy,
                    **result,
                })

    df = pd.DataFrame(rows)

    # ========================================================
    # 11. SUMMARY
    # ========================================================

    summary = df.groupby(
        "policy",
        sort=False,
    ).agg(
        accuracy192=(
            "correct192",
            "mean",
        ),
        coverage=(
            "accepted",
            "mean",
        ),
        accepted=(
            "accepted",
            "sum",
        ),
        correct_accepted=(
            "correct_accepted",
            "sum",
        ),
        mean_shots_or_cap=(
            "shots_or_cap",
            "mean",
        ),
        mean_operational_loss=(
            "realized_operational_loss",
            "mean",
        ),
        mean_forced_loss192=(
            "realized_loss192",
            "mean",
        ),
    )

    summary["selective_risk"] = (
        1
        - summary["correct_accepted"]
        / summary["accepted"]
    )

    print(
        "\n=== MISSION 08 BENCHMARK ==="
    )

    print(
        summary[
            [
                "accuracy192",
                "coverage",
                "selective_risk",
                "mean_operational_loss",
                "mean_shots_or_cap",
            ]
        ].round(3).to_string()
    )

    # ========================================================
    # 12. PER-SCENARIO AUDIT
    # ========================================================

    per_scenario = df.groupby(
        ["scenario", "policy"],
        sort=False,
    ).agg(
        accuracy=(
            "correct192",
            "mean",
        ),
        coverage=(
            "accepted",
            "mean",
        ),
        accepted=(
            "accepted",
            "sum",
        ),
        correct=(
            "correct_accepted",
            "sum",
        ),
    ).reset_index()

    per_scenario["risk"] = (
        1
        - per_scenario["correct"]
        / per_scenario["accepted"].replace(
            0,
            np.nan,
        )
    )

    print(
        "\n=== BOUNDARY CASES ==="
    )

    boundary = per_scenario[
        per_scenario["scenario"].str.contains(
            "boundary"
        )
    ]

    print(
        boundary[
            [
                "scenario",
                "policy",
                "accuracy",
                "coverage",
                "risk",
            ]
        ].round(3).to_string(
            index=False
        )
    )

    # ========================================================
    # 13. SAVE RESULTS
    # ========================================================

    output = Path("results")

    output.mkdir(
        exist_ok=True
    )

    df.to_csv(
        output / "mission08_trials.csv",
        index=False,
    )

    summary.to_csv(
        output / "mission08_summary.csv"
    )

    per_scenario.to_csv(
        output / "mission08_scenarios.csv",
        index=False,
    )

    # ========================================================
    # 14. VISUALIZATION
    # ========================================================

    by_class = (
        df.groupby(
            ["policy", "true_class"]
        )["correct192"]
        .mean()
        .unstack()
        .reindex(POLICIES)
    )

    ax = by_class.plot(
        kind="bar",
        figsize=(10, 5),
        rot=15,
    )

    ax.set(
        title=(
            "Q-TRACE: Observable-Impact-Aware "
            "Classification (Synthetic)"
        ),
        ylabel="Accuracy at 192 shots",
        xlabel="Measurement policy",
        ylim=(0, 1.05),
    )

    ax.legend(
        title="True operational class"
    )

    ax.figure.tight_layout()

    ax.figure.savefig(
        output / "mission08_by_class.png",
        dpi=160,
    )

    plt.close(ax.figure)

    print(
        "\nSaved results/mission08_*"
    )

    print(
        "Synthetic measurements; Qiskit derives "
        "the observable impact coefficients."
    )


if __name__ == "__main__":
    main()
