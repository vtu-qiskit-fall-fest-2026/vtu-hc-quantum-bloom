
"""
Q-TRACE | Mission 09
Information-Theoretic Limits of Quantum Fault Detectability

Research objectives:
1. Derive task-specific Pauli error impact with Qiskit.
2. Calculate exact binary Bayes classification limits.
3. Estimate minimum measurement budgets.
4. Validate the assumed shot distributions with Qiskit Aer.

The oracle knows the physical fault and the two
candidate error probabilities.

This is an optimistic, restricted-model benchmark,
NOT an experimental real-hardware performance bound.

Run:
    python notebooks/09_information_limits.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import binom, binomtest

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import pauli_error


# ============================================================
# 1. CONFIGURATION
# ============================================================

EPSILON = 0.06

BUDGETS = [
    16, 32, 64, 96, 128,
    192, 384, 768, 1536, 4096,
]

AER_SHOTS = 4096
SEED = 2026

OUTPUT = Path("results")

# Each scenario:
# name, Pauli error, measurement observable,
# safe error rate, harmful error rate.

SCENARIOS = [
    ("X_boundary", "X", "ZZ", 0.040, 0.046),
    ("Z_boundary", "Z", "XX", 0.095, 0.105),
    ("X_separated", "X", "ZZ", 0.025, 0.120),
    ("Z_separated", "Z", "XX", 0.080, 0.160),
]


# ============================================================
# 2. QUANTUM STATE PREPARATION
# ============================================================

def bell():

    qc = QuantumCircuit(2)

    qc.h(0)
    qc.cx(0, 1)

    return qc


# ============================================================
# 3. QUANTUM OBSERVABLE IMPACT
# ============================================================

def impact_coefficients():
    """
    Compute observable sensitivity from Qiskit.

    Target observable:
        O = 0.7 ZZ + 0.3 XX

    A stochastic Pauli fault of probability p
    produces an observable deviation:

        Delta = coefficient * p
    """

    state = Statevector.from_instruction(
        bell()
    ).data

    target = (
        0.7 * Pauli("ZZ").to_matrix()
        + 0.3 * Pauli("XX").to_matrix()
    )

    ideal = np.vdot(
        state,
        target @ state,
    ).real

    coefficients = {}

    # Qiskit labels are in q1-q0 order.
    # IX means X on physical qubit 0.
    # IZ means Z on physical qubit 0.

    for gate, pauli_label in [
        ("X", "IX"),
        ("Z", "IZ"),
    ]:

        error_operator = Pauli(
            pauli_label
        ).to_matrix()

        faulted = error_operator @ state

        faulty_value = np.vdot(
            faulted,
            target @ faulted,
        ).real

        coefficients[gate] = float(
            abs(ideal - faulty_value)
        )

    assert np.isclose(ideal, 1.0)

    assert np.isclose(
        coefficients["X"], 1.4
    )

    assert np.isclose(
        coefficients["Z"], 0.6
    )

    return coefficients


# ============================================================
# 4. BERNOULLI KL DIVERGENCE
# ============================================================

def bernoulli_kl(p, q):
    """
    D_KL(Bernoulli(p) || Bernoulli(q))

    Unit: nats per informative measurement shot.
    """

    return float(
        p * np.log(p / q)
        + (1 - p)
        * np.log((1 - p) / (1 - q))
    )


# ============================================================
# 5. EXACT OPTIMAL BINARY CLASSIFICATION ERROR
# ============================================================

def exact_bayes_error(
    n,
    p_safe,
    p_harmful,
):
    """
    Minimum binary classification error.

    Assumptions:
    - Equal prior probability for both hypotheses.
    - Independent informative Bernoulli measurements.
    - Exact knowledge of both candidate noise rates.
    - Fixed measurement budget n.

    The likelihood-ratio test is optimal under
    these assumptions.
    """

    if n == 0:
        return 0.5

    if not (
        0 < p_safe < p_harmful < 1
    ):
        raise ValueError(
            "Require 0 < p_safe < p_harmful < 1"
        )

    # Choose the harmful hypothesis when
    # the number of negative outcomes reaches
    # the likelihood-ratio decision threshold.

    threshold = (
        n
        * np.log(
            (1 - p_safe)
            / (1 - p_harmful)
        )
        / np.log(
            p_harmful
            * (1 - p_safe)
            / (
                p_safe
                * (1 - p_harmful)
            )
        )
    )

    cutoff = int(
        np.ceil(threshold)
    )

    # False alarm:
    # True state is safe, classified as harmful.

    false_alarm = binom.sf(
        cutoff - 1,
        n,
        p_safe,
    )

    # Missed harmful fault:
    # True state is harmful, classified as safe.

    missed_fault = binom.cdf(
        cutoff - 1,
        n,
        p_harmful,
    )

    return float(
        0.5 * (
            false_alarm + missed_fault
        )
    )


# ============================================================
# 6. MINIMUM REQUIRED MEASUREMENT BUDGET
# ============================================================

def shots_to_error_below(
    p_safe,
    p_harmful,
    target,
    cap=200_000,
):
    """
    Find the smallest integer n such that
    optimal Bayes error <= target.

    Binary search is valid because the optimal
    classification error cannot increase
    when an additional sample is available.
    """

    if exact_bayes_error(
        cap,
        p_safe,
        p_harmful,
    ) > target:

        return None

    low = 0
    high = cap

    while low < high:

        mid = (low + high) // 2

        error = exact_bayes_error(
            mid,
            p_safe,
            p_harmful,
        )

        if error <= target:
            high = mid

        else:
            low = mid + 1

    return low


# ============================================================
# 7. BUILD PHYSICAL QUANTUM NOISE EXPERIMENT
# ============================================================

def noisy_probe(
    error_gate,
    observable,
    p,
):
    """
    Prepare a Bell state.

    Apply stochastic X or Z error on q0
    immediately after state preparation.

    Measure:
        ZZ for X errors
        XX for Z errors

    In this experimental setting:
        P(measured eigenvalue -1) = p
    """

    qc = bell()

    error = pauli_error([
        ("I", 1 - p),
        (error_gate, p),
    ])

    qc.append(
        error.to_instruction(),
        [0],
    )

    # Convert X measurements to Z measurements.

    if observable == "XX":

        qc.h([0, 1])

    elif observable != "ZZ":

        raise ValueError(
            f"Unsupported observable: {observable}"
        )

    qc.measure_all()

    return qc


# ============================================================
# 8. EXTRACT NEGATIVE PAULI MEASUREMENT OUTCOMES
# ============================================================

def minus_count(counts):
    """
    The product observable XX or ZZ has eigenvalue:

        +1 -> even measurement parity
        -1 -> odd measurement parity
    """

    return sum(
        count
        for bits, count in counts.items()
        if (
            bits.replace(" ", "").count("1") % 2
        )
    )


# ============================================================
# 9. MAIN RESEARCH EXPERIMENT
# ============================================================

def main():

    OUTPUT.mkdir(
        exist_ok=True
    )

    coefficients = impact_coefficients()

    print(
        "\n=== Q-TRACE MISSION 09 ==="
    )

    print(
        "Observable: O = 0.7 ZZ + 0.3 XX"
    )

    print(
        "Impact threshold:",
        EPSILON,
    )

    print(
        "Qiskit-derived impact coefficients:",
        coefficients,
    )

    theory = []
    curves = []

    circuits = []
    configs = []

    for (
        name,
        gate,
        observable,
        p_safe,
        p_harmful,
    ) in SCENARIOS:

        # Confirm that the two probabilities
        # belong to opposite sides of the
        # operational impact threshold.

        safe_impact = (
            coefficients[gate]
            * p_safe
        )

        harmful_impact = (
            coefficients[gate]
            * p_harmful
        )

        assert (
            safe_impact
            < EPSILON
            < harmful_impact
        ), name

        # ----------------------------------------
        # Analytical information limits
        # ----------------------------------------

        kl = bernoulli_kl(
            p_safe,
            p_harmful,
        )

        oracle_error = exact_bayes_error(
            192,
            p_safe,
            p_harmful,
        )

        # Pinsker-based lower bound on the
        # equal-prior binary Bayes error.
        #
        # Total KL for n independent shots = n * KL.

        pinsker_bound = max(
            0.0,
            0.5 * (
                1.0
                - np.sqrt(
                    192 * kl / 2
                )
            ),
        )

        n10 = shots_to_error_below(
            p_safe,
            p_harmful,
            target=0.10,
        )

        n05 = shots_to_error_below(
            p_safe,
            p_harmful,
            target=0.05,
        )

        theory.append({
            "scenario": name,
            "p_safe": p_safe,
            "p_harmful": p_harmful,
            "safe_impact": safe_impact,
            "harmful_impact": harmful_impact,
            "KL_per_shot": kl,
            "oracle_error_192": oracle_error,
            "pinsker_lower_bound_192": pinsker_bound,
            "shots_for_10pct_error": n10,
            "shots_for_5pct_error": n05,
        })

        # ----------------------------------------
        # Bayes error curve vs shot budget
        # ----------------------------------------

        for shots in BUDGETS:

            curves.append({
                "scenario": name,
                "shots": shots,
                "minimum_pairwise_error": (
                    exact_bayes_error(
                        shots,
                        p_safe,
                        p_harmful,
                    )
                ),
            })

        # ----------------------------------------
        # Build corresponding Qiskit circuits
        # ----------------------------------------

        for case, p in [
            ("safe", p_safe),
            ("harmful", p_harmful),
        ]:

            configs.append(
                (name, case, p)
            )

            circuits.append(
                noisy_probe(
                    gate,
                    observable,
                    p,
                )
            )

    theory_df = pd.DataFrame(
        theory
    )

    curve_df = pd.DataFrame(
        curves
    )

    print(
        "\n=== ORACLE INFORMATION LIMITS ==="
    )

    print(
        theory_df.round(5).to_string(
            index=False
        )
    )

    # ========================================================
    # 10. QISKIT AER VALIDATION
    # ========================================================

    simulator = AerSimulator(
        method="stabilizer"
    )

    result = simulator.run(
        circuits,
        shots=AER_SHOTS,
        seed_simulator=SEED,
    ).result()

    aer_rows = []

    for index, (
        name,
        case,
        p,
    ) in enumerate(configs):

        counts = result.get_counts(
            index
        )

        observed_minus = minus_count(
            counts
        )

        estimated_p = (
            observed_minus / AER_SHOTS
        )

        # Wilson confidence interval for
        # measured negative-outcome probability.

        interval = binomtest(
            observed_minus,
            AER_SHOTS,
        ).proportion_ci(
            confidence_level=0.95,
            method="wilson",
        )

        aer_rows.append({
            "scenario": name,
            "case": case,
            "theoretical_Pminus": p,
            "aer_Pminus": estimated_p,
            "minus_outcomes": observed_minus,
            "shots": AER_SHOTS,
            "wilson_low": interval.low,
            "wilson_high": interval.high,
        })

    aer_df = pd.DataFrame(
        aer_rows
    )

    print(
        "\n=== QISKIT AER VALIDATION ==="
    )

    print(
        aer_df.round(5).to_string(
            index=False
        )
    )

    # ========================================================
    # 11. SAVE RESULTS
    # ========================================================

    theory_df.to_csv(
        OUTPUT / "mission09_information_limits.csv",
        index=False,
    )

    curve_df.to_csv(
        OUTPUT / "mission09_error_curve.csv",
        index=False,
    )

    aer_df.to_csv(
        OUTPUT / "mission09_aer_validation.csv",
        index=False,
    )

    # ========================================================
    # 12. VISUALIZE INFORMATION-THEORETIC LIMITS
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for name, group in curve_df.groupby(
        "scenario",
        sort=False,
    ):

        ax.plot(
            group["shots"],
            group["minimum_pairwise_error"],
            marker="o",
            label=name,
        )

    ax.axhline(
        0.10,
        linestyle="--",
        alpha=0.6,
        label="10% error target",
    )

    ax.axvline(
        192,
        linestyle=":",
        alpha=0.65,
        label="192-shot budget",
    )

    ax.set(
        xlabel="Number of informative shots",
        ylabel="Optimal pairwise error probability",
        title=(
            "Q-TRACE: Information-Theoretic Limits "
            "of Fault Detectability"
        ),
        ylim=(0, 0.52),
    )

    ax.set_xscale(
        "log"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT / "mission09_information_limits.png",
        dpi=180,
    )

    plt.close(fig)

    print(
        "\nFiles saved to results/mission09_*"
    )

    print(
        "\nIMPORTANT ASSUMPTIONS:"
    )

    print(
        "- Binary oracle knows the fault type/location."
    )

    print(
        "- Each informative shot samples Bernoulli(p)."
    )

    print(
        "- Both hypotheses have equal prior probability."
    )

    print(
        "- Qiskit Aer validates the measurement model."
    )

    print(
        "- These are NOT universal bounds "
        "for arbitrary quantum protocols."
    )


# ============================================================
# 13. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
