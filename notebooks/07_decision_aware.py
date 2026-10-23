
"""
Q-TRACE | Mission 07: Detectability-Aware Active Diagnosis

Synthetic Pauli measurement model, independently validated
against Qiskit Aer for simple probes in Mission 02.

No real quantum hardware calls are made here.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import xlogy
from scipy.stats import binom, binomtest


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

SENS = np.array([
    [0, 1, 1, 0, 0],  # bell_ZZ
    [0, 0, 0, 1, 1],  # bell_XX
    [0, 1, 0, 0, 0],  # zero_Z0
    [0, 0, 1, 0, 0],  # zero_Z1
    [0, 0, 0, 1, 0],  # plus_X0
    [0, 0, 0, 0, 1],  # plus_X1
], dtype=float)

PROBES = [
    "bell_ZZ",
    "bell_XX",
    "zero_Z0",
    "zero_Z1",
    "plus_X0",
    "plus_X1",
]

LABELS = [
    "below_threshold",
    "X0",
    "X1",
    "Z0",
    "Z1",
]

P_DETECT = 0.03

BATCH = 16
BUDGET = 192
TRIALS_PER_LABEL = 120

SEED = 2026

ABSTAIN_COST = 0.65

POLICIES = [
    "label_eig",
    "latent_eig",
    "fixed4",
    "round6",
    "random6",
]


# ============================================================
# 2. LATENT FAULT MODEL
# ============================================================

# Latent states include:
#
# 1. No physical fault
# 2. Weak faults below the operational threshold
# 3. Detectable faults at or above the threshold
#
# Operational class:
#
# 0 = below_threshold
# 1 = X0
# 2 = X1
# 3 = Z0
# 4 = Z1

state_fault = [0]
state_p = [0.0]
state_label = [0]
state_prior = [0.10]

for fault in range(1, 5):

    # Weak faults:
    # These belong to the operationally negative class.

    for p in [0.005, 0.015, 0.025]:

        state_fault.append(fault)
        state_p.append(p)
        state_label.append(0)

        state_prior.append(
            0.10 / (4 * 3)
        )

    # Detectable faults.

    for p in [0.03, 0.05, 0.10, 0.20]:

        state_fault.append(fault)
        state_p.append(p)
        state_label.append(fault)

        state_prior.append(
            0.80 / (4 * 4)
        )


state_fault = np.array(state_fault)
state_p = np.array(state_p)
state_label = np.array(state_label)

PRIOR = np.array(
    state_prior,
    dtype=float,
)

assert np.isclose(
    PRIOR.sum(),
    1.0,
)

assert np.allclose(
    PRIOR @ np.eye(5)[state_label],
    np.full(5, 0.2),
)

assert np.all(
    state_p[state_label == 0] < P_DETECT
)

assert np.all(
    state_p[state_label != 0] >= P_DETECT
)


# ============================================================
# 3. CLASS MARGINALIZATION
# ============================================================

# Convert the posterior over all latent states
# into a posterior over five operational classes.

CLASS_MAP = np.eye(5)[state_label]


# ============================================================
# 4. QUANTUM MEASUREMENT LIKELIHOODS
# ============================================================

# For sensitive Pauli probes:
#
# P(measurement outcome = -1) = p
#
# For insensitive probes:
#
# P(measurement outcome = -1) = 0

OUTCOME = np.arange(
    BATCH + 1
)[None, :, None]

P_MINUS = (
    SENS[:, state_fault]
    * state_p[None, :]
)

LIKELIHOOD = binom.pmf(
    OUTCOME,
    BATCH,
    P_MINUS[:, None, :],
)

assert np.allclose(
    LIKELIHOOD.sum(axis=1),
    1.0,
)


# ============================================================
# 5. COST-SENSITIVE DECISION MODEL
# ============================================================

# Rows = diagnostic decisions
# Columns = true operational labels
#
# Correct diagnosis: 0
# Wrong fault location: 3
# Missed detectable fault: 8
# False alarm: 4

LOSS = np.full(
    (5, 5),
    3.0,
)

np.fill_diagonal(
    LOSS,
    0.0,
)

LOSS[0, 1:] = 8.0

LOSS[1:, 0] = 4.0


# ============================================================
# 6. SHANNON ENTROPY
# ============================================================

def entropy(p):

    return float(
        -np.sum(
            xlogy(p, p)
        )
    )


# ============================================================
# 7. EXPECTED INFORMATION GAIN
# ============================================================

def eig(prior, probe, grouping):
    """
    Calculate expected information gain.

    The target may be:

    - Operational fault labels
    - Full latent fault states

    EIG = H(prior) - E[H(posterior)]
    """

    like = LIKELIHOOD[probe]

    predictive = like @ prior

    valid = predictive > 1e-14

    post = (
        like[valid]
        * prior[None, :]
        / predictive[valid, None]
    )

    post_target = post @ grouping

    expected_entropy = float(
        predictive[valid]
        @ (
            -np.sum(
                xlogy(
                    post_target,
                    post_target,
                ),
                axis=1,
            )
        )
    )

    current_entropy = entropy(
        prior @ grouping
    )

    return (
        current_entropy
        - expected_entropy
    )


# ============================================================
# 8. ADAPTIVE PROBE SELECTION
# ============================================================

def select_probe(
    policy,
    posterior,
    step,
    rng_schedule,
):

    if policy in {
        "label_eig",
        "latent_eig",
    }:

        if policy == "label_eig":

            grouping = CLASS_MAP

        else:

            grouping = np.eye(
                len(PRIOR)
            )

        scores = [
            eig(
                posterior,
                k,
                grouping,
            )
            for k in range(len(PROBES))
        ]

        return int(
            np.argmax(scores)
        )

    if policy == "fixed4":

        return [
            0, 1, 2, 4
        ][step % 4]

    if policy == "round6":

        return step % 6

    if policy == "random6":

        return int(
            rng_schedule[step]
        )

    raise ValueError(policy)


# ============================================================
# 9. BAYESIAN DECISION RULE
# ============================================================

def action_and_risk(label_posterior):
    """
    Choose the decision with minimum
    expected terminal loss.
    """

    risks = LOSS @ label_posterior

    action = int(
        np.argmin(risks)
    )

    minimum_risk = float(
        risks[action]
    )

    return action, minimum_risk


# ============================================================
# 10. SIMULATE ONE DIAGNOSTIC TRIAL
# ============================================================

def simulate(
    policy,
    physical_fault,
    p_true,
    target_label,
    uniforms,
    schedule,
):

    posterior = PRIOR.copy()

    # -1 indicates no accepted decision.
    first_action = -1

    first_shots = BUDGET

    accuracy64 = np.nan

    for step in range(
        BUDGET // BATCH
    ):

        # ----------------------------------------
        # Select the next measurement probe
        # ----------------------------------------

        probe = select_probe(
            policy,
            posterior,
            step,
            schedule,
        )

        # ----------------------------------------
        # Generate synthetic measurement outcomes
        # ----------------------------------------

        minus_prob = (
            SENS[probe, physical_fault]
            * p_true
        )

        nminus = int(
            np.count_nonzero(
                uniforms[step] < minus_prob
            )
        )

        # ----------------------------------------
        # Bayesian posterior update
        # ----------------------------------------

        posterior *= LIKELIHOOD[
            probe,
            nminus,
        ]

        posterior /= posterior.sum()

        # ----------------------------------------
        # Marginalize the latent noise strength
        # ----------------------------------------

        label_posterior = (
            posterior @ CLASS_MAP
        )

        decision, min_risk = (
            action_and_risk(
                label_posterior
            )
        )

        shots = (
            step + 1
        ) * BATCH

        if shots == 64:

            accuracy64 = int(
                decision == target_label
            )

        # ----------------------------------------
        # Risk-aware stopping
        # ----------------------------------------

        if (
            first_action == -1
            and min_risk <= ABSTAIN_COST
        ):

            first_action = decision
            first_shots = shots

    # --------------------------------------------
    # Evaluate at the full measurement budget
    # --------------------------------------------

    final_label_post = (
        posterior @ CLASS_MAP
    )

    final_decision, final_risk = (
        action_and_risk(
            final_label_post
        )
    )

    return {

        "accuracy64":
            accuracy64,

        "accuracy192":
            int(
                final_decision == target_label
            ),

        "accepted":
            int(
                first_action != -1
            ),

        "correct_accepted":
            int(
                first_action == target_label
            ),

        "shots_to_decision_or_cap":
            first_shots,

        "final_pred":
            final_decision,

        "final_min_risk":
            final_risk,
    }


# ============================================================
# 11. MONTE CARLO EXPERIMENT
# ============================================================

def main():

    rng = np.random.default_rng(
        SEED
    )

    rows = []

    for target in range(5):

        for repeat in range(
            TRIALS_PER_LABEL
        ):

            # ------------------------------------
            # Construct ground-truth scenarios
            # ------------------------------------

            if target == 0:

                # Operationally negative:
                # no fault or weak fault.

                if repeat % 3 == 0:

                    physical_fault = 0
                    p_true = 0.0

                else:

                    physical_fault = int(
                        rng.integers(1, 5)
                    )

                    p_true = [
                        0.015,
                        0.025,
                    ][repeat % 2]

            else:

                physical_fault = target

                # Out-of-grid strengths.

                p_true = [
                    0.035,
                    0.08,
                ][repeat % 2]

            # ------------------------------------
            # Common random numbers
            # ------------------------------------

            uniforms = rng.random(
                (
                    BUDGET // BATCH,
                    BATCH,
                )
            )

            schedule = rng.integers(
                0,
                len(PROBES),
                size=BUDGET // BATCH,
            )

            # ------------------------------------
            # Evaluate all policies
            # ------------------------------------

            for policy in POLICIES:

                metrics = simulate(
                    policy,
                    physical_fault,
                    p_true,
                    target,
                    uniforms,
                    schedule,
                )

                rows.append({

                    "true_label":
                        LABELS[target],

                    "physical_fault":
                        (
                            "None"
                            if physical_fault == 0
                            else LABELS[physical_fault]
                        ),

                    "true_p":
                        p_true,

                    "repeat":
                        repeat,

                    "policy":
                        policy,

                    **metrics,
                })

    trials = pd.DataFrame(rows)


    # ========================================================
    # 12. POLICY BENCHMARK
    # ========================================================

    summary = trials.groupby(
        "policy",
        sort=False,
    ).agg(

        accuracy64=(
            "accuracy64",
            "mean",
        ),

        accuracy192=(
            "accuracy192",
            "mean",
        ),

        coverage=(
            "accepted",
            "mean",
        ),

        mean_shots_or_cap=(
            "shots_to_decision_or_cap",
            "mean",
        ),

        accepted_count=(
            "accepted",
            "sum",
        ),

        correct_accepted=(
            "correct_accepted",
            "sum",
        ),
    )

    summary["selective_risk"] = (
        1
        - summary["correct_accepted"]
        / summary["accepted_count"]
    )

    print(
        "\n=== MISSION 07: OPERATIONAL DETECTABILITY ==="
    )

    print(
        f"Minimum fault strength: {P_DETECT}; "
        f"batch={BATCH}; budget={BUDGET}"
    )

    print(
        "Terminal losses: false alarm=4, "
        "missed fault=8, mislocation=3; "
        f"abstain={ABSTAIN_COST}"
    )

    print(
        "True test strengths 0.035 and 0.08 "
        "are not in the inference grid."
    )

    print(
        "\n=== POLICY BENCHMARK (SYNTHETIC DATA) ==="
    )

    print(
        summary[
            [
                "accuracy64",
                "accuracy192",
                "coverage",
                "selective_risk",
                "mean_shots_or_cap",
            ]
        ].round(3).to_string()
    )


    # ========================================================
    # 13. PAIRED STATISTICAL SIGNIFICANCE TEST
    # ========================================================

    paired = trials.pivot(
        index=[
            "true_label",
            "repeat",
        ],
        columns="policy",
        values="accuracy192",
    )

    a = paired[
        "label_eig"
    ].to_numpy(dtype=int)

    b = paired[
        "latent_eig"
    ].to_numpy(dtype=int)

    wins = int(
        np.sum(
            (a == 1)
            & (b == 0)
        )
    )

    losses = int(
        np.sum(
            (a == 0)
            & (b == 1)
        )
    )

    discordant = wins + losses

    if discordant > 0:

        p_value = float(
            binomtest(
                wins,
                discordant,
                p=0.5,
            ).pvalue
        )

    else:

        p_value = 1.0

    print(
        f"\nPaired label_eig vs latent_eig: "
        f"{wins} wins, "
        f"{losses} losses, "
        f"exact McNemar p={p_value:.4f}"
    )


    # ========================================================
    # 14. PER-LABEL SCIENTIFIC AUDIT
    # ========================================================

    by_label = trials.groupby(
        [
            "policy",
            "true_label",
        ],
        sort=False,
    ).agg(

        coverage=(
            "accepted",
            "mean",
        ),

        fixed_budget_accuracy=(
            "accuracy192",
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

    by_label["risk_if_accepted"] = (
        1
        - by_label["correct"]
        / by_label["accepted"].replace(
            0,
            np.nan,
        )
    )

    print(
        "\n=== PER-LABEL AUDIT ==="
    )

    print(
        by_label[
            [
                "policy",
                "true_label",
                "coverage",
                "fixed_budget_accuracy",
                "risk_if_accepted",
            ]
        ].round(3).to_string(
            index=False
        )
    )


    # ========================================================
    # 15. SAVE EXPERIMENTAL RESULTS
    # ========================================================

    out = Path("results")

    out.mkdir(
        exist_ok=True
    )

    trials.to_csv(
        out / "mission07_trials.csv",
        index=False,
    )

    summary.to_csv(
        out / "mission07_summary.csv"
    )

    by_label.to_csv(
        out / "mission07_by_label.csv",
        index=False,
    )


    # ========================================================
    # 16. RISK-COVERAGE ANALYSIS
    # ========================================================

    # Evaluate different terminal risk thresholds.
    #
    # The inference model and dataset remain fixed.
    # Only the acceptance threshold changes.

    records = []

    for policy in POLICIES:

        subset = trials[
            trials.policy == policy
        ]

        for risk_limit in np.linspace(
            0.05,
            2.0,
            30,
        ):

            accepted = (
                subset.final_min_risk.to_numpy()
                <= risk_limit
            )

            accuracy = (
                subset.accuracy192.to_numpy(
                    dtype=bool
                )
            )

            records.append({

                "policy":
                    policy,

                "risk_limit":
                    risk_limit,

                "coverage":
                    float(
                        accepted.mean()
                    ),

                "selective_risk":
                    (
                        float(
                            (~accuracy[accepted]).mean()
                        )
                        if accepted.any()
                        else np.nan
                    ),
            })

    sweep = pd.DataFrame(records)

    sweep.to_csv(
        out / "mission07_risk_coverage.csv",
        index=False,
    )


    # ========================================================
    # 17. VISUALIZATION
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for policy in POLICIES:

        data = sweep[
            sweep.policy == policy
        ]

        ax.plot(
            data.coverage,
            data.selective_risk,
            marker=".",
            label=policy,
        )

    ax.set(
        xlabel="Diagnostic coverage at 192 shots",
        ylabel="Misdiagnosis rate among accepted cases",
        title=(
            "Q-TRACE: Risk-Coverage Trade-off "
            "(Synthetic Noise)"
        ),
        xlim=(0, 1.05),
        ylim=(0, 1.05),
    )

    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        out / "mission07_risk_coverage.png",
        dpi=180,
    )

    plt.close(fig)

    print(
        "\nSaved mission07_*.csv "
        "and mission07_risk_coverage.png"
    )

    print(
        "NOTE: Synthetic model; "
        "no Aer or hardware inference in Mission 07."
    )
if __name__ == "__main__":
    main()
