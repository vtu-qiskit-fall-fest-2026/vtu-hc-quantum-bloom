
"""
Q-TRACE | Mission 11B
Operational Loss Sensitivity and Paired Bootstrap Audit

Uses the completed Mission 11 Aer trials.

This is a counterfactual evaluation of loss weights.
Stopping decisions remain unchanged.

Run:
    python notebooks/11b_loss_sensitivity.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT = Path("results/mission11_aer_trials.csv")
OUTPUT = Path("results")

POLICIES = [
    "adaptive_eig",
    "optimized_fixed",
    "round_robin",
]

WRONG_COSTS = [
    0.5, 1, 2, 3, 4, 5, 8, 10
]

ABSTAIN_COST = 0.25
SHOT_COST = 0.001

BOOTSTRAP_REPS = 2000
SEED = 2026


# ============================================================
# 1. LOAD AND VALIDATE EXPERIMENTS
# ============================================================

def load_trials():

    df = pd.read_csv(
        INPUT,
        keep_default_na=False,
    )

    required = {
        "scenario",
        "fault",
        "repeat",
        "policy",
        "accepted",
        "wrong_accepted",
        "used_shots",
    }

    assert required.issubset(
        set(df.columns)
    )

    assert not df.duplicated(
        ["scenario", "repeat", "policy"]
    ).any()

    assert set(df["policy"]) == set(POLICIES)

    assert (
        df["wrong_accepted"]
        <= df["accepted"]
    ).all()

    return df


# ============================================================
# 2. RECOMPUTE OPERATIONAL LOSS
# ============================================================

def calculate_loss(df, wrong_cost):

    return (
        wrong_cost * df["wrong_accepted"]
        + ABSTAIN_COST * (1 - df["accepted"])
        + SHOT_COST * df["used_shots"]
    )


# ============================================================
# 3. PAIRED STRATIFIED BOOTSTRAP
# ============================================================

def paired_bootstrap(
    paired,
    policy_a,
    policy_b,
    rng,
):

    diff = (
        paired[policy_a]
        - paired[policy_b]
    )

    # Preserve the number of trials per scenario.
    blocks = [
        group.to_numpy()
        for _, group in diff.groupby(level="scenario")
    ]

    estimates = np.empty(
        BOOTSTRAP_REPS
    )

    for repeat in range(BOOTSTRAP_REPS):

        samples = [
            rng.choice(
                block,
                size=len(block),
                replace=True,
            )
            for block in blocks
        ]

        estimates[repeat] = np.concatenate(
            samples
        ).mean()

    return (
        float(diff.mean()),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


# ============================================================
# 4. BREAK-EVEN MISDIAGNOSIS COST
# ============================================================

def break_even_cost(df):

    grouped = df.groupby("policy").agg(
        wrong_rate=("wrong_accepted", "mean"),
        coverage=("accepted", "mean"),
        shots=("used_shots", "mean"),
    )

    adaptive = grouped.loc["adaptive_eig"]
    fixed = grouped.loc["optimized_fixed"]

    delta_wrong = (
        adaptive["wrong_rate"]
        - fixed["wrong_rate"]
    )

    delta_abstain = (
        (1 - adaptive["coverage"])
        - (1 - fixed["coverage"])
    )

    delta_shots = (
        adaptive["shots"]
        - fixed["shots"]
    )

    non_error_difference = (
        ABSTAIN_COST * delta_abstain
        + SHOT_COST * delta_shots
    )

    if abs(delta_wrong) < 1e-12:
        return np.nan

    return float(
        -non_error_difference / delta_wrong
    )


# ============================================================
# 5. PER-SCENARIO AUDIT
# ============================================================

def scenario_audit(df):

    rows = []

    for (scenario, policy), group in df.groupby(
        ["scenario", "policy"]
    ):

        accepted = int(
            group["accepted"].sum()
        )

        wrong = int(
            group["wrong_accepted"].sum()
        )

        rows.append({
            "scenario": scenario,
            "policy": policy,
            "n": len(group),
            "coverage": group["accepted"].mean(),
            "wrong_per_trial": (
                group["wrong_accepted"].mean()
            ),
            "risk_if_accepted": (
                wrong / accepted
                if accepted else np.nan
            ),
            "mean_shots": group["used_shots"].mean(),
        })

    return pd.DataFrame(rows)


# ============================================================
# 6. MAIN ANALYSIS
# ============================================================

def main():

    OUTPUT.mkdir(
        exist_ok=True
    )

    df = load_trials()

    rng = np.random.default_rng(SEED)

    print("\n=== Q-TRACE LOSS SENSITIVITY ===")

    print("Total records:", len(df))
    print(
        "Unique paired trials:",
        df[["scenario", "repeat"]]
        .drop_duplicates()
        .shape[0]
    )

    print(
        "\n=== BASELINE OPERATING CHARACTERISTICS ==="
    )

    base = df.groupby("policy").agg(
        coverage=("accepted", "mean"),
        wrong_per_trial=("wrong_accepted", "mean"),
        avg_shots=("used_shots", "mean"),
    )

    print(
        base.round(4).to_string()
    )

    # --------------------------------------------------------
    # Loss decomposition and break-even point
    # --------------------------------------------------------

    threshold = break_even_cost(df)

    print(
        "\n=== EMPIRICAL BREAK-EVEN WRONG COST ==="
    )

    print(
        f"Wrong diagnosis cost: {threshold:.4f}"
    )

    print(
        "This threshold holds only for the "
        "recorded decisions and test scenarios."
    )

    # --------------------------------------------------------
    # Sweep misdiagnosis costs
    # --------------------------------------------------------

    loss_rows = []
    comparison_rows = []

    for wrong_cost in WRONG_COSTS:

        working = df.copy()

        working["evaluated_loss"] = calculate_loss(
            working,
            wrong_cost,
        )

        mean_losses = (
            working.groupby("policy")["evaluated_loss"]
            .mean()
        )

        for policy in POLICIES:

            loss_rows.append({
                "wrong_cost": wrong_cost,
                "policy": policy,
                "mean_loss": float(
                    mean_losses[policy]
                ),
            })

        paired = working.pivot(
            index=["scenario", "repeat"],
            columns="policy",
            values="evaluated_loss",
        )

        assert not paired.isna().any().any()

        for comparison in [
            "optimized_fixed",
            "round_robin",
        ]:

            mean, low, high = paired_bootstrap(
                paired,
                "adaptive_eig",
                comparison,
                rng,
            )

            comparison_rows.append({
                "wrong_cost": wrong_cost,
                "comparison": comparison,
                "adaptive_minus_baseline": mean,
                "ci_low": low,
                "ci_high": high,
            })

    losses = pd.DataFrame(loss_rows)
    comparisons = pd.DataFrame(comparison_rows)

    print(
        "\n=== OPERATIONAL LOSS BY WRONG COST ==="
    )

    print(
        losses.pivot(
            index="wrong_cost",
            columns="policy",
            values="mean_loss",
        ).round(4).to_string()
    )

    print(
        "\n=== PAIRED BOOTSTRAP COMPARISON ==="
    )

    print(
        comparisons.round(4).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Per-scenario audit
    # --------------------------------------------------------

    scenarios = scenario_audit(df)

    print(
        "\n=== NO-FAULT SCENARIO AUDIT ==="
    )

    print(
        scenarios[
            scenarios["scenario"].str.startswith("None_")
        ].round(3).to_string(index=False)
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    losses.to_csv(
        OUTPUT / "mission11b_loss_sweep.csv",
        index=False,
    )

    comparisons.to_csv(
        OUTPUT / "mission11b_paired_bootstrap.csv",
        index=False,
    )

    scenarios.to_csv(
        OUTPUT / "mission11b_scenario_audit.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Plot loss sensitivity
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 5))

    for policy in POLICIES:

        sub = losses[
            losses["policy"] == policy
        ]

        ax.plot(
            sub["wrong_cost"],
            sub["mean_loss"],
            marker="o",
            label=policy,
        )

    if np.isfinite(threshold):
        ax.axvline(
            threshold,
            linestyle="--",
            label=f"Break-even ≈ {threshold:.2f}",
        )

    ax.set(
        xlabel="Cost of a wrong diagnosis",
        ylabel="Mean realized operational loss",
        title="Q-TRACE: Sensitivity to Misdiagnosis Cost",
    )

    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT / "mission11b_loss_sensitivity.png",
        dpi=180,
    )

    plt.close(fig)

    print("\nSaved results/mission11b_*")

    print(
        "\nIMPORTANT: Policy choices and stopping times "
        "are held fixed during the cost sweep."
    )


if __name__ == "__main__":
    main()
