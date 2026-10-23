
"""
Q-TRACE | Mission 12
Cost-Aligned Stopping with Qiskit Aer

Reuse Mission 11's actual quantum measurement pipeline.

Compare:
- Fixed 95% confidence stopping
- Loss-aligned Bayesian acceptance

Run:
python notebooks/12_loss_aligned_stopping.py --repeats 10
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. IMPORT EXISTING MISSION 11 IMPLEMENTATION
# ============================================================

M11_PATH = Path(__file__).with_name(
    "11_aer_closed_loop.py"
)

if not M11_PATH.exists():
    raise FileNotFoundError(
        f"Missing dependency: {M11_PATH}"
    )

spec = importlib.util.spec_from_file_location(
    "qtrace_m11",
    M11_PATH,
)

m11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m11)


# ============================================================
# 2. EXPERIMENT CONFIGURATION
# ============================================================

ABSTAIN_COST = 0.25
SHOT_COST = 0.001

WRONG_COSTS = (1.0, 5.0)

STOP_RULES = (
    "fixed95",
    "loss_aligned",
)

POLICIES = (
    "adaptive_eig",
    "optimized_fixed",
    "round_robin",
)


# ============================================================
# 3. LOSS-ALIGNED ACCEPTANCE THRESHOLD
# ============================================================

def stopping_threshold(wrong_cost, rule):

    if rule == "fixed95":
        return 0.95

    if rule == "loss_aligned":

        threshold = (
            1.0
            - ABSTAIN_COST / wrong_cost
        )

        return float(
            np.clip(threshold, 0.0, 1.0)
        )

    raise ValueError(rule)


# ============================================================
# 4. CLOSED-LOOP DIAGNOSIS
# ============================================================

def diagnose(
    policy,
    rule,
    wrong_cost,
    fault,
    rate,
    seed_table,
    batch,
    budget,
    prior,
    classes,
    likes,
    fixed,
    aer,
    cache,
):

    posterior = prior.copy()

    threshold = stopping_threshold(
        wrong_cost,
        rule,
    )

    accepted = False
    used = 0

    label = "ABSTAIN"
    confidence = float("nan")

    for step in range(budget // batch):

        # ----------------------------------------
        # Choose the next diagnostic probe
        # ----------------------------------------

        if policy == "adaptive_eig":

            scores = [
                m11.eig_class(
                    posterior,
                    classes,
                    likes[i],
                )
                for i in range(len(m11.PROBES))
            ]

            probe = int(np.argmax(scores))

        elif policy == "optimized_fixed":

            probe = fixed[step]

        elif policy == "round_robin":

            probe = step % len(m11.PROBES)

        else:

            raise ValueError(policy)

        # ----------------------------------------
        # Construct or retrieve quantum circuit
        # ----------------------------------------

        key = (probe, fault, rate)

        if key not in cache:

            cache[key] = m11.make_aer_circuit(
                probe,
                fault,
                rate,
            )

        # ----------------------------------------
        # Execute actual Qiskit Aer measurements
        # ----------------------------------------

        counts = aer.run(
            cache[key],
            shots=batch,
            seed_simulator=int(
                seed_table[step, probe]
            ),
        ).result().get_counts()

        # ----------------------------------------
        # Extract negative Pauli outcomes
        # ----------------------------------------

        minus = m11.negative_outcomes(
            counts,
            m11.PROBES[probe][2],
        )

        # ----------------------------------------
        # Bayesian posterior update
        # ----------------------------------------

        weights = (
            posterior
            * likes[probe, minus]
        )

        normalizer = float(weights.sum())

        if normalizer <= 0:

            raise RuntimeError(
                "Observation has zero probability "
                "under the inference model"
            )

        posterior = weights / normalizer

        class_posterior = posterior @ classes

        predicted_index = int(
            np.argmax(class_posterior)
        )

        confidence = float(
            class_posterior[predicted_index]
        )

        used += batch

        # ----------------------------------------
        # Stop when acceptance rule is satisfied
        # ----------------------------------------

        if confidence >= threshold - 1e-12:

            accepted = True

            label = m11.FAULTS[
                predicted_index
            ]

            break

    # ========================================================
    # 5. REALIZED OPERATIONAL LOSS
    # ========================================================

    correct = int(
        accepted and label == fault
    )

    wrong = int(
        accepted and label != fault
    )

    realized_loss = (
        wrong_cost * wrong
        + ABSTAIN_COST * (not accepted)
        + SHOT_COST * used
    )

    return {
        "accepted": int(accepted),
        "correct_accepted": correct,
        "wrong_accepted": wrong,
        "used_shots": used,
        "prediction": label,
        "confidence": confidence,
        "realized_loss": float(realized_loss),
    }


# ============================================================
# 6. WILSON CONFIDENCE INTERVAL
# ============================================================

def wilson(k, n, z=1.96):

    if n == 0:
        return (
            float("nan"),
            float("nan"),
        )

    p = k / n

    denominator = 1 + z * z / n

    center = (
        p + z * z / (2 * n)
    ) / denominator

    radius = (
        z * np.sqrt(
            p * (1 - p) / n
            + z * z / (4 * n * n)
        )
    ) / denominator

    return (
        center - radius,
        center + radius,
    )


# ============================================================
# 7. STRATIFIED PAIRED BOOTSTRAP
# ============================================================

def paired_bootstrap(
    a,
    b,
    scenario_labels,
    rng,
    repetitions=1500,
):

    differences = (
        np.asarray(a)
        - np.asarray(b)
    )

    scenario_labels = np.asarray(
        scenario_labels
    )

    blocks = [
        differences[
            scenario_labels == scenario
        ]

        for scenario in np.unique(
            scenario_labels
        )
    ]

    means = np.empty(repetitions)

    for repeat in range(repetitions):

        samples = [
            rng.choice(
                block,
                size=len(block),
                replace=True,
            )

            for block in blocks
        ]

        means[repeat] = (
            np.concatenate(samples).mean()
        )

    return (
        float(differences.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


# ============================================================
# 8. MAIN EXPERIMENT
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    if (
        args.repeats < 1
        or args.batch < 1
        or args.budget < args.batch
        or args.budget % args.batch
    ):

        parser.error(
            "Choose positive repeats/batch and "
            "a budget divisible by batch"
        )

    # ----------------------------------------
    # Qiskit-derived signatures and model
    # ----------------------------------------

    sens = m11.sensitivity_matrix()

    prior, classes, likelihood = (
        m11.make_model(
            sens,
            args.batch,
        )
    )

    fixed = m11.optimized_fixed_schedule(
        sens,
        args.budget // args.batch,
    )

    aer = m11.AerSimulator(
        method="stabilizer"
    )

    circuit_cache = {}

    # Same test scenarios as Mission 11.

    scenarios = [("None", 0.0)] + [
        (fault, rate)

        for rate in (0.04, 0.10)

        for fault in m11.FAULTS[1:]
    ]

    rows = []

    # ========================================================
    # 9. EXECUTE PAIRED AER TRIALS
    # ========================================================

    for sid, (fault, rate) in enumerate(scenarios):

        for repeat in range(args.repeats):

            rng = np.random.default_rng(
                np.random.SeedSequence([
                    2026,
                    sid,
                    repeat,
                ])
            )

            seeds = rng.integers(
                1,
                2**31 - 1,
                size=(
                    args.budget // args.batch,
                    len(m11.PROBES),
                ),
            )

            for cost in WRONG_COSTS:

                for rule in STOP_RULES:

                    for policy in POLICIES:

                        result = diagnose(
                            policy,
                            rule,
                            cost,
                            fault,
                            rate,
                            seeds,
                            args.batch,
                            args.budget,
                            prior,
                            classes,
                            likelihood,
                            fixed,
                            aer,
                            circuit_cache,
                        )

                        rows.append({
                            "scenario":
                                f"{fault}_p{rate:.2f}",
                            "fault": fault,
                            "true_rate": rate,
                            "repeat": repeat,
                            "wrong_cost": cost,
                            "rule": rule,
                            "policy": policy,
                            "threshold":
                                stopping_threshold(
                                    cost, rule
                                ),
                            **result,
                        })

    df = pd.DataFrame(rows)

    # ========================================================
    # 10. CORRECTNESS TEST
    # ========================================================

    # With wrong_cost=5:
    # 1 - 0.25/5 = 0.95
    #
    # Therefore fixed95 and loss_aligned
    # must produce identical outcomes.

    at5 = df[
        df.wrong_cost == 5.0
    ]

    matched = at5.pivot(
        index=[
            "scenario",
            "repeat",
            "policy",
        ],
        columns="rule",
        values=[
            "used_shots",
            "accepted",
            "wrong_accepted",
        ],
    )

    for field in [
        "used_shots",
        "accepted",
        "wrong_accepted",
    ]:

        assert (
            matched[(field, "fixed95")]
            == matched[(field, "loss_aligned")]
        ).all(), field

    print(
        "\nPASS: Cost=5 stopping rules "
        "produce identical outcomes."
    )

    # ========================================================
    # 11. BENCHMARK SUMMARY
    # ========================================================

    summary_rows = []

    for (
        cost,
        rule,
        policy,
    ), group in df.groupby(
        [
            "wrong_cost",
            "rule",
            "policy",
        ],
        sort=True,
    ):

        n = len(group)

        accepted = int(
            group.accepted.sum()
        )

        good = int(
            group.correct_accepted.sum()
        )

        coverage_ci = wilson(
            accepted,
            n,
        )

        summary_rows.append({
            "wrong_cost": cost,
            "rule": rule,
            "policy": policy,
            "n": n,

            "coverage": accepted / n,

            "coverage_ci_low":
                coverage_ci[0],

            "coverage_ci_high":
                coverage_ci[1],

            "accuracy_if_accepted": (
                good / accepted
                if accepted else np.nan
            ),

            "avg_shots": float(
                group.used_shots.mean()
            ),

            "wrong_per_trial": float(
                group.wrong_accepted.mean()
            ),

            "mean_loss": float(
                group.realized_loss.mean()
            ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    print(
        "\n=== LOSS-ALIGNED AER STOPPING ==="
    )

    print(
        "Abstention cost:",
        ABSTAIN_COST,
        "Shot cost:",
        SHOT_COST,
    )

    print(
        "\nFixed schedule:",
        [m11.PROBES[k][0] for k in fixed]
    )

    print(
        summary[
            [
                "wrong_cost",
                "rule",
                "policy",
                "n",
                "coverage",
                "accuracy_if_accepted",
                "wrong_per_trial",
                "avg_shots",
                "mean_loss",
            ]
        ].round(3).to_string(
            index=False
        )
    )

    # ========================================================
    # 12. PAIRED POLICY COMPARISONS
    # ========================================================

    comparison_rows = []

    bootstrap_rng = np.random.default_rng(
        9702
    )

    for cost in WRONG_COSTS:

        # ------------------------------------
        # Compare acceptance rules
        # ------------------------------------

        for policy in POLICIES:

            subset = df[
                (df.wrong_cost == cost)
                & (df.policy == policy)
            ]

            paired = subset.pivot(
                index=[
                    "scenario",
                    "repeat",
                ],
                columns="rule",
                values="realized_loss",
            )

            avg, low, high = paired_bootstrap(
                paired["loss_aligned"].values,
                paired["fixed95"].values,
                paired.index.get_level_values(
                    "scenario"
                ),
                bootstrap_rng,
            )

            comparison_rows.append({
                "wrong_cost": cost,
                "policy": policy,
                "comparison":
                    "loss_aligned - fixed95",
                "mean_diff": avg,
                "ci_low": low,
                "ci_high": high,
            })

        # ------------------------------------
        # Adaptive vs fixed under the
        # same loss-aligned acceptance rule
        # ------------------------------------

        subset = df[
            (df.wrong_cost == cost)
            & (df.rule == "loss_aligned")
        ]

        paired = subset.pivot(
            index=[
                "scenario",
                "repeat",
            ],
            columns="policy",
            values="realized_loss",
        )

        avg, low, high = paired_bootstrap(
            paired["adaptive_eig"].values,
            paired["optimized_fixed"].values,
            paired.index.get_level_values(
                "scenario"
            ),
            bootstrap_rng,
        )

        comparison_rows.append({
            "wrong_cost": cost,
            "policy": "adaptive_vs_fixed",
            "comparison":
                "adaptive_eig - optimized_fixed",
            "mean_diff": avg,
            "ci_low": low,
            "ci_high": high,
        })

    comparisons = pd.DataFrame(
        comparison_rows
    )

    print(
        "\n=== PAIRED STRATIFIED BOOTSTRAP ==="
    )

    print(
        comparisons.round(4).to_string(
            index=False
        )
    )

    # ========================================================
    # 13. PER-FAULT AUDIT
    # ========================================================

    by_fault = (
        df.groupby(
            [
                "wrong_cost",
                "rule",
                "policy",
                "fault",
            ],
            sort=True,
        )[
            [
                "accepted",
                "wrong_accepted",
                "used_shots",
                "realized_loss",
            ]
        ]
        .mean()
        .reset_index()
    )

    print(
        "\n=== ADAPTIVE PER-FAULT AUDIT ==="
    )

    print(
        by_fault[
            by_fault.policy == "adaptive_eig"
        ].round(3).to_string(
            index=False
        )
    )

    print(
        "\n=== NO-FAULT COVERAGE ==="
    )

    print(
        df[
            df.fault == "None"
        ].groupby(
            [
                "wrong_cost",
                "rule",
                "policy",
            ]
        )[
            [
                "accepted",
                "wrong_accepted",
                "used_shots",
            ]
        ].mean().round(3).to_string()
    )

    # ========================================================
    # 14. SAVE RESULTS
    # ========================================================

    output = Path("results")

    output.mkdir(
        exist_ok=True
    )

    df.to_csv(
        output / "mission12_trials.csv",
        index=False,
    )

    summary.to_csv(
        output / "mission12_summary.csv",
        index=False,
    )

    comparisons.to_csv(
        output / "mission12_paired.csv",
        index=False,
    )

    by_fault.to_csv(
        output / "mission12_by_fault.csv",
        index=False,
    )

    print(
        "\nSaved results/mission12_*.csv"
    )

    print(
        "\nIMPORTANT: Loss-aligned acceptance is "
        "not globally optimal sequential stopping."
    )

    print(
        "The policy still measures until "
        "acceptance or the budget cap."
    )


# ============================================================
# 15. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
