
"""
Q-TRACE | Mission 06
Scientific Audit of Selective Quantum Fault Diagnosis

Uses saved Mission 05 Monte Carlo trials.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT = Path("results/mission05_trials.csv")
OUTPUT = Path("results")
OUTPUT.mkdir(exist_ok=True)

df = pd.read_csv(
    INPUT,
    keep_default_na=False,
    na_values=[""],
)

assert df["fault"].notna().all()
assert set(df["fault"]) == {"None", "X0", "X1", "Z0", "Z1"}

print("\nFault counts:")
print(df["fault"].value_counts())

required = [
    "true_rate", "fault", "repeat", "policy",
    "accuracy192", "reached_confidence",
    "accuracy_at_first_confidence",
]

assert set(required).issubset(df.columns)

# Did the algorithm actually issue a confident diagnosis?
df["accepted"] = df["reached_confidence"].astype(int)

# A correct confident diagnosis.
df["correct_accepted"] = np.where(
    df["accepted"] == 1,
    df["accuracy_at_first_confidence"].fillna(0),
    0,
).astype(int)

# A confidently incorrect diagnosis.
df["wrong_accepted"] = (
    df["accepted"] - df["correct_accepted"]
)

assert (df["wrong_accepted"] >= 0).all()


# ---------------------------------------------------
# Wilson 95% confidence interval for a proportion
# ---------------------------------------------------

def wilson_interval(successes, total):
    if total == 0:
        return np.nan, np.nan

    z = 1.96
    p = successes / total

    denominator = 1 + z**2 / total

    center = (
        p + z**2 / (2 * total)
    ) / denominator

    half_width = (
        z * np.sqrt(
            p * (1 - p) / total
            + z**2 / (4 * total**2)
        )
    ) / denominator

    return center - half_width, center + half_width


# ---------------------------------------------------
# Aggregate accuracy and selective risk
# ---------------------------------------------------

def summarize(group):
    n = len(group)

    accepted = int(group["accepted"].sum())
    correct = int(group["correct_accepted"].sum())
    wrong = accepted - correct

    cov_low, cov_high = wilson_interval(
        accepted, n
    )

    acc_low, acc_high = wilson_interval(
        correct, accepted
    )

    return {
        "n": n,
        "coverage": accepted / n,
        "selective_accuracy": (
            correct / accepted if accepted else np.nan
        ),
        "selective_risk": (
            wrong / accepted if accepted else np.nan
        ),
        "false_confidence_per_trial": wrong / n,
        "accuracy_at_192": group["accuracy192"].mean(),
        "coverage_ci_low": cov_low,
        "coverage_ci_high": cov_high,
        "accuracy_ci_low": acc_low,
        "accuracy_ci_high": acc_high,
    }


rows = []

for (rate, policy), group in df.groupby(
    ["true_rate", "policy"]
):
    rows.append({
        "true_rate": rate,
        "policy": policy,
        **summarize(group)
    })

summary = pd.DataFrame(rows)

print("\n=== SELECTIVE DIAGNOSIS AUDIT ===")
print(
    summary[
        [
            "true_rate",
            "policy",
            "coverage",
            "selective_accuracy",
            "selective_risk",
            "false_confidence_per_trial",
        ]
    ].round(4).to_string(index=False)
)


# ---------------------------------------------------
# Per-fault audit: identify hidden selection bias
# ---------------------------------------------------

fault_rows = []

for (rate, fault, policy), group in df.groupby(
    ["true_rate", "fault", "policy"]
):
    fault_rows.append({
        "true_rate": rate,
        "fault": fault,
        "policy": policy,
        **summarize(group)
    })

by_fault = pd.DataFrame(fault_rows)

for rate in [0.02, 0.10]:
    print(f"\n=== PER-FAULT COVERAGE: p={rate} ===")

    subset = by_fault[
        np.isclose(by_fault["true_rate"], rate)
        & by_fault["policy"].isin([
            "point_active",
            "robust_active",
        ])
    ]

    print(
        subset[
            [
                "fault",
                "policy",
                "coverage",
                "selective_accuracy",
                "accuracy_at_192",
            ]
        ].round(3).to_string(index=False)
    )


# ---------------------------------------------------
# Risk-coverage comparison
# ---------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

for policy in ["point_active", "robust_active"]:

    subset = summary[
        (summary["policy"] == policy)
        & (summary["true_rate"] > 0)
        & (summary["coverage"] > 0)
    ].sort_values("true_rate")

    ax.plot(
        subset["coverage"],
        subset["selective_risk"],
        marker="o",
        label=policy,
    )

    for _, row in subset.iterrows():
        ax.annotate(
            f"p={row['true_rate']:.2f}",
            (
                row["coverage"],
                row["selective_risk"],
            ),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )

ax.set(
    xlabel="Diagnostic coverage",
    ylabel="Error rate among confident diagnoses",
    title="Q-TRACE: Selective Risk vs Coverage",
    xlim=(0, 1.05),
    ylim=(-0.02, 0.5),
)

ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()

fig.savefig(
    OUTPUT / "mission06_risk_coverage.png",
    dpi=180,
)

plt.close(fig)


# ---------------------------------------------------
# Save audit results
# ---------------------------------------------------

summary.to_csv(
    OUTPUT / "mission06_summary.csv",
    index=False,
)

by_fault.to_csv(
    OUTPUT / "mission06_per_fault.csv",
    index=False,
)

print("\nSaved Mission 06 audit files.")
print("NOTE: Different points correspond to different")
print("noise regimes, not different stopping thresholds.")
