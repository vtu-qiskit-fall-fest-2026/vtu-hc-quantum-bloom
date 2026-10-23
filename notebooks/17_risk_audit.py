from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


KEY = ["condition", "budget", "target", "case", "rep"]
POLICIES = ["fresh_only", "fresh_fixed"]
METRICS = [
    "squared_error",
    "conditional_risk",
    "gain_mismatch",
    "application_variance",
    "diagnostic_shots",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bootstrap_ci(target_means, rng, draws=5000):
    """Resample targets, keeping all trials within each target together."""
    x = np.asarray(target_means, dtype=float)
    require(len(x) >= 2, "Need at least two target clusters.")
    samples = rng.choice(x, size=(draws, len(x)), replace=True)
    return np.quantile(samples.mean(axis=1), [0.025, 0.975])


def audit(source):
    config = json.loads((source / "config.json").read_text())
    source_config = config["source_config"]

    df = pd.read_csv(
        source / "trials.csv",
        keep_default_na=False,
        na_values=[""],
    )
    df = df[df["strategy"].isin(POLICIES)].copy()

    required = set(KEY + METRICS[:1] + [
        "strategy", "observable", "fault", "application_rate",
        "ideal", "estimate", "gain", "raw_same_shots",
        "diagnostic_shots", "application_shots", "rate",
    ])
    require(required <= set(df.columns), "Missing required trial columns.")
    require(not df.empty, "No fresh policies found.")
    require(
        not df.duplicated(KEY + ["strategy"]).any(),
        "Duplicate trial keys.",
    )
    require(
        (df["diagnostic_shots"] + df["application_shots"]
         == df["budget"]).all(),
        "Shot budget mismatch.",
    )
    require((df["application_shots"] > 0).all(), "No application shots.")
    require(
        set(df["observable"]) <= {"XX", "ZZ"},
        "This audit implements only XX and ZZ.",
    )
    require(
        set(df["fault"]) <= {"None", "X0", "X1", "Z0", "Z1"},
        "Unexpected fault labels.",
    )
    require(
        np.allclose(
            df["estimate"],
            df["gain"] * df["raw_same_shots"],
            rtol=0,
            atol=1e-10,
        ),
        "Estimator is clipped/nonlinear; this risk formula does not apply.",
    )
    require(
        np.allclose(
            df["squared_error"],
            (df["estimate"] - df["ideal"]) ** 2,
            rtol=0,
            atol=1e-10,
        ),
        "Stored squared_error is inconsistent.",
    )

    readout = {
        name: probability
        for name, probability, _ in source_config["conditions"]
    }
    eps = df["condition"].map(readout)
    require(eps.notna().all(), "Unknown noise condition.")

    affected = (
        ((df["observable"] == "ZZ") & df["fault"].str.startswith("X"))
        | ((df["observable"] == "XX") & df["fault"].str.startswith("Z"))
    )

    # EVALUATION ONLY: true application noise and ideal expectation.
    # These quantities must never enter the deployed decision policy.
    attenuation = (
        (1.0 - 2.0 * df["application_rate"] * affected.astype(float))
        * (1.0 - 2.0 * eps) ** 2
    )
    noisy_mean = attenuation * df["ideal"]

    require(
        (np.abs(noisy_mean) <= 1.0 + 1e-10).all(),
        "Invalid measurement expectation.",
    )

    # Conditional on the observed calibration data and chosen gain:
    # E[(g * sample_mean - ideal)^2 | calibration]
    # = squared conditional bias + conditional application variance.
    df["gain_mismatch"] = (
        (df["gain"] * attenuation - 1.0) * df["ideal"]
    ) ** 2
    df["application_variance"] = (
        df["gain"] ** 2
        * np.maximum(0.0, 1.0 - noisy_mean ** 2)
        / df["application_shots"]
    )
    df["conditional_risk"] = (
        df["gain_mismatch"] + df["application_variance"]
    )

    summary = (
        df.groupby(["condition", "budget", "strategy"])[METRICS]
        .mean()
        .reset_index()
    )

    # Strict pairing: adaptive and fixed must share every evaluation key.
    paired = df.pivot(
        index=KEY, columns="strategy", values=METRICS
    )
    require(not paired.isna().any().any(), "Missing paired policy results.")

    rng = np.random.default_rng(170031)
    comparisons = []

    for (condition, budget), group in paired.groupby(
        level=["condition", "budget"], sort=True
    ):
        for metric in METRICS:
            delta = (
                group[(metric, "fresh_only")]
                - group[(metric, "fresh_fixed")]
            )
            target_means = delta.groupby(level="target").mean()
            low, high = bootstrap_ci(target_means, rng)
            comparisons.append({
                "condition": condition,
                "budget": budget,
                "metric": metric,
                "delta_adaptive_minus_fixed": delta.mean(),
                "ci_low": low,
                "ci_high": high,
                "target_clusters": len(target_means),
            })

    comparisons = pd.DataFrame(comparisons)

    scenario_keys = [
        "condition", "budget", "observable", "fault", "rate"
    ]
    scenario = df.groupby(
        scenario_keys + ["strategy"]
    )[METRICS].mean().unstack("strategy")

    deltas = pd.DataFrame(index=scenario.index)
    for metric in METRICS:
        deltas[f"delta_{metric}"] = (
            scenario[(metric, "fresh_only")]
            - scenario[(metric, "fresh_fixed")]
        )

    # Weight each scenario by its actual representation in the dataset.
    counts = (
        df[df["strategy"] == "fresh_only"]
        .groupby(scenario_keys)
        .size()
    )
    weights = counts / counts.groupby(
        level=["condition", "budget"]
    ).transform("sum")
    deltas["scenario_weight"] = weights
    deltas["risk_gap_contribution"] = (
        weights * deltas["delta_conditional_risk"]
    )
    deltas = deltas.reset_index().sort_values(
        ["condition", "budget", "risk_gap_contribution"],
        ascending=[True, True, False],
    )

    # Descriptive only: stopping groups are selected by the policy.
    # They are not randomized treatment groups.
    stop_groups = (
        df[df["strategy"] == "fresh_only"]
        .groupby(["condition", "budget", "diagnostic_shots"])
        .agg(
            trials=("conditional_risk", "size"),
            conditional_risk=("conditional_risk", "mean"),
            gain_mismatch=("gain_mismatch", "mean"),
            application_variance=("application_variance", "mean"),
            mean_gain=("gain", "mean"),
        )
        .reset_index()
    )

    return config, summary, comparisons, deltas, stop_groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/calibration_16b")
    parser.add_argument("--output", default="results/risk_audit_17")
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    require(
        not output.exists() or not any(output.iterdir()),
        f"Output directory is not empty: {output}",
    )

    config, summary, comparisons, scenarios, stops = audit(source)

    output.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("summary", summary),
        ("comparisons", comparisons),
        ("scenario_contributions", scenarios),
        ("stopping_groups", stops),
    ]:
        frame.to_csv(output / f"{name}.csv", index=False)

    metadata = {
        "source": str(source.resolve()),
        "source_config": config,
        "policy_changes": "None; offline evaluation only.",
        "risk": "Conditional on realized calibration; integrates application shots.",
        "assumptions": [
            "Independent application shots after calibration.",
            "Linear estimator: estimate = gain * raw_same_shots.",
            "Single Pauli channel and symmetric independent readout.",
            "XX/ZZ observables; stationary noise during fresh phase.",
        ],
        "limitations": [
            "True noise and ideal values used only by evaluator.",
            "Posterior/predicted stopping risk unavailable in input.",
            "Gain mismatch includes variability across calibration outcomes.",
            "Stopping-group comparisons are descriptive, not causal.",
            "Retrospective target-cluster CIs; no multiplicity correction.",
        ],
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2))

    print("\nRisk decomposition:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nConditional-risk comparison; negative favors adaptive:")
    selected = comparisons[comparisons["metric"] == "conditional_risk"]
    print(selected.to_string(index=False))

    print("\nLargest contributors to readout_3pct / 512 risk gap:")
    selected = scenarios[
        (scenarios["condition"] == "readout_3pct")
        & (scenarios["budget"] == 512)
    ]
    print(selected.head(10).to_string(index=False))

    print(f"\nSaved audit to {output}")


if __name__ == "__main__":
    main()