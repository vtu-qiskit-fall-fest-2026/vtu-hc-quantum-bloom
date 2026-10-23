from pathlib import Path
import argparse
import importlib.util
import json

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import binom
from qtrace.policies.stopping import BATCH, M2, Plan


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "mission15", HERE / "15_robustness.py"
)
s = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(s)

m = s.m
runner = s.runner

FIXED_SHOTS = {128: 16, 512: 128}
POLICIES = [
    "raw",
    "old_fixed",
    "old_adaptive",
    "effective_fixed",
    "effective_adaptive",
]
KEY = ["condition", "budget", "target", "case", "rep"]




def measure_trial(backend, case, budget, seed, policy, drift, plans):
    _, _, circuit, observable, fault, rate, _ = case
    application_rate = s.application_rate(fault, rate, drift)

    # Simulator/evaluator owns true noise. The policy only gets counts.
    def measure(qc, obs, shots, measurement_seed):
        return backend.minus(
            qc, obs, fault, application_rate, shots, measurement_seed
        )

    t = 0
    minus_count = 0
    posterior_mean = np.nan
    posterior_variance = np.nan
    predicted_risk = np.nan
    continuation_risk = np.nan

    if policy == "raw":
        gain = 1.0
        reason = "raw"
    else:
        prior_name, allocation = policy.split("_", 1)
        plan = plans[(prior_name, budget)]
        probe = 0 if observable == "ZZ" else 1
        probe_mode, probe_observable = m.PROBES[probe]
        probe_circuit = m.prepare(probe_mode)
        fixed_batches = FIXED_SHOTS[budget] // BATCH

        while True:
            state = plan.state(t, minus_count)
            if allocation == "fixed":
                proceed = t < fixed_batches
            else:
                proceed = bool(state["continue"])

            if not proceed:
                break

            minus_count += int(measure(
                probe_circuit,
                probe_observable,
                BATCH,
                seed + 300003 + t * 101 + probe,
            ))
            t += 1

        gain = float(state["gain"])
        posterior_mean = float(state["mean"])
        posterior_variance = float(state["variance"])
        predicted_risk = float(state["risk"])
        continuation_risk = float(state["continuation"])
        reason = (
            "fixed_allocation"
            if allocation == "fixed"
            else "budget_limit"
            if t == plan.last
            else "stop_risk_le_continuation"
        )

    diagnostic_shots = t * BATCH
    application_shots = budget - diagnostic_shots
    minus = measure(
        circuit, observable, application_shots, seed + 1000003
    )
    raw = 1.0 - 2.0 * minus / application_shots

    assert diagnostic_shots + application_shots == budget
    return {
        "estimate": gain * raw,
        "raw_same_shots": raw,
        "gain": gain,
        "diagnostic_shots": diagnostic_shots,
        "application_shots": application_shots,
        "diagnostic_minus_count": minus_count,
        "posterior_lambda_mean": posterior_mean,
        "posterior_lambda_variance": posterior_variance,
        "predicted_stop_risk": predicted_risk,
        "predicted_continuation_risk": continuation_risk,
        "stop_reason": reason,
    }


def comparisons(df):
    pairs = [
        ("old_adaptive", "old_fixed"),
        ("effective_adaptive", "effective_fixed"),
        ("effective_adaptive", "old_adaptive"),
        ("effective_fixed", "old_fixed"),
        ("effective_adaptive", "raw"),
    ]
    rng = np.random.default_rng(180901)
    rows = []

    for metric in ["squared_error", "conditional_risk"]:
        wide = df.pivot(index=KEY, columns="strategy", values=metric)
        if wide.isna().any().any():
            raise ValueError("Incomplete paired results.")

        for (condition, budget), group in wide.groupby(
            level=["condition", "budget"]
        ):
            for method, baseline in pairs:
                delta = group[method] - group[baseline]
                clusters = delta.groupby(level="target").mean().to_numpy()
                draws = rng.choice(
                    clusters, size=(5000, len(clusters)), replace=True
                ).mean(axis=1)
                low, high = np.quantile(draws, [0.025, 0.975])
                rows.append({
                    "condition": condition,
                    "budget": budget,
                    "metric": metric,
                    "method": method,
                    "baseline": baseline,
                    "delta": delta.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "target_clusters": len(clusters),
                })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["aer", "analytic"], default="aer")
    parser.add_argument("--targets", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=180031)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.smoke:
        args.targets = 2
        args.repeats = 1

    if args.targets < 2 or args.repeats < 1:
        raise ValueError("Need targets >= 2 and repeats >= 1.")

    budgets = [128] if args.smoke else [128, 512]
    output = Path(args.output or (
        "results/prior_ablation_18_smoke"
        if args.smoke else "results/prior_ablation_18"
    ))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")

    plans = {
        (name, budget): Plan(name, budget)
        for name in ["old", "effective"]
        for budget in budgets
    }
    cases = list(runner.cases(
        args.seed, args.targets, [0.04, 0.12, 0.24]
    ))

    output.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "budgets": budgets,
        "batch": BATCH,
        "second_moment": M2,
        "fixed_shots": FIXED_SHOTS,
        "conditions": s.CONDITIONS,
        "policies": POLICIES,
        "old_prior": "0.6 atom at p_eff=0 + 0.4 uniform midpoint grid",
        "effective_prior": "uniform midpoint grid, 80 points in (0, 0.4)",
        "tuning": "None within this run; fixed allocations inherited.",
        "evaluation_status": "New-seed development ablation, not final confirmation.",
        "limitations": [
            "Fixed allocation not retuned for the new prior.",
            "Fresh calibration and application share stationary noise.",
            "Effective attenuation is not physical fault identification.",
            "Predicted risk averages over the frozen target second moment.",
            "Exploratory target-cluster CIs without multiplicity adjustment.",
        ],
    }
    (output / "config.json").write_text(json.dumps(config, indent=2))

    rows = []
    for condition, readout, drift in s.CONDITIONS:
        backend = s.StressBackend(args.backend, readout)

        for budget in budgets:
            for index, case in enumerate(cases):
                target_id, case_id, _, observable, fault, rate, ideal = case
                label = ["None", "X0", "X1", "Z0", "Z1"][int(fault)]
                app_rate = s.application_rate(fault, rate, drift)
                affected = (
                    (observable == "ZZ" and label.startswith("X"))
                    or (observable == "XX" and label.startswith("Z"))
                )

                # Truth appears only in simulation and evaluation.
                true_lambda = (
                    (1.0 - 2.0 * app_rate * int(affected))
                    * (1.0 - 2.0 * readout) ** 2
                )

                for rep in range(args.repeats):
                    seed = int(np.random.SeedSequence(
                        [args.seed, 7001, index, rep]
                    ).generate_state(1)[0])

                    for policy in POLICIES:
                        result = measure_trial(
                            backend, case, budget, seed,
                            policy, drift, plans,
                        )
                        gain = result["gain"]
                        gain_mismatch = (
                            (gain * true_lambda - 1.0) * ideal
                        ) ** 2
                        application_variance = (
                            gain ** 2
                            * max(0.0, 1.0 - (true_lambda * ideal) ** 2)
                            / result["application_shots"]
                        )
                        rows.append({
                            "condition": condition,
                            "budget": budget,
                            "target": target_id,
                            "case": case_id,
                            "rep": rep,
                            "strategy": policy,
                            "observable": observable,
                            "fault": label,
                            "rate": rate,
                            "application_rate": app_rate,
                            "ideal": ideal,
                            **result,
                            "squared_error": (result["estimate"] - ideal) ** 2,
                            "gain_mismatch": gain_mismatch,
                            "application_variance": application_variance,
                            "conditional_risk": (
                                gain_mismatch + application_variance
                            ),
                        })

                backend.cache.clear()
                if (index + 1) % 26 == 0:
                    print(
                        f"{condition} B={budget}: "
                        f"{index + 1}/{len(cases)} cases",
                        flush=True,
                    )

            pd.DataFrame(rows).to_csv(output / "trials.csv", index=False)
            print(f"Saved {condition}, budget {budget}", flush=True)

    df = pd.DataFrame(rows)
    assert not df.duplicated(KEY + ["strategy"]).any()
    assert (
        df["diagnostic_shots"] + df["application_shots"] == df["budget"]
    ).all()

    metrics = [
        "squared_error", "conditional_risk",
        "gain_mismatch", "application_variance",
        "diagnostic_shots", "application_shots",
        "predicted_stop_risk",
    ]
    summary = df.groupby(
        ["condition", "budget", "strategy"]
    )[metrics].mean().reset_index()
    summary.to_csv(output / "summary.csv", index=False)

    by_scenario = df.groupby([
        "condition", "budget", "strategy",
        "observable", "fault", "rate",
    ])[metrics].mean().reset_index()
    by_scenario.to_csv(output / "by_scenario.csv", index=False)

    comp = comparisons(df)
    comp.to_csv(output / "comparisons.csv", index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nMSE comparisons; negative favors method:")
    print(comp[comp.metric == "squared_error"].to_string(index=False))
    print(f"\nResults: {output}")


if __name__ == "__main__":
    main()