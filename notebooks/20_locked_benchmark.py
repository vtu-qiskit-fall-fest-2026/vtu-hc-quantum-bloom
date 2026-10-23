from pathlib import Path
import argparse
import importlib.util
import json

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "mission18", HERE / "18_prior_ablation.py"
)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

VALIDATION_SEED = 200031
TEST_SEED = 200071
PRIORS = ["old", "effective"]
KEY = ["condition", "budget", "target", "case", "rep"]


def candidates(budget, family):
    if family == "adaptive":
        return [
            {"prior": prior, "family": family, "shots": None}
            for prior in PRIORS
        ]
    return [
        {"prior": prior, "family": family, "shots": shots}
        for prior in PRIORS
        for shots in range(0, budget, v.BATCH)
    ]


def candidate_name(config):
    return (
        f"{config['prior']}_{config['family']}"
        f"_{config['shots']}"
    )


def run(backend, case, budget, seed, config, drift, plans):
    if config is None:
        policy = "raw"
    else:
        policy = f"{config['prior']}_{config['family']}"
        if config["family"] == "fixed":
            # Sequential calls only; measure_trial reads this allocation.
            v.FIXED_SHOTS[budget] = int(config["shots"])

    return v.measure_trial(
        backend, case, budget, seed, policy, drift, plans
    )


def trial_seed(split_seed, index, rep):
    return int(np.random.SeedSequence(
        [split_seed, 7001, index, rep]
    ).generate_state(1)[0])


def select_on_validation(plans, budgets, targets, repeats, output):
    cases = list(v.runner.cases(
        VALIDATION_SEED, targets, [0.04, 0.12, 0.24]
    ))
    locked = {}
    validation_rows = []

    for budget in budgets:
        configs = (
            candidates(budget, "fixed")
            + candidates(budget, "adaptive")
        )
        totals = {
            candidate_name(c): {
                "error": 0.0, "count": 0, "diagnostic_shots": 0
            }
            for c in configs
        }

        for condition, readout, drift in v.s.CONDITIONS:
            backend = v.s.StressBackend("analytic", readout)

            for index, case in enumerate(cases):
                ideal = case[-1]
                for rep in range(repeats):
                    seed = trial_seed(VALIDATION_SEED, index, rep)
                    for config in configs:
                        result = run(
                            backend, case, budget, seed,
                            config, drift, plans,
                        )
                        item = totals[candidate_name(config)]
                        item["error"] += (result["estimate"] - ideal) ** 2
                        item["count"] += 1
                        item["diagnostic_shots"] += result["diagnostic_shots"]

                backend.cache.clear()

            print(
                f"Validation: budget={budget}, condition={condition}",
                flush=True,
            )

        for config in configs:
            item = totals[candidate_name(config)]
            validation_rows.append({
                "budget": budget,
                **config,
                "mse": item["error"] / item["count"],
                "trials": item["count"],
                "mean_diagnostic_shots": (
                    item["diagnostic_shots"] / item["count"]
                ),
            })

        locked[budget] = {}
        for family in ["fixed", "adaptive"]:
            options = [c for c in configs if c["family"] == family]
            # Tie-breaking follows the declared candidate order.
            best = min(
                options,
                key=lambda c: (
                    totals[candidate_name(c)]["error"]
                    / totals[candidate_name(c)]["count"]
                ),
            )
            locked[budget][family] = dict(best)

        print(f"Locked budget {budget}: {locked[budget]}", flush=True)

    pd.DataFrame(validation_rows).to_csv(
        output / "validation.csv", index=False
    )
    (output / "locked.json").write_text(json.dumps(locked, indent=2))
    return locked


def paired_comparisons(df):
    rng = np.random.default_rng(200901)
    results = []

    # Overall comparison is primary; individual conditions exploratory.
    scopes = [("overall", df)]
    scopes += list(df.groupby("condition"))

    for scope, subset in scopes:
        for budget, group in subset.groupby("budget"):
            wide = group.pivot(
                index=KEY, columns="strategy", values="squared_error"
            )
            if wide.isna().any().any():
                raise ValueError("Incomplete paired results.")

            for baseline in ["fixed", "raw"]:
                delta = wide["adaptive"] - wide[baseline]

                # Keep conditions, noise cases and repeats of a target together.
                clusters = delta.groupby(level="target").mean().to_numpy()
                draws = rng.choice(
                    clusters, size=(5000, len(clusters)), replace=True
                ).mean(axis=1)
                low, high = np.quantile(draws, [0.025, 0.975])
                results.append({
                    "scope": scope,
                    "budget": budget,
                    "method": "adaptive",
                    "baseline": baseline,
                    "delta_mse": delta.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "target_clusters": len(clusters),
                })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["aer", "analytic"], default="aer")
    parser.add_argument("--validation-targets", type=int, default=16)
    parser.add_argument("--validation-repeats", type=int, default=5)
    parser.add_argument("--targets", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.smoke:
        args.validation_targets = args.targets = 2
        args.validation_repeats = args.repeats = 1

    if min(args.validation_targets, args.targets) < 2:
        raise ValueError("Need at least two targets per split.")
    if min(args.validation_repeats, args.repeats) < 1:
        raise ValueError("Repeats must be positive.")

    budgets = [128] if args.smoke else [128, 512]
    output = Path(args.output or (
        "results/locked_benchmark_20_smoke"
        if args.smoke else "results/locked_benchmark_20"
    ))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    protocol = {
        **vars(args),
        "budgets": budgets,
        "validation_seed": VALIDATION_SEED,
        "test_seed": TEST_SEED,
        "validation_backend": "analytic",
        "test_backend": args.backend,
        "conditions": v.s.CONDITIONS,
        "rates": [0.04, 0.12, 0.24],
        "second_moment": v.M2,
        "selection": "Minimum aggregate validation MSE per family and budget",
        "weighting": "Equal conditions; equal cases and repeats within condition",
        "fixed_candidates": "Both priors; 0 to budget-16 shots in steps of 16",
        "adaptive_candidates": "Both priors; unchanged DP",
        "primary_comparison": "Aggregate test MSE: adaptive minus fixed",
        "limits": [
            "New target and measurement seeds; same noise scenario family.",
            "Validation simulation cost is offline development cost.",
            "Each test trial counts calibration plus application shots.",
            "Target-cluster CIs condition on selected configurations.",
            "Per-condition comparisons are exploratory; no multiplicity correction.",
            "No claim of hardware performance or QEC.",
        ],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    plans = {
        (prior, budget): v.Plan(prior, budget)
        for prior in PRIORS
        for budget in budgets
    }

    locked = select_on_validation(
        plans, budgets,
        args.validation_targets, args.validation_repeats,
        output,
    )

    # Test cases are generated only after all configurations are locked.
    cases = list(v.runner.cases(
        TEST_SEED, args.targets, [0.04, 0.12, 0.24]
    ))
    rows = []

    for condition, readout, drift in v.s.CONDITIONS:
        backend = v.s.StressBackend(args.backend, readout)

        for budget in budgets:
            for index, case in enumerate(cases):
                target, case_id, _, observable, fault, rate, ideal = case

                for rep in range(args.repeats):
                    seed = trial_seed(TEST_SEED, index, rep)
                    for family in ["raw", "fixed", "adaptive"]:
                        config = (
                            None if family == "raw"
                            else locked[budget][family]
                        )
                        result = run(
                            backend, case, budget, seed,
                            config, drift, plans,
                        )
                        rows.append({
                            "condition": condition,
                            "budget": budget,
                            "target": target,
                            "case": case_id,
                            "rep": rep,
                            "strategy": family,
                            "selected_prior": (
                                "none" if config is None else config["prior"]
                            ),
                            "observable": observable,
                            "fault": ["None", "X0", "X1", "Z0", "Z1"][int(fault)],
                            "rate": rate,
                            "ideal": ideal,
                            **result,
                            "squared_error": (result["estimate"] - ideal) ** 2,
                        })

                backend.cache.clear()
                if (index + 1) % 26 == 0:
                    print(
                        f"Test {condition}, B={budget}: "
                        f"{index + 1}/{len(cases)} cases",
                        flush=True,
                    )

            pd.DataFrame(rows).to_csv(output / "trials.csv", index=False)

    df = pd.DataFrame(rows)
    assert not df.duplicated(KEY + ["strategy"]).any()
    assert (
        df["diagnostic_shots"] + df["application_shots"] == df["budget"]
    ).all()

    metrics = ["squared_error", "diagnostic_shots", "application_shots"]
    df.groupby(["condition", "budget", "strategy"])[metrics].mean().to_csv(
        output / "summary.csv"
    )
    overall = df.groupby(["budget", "strategy"])[metrics].mean()
    overall.to_csv(output / "overall.csv")

    comp = paired_comparisons(df)
    comp.to_csv(output / "comparisons.csv", index=False)

    print("\nLocked configurations:")
    print(json.dumps(locked, indent=2))
    print("\nOverall test results:")
    print(overall.to_string())
    print("\nPrimary comparisons:")
    print(comp[comp.scope == "overall"].to_string(index=False))
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()