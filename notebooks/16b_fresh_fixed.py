"""Add a fresh-fixed baseline to existing Mission16 results.

Requires:
    16_calibration_check.py and its dependencies
    results/calibration_16/config.json
    results/calibration_16/trials.csv

Only the missing baseline is simulated.
This is a retrospective exploratory comparison, not a new holdout.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


source = Path(__file__).with_name("16_calibration_check.py")
if not source.exists():
    raise FileNotFoundError(source)

spec = importlib.util.spec_from_file_location("mission16", source)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

m = v.m
s = v.s
runner = v.runner

KEY = ["condition", "budget", "target", "case", "rep"]


def fresh_fixed(backend, case, budget, seed, drift, fraction):
    _, _, qc, obs, fault, rate, _ = case

    probe = {"ZZ": 0, "XX": 1}[obs]
    mode, measured = m.PROBES[probe]
    probe_circuit = m.prepare(mode)

    probabilities = (
        m.SENS[probe, m.STATE_FAULT] * m.STATE_RATE
    )
    post = m.PRIOR.copy()

    steps = int(budget * fraction) // m.BATCH
    diagnostic_shots = steps * m.BATCH
    application_shots = budget - diagnostic_shots
    assert application_shots > 0

    # Hidden truth is used only to generate measurements.
    p_app = s.application_rate(fault, rate, drift)

    for step in range(steps):
        k = backend.minus(
            probe_circuit,
            measured,
            fault,
            p_app,
            m.BATCH,
            seed + 300003 + step * 101 + probe,
        )
        post = v.update(
            post, k, m.BATCH, probabilities
        )

    gain = float(
        m.terminal(
            post, m.attenuation(obs),
            application_shots, v.M2
        )[1][0]
    )

    k = backend.minus(
        qc, obs, fault, p_app,
        application_shots, seed + 1000003
    )
    raw = 1.0 - 2.0 * k / application_shots

    return {
        "estimate": gain * raw,
        "raw_same_shots": raw,
        "diagnostic_shots": diagnostic_shots,
        "application_shots": application_shots,
        "mitigated": int(gain > 1.0 + 1e-10),
        "gain": gain,
        "old_shots": 0,
        "fresh_shots": diagnostic_shots,
        "alarm": np.nan,
        "predictive_tail": np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("results/calibration_16")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/calibration_16b")
    )
    args = parser.parse_args()

    cfg = json.loads(
        (args.input / "config.json").read_text()
    )
    old = pd.read_csv(
        args.input / "trials.csv",
        keep_default_na=False,
        na_values=[""],
    )

    if cfg["smoke"]:
        parser.error("Use the full Mission16 results, not smoke.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output is not empty; use a new --output.")

    # Do not silently compare results from different inference models.
    assert np.allclose(cfg["noise_grid"], m.RATES)
    assert np.allclose(cfg["prior"], m.PRIOR)
    assert np.isclose(cfg["second_moment"], v.M2)

    existing_fresh = old[
        old.strategy == "fresh_only"
    ].copy()

    assert not existing_fresh.duplicated(KEY).any()
    reference = existing_fresh.set_index(KEY)

    cases = runner.cases(
        cfg["seed"], cfg["targets"],
        [0.04, 0.12, 0.24]
    )
    expected_rows = (
        len(cfg["conditions"])
        * len(cfg["budgets"])
        * len(cases)
        * cfg["repeats"]
    )
    assert len(existing_fresh) == expected_rows
    assert (
        existing_fresh.diagnostic_shots
        + existing_fresh.application_shots
        == existing_fresh.budget
    ).all()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []

    for condition, readout, drift in cfg["conditions"]:
        backend = s.StressBackend(cfg["backend"], readout)

        for budget in cfg["budgets"]:
            fraction = float(
                cfg["fixed_fractions"][str(budget)]
            )

            for index, case in enumerate(cases):
                target_id, case_id, _, _, _, _, ideal = case

                for rep in range(cfg["repeats"]):
                    seed = int(
                        np.random.SeedSequence([
                            cfg["seed"], 7001, index, rep
                        ]).generate_state(1)[0]
                    )
                    key = (
                        condition, budget,
                        target_id, case_id, rep
                    )
                    previous = reference.loc[key]
                    assert np.isclose(
                        previous["ideal"], ideal,
                        atol=1e-12, rtol=0
                    )

                    # Reproduce one stored fresh-only trial per
                    # condition/budget before reusing its results.
                    if index == 0 and rep == 0:
                        check = v.run_policy(
                            backend, case, budget, seed,
                            "fresh_only", drift
                        )
                        for field in [
                            "estimate", "gain",
                            "diagnostic_shots", "application_shots",
                        ]:
                            if not np.isclose(
                                check[field], previous[field],
                                atol=1e-12, rtol=0
                            ):
                                raise RuntimeError(
                                    "Stored fresh_only result did not "
                                    f"reproduce: {condition}, {budget}, "
                                    f"{field}. Check code/package versions."
                                )

                    result = fresh_fixed(
                        backend, case, budget,
                        seed, drift, fraction
                    )

                    row = previous.to_dict()
                    row.update(dict(zip(KEY, key)))
                    row.update(result)
                    row["strategy"] = "fresh_fixed"
                    row["squared_error"] = (
                        result["estimate"] - ideal
                    ) ** 2
                    rows.append(row)

                backend.cache.clear()
                if (index + 1) % 26 == 0:
                    print(
                        f"{condition}, budget={budget}: "
                        f"{index + 1}/{len(cases)} cases",
                        flush=True,
                    )

            pd.DataFrame(rows).to_csv(
                args.output / "fresh_fixed_trials.csv",
                index=False,
            )

    new = pd.DataFrame(rows)
    assert len(new) == expected_rows
    assert (
        new.diagnostic_shots + new.application_shots
        == new.budget
    ).all()

    combined = pd.concat([
        old[old.strategy.isin([
            "raw", "oracle", "fresh_only"
        ])],
        new,
    ], ignore_index=True)

    assert not combined.duplicated(
        KEY + ["strategy"]
    ).any()

    combined.to_csv(
        args.output / "trials.csv", index=False
    )

    summary = combined.groupby([
        "condition", "budget", "strategy"
    ]).agg(
        mse=("squared_error", "mean"),
        diagnostic_shots=("diagnostic_shots", "mean"),
        application_shots=("application_shots", "mean"),
    )
    summary.to_csv(args.output / "summary.csv")

    combined.groupby([
        "condition", "budget", "strategy",
        "observable", "fault", "rate",
    ]).agg(
        mse=("squared_error", "mean"),
        n=("squared_error", "size"),
    ).to_csv(args.output / "by_scenario.csv")

    rng = np.random.default_rng(160207)
    comparisons = []

    for (condition, budget), group in combined.groupby(
        ["condition", "budget"]
    ):
        wide = group.pivot(
            index=["target", "case", "rep"],
            columns="strategy",
            values="squared_error",
        )

        for baseline in ["fresh_fixed", "raw", "oracle"]:
            delta = wide["fresh_only"] - wide[baseline]
            means = (
                delta.groupby(level="target").mean().to_numpy()
            )
            boot = rng.choice(
                means,
                size=(5000, len(means)),
                replace=True,
            ).mean(axis=1)
            low, high = np.quantile(boot, [0.025, 0.975])

            comparisons.append({
                "condition": condition,
                "budget": budget,
                "method": "fresh_only",
                "baseline": baseline,
                "delta_mse": float(delta.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
                "target_clusters": len(means),
            })

    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(
        args.output / "comparisons.csv", index=False
    )

    provenance = {
        "experiment": "16b: adaptive vs fixed, both fresh",
        "source_directory": str(args.input.resolve()),
        "source_config": cfg,
        "fixed_allocation": "Frozen from earlier clean validation",
        "reused": ["fresh_only", "raw", "oracle"],
        "newly_simulated": ["fresh_fixed"],
        "comparison": "fresh_only MSE minus baseline MSE",
        "inference": (
            "Retrospective exploratory target-cluster bootstrap; "
            "no multiple-comparison adjustment."
        ),
    }
    (args.output / "config.json").write_text(
        json.dumps(provenance, indent=2)
    )

    print(summary.round(6).to_string())
    print(comparisons.round(6).to_string(index=False))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()