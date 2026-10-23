"""Calibration freshness experiment.

Requires 15_robustness.py and its dependencies.
All old calibration, fresh checks and application shots are charged.
No test truth enters the calibration alarm or gain decision.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import binom


source = Path(__file__).with_name("15_robustness.py")
if not source.exists():
    raise FileNotFoundError(source)

spec = importlib.util.spec_from_file_location("mission15", source)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

m = s.m
runner = s.runner
dp = s.e.d
M2 = runner.SECOND_MOMENT

CHECK_SHOTS = {128: 16, 512: 64}
ALPHA = 0.05

POLICIES = [
    "raw",
    "stale_qtrace",
    "target_fixed",
    "qtrace",
    "fresh_only",
    "oracle",
]


def update(post, k, n, probabilities):
    with np.errstate(divide="ignore"):
        log_weights = (
            np.log(post)
            + binom.logpmf(k, n, probabilities)
        )
    normalizer = logsumexp(log_weights)
    if not np.isfinite(normalizer):
        raise RuntimeError("Observation impossible under model.")
    return np.exp(log_weights - normalizer)


def predictive_tail(post, k, n, probabilities):
    """Conservative two-sided posterior-predictive tail score.

    This is a model-based alarm score, not a universal frequentist
    false-positive guarantee under misspecified noise.
    """
    outcomes = np.arange(n + 1)
    predictive = (
        binom.pmf(
            outcomes[:, None],
            n,
            probabilities[None, :],
        )
        @ post
    )
    return float(min(
        1.0,
        2.0 * min(
            predictive[:k + 1].sum(),
            predictive[k:].sum(),
        ),
    ))


def run_policy(backend, case, budget, seed, policy, drift):
    _, _, qc, obs, fault, rate, _ = case

    # Existing strategies retain their original implementation.
    if policy in ("raw", "oracle", "stale_qtrace"):
        original_name = (
            "qtrace" if policy == "stale_qtrace" else policy
        )
        result = s.run_policy(
            backend, case, budget, seed, original_name, drift
        )
        result.update(
            old_shots=result["diagnostic_shots"],
            fresh_shots=0,
            alarm=np.nan,
            predictive_tail=np.nan,
        )
        return result

    probe = {"ZZ": 0, "XX": 1}[obs]
    mode, measured = m.PROBES[probe]
    probe_circuit = m.prepare(mode)
    probabilities = (
        m.SENS[probe, m.STATE_FAULT] * m.STATE_RATE
    )
    lam = m.attenuation(obs)

    # Hidden parameters are used only by this measurement closure.
    def measure(circuit, observable, shots, draw_seed, fresh):
        effective_rate = (
            s.application_rate(fault, rate, drift)
            if fresh else rate
        )
        return backend.minus(
            circuit, observable, fault,
            effective_rate, shots, draw_seed
        )

    post = m.PRIOR.copy()
    step = 0
    minus_total = 0
    alarm = np.nan
    tail = np.nan

    if policy == "fresh_only":
        # Strong baseline: all characterization occurs after the change.
        plan = dp.build_plan(budget, obs, float(M2))

        while plan["continue"][step][minus_total]:
            k = measure(
                probe_circuit, measured, m.BATCH,
                seed + 300003 + step * 101 + probe,
                fresh=True,
            )
            post = update(post, k, m.BATCH, probabilities)
            minus_total += int(k)
            step += 1

        old_shots = 0
        fresh_shots = step * m.BATCH

    else:
        check_shots = CHECK_SHOTS[budget]

        if policy == "qtrace":
            # Reserve fresh-check shots before old-phase planning.
            plan = dp.build_plan(
                budget - check_shots, obs, float(M2)
            )
            fixed_steps = None
        else:
            plan = None
            fixed_steps = (
                int(budget * s.FIXED_FRACTIONS[budget])
                // m.BATCH
            )

        while True:
            take_more = (
                plan["continue"][step][minus_total]
                if plan is not None
                else step < fixed_steps
            )
            if not take_more:
                break

            k = measure(
                probe_circuit, measured, m.BATCH,
                seed + step * 101 + probe,
                fresh=False,
            )
            post = update(post, k, m.BATCH, probabilities)
            minus_total += int(k)
            step += 1

        old_shots = step * m.BATCH

        fresh_k = measure(
            probe_circuit, measured, check_shots,
            seed + 200003,
            fresh=True,
        )
        tail = predictive_tail(
            post, fresh_k, check_shots, probabilities
        )
        alarm = int(tail < ALPHA)

        # Alarm: discard old observations.
        # No alarm: pool old and fresh observations.
        starting_post = m.PRIOR if alarm else post
        post = update(
            starting_post, fresh_k, check_shots, probabilities
        )
        fresh_shots = check_shots

    diagnostic_shots = old_shots + fresh_shots
    application_shots = budget - diagnostic_shots
    assert application_shots > 0

    gain = float(
        m.terminal(post, lam, application_shots, M2)[1][0]
    )
    k = measure(
        qc, obs, application_shots,
        seed + 1000003,
        fresh=True,
    )
    raw = 1.0 - 2.0 * k / application_shots

    assert diagnostic_shots + application_shots == budget

    return {
        "estimate": gain * raw,
        "raw_same_shots": raw,
        "diagnostic_shots": diagnostic_shots,
        "application_shots": application_shots,
        "mitigated": int(gain > 1.0 + 1e-10),
        "gain": gain,
        "old_shots": old_shots,
        "fresh_shots": fresh_shots,
        "alarm": alarm,
        "predictive_tail": tail,
    }


def save_results(rows, output):
    df = pd.DataFrame(rows)
    df.to_csv(output / "trials.csv", index=False)

    summary = df.groupby(
        ["condition", "budget", "strategy"]
    ).agg(
        mse=("squared_error", "mean"),
        old_shots=("old_shots", "mean"),
        fresh_shots=("fresh_shots", "mean"),
        diagnostic_shots=("diagnostic_shots", "mean"),
        application_shots=("application_shots", "mean"),
        alarm_rate=("alarm", "mean"),
    )
    summary.to_csv(output / "summary.csv")

    df.groupby([
        "condition", "budget", "strategy",
        "observable", "fault", "rate",
    ]).agg(
        mse=("squared_error", "mean"),
        alarm_rate=("alarm", "mean"),
        n=("squared_error", "size"),
    ).to_csv(output / "by_scenario.csv")

    # Effective change refers to the observable attenuation.
    # A commuting Pauli fault changing rate does not change this observable.
    df[df["alarm"].notna()].groupby([
        "condition", "budget", "strategy", "effective_changed",
    ]).agg(
        alarm_rate=("alarm", "mean"),
        n=("alarm", "size"),
    ).to_csv(output / "alarms.csv")

    rng = np.random.default_rng(160029)
    comparisons = []

    for (condition, budget), group in df.groupby(
        ["condition", "budget"]
    ):
        wide = group.pivot(
            index=["target", "case", "rep"],
            columns="strategy",
            values="squared_error",
        )
        for baseline in [
            "raw", "stale_qtrace",
            "target_fixed", "fresh_only", "oracle",
        ]:
            delta = wide["qtrace"] - wide[baseline]
            target_means = (
                delta.groupby(level="target").mean().to_numpy()
            )
            bootstrap = rng.choice(
                target_means,
                size=(5000, len(target_means)),
                replace=True,
            ).mean(axis=1)
            low, high = np.quantile(
                bootstrap, [0.025, 0.975]
            )
            comparisons.append({
                "condition": condition,
                "budget": budget,
                "baseline": baseline,
                "delta_mse": float(delta.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
                "target_clusters": len(target_means),
            })

    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(output / "comparisons.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["aer", "analytic"], default="aer"
    )
    parser.add_argument("--targets", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=160031)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.targets < 2 or args.repeats < 1 or args.seed < 0:
        parser.error("Require targets>=2, repeats>=1, seed>=0.")

    budgets = [128, 512]
    if args.smoke:
        args.targets, args.repeats = 2, 1
        budgets = [128]

    output = args.output or Path(
        "results/calibration_16_smoke"
        if args.smoke else "results/calibration_16"
    )
    if output.exists() and any(output.iterdir()):
        parser.error("Output is not empty; choose a new --output.")
    output.mkdir(parents=True, exist_ok=True)

    m.checks()
    s.e.d.c.check_gain()

    config = {
        "backend": args.backend,
        "smoke": args.smoke,
        "targets": args.targets,
        "repeats": args.repeats,
        "seed": args.seed,
        "budgets": budgets,
        "check_shots": CHECK_SHOTS,
        "alarm_threshold": ALPHA,
        "fixed_fractions": s.FIXED_FRACTIONS,
        "second_moment": M2,
        "noise_grid": m.RATES.tolist(),
        "prior": m.PRIOR.tolist(),
        "conditions": s.CONDITIONS,
        "policies": POLICIES,
        "on_alarm": "Discard old data; infer from fresh check.",
        "no_alarm": "Pool old and fresh observations.",
        "tuning": "None on this evaluation set.",
        "limitations": [
            "One change between old and fresh measurement phases.",
            "Stable noise throughout fresh phase and application.",
            "DP for old phase does not optimize the future alarm rule.",
            "Exploratory CIs; no multiple-comparison adjustment.",
            "Oracle knows true application attenuation.",
        ],
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2)
    )

    cases = runner.cases(
        args.seed, args.targets, [0.04, 0.12, 0.24]
    )
    rows = []

    for condition, readout, drift in s.CONDITIONS:
        backend = s.StressBackend(args.backend, readout)

        for budget in budgets:
            for index, case in enumerate(cases):
                target, case_id, _, obs, fault, rate, ideal = case
                p_app = s.application_rate(fault, rate, drift)
                affected = m.pauli(fault).anticommutes(
                    m.Pauli(obs)
                )
                effective_changed = bool(
                    affected and not np.isclose(rate, p_app)
                )

                for rep in range(args.repeats):
                    seed = int(np.random.SeedSequence([
                        args.seed, 7001, index, rep
                    ]).generate_state(1)[0])

                    for policy in POLICIES:
                        result = run_policy(
                            backend, case, budget,
                            seed, policy, drift
                        )
                        rows.append({
                            "condition": condition,
                            "budget": budget,
                            "target": target,
                            "case": case_id,
                            "rep": rep,
                            "strategy": policy,
                            "observable": obs,
                            "fault": m.FAULTS[fault],
                            "rate": rate,
                            "application_rate": p_app,
                            "effective_changed": effective_changed,
                            "ideal": ideal,
                            **result,
                            "squared_error": (
                                result["estimate"] - ideal
                            ) ** 2,
                        })

                backend.cache.clear()
                if (index + 1) % 26 == 0:
                    print(
                        f"{condition}, budget={budget}: "
                        f"{index + 1}/{len(cases)} cases",
                        flush=True,
                    )

            pd.DataFrame(rows).to_csv(
                output / "trials.csv", index=False
            )

    summary = save_results(rows, output)
    print(summary.round(6).to_string())
    print(f"Saved to {output}")

    if args.smoke:
        print("SMOKE ONLY: do not interpret performance or CIs.")


if __name__ == "__main__":
    main()