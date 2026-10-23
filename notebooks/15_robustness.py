"""Frozen-policy robustness test for Q-TRACE 14e.

Requires 14e_fine_noise.py and its dependencies beside this file.

No tuning on stress-test data.
Readout: independent symmetric assignment errors.
Drift: one change in Pauli error probability before application.
All diagnostic and application shots count toward the total budget.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError


SOURCE = Path(__file__).with_name("14e_fine_noise.py")
if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

spec = importlib.util.spec_from_file_location("mission14e", SOURCE)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)

e.configure_model()
m = e.m
runner = e.runner

CONDITIONS = [
    ("clean", 0.00, 0.00),
    ("readout_1pct", 0.01, 0.00),
    ("readout_3pct", 0.03, 0.00),
    ("drift_up", 0.00, 0.03),
    ("drift_down", 0.00, -0.03),
    ("readout_and_drift", 0.01, 0.03),
]

POLICIES = ["raw", "target_fixed", "qtrace", "oracle"]

# Locked from the earlier clean validation, for both XX and ZZ.
FIXED_FRACTIONS = {128: 0.125, 512: 0.25}


def application_rate(fault, rate, drift):
    if fault == 0:
        return 0.0
    return float(np.clip(rate + drift, 0.0, 0.49))


class StressBackend:
    """Measurement backend; hidden noise parameters stay here."""

    def __init__(self, name, readout):
        self.name = name
        self.readout = float(readout)
        self.base = m.Backend(name)
        self.cache = {}

        if name == "aer":
            noise = NoiseModel()
            if readout > 0:
                noise.add_all_qubit_readout_error(
                    ReadoutError([
                        [1.0 - readout, readout],
                        [readout, 1.0 - readout],
                    ])
                )
            self.base.sim = AerSimulator(
                method="density_matrix",
                noise_model=noise,
                max_parallel_threads=1,
            )

    def minus(self, qc, obs, fault, rate, shots, seed):
        circuit_key = tuple(
            (
                item.operation.name,
                tuple(float(p) for p in item.operation.params),
                tuple(qc.find_bit(q).index for q in item.qubits),
            )
            for item in qc.data
        )
        key = (circuit_key, obs, fault, rate, shots, seed)

        if key not in self.cache:
            if self.name == "aer":
                k = self.base.minus(
                    qc, obs, fault, rate, shots, seed
                )
            else:
                # Exact binomial model for these assumptions.
                ideal = float(
                    m.Statevector.from_instruction(qc)
                    .expectation_value(m.Pauli(obs)).real
                )
                affected = int(
                    m.pauli(fault).anticommutes(m.Pauli(obs))
                )
                weight = sum(axis != "I" for axis in obs)
                attenuation = (
                    (1.0 - 2.0 * rate * affected)
                    * (1.0 - 2.0 * self.readout) ** weight
                )
                probability = np.clip(
                    (1.0 - ideal * attenuation) / 2.0,
                    0.0, 1.0,
                )
                k = np.random.default_rng(seed).binomial(
                    shots, probability
                )

            self.cache[key] = int(k)

        return self.cache[key]


class TrialMeasurements:
    """Separate diagnostic and application noise for one trial.

    Existing 14-series trials identify their single application
    measurement using seed + 1000003.
    """

    def __init__(self, backend, seed, drift):
        self.backend = backend
        self.application_seed = seed + 1000003
        self.drift = drift

    def minus(self, qc, obs, fault, rate, shots, seed):
        if seed == self.application_seed:
            rate = application_rate(fault, rate, self.drift)
        return self.backend.minus(
            qc, obs, fault, rate, shots, seed
        )


def run_policy(backend, case, budget, seed, policy, drift):
    _, _, qc, obs, fault, rate, _ = case
    measurements = TrialMeasurements(backend, seed, drift)

    internal = {
        "raw": "raw",
        "target_fixed": "fixed_selective",
        "qtrace": "adaptive_selective",
        "oracle": "raw",
    }[policy]

    result = m.trial(
        measurements,
        qc, obs, fault, rate, budget, seed,
        internal,
        {
            "schedule": f"target_{obs}",
            "fraction": FIXED_FRACTIONS[budget],
        },
        runner.SECOND_MOMENT,
    )

    if policy == "oracle":
        # Privileged reference: knows application noise and readout.
        p_app = application_rate(fault, rate, drift)
        affected = int(
            m.pauli(fault).anticommutes(m.Pauli(obs))
        )
        weight = sum(axis != "I" for axis in obs)
        attenuation = (
            (1.0 - 2.0 * p_app * affected)
            * (1.0 - 2.0 * backend.readout) ** weight
        )
        gain = 1.0 / attenuation

        result.update(
            estimate=gain * result["raw_same_shots"],
            gain=gain,
            mitigated=int(gain > 1.0 + 1e-10),
        )

    assert (
        result["diagnostic_shots"]
        + result["application_shots"]
        == budget
    )
    return result


def save_tables(rows, output):
    df = pd.DataFrame(rows)
    df.to_csv(output / "trials.csv", index=False)

    summary = df.groupby(
        ["condition", "budget", "strategy"]
    ).agg(
        mse=("squared_error", "mean"),
        diagnostic_shots=("diagnostic_shots", "mean"),
        application_shots=("application_shots", "mean"),
        mitigation_rate=("mitigated", "mean"),
    )
    summary.to_csv(output / "summary.csv")

    df.groupby([
        "condition", "budget", "strategy",
        "observable", "fault", "rate",
    ]).agg(
        mse=("squared_error", "mean"),
        n=("squared_error", "size"),
    ).to_csv(output / "by_scenario.csv")

    comparisons = []
    rng = np.random.default_rng(150027)

    for (condition, budget), group in df.groupby(
        ["condition", "budget"]
    ):
        wide = group.pivot(
            index=["target", "case", "rep"],
            columns="strategy",
            values="squared_error",
        )
        for baseline in ["raw", "target_fixed", "oracle"]:
            delta = wide["qtrace"] - wide[baseline]
            cluster_means = (
                delta.groupby(level="target").mean().to_numpy()
            )
            boot = rng.choice(
                cluster_means,
                size=(5000, len(cluster_means)),
                replace=True,
            ).mean(axis=1)
            low, high = np.quantile(boot, [0.025, 0.975])

            comparisons.append({
                "condition": condition,
                "budget": budget,
                "baseline": baseline,
                "delta_mse": float(delta.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
                "target_clusters": len(cluster_means),
            })

    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(output / "comparisons.csv", index=False)
    return summary, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["aer", "analytic"], default="aer"
    )
    parser.add_argument("--targets", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=150031)
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
        "results/robustness_15_smoke"
        if args.smoke else "results/robustness_15"
    )
    if output.exists() and any(output.iterdir()):
        parser.error(
            f"{output} is not empty. Use a new --output directory."
        )
    output.mkdir(parents=True, exist_ok=True)

    m.checks()
    e.d.c.check_gain()

    config = {
        "backend": args.backend,
        "smoke": args.smoke,
        "targets": args.targets,
        "repeats": args.repeats,
        "test_seed": args.seed,
        "budgets": budgets,
        "conditions": [
            {"name": name, "readout": ro, "drift": drift}
            for name, ro, drift in CONDITIONS
        ],
        "fixed_fractions": FIXED_FRACTIONS,
        "second_moment": runner.SECOND_MOMENT,
        "noise_grid": m.RATES.tolist(),
        "latent_prior": m.PRIOR.tolist(),
        "policy": "Frozen 14e DP, gain >= 1",
        "tuning": "None on stress-test data",
        "oracle": "Known application attenuation, inverse gain",
        "ci": "Exploratory target-cluster bootstrap; unadjusted",
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2)
    )

    # Same circuits and base trial seeds across stress conditions.
    cases = runner.cases(
        args.seed, args.targets, [0.04, 0.12, 0.24]
    )
    rows = []

    for condition, readout, drift in CONDITIONS:
        backend = StressBackend(args.backend, readout)

        for budget in budgets:
            for index, case in enumerate(cases):
                target_id, case_id, _, obs, fault, rate, ideal = case

                for rep in range(args.repeats):
                    seed = int(
                        np.random.SeedSequence([
                            args.seed, 7001, index, rep
                        ]).generate_state(1)[0]
                    )
                    for policy in POLICIES:
                        result = run_policy(
                            backend, case, budget,
                            seed, policy, drift
                        )
                        rows.append({
                            "condition": condition,
                            "budget": budget,
                            "target": target_id,
                            "case": case_id,
                            "rep": rep,
                            "strategy": policy,
                            "observable": obs,
                            "fault": m.FAULTS[fault],
                            "rate": rate,
                            "application_rate": application_rate(
                                fault, rate, drift
                            ),
                            "readout": readout,
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

            # Save completed work after each condition/budget.
            pd.DataFrame(rows).to_csv(
                output / "trials.csv", index=False
            )

    summary, comparisons = save_tables(rows, output)
    print(summary.round(6).to_string())
    print(comparisons.round(6).to_string(index=False))
    print(f"Saved to {output}")

    if args.smoke:
        print("SMOKE ONLY: do not interpret performance or CIs.")


if __name__ == "__main__":
    main()