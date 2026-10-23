"""
Q-TRACE — Target-aware fixed-baseline validation.

Requires 14_mitigation_pilot.py in the same directory.

Changes from the original pilot:
- No changes to the adaptive policy.
- Adds a target-aware fixed baseline.
- Uses new target circuits and sampling seeds.
- Crosses every target with ZZ/XX and all noise scenarios.

Scope:
- Qiskit Aer simulation.
- Ideal readout.
- One stationary Pauli channel shared between diagnostic
  and application circuits.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. IMPORT MISSION 14
# ============================================================

SOURCE = Path(__file__).with_name(
    "14_mitigation_pilot.py"
)

if not SOURCE.exists():
    raise FileNotFoundError(
        "Put 14_mitigation_pilot.py beside this file: "
        f"{SOURCE}"
    )

spec = importlib.util.spec_from_file_location(
    "mission14",
    SOURCE,
)

m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


# ============================================================
# 2. CONFIGURATION
# ============================================================

POLICIES = [
    "raw",
    "target_fixed",
    "adaptive_selective",
    "adaptive_always",
    "oracle",
]

# Candidate fractions of total shots spent on diagnosis.
# Selected using validation data only.
FRACTIONS = [
    0.0,
    0.0625,
    0.125,
    0.25,
    0.5,
    0.75,
]

# Frozen value from Mission 14 validation.
# The adaptive policy remains unchanged.
SECOND_MOMENT = 0.26497528958302274

# Target-aware fixed schedules:
# ZZ target -> repeatedly measure bell_ZZ.
# XX target -> repeatedly measure bell_XX.
m.SCHEDULES["target_ZZ"] = [0]
m.SCHEDULES["target_XX"] = [1]


# ============================================================
# 3. CACHE IDENTICAL SIMULATOR CALLS
# ============================================================

class CachedBackend(m.Backend):
    """
    Reuse identical simulated draws across paired methods.

    This saves simulator execution time, NOT measurement
    budget: each policy is still charged its own shots.
    """

    def __init__(self, name):
        super().__init__(name)
        self.cache = {}

    def minus(
        self,
        qc,
        obs,
        fault,
        rate,
        shots,
        seed,
    ):
        circuit_key = tuple(
            (
                instruction.operation.name,
                tuple(
                    float(parameter)
                    for parameter
                    in instruction.operation.params
                ),
                tuple(
                    qc.find_bit(qubit).index
                    for qubit in instruction.qubits
                ),
            )
            for instruction in qc.data
        )

        key = (
            circuit_key,
            obs,
            fault,
            rate,
            shots,
            seed,
        )

        if key not in self.cache:
            self.cache[key] = super().minus(
                qc,
                obs,
                fault,
                rate,
                shots,
                seed,
            )

        return self.cache[key]


# ============================================================
# 4. FULLY CROSSED TARGET / OBSERVABLE / NOISE CASES
# ============================================================

def cases(seed, targets, rates):
    rng = np.random.default_rng(seed)

    result = []

    for target_id in range(targets):
        angles = rng.uniform(
            -np.pi,
            np.pi,
            4,
        )

        qc = m.target(angles)

        for obs in ["ZZ", "XX"]:
            # Used for evaluation only.
            ideal = float(
                m.Statevector.from_instruction(qc)
                .expectation_value(m.Pauli(obs))
                .real
            )

            # One no-fault case, then every fault identity
            # at every supplied noise rate.
            scenarios = [
                (0, 0.0)
            ] + [
                (fault, float(rate))
                for fault in range(1, 5)
                for rate in rates
            ]

            for fault, rate in scenarios:
                case_id = (
                    f"t{target_id}_{obs}"
                    f"_f{fault}_p{rate}"
                )

                result.append(
                    (
                        target_id,
                        case_id,
                        qc,
                        obs,
                        fault,
                        rate,
                        ideal,
                    )
                )

    expected = (
        targets
        * 2
        * (1 + 4 * len(rates))
    )

    assert len(result) == expected

    return result


# ============================================================
# 5. RUN ONE TRIAL USING THE EXISTING MISSION 14 ENGINE
# ============================================================

def run(
    backend,
    case,
    budget,
    seed,
    policy,
    fraction,
):
    (
        _,
        _,
        qc,
        obs,
        fault,
        rate,
        _,
    ) = case

    config = {
        "schedule": f"target_{obs}",
        "fraction": fraction,
    }

    # Reuse the same selective estimator as Mission 14.
    if policy == "target_fixed":
        internal_policy = "fixed_selective"
    else:
        internal_policy = policy

    return m.trial(
        backend,
        qc,
        obs,
        fault,
        rate,
        budget,
        seed,
        internal_policy,
        config,
        SECOND_MOMENT,
    )


# ============================================================
# 6. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--backend",
        choices=["aer", "analytic"],
        default="aer",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--targets",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[128, 512],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=41873,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        type=Path,
    )

    args = parser.parse_args()

    if args.smoke:
        args.targets = 2
        args.repeats = 1
        args.budgets = [128]

    if (
        args.targets < 2
        or args.repeats < 1
        or min(args.budgets) < 32
        or args.seed < 0
    ):
        parser.error(
            "Require targets>=2, repeats>=1, "
            "budgets>=32, seed>=0"
        )

    if args.output is not None:
        output = args.output

    elif args.smoke:
        output = Path(
            "results/mitigation_14b_smoke"
        )

    else:
        output = Path(
            "results/mitigation_14b"
        )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Run Mission 14's mathematical and isolation checks.
    m.checks()

    backend = CachedBackend(args.backend)

    validation = cases(
        seed=29173,
        targets=2 if args.smoke else 8,
        rates=[0.05, 0.15, 0.30],
    )

    val_repeats = (
        1 if args.smoke else 3
    )

    chosen = {}
    tuning = []

    # ========================================================
    # 7. TUNE TARGET-FIXED BASELINE ON VALIDATION ONLY
    # ========================================================

    for budget in args.budgets:
        for obs in ["ZZ", "XX"]:
            best = (
                float("inf"),
                None,
            )

            for fraction in FRACTIONS:
                errors = []

                for idx, case in enumerate(validation):
                    if case[3] != obs:
                        continue

                    for rep in range(val_repeats):
                        seed = int(
                            np.random.SeedSequence(
                                [331, idx, rep]
                            ).generate_state(1)[0]
                        )

                        result = run(
                            backend,
                            case,
                            budget,
                            seed,
                            "target_fixed",
                            fraction,
                        )

                        ideal = case[-1]

                        errors.append(
                            (
                                result["estimate"]
                                - ideal
                            ) ** 2
                        )

                score = float(
                    np.mean(errors)
                )

                tuning.append(
                    {
                        "budget": budget,
                        "observable": obs,
                        "fraction": fraction,
                        "mse": score,
                    }
                )

                if score < best[0]:
                    best = (
                        score,
                        fraction,
                    )

            chosen[
                f"{budget}_{obs}"
            ] = best[1]

            print(
                "Validation locked: "
                f"budget={budget}, "
                f"observable={obs}, "
                f"fraction={best[1]}",
                flush=True,
            )

        backend.cache.clear()

    pd.DataFrame(tuning).to_csv(
        output / "validation.csv",
        index=False,
    )

    configuration = {
        "backend": args.backend,
        "smoke": args.smoke,
        "targets": args.targets,
        "repeats": args.repeats,
        "budgets": args.budgets,
        "test_seed": args.seed,
        "validation_seed": 29173,
        "fixed_fractions": chosen,
        "second_moment": SECOND_MOMENT,
        "aggregation": (
            "Equal weight per target x observable "
            "x 13 noise scenarios"
        ),
        "note": (
            "Fixed baseline tuned over a finite "
            "fraction grid; not globally optimal."
        ),
    }

    (
        output / "config.json"
    ).write_text(
        json.dumps(
            configuration,
            indent=2,
        )
    )

    # ========================================================
    # 8. NEW TEST CIRCUITS AND OFF-GRID NOISE RATES
    # ========================================================

    test = cases(
        seed=args.seed,
        targets=args.targets,
        rates=[0.04, 0.12, 0.24],
    )

    rows = []

    for budget in args.budgets:
        for idx, case in enumerate(test):
            (
                target_id,
                case_id,
                _,
                obs,
                fault,
                rate,
                ideal,
            ) = case

            for rep in range(args.repeats):
                seed = int(
                    np.random.SeedSequence(
                        [
                            args.seed,
                            7001,
                            idx,
                            rep,
                        ]
                    ).generate_state(1)[0]
                )

                for policy in POLICIES:
                    result = run(
                        backend,
                        case,
                        budget,
                        seed,
                        policy,
                        chosen[
                            f"{budget}_{obs}"
                        ],
                    )

                    rows.append(
                        {
                            "budget": budget,
                            "target": target_id,
                            "case": case_id,
                            "rep": rep,
                            "strategy": policy,
                            "observable": obs,
                            "fault": m.FAULTS[fault],
                            "rate": rate,
                            "ideal": ideal,
                            **result,
                        }
                    )

            backend.cache.clear()

            if (idx + 1) % 26 == 0:
                print(
                    f"Test budget {budget}: "
                    f"{idx + 1}/{len(test)} cases",
                    flush=True,
                )

        # Save a checkpoint after each budget.
        pd.DataFrame(rows).to_csv(
            output / "trials.csv",
            index=False,
        )

    # ========================================================
    # 9. SUMMARY AND PER-SCENARIO RESULTS
    # ========================================================

    df = pd.DataFrame(rows)

    df["squared_error"] = (
        df.estimate - df.ideal
    ) ** 2

    assert (
        (
            df.diagnostic_shots
            + df.application_shots
        )
        == df.budget
    ).all()

    df.to_csv(
        output / "trials.csv",
        index=False,
    )

    summary = df.groupby(
        ["budget", "strategy"]
    ).agg(
        mse=(
            "squared_error",
            "mean",
        ),
        diagnostic_shots=(
            "diagnostic_shots",
            "mean",
        ),
        application_shots=(
            "application_shots",
            "mean",
        ),
        mitigation_rate=(
            "mitigated",
            "mean",
        ),
    )

    summary.to_csv(
        output / "summary.csv"
    )

    by_scenario = df.groupby(
        [
            "budget",
            "strategy",
            "observable",
            "fault",
            "rate",
        ]
    ).agg(
        mse=(
            "squared_error",
            "mean",
        ),
        n=(
            "squared_error",
            "size",
        ),
    )

    by_scenario.to_csv(
        output / "by_scenario.csv"
    )

    # ========================================================
    # 10. PAIRED TARGET-CLUSTER BOOTSTRAP
    # ========================================================

    comparisons = []

    rng = np.random.default_rng(
        91841
    )

    for budget in args.budgets:
        wide = df[
            df.budget == budget
        ].pivot(
            index=[
                "target",
                "case",
                "rep",
            ],
            columns="strategy",
            values="squared_error",
        )

        for baseline in [
            "raw",
            "target_fixed",
            "adaptive_always",
            "oracle",
        ]:
            delta = (
                wide.adaptive_selective
                - wide[baseline]
            )

            # Resample entire targets, preserving all their
            # observables, noise scenarios, repetitions,
            # and paired strategy comparisons.
            #
            # These intervals remain exploratory:
            # only 12 target clusters by default.
            means = (
                delta
                .groupby(level="target")
                .mean()
                .to_numpy()
            )

            boot = rng.choice(
                means,
                size=(
                    5000,
                    len(means),
                ),
                replace=True,
            ).mean(axis=1)

            lo, hi = np.quantile(
                boot,
                [0.025, 0.975],
            )

            comparisons.append(
                {
                    "budget": budget,
                    "baseline": baseline,
                    "delta_mse": delta.mean(),
                    "target_cluster_ci_low": lo,
                    "target_cluster_ci_high": hi,
                    "target_clusters": len(means),
                    "smoke": args.smoke,
                }
            )

    pd.DataFrame(comparisons).to_csv(
        output / "comparisons.csv",
        index=False,
    )

    # ========================================================
    # 11. PRINT RESULTS
    # ========================================================

    print(
        summary
        .round(6)
        .to_string()
    )

    print(
        pd.DataFrame(comparisons)
        .round(6)
        .to_string(index=False)
    )

    print(
        f"Saved to {output}"
    )

    if args.smoke:
        print(
            "SMOKE TEST ONLY: ignore performance "
            "estimates and intervals."
        )

    else:
        print(
            "Exploratory target-cluster CIs; "
            "no hardware, drift, or "
            "readout-noise validation."
        )


if __name__ == "__main__":
    main()