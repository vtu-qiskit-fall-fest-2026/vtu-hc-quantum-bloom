
"""
Q-TRACE | Mission 11
Closed-Loop Bayesian Fault Diagnosis with Qiskit Aer

Goals:
- Derive Pauli fault signatures using Qiskit.
- Execute diagnostic circuits on Qiskit Aer.
- Update Bayesian beliefs using actual shot counts.
- Compare adaptive probing to optimized static allocation.
- Evaluate actual stopping, abstention, shot costs and
  paired statistical uncertainty.

Scope:
- Two qubits
- A single stochastic X/Z Pauli fault, or no fault
- Ideal state preparation and measurement
- Local simulation, NOT real quantum hardware

Run:
    python notebooks/11_aer_closed_loop.py --demo
"""

import argparse

from collections import Counter
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import binom

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

from qiskit_aer import AerSimulator
from qiskit_aer.noise import pauli_error


# ============================================================
# 1. EXPERIMENT SETTINGS
# ============================================================

FAULTS = ["None", "X0", "X1", "Z0", "Z1"]

PROBES = [
    ("bell_ZZ", "bell", "ZZ"),
    ("bell_XX", "bell", "XX"),
    ("zero_Z0", "zero", "IZ"),
    ("zero_Z1", "zero", "ZI"),
    ("plus_X0", "plus", "IX"),
    ("plus_X1", "plus", "XI"),
]

RATE_GRID = np.array([
    0.02,
    0.05,
    0.10,
    0.20,
])

POLICIES = [
    "adaptive_eig",
    "optimized_fixed",
    "round_robin",
]

CONFIDENCE = 0.95

ABSTAIN_LOSS = 0.25
SHOT_COST = 0.001


# ============================================================
# 2. QUANTUM STATE PREPARATION
# ============================================================

def prepared(mode):

    qc = QuantumCircuit(2)

    if mode == "bell":
        qc.h(0)
        qc.cx(0, 1)

    elif mode == "plus":
        qc.h([0, 1])

    elif mode != "zero":
        raise ValueError(mode)

    return qc


# ============================================================
# 3. LOCAL PAULI FAULTS
# ============================================================

def local_pauli(fault):

    if fault == "None":
        return Pauli("II")

    gate = fault[0]
    qubit = int(fault[1])

    # Qiskit Pauli ordering is q1, q0.
    label = ["I", "I"]
    label[1 - qubit] = gate

    return Pauli("".join(label))


# ============================================================
# 4. DERIVE SENSITIVITY USING QISKIT
# ============================================================

def sensitivity_matrix():

    sens = np.zeros(
        (len(PROBES), len(FAULTS)),
        dtype=int,
    )

    for a, (_, mode, observable) in enumerate(PROBES):

        state = Statevector.from_instruction(
            prepared(mode)
        )

        target = Pauli(observable)

        ideal = float(
            np.real(state.expectation_value(target))
        )

        assert np.isclose(ideal, 1.0)

        for j, fault in enumerate(FAULTS):

            error = local_pauli(fault)

            sens[a, j] = int(
                error.anticommutes(target)
            )

    # No-fault hypothesis is never sensitive.
    assert sens[:, 0].sum() == 0

    # All five hypotheses must have distinct signatures.
    signatures = {
        tuple(sens[:, j])
        for j in range(len(FAULTS))
    }

    assert len(signatures) == len(FAULTS)

    return sens


# ============================================================
# 5. BAYESIAN LATENT NOISE MODEL
# ============================================================

def make_model(sens, batch):

    # One no-fault state.
    # Four fault identities, each with several possible rates.

    state_fault = np.array(
        [0] + [
            j
            for j in range(1, 5)
            for _ in RATE_GRID
        ]
    )

    state_rate = np.array(
        [0.0] + list(np.tile(RATE_GRID, 4))
    )

    # Equal prior for each physical fault identity.
    # Spread each fault's mass across candidate rates.

    prior = np.array(
        [0.2]
        + [0.2 / len(RATE_GRID)]
        * (4 * len(RATE_GRID))
    )

    # Map latent states to five fault labels.
    classes = np.eye(len(FAULTS))[state_fault]

    # Probability of negative Pauli measurement.
    p_minus = (
        sens[:, state_fault]
        * state_rate[None, :]
    )

    outcomes = np.arange(
        batch + 1
    )[None, :, None]

    likelihood = binom.pmf(
        outcomes,
        batch,
        p_minus[:, None, :],
    )

    assert np.allclose(
        likelihood.sum(axis=1),
        1.0,
    )

    return prior, classes, likelihood


# ============================================================
# 6. INFORMATION THEORY
# ============================================================

def entropy_rows(arr):

    arr = np.asarray(arr)

    # Avoid log(0).
    safe = np.clip(
        arr,
        1e-300,
        1.0,
    )

    return -np.sum(
        np.where(
            arr > 0,
            arr * np.log(safe),
            0.0,
        ),
        axis=-1,
    )


def eig_class(posterior, classes, likelihood):

    predictive = likelihood @ posterior

    valid = predictive > 1e-14

    after = (
        likelihood[valid]
        * posterior[None, :]
        / predictive[valid, None]
    )

    before_class = posterior @ classes
    after_class = after @ classes

    return float(
        entropy_rows(before_class)
        - predictive[valid]
        @ entropy_rows(after_class)
    )


# ============================================================
# 7. OPTIMIZED NON-ADAPTIVE BASELINE
# ============================================================

def optimized_fixed_schedule(sens, batches):
    """
    Find a static probe allocation maximizing the
    minimum pairwise Bhattacharyya separation.

    Optimization uses only the inference model,
    never observations or test labels.

    This is a stronger baseline than fixed4,
    but NOT globally Bayes-optimal allocation.
    """

    # Approximate candidate fault strength for
    # the static design objective.

    p = sens * float(
        np.mean(RATE_GRID)
    )

    pair_distances = []

    for i in range(len(FAULTS)):

        for j in range(i + 1, len(FAULTS)):

            bc = (
                np.sqrt(p[:, i] * p[:, j])
                + np.sqrt(
                    (1 - p[:, i])
                    * (1 - p[:, j])
                )
            )

            distance = -np.log(
                np.clip(bc, 1e-14, 1.0)
            )

            pair_distances.append(distance)

    pair_distances = np.array(pair_distances)

    best_combo = None
    best_score = (-np.inf, -np.inf)

    # Enumerate all count allocations,
    # ignoring order initially.

    for combo in combinations_with_replacement(
        range(len(PROBES)),
        batches,
    ):

        counts = np.bincount(
            combo,
            minlength=len(PROBES),
        )

        separations = pair_distances @ counts

        score = (
            float(separations.min()),
            float(separations.sum()),
        )

        if score > best_score:
            best_combo = combo
            best_score = score

    # Spread selected settings across the schedule
    # instead of running all instances consecutively.

    remaining = Counter(best_combo)

    interleaved = []

    while len(interleaved) < batches:

        for probe in range(len(PROBES)):

            if remaining[probe] > 0:
                interleaved.append(probe)
                remaining[probe] -= 1

    return interleaved


# ============================================================
# 8. BUILD ACTUAL NOISY QUANTUM CIRCUIT
# ============================================================

def make_aer_circuit(probe_id, fault, p_true):

    _, mode, observable = PROBES[probe_id]

    qc = prepared(mode)

    if fault != "None":

        axis = fault[0]
        qubit = int(fault[1])

        error = pauli_error([
            ("I", 1 - p_true),
            (axis, p_true),
        ])

        qc.append(
            error.to_instruction(),
            [qubit],
        )

    # Rotate measurements into Z basis.

    for q, axis in enumerate(observable[::-1]):

        if axis == "X":
            qc.h(q)

        elif axis not in ("I", "Z"):
            raise ValueError(axis)

    qc.measure_all()

    return qc


# ============================================================
# 9. READ ACTUAL QUANTUM MEASUREMENT COUNTS
# ============================================================

def negative_outcomes(counts, observable):

    total = 0

    for bits, frequency in counts.items():

        bits = bits.replace(" ", "")

        parity = sum(
            int(bits[-1 - q])
            for q, axis in enumerate(observable[::-1])
            if axis != "I"
        )

        if parity % 2:
            total += frequency

    return total


# ============================================================
# 10. WILSON CONFIDENCE INTERVAL
# ============================================================

def wilson(successes, total):

    if total == 0:
        return np.nan, np.nan

    z = 1.96

    p = successes / total

    denominator = 1 + z**2 / total

    center = (
        p + z**2 / (2 * total)
    ) / denominator

    radius = (
        z * np.sqrt(
            p * (1 - p) / total
            + z**2 / (4 * total**2)
        )
    ) / denominator

    return (
        center - radius,
        center + radius,
    )


# ============================================================
# 11. CLOSED-LOOP QUANTUM DIAGNOSIS
# ============================================================

def run_one(
    policy,
    fault,
    rate,
    seed_table,
    batch,
    budget,
    sens,
    prior,
    classes,
    likelihood,
    fixed,
    aer,
    cache,
    trace=False,
):

    posterior = prior.copy()

    n_batches = budget // batch

    history = []

    for step in range(n_batches):

        # ----------------------------------------
        # Choose next probe
        # ----------------------------------------

        if policy == "adaptive_eig":

            scores = [
                eig_class(
                    posterior,
                    classes,
                    likelihood[a],
                )
                for a in range(len(PROBES))
            ]

            probe = int(np.argmax(scores))

        elif policy == "optimized_fixed":

            probe = fixed[step]

        elif policy == "round_robin":

            probe = step % len(PROBES)

        else:

            raise ValueError(policy)

        # ----------------------------------------
        # Retrieve or construct Qiskit circuit
        # ----------------------------------------

        key = (
            probe,
            fault,
            rate,
        )

        if key not in cache:

            cache[key] = make_aer_circuit(
                probe,
                fault,
                rate,
            )

        qc = cache[key]

        # ----------------------------------------
        # ACTUAL QISKIT AER EXECUTION
        # ----------------------------------------

        counts = aer.run(
            qc,
            shots=batch,
            seed_simulator=int(
                seed_table[step, probe]
            ),
        ).result().get_counts()

        # ----------------------------------------
        # Convert quantum counts to Pauli outcomes
        # ----------------------------------------

        nminus = negative_outcomes(
            counts,
            PROBES[probe][2],
        )

        # ----------------------------------------
        # Bayesian update
        # ----------------------------------------

        weights = (
            posterior
            * likelihood[probe, nminus]
        )

        normalizer = weights.sum()

        if normalizer <= 0:

            raise RuntimeError(
                "Observation impossible under inference model"
            )

        posterior = weights / normalizer

        label_posterior = (
            posterior @ classes
        )

        prediction = int(
            np.argmax(label_posterior)
        )

        confidence = float(
            label_posterior[prediction]
        )

        used = (step + 1) * batch

        if trace:

            history.append((
                used,
                PROBES[probe][0],
                nminus,
                FAULTS[prediction],
                confidence,
            ))

        # ----------------------------------------
        # REAL STOPPING
        # ----------------------------------------

        if confidence >= CONFIDENCE:

            # Stop quantum execution immediately.
            # No more shots after acceptance.
            break

    accepted = int(
        confidence >= CONFIDENCE
    )

    true_idx = FAULTS.index(fault)

    correct = int(
        bool(accepted)
        and prediction == true_idx
    )

    wrong = int(
        bool(accepted)
        and prediction != true_idx
    )

    # Realized operational loss, with explicit
    # shot consumption and abstention cost.

    loss = (
        wrong
        + (1 - accepted) * ABSTAIN_LOSS
        + used * SHOT_COST
    )

    metrics = {

        "accepted": accepted,

        "correct_accepted": correct,

        "wrong_accepted": wrong,

        "used_shots": used,

        "predicted": (
            FAULTS[prediction]
            if accepted
            else "ABSTAIN"
        ),

        "posterior_confidence": confidence,

        "operational_loss": loss,
    }

    return metrics, history


# ============================================================
# 12. MAIN BENCHMARK
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repeats",
        type=int,
        default=4,
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

    parser.add_argument(
        "--demo",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        args.repeats < 1
        or args.batch < 1
        or args.budget % args.batch != 0
    ):

        parser.error(
            "repeats and batch must be positive; "
            "budget must be divisible by batch"
        )

    batches = args.budget // args.batch

    # ----------------------------------------
    # Generate actual Qiskit-derived signatures
    # ----------------------------------------

    sens = sensitivity_matrix()

    prior, classes, likelihood = make_model(
        sens,
        args.batch,
    )

    fixed = optimized_fixed_schedule(
        sens,
        batches,
    )

    print(
        "\n=== QISKIT-DERIVED SIGNATURE MATRIX ==="
    )

    print(
        pd.DataFrame(
            sens,
            index=[p[0] for p in PROBES],
            columns=FAULTS,
        )
    )

    print(
        "\n=== OPTIMIZED FIXED SCHEDULE ==="
    )

    print(
        [PROBES[i][0] for i in fixed]
    )

    print(
        "\nStatic allocation optimized before test observations."
    )

    # ----------------------------------------
    # Scenarios
    # ----------------------------------------

    scenarios = [("None", 0.0)] + [

        (fault, rate)

        for rate in (0.04, 0.10)

        for fault in FAULTS[1:]
    ]

    aer = AerSimulator(
        method="stabilizer"
    )

    cache = {}

    rows = []

    # ----------------------------------------
    # Paired Monte Carlo trials
    # ----------------------------------------

    for scen_id, (fault, rate) in enumerate(scenarios):

        for repeat in range(args.repeats):

            rng = np.random.default_rng(
                np.random.SeedSequence([
                    2026,
                    scen_id,
                    repeat,
                ])
            )

            # Same potential simulator seed whenever
            # policies select the same probe at the
            # same step within the same scenario/trial.

            seed_table = rng.integers(
                1,
                2**31 - 1,
                size=(
                    batches,
                    len(PROBES),
                ),
            )

            for policy in POLICIES:

                result, history = run_one(
                    policy,
                    fault,
                    rate,
                    seed_table,
                    args.batch,
                    args.budget,
                    sens,
                    prior,
                    classes,
                    likelihood,
                    fixed,
                    aer,
                    cache,
                    trace=(
                        args.demo
                        and scen_id == 1
                        and repeat == 0
                        and policy == "adaptive_eig"
                    ),
                )

                rows.append({

                    "scenario": (
                        f"{fault}_p{rate:.2f}"
                    ),

                    "fault": fault,

                    "true_rate": rate,

                    "repeat": repeat,

                    "policy": policy,

                    **result,
                })

                # Show one real measurement trace.

                if history:

                    print(
                        "\n=== LIVE AER DECISION TRACE ==="
                    )

                    for (
                        used,
                        name,
                        count,
                        prediction,
                        confidence,
                    ) in history:

                        print(
                            f"{used:3d} shots | "
                            f"{name:8s} | "
                            f"negatives={count:2d} | "
                            f"best={prediction:4s} | "
                            f"posterior={confidence:.3f}"
                        )

    df = pd.DataFrame(rows)

    # ----------------------------------------
    # Save complete trials
    # ----------------------------------------

    output = Path("results")

    output.mkdir(
        exist_ok=True
    )

    df.to_csv(
        output / "mission11_aer_trials.csv",
        index=False,
    )

    # ========================================================
    # 13. SUMMARY AND CONFIDENCE INTERVALS
    # ========================================================

    summary_rows = []

    for policy, group in df.groupby(
        "policy",
        sort=False,
    ):

        n = len(group)

        accepted = int(
            group["accepted"].sum()
        )

        good = int(
            group["correct_accepted"].sum()
        )

        bad = int(
            group["wrong_accepted"].sum()
        )

        coverage_ci = wilson(
            accepted,
            n,
        )

        accuracy_ci = wilson(
            good,
            accepted,
        )

        summary_rows.append({

            "policy": policy,

            "n": n,

            "coverage": accepted / n,

            "cov_ci_low": coverage_ci[0],

            "cov_ci_high": coverage_ci[1],

            "accuracy_if_accepted": (
                good / accepted
                if accepted
                else np.nan
            ),

            "acc_ci_low": accuracy_ci[0],

            "acc_ci_high": accuracy_ci[1],

            "wrong_per_trial": bad / n,

            "avg_shots": (
                group["used_shots"].mean()
            ),

            "mean_operational_loss": (
                group["operational_loss"].mean()
            ),
        })

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        output / "mission11_aer_summary.csv",
        index=False,
    )

    # ========================================================
    # 14. PER-SCENARIO AUDIT
    # ========================================================

    by_scenario = df.groupby(
        ["scenario", "policy"],
        sort=False,
    ).agg(

        coverage=(
            "accepted",
            "mean",
        ),

        wrong_per_trial=(
            "wrong_accepted",
            "mean",
        ),

        mean_loss=(
            "operational_loss",
            "mean",
        ),

        mean_shots=(
            "used_shots",
            "mean",
        ),
    ).reset_index()

    by_scenario.to_csv(
        output / "mission11_aer_scenarios.csv",
        index=False,
    )

    # ========================================================
    # 15. PAIRED BOOTSTRAP COMPARISON
    # ========================================================

    paired = df.pivot(
        index=[
            "scenario",
            "repeat",
        ],
        columns="policy",
        values="operational_loss",
    )

    paired["difference"] = (
        paired["adaptive_eig"]
        - paired["optimized_fixed"]
    )

    bootstrap_rng = np.random.default_rng(
        9001
    )

    # Stratify by scenario so the test scenario
    # composition stays balanced.

    scenario_blocks = [

        group["difference"].to_numpy()

        for _, group in paired.groupby(
            level="scenario"
        )
    ]

    bootstrap_means = []

    for _ in range(1500):

        samples = np.concatenate([

            bootstrap_rng.choice(
                block,
                size=len(block),
                replace=True,
            )

            for block in scenario_blocks
        ])

        bootstrap_means.append(
            float(samples.mean())
        )

    ci_low, ci_high = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    # ========================================================
    # 16. REPORT
    # ========================================================

    print(
        "\n=== REAL AER CLOSED-LOOP BENCHMARK ==="
    )

    print(
        summary.round(3).to_string(
            index=False
        )
    )

    print(
        "\n=== PER-SCENARIO AUDIT ==="
    )

    print(
        by_scenario.round(3).to_string(
            index=False
        )
    )

    print(
        "\n=== PAIRED OPERATIONAL LOSS COMPARISON ==="
    )

    print(
        "Adaptive EIG - Optimized Fixed"
    )

    print(
        f"Mean difference: "
        f"{paired['difference'].mean():.4f}"
    )

    print(
        f"Stratified bootstrap 95% CI: "
        f"[{ci_low:.4f}, {ci_high:.4f}]"
    )

    print(
        "Negative difference favors adaptive."
    )

    print(
        "If the interval contains zero, "
        "the result is inconclusive."
    )

    print(
        "\nOperational loss = "
        "wrong accepted + "
        "0.25 * abstain + "
        "0.001 * shots"
    )

    print(
        "\nModel limitations:"
    )

    print(
        "- Local Qiskit Aer simulator"
    )

    print(
        "- Single stochastic Pauli fault"
    )

    print(
        "- Ideal preparation and measurement"
    )

    print(
        "- No device calibration data"
    )

    print(
        "- No demonstrated quantum advantage"
    )

    print(
        "\nSaved results/mission11_aer_*.csv"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
