from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from qiskit import (
    QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
)
from qiskit_aer import AerSimulator
from qiskit_aer.noise import pauli_error


STATES = ["0", "1", "+", "-", "+i", "-i"]

# Error mask: bit i means X on data qubit i.
# Index order: I, X0, X1, X0X1, X2, X0X2, X1X2, X0X1X2.
ENVIRONMENTS = {
    "single_dominant": [
        .70, .08, .08, .01, .10, .01, .01, .01
    ],
    "correlated": [
        .65, .03, .03, .22, .02, .02, .02, .01
    ],
}

STANDARD = {0: 0, 1: 1, 2: 4, 3: 2}


def syndrome(mask):
    b0 = mask & 1
    b1 = (mask >> 1) & 1
    b2 = (mask >> 2) & 1
    return (b0 ^ b1) + 2 * (b1 ^ b2)


def choose_decoder(probabilities):
    """Choose most probable X mask compatible with each syndrome."""
    table = {}
    for syn in range(4):
        compatible = [e for e in range(8) if syndrome(e) == syn]
        best = max(probabilities[e] for e in compatible)
        tied = [
            e for e in compatible
            if abs(probabilities[e] - best) < 1e-12
        ]
        # Predeclared tie-breaking: prefer standard correction.
        table[syn] = (
            STANDARD[syn] if STANDARD[syn] in tied else min(tied)
        )
    return table


def learn_decoder(counts):
    # Symmetric Dirichlet(1,...,1) prior.
    posterior = np.ones(8, dtype=float)
    for bits, count in counts.items():
        posterior[int(bits.replace(" ", ""), 2)] += count
    posterior /= posterior.sum()
    return choose_decoder(posterior), posterior


def make_channel(probabilities):
    assert np.isclose(sum(probabilities), 1.0)
    terms = []
    for mask, probability in enumerate(probabilities):
        # Qiskit labels are ordered q2 q1 q0.
        label = "".join(
            "X" if mask & (1 << q) else "I"
            for q in [2, 1, 0]
        )
        terms.append((label, probability))
    return pauli_error(terms).to_instruction()


def calibration_circuit(channel):
    qc = QuantumCircuit(3, 3)
    qc.append(channel, [0, 1, 2])
    qc.measure(range(3), range(3))
    return qc


def qec_circuit(state, channel, table):
    data = QuantumRegister(3, "q")
    anc = QuantumRegister(2, "anc")
    out = ClassicalRegister(3, "out")
    syn = ClassicalRegister(2, "syn")
    qc = QuantumCircuit(data, anc, out, syn)

    if state == "1":
        qc.x(data[0])
    elif state in ["+", "-", "+i", "-i"]:
        qc.h(data[0])
        if state == "-":
            qc.z(data[0])
        elif state == "+i":
            qc.s(data[0])
        elif state == "-i":
            qc.sdg(data[0])

    qc.cx(data[0], data[1])
    qc.cx(data[0], data[2])
    qc.barrier()
    qc.append(channel, list(data))
    qc.barrier()

    qc.cx(data[0], anc[0])
    qc.cx(data[1], anc[0])
    qc.cx(data[1], anc[1])
    qc.cx(data[2], anc[1])
    qc.measure(anc, syn)

    # Only measured syndrome chooses the action during execution.
    for value, mask in table.items():
        if mask:
            with qc.if_test((syn, value)):
                for qubit in range(3):
                    if mask & (1 << qubit):
                        qc.x(data[qubit])

    qc.barrier()
    qc.cx(data[0], data[2])
    qc.cx(data[0], data[1])

    if state in ["+", "-"]:
        qc.h(data[0])
    elif state in ["+i", "-i"]:
        qc.sdg(data[0])
        qc.h(data[0])

    qc.measure(data, out)
    return qc


def score(counts, state):
    expected = int(state in ["1", "-", "-i"])
    shots = sum(counts.values())
    full = logical = 0

    for bits, count in counts.items():
        value = int(bits.replace(" ", ""), 2) & 7
        logical += count * ((value & 1) == expected)
        full += count * (value == expected)

    return {
        "shots": shots,
        "logical_success": logical / shots,
        "decoded_full_success": full / shots,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-shots", type=int, default=128)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=230031)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.smoke:
        args.repeats = 2
        args.shots = 128
    if min(args.calibration_shots, args.shots, args.repeats) < 1:
        raise ValueError("Shots and repeats must be positive.")

    output = Path(args.output or (
        "results/calibration_decoder_23_smoke"
        if args.smoke else "results/calibration_decoder_23"
    ))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    protocol = {
        **vars(args),
        "environments": ENVIRONMENTS,
        "states": STATES,
        "standard_decoder": STANDARD,
        "prior": "Dirichlet(1,...,1)",
        "calibration": "Independent 000 probes through the same X-only channel",
        "primary_metric": "decoded_full_success",
        "resource_accounting": {
            "standard_shots_per_repeat": 6 * args.shots,
            "learned_shots_per_repeat": (
                args.calibration_shots + 6 * args.shots
            ),
            "oracle_shots_per_repeat": 6 * args.shots,
        },
        "limitations": [
            "Aer mechanism pilot; not a hardware run.",
            "Ideal gates, readout and syndrome extraction outside injected channel.",
            "Channel stationary between calibration and test.",
            "Learned method pays extra calibration shots.",
            "Not an equal-total-budget comparison.",
            "Oracle is privileged and unavailable in deployment.",
            "No novelty claim for maximum-probability decoding.",
        ],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    backend = AerSimulator()
    rows, calibration_logs = [], []

    for env_index, (name, probabilities) in enumerate(ENVIRONMENTS.items()):
        channel = make_channel(probabilities)
        cal = transpile(
            calibration_circuit(channel),
            backend,
            optimization_level=0,
        )
        oracle = choose_decoder(probabilities)

        for rep in range(args.repeats):
            seeds = np.random.SeedSequence(
                [args.seed, env_index, rep]
            ).generate_state(3)

            cal_counts = backend.run(
                cal,
                shots=args.calibration_shots,
                seed_simulator=int(seeds[0]),
            ).result().get_counts()

            # This function never receives environment probabilities.
            learned, posterior = learn_decoder(cal_counts)

            calibration_logs.append({
                "environment": name,
                "rep": rep,
                "calibration_counts": cal_counts,
                "posterior_mean": posterior.tolist(),
                "locked_decoder": learned,
            })

            # Persist learned decoders before executing their test circuits.
            (output / "calibration.json").write_text(
                json.dumps(calibration_logs, indent=2)
            )

            configs = {
                "standard": STANDARD,
                "learned": learned,
                "oracle": oracle,
            }
            circuits, labels = [], []
            for method, table in configs.items():
                for state in STATES:
                    circuits.append(qec_circuit(state, channel, table))
                    labels.append((method, state))

            compiled = transpile(
                circuits,
                backend,
                optimization_level=0,
                seed_transpiler=int(seeds[1]),
            )
            result = backend.run(
                compiled,
                shots=args.shots,
                seed_simulator=int(seeds[2]),
            ).result()

            for index, (method, state) in enumerate(labels):
                table = configs[method]
                # Evaluator only: exact probability of cancelling the X mask.
                exact_mask_success = sum(
                    probabilities[e]
                    for e in range(8)
                    if table[syndrome(e)] == e
                )
                rows.append({
                    "environment": name,
                    "rep": rep,
                    "method": method,
                    "state": state,
                    **score(result.get_counts(index), state),
                    "exact_mask_success_reference": exact_mask_success,
                    # Calibration is paid once per repeat, shared by six states.
                    "calibration_shots_amortized": (
                        args.calibration_shots / len(STATES)
                        if method == "learned" else 0.0
                    ),
                })

            print(
                f"{name}: repeat {rep + 1}/{args.repeats}; "
                f"decoder={learned}",
                flush=True,
            )

        pd.DataFrame(rows).to_csv(output / "trials.csv", index=False)

    df = pd.DataFrame(rows)
    summary = df.groupby(["environment", "method"])[[
        "logical_success",
        "decoded_full_success",
        "calibration_shots_amortized",
    ]].mean()
    summary.to_csv(output / "summary.csv")

    # Cluster by calibration repeat, because all six states share its decoder.
    paired = df.groupby(
        ["environment", "rep", "method"]
    ).decoded_full_success.mean().unstack("method")

    rng = np.random.default_rng(args.seed + 1)
    comparisons = []
    for environment, group in paired.groupby(level="environment"):
        for baseline in ["standard", "oracle"]:
            delta = (group["learned"] - group[baseline]).to_numpy()
            draws = rng.choice(
                delta, size=(5000, len(delta)), replace=True
            ).mean(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
            comparisons.append({
                "environment": environment,
                "method": "learned",
                "baseline": baseline,
                "delta_success": delta.mean(),
                "ci_low": low,
                "ci_high": high,
                "calibration_repeats": len(delta),
            })

    comp = pd.DataFrame(comparisons)
    comp.to_csv(output / "comparisons.csv", index=False)

    print("\nSummary:")
    print(summary.to_string())
    print("\nPositive delta favors learned decoder:")
    print(comp.to_string(index=False))
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()