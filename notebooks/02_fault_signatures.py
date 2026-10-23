
"""
Q-TRACE | Mission 02
Pauli Fault Signature Matrix & Identifiability

Requires: qiskit, qiskit-aer, numpy, pandas
"""

from itertools import combinations

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import pauli_error


P_ERROR = 0.10
SHOTS = 4096
SEED = 42

FAULTS = {
    "None": None,
    "X0": ("X", 0),
    "X1": ("X", 1),
    "Z0": ("Z", 0),
    "Z1": ("Z", 1),
}

# Each probe consists of (preparation, observable).
# Qiskit uses q1-q0 ordering for Pauli labels.
PROBES = {
    "bell_ZZ": ("bell", "ZZ"),
    "bell_XX": ("bell", "XX"),
    "zero_Z0": ("zero", "IZ"),
    "zero_Z1": ("zero", "ZI"),
    "plus_X0": ("plus", "IX"),
    "plus_X1": ("plus", "XI"),
}


# --------------------------------------------------
# 1. Prepare different quantum input states
# --------------------------------------------------

def prepare_state(mode):
    qc = QuantumCircuit(2)

    if mode == "bell":
        qc.h(0)
        qc.cx(0, 1)

    elif mode == "plus":
        qc.h([0, 1])

    elif mode != "zero":
        raise ValueError(f"Unknown preparation: {mode}")

    return qc


# --------------------------------------------------
# 2. Represent a local Pauli error
# --------------------------------------------------

def fault_pauli(fault):
    gate, qubit = fault

    labels = ["I", "I"]
    labels[1 - qubit] = gate

    return Pauli("".join(labels))


# --------------------------------------------------
# 3. Calculate theoretical observable expectation
# --------------------------------------------------

def theoretical_expectation(mode, observable, fault):
    state = Statevector.from_instruction(
        prepare_state(mode)
    )

    target = Pauli(observable)

    ideal = float(
        np.real(state.expectation_value(target))
    )

    if fault is None:
        return ideal

    error = fault_pauli(fault)

    if error.anticommutes(target):
        return ideal * (1 - 2 * P_ERROR)

    return ideal


# --------------------------------------------------
# 4. Build an experiment for Qiskit Aer
# --------------------------------------------------

def build_experiment(mode, observable, fault):
    qc = prepare_state(mode)

    if fault is not None:
        gate, qubit = fault

        noise = pauli_error([
            ("I", 1 - P_ERROR),
            (gate, P_ERROR),
        ])

        qc.append(noise.to_instruction(), [qubit])

    # Rotate X measurements into the Z basis.
    for qubit, operator in enumerate(observable[::-1]):
        if operator == "X":
            qc.h(qubit)

    qc.measure_all()

    return qc


# --------------------------------------------------
# 5. Convert counts to Pauli expectation
# --------------------------------------------------

def expectation_from_counts(counts, observable):
    total = sum(counts.values())
    weighted = 0

    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")

        parity = sum(
            int(bit)
            for bit, op in zip(bits, observable)
            if op != "I"
        )

        eigenvalue = (-1) ** parity
        weighted += eigenvalue * count

    return weighted / total


# --------------------------------------------------
# 6. Construct theoretical sensitivity matrix
# --------------------------------------------------

matrix = pd.DataFrame(
    index=list(PROBES.keys()),
    columns=list(FAULTS.keys()),
    dtype=float,
)

for probe_name, (mode, observable) in PROBES.items():
    for fault_name, fault in FAULTS.items():

        matrix.loc[probe_name, fault_name] = (
            theoretical_expectation(
                mode, observable, fault
            )
        )

print("\n=== THEORETICAL FAULT SIGNATURE MATRIX ===")
print(matrix.round(3))


# --------------------------------------------------
# 7. Find observationally equivalent faults
# --------------------------------------------------

def fault_groups(selected_probes):
    groups = {}

    for fault_name in FAULTS:
        signature = tuple(
            round(float(matrix.loc[p, fault_name]), 8)
            for p in selected_probes
        )

        groups.setdefault(signature, []).append(
            fault_name
        )

    return list(groups.values())


print("\n=== BELL-ONLY IDENTIFIABILITY ===")
print(fault_groups(["bell_ZZ", "bell_XX"]))

print("\n=== ALL-PROBE IDENTIFIABILITY ===")
print(fault_groups(list(PROBES)))

assert len(fault_groups(["bell_ZZ", "bell_XX"])) == 3
assert len(fault_groups(list(PROBES))) == len(FAULTS)


# --------------------------------------------------
# 8. Find the smallest identifying subset
#    within our six candidate measurement settings
# --------------------------------------------------

best_subset = None

for size in range(1, len(PROBES) + 1):
    for subset in combinations(PROBES.keys(), size):

        if len(fault_groups(subset)) == len(FAULTS):
            best_subset = subset
            break

    if best_subset is not None:
        break

print("\n=== MINIMUM IDENTIFYING SUBSET ===")
print("Number of settings:", len(best_subset))
print("Selected probes:", best_subset)


# --------------------------------------------------
# 9. Validate all signatures using Qiskit Aer
# --------------------------------------------------

simulator = AerSimulator(method="stabilizer")

circuits = []
metadata = []

for probe_name, (mode, observable) in PROBES.items():
    for fault_name, fault in FAULTS.items():

        circuits.append(
            build_experiment(mode, observable, fault)
        )

        metadata.append(
            (probe_name, observable, fault_name)
        )

result = simulator.run(
    circuits,
    shots=SHOTS,
    seed_simulator=SEED,
).result()

sim_matrix = pd.DataFrame(
    index=matrix.index,
    columns=matrix.columns,
    dtype=float,
)

for index, (probe_name, observable, fault_name) in enumerate(metadata):

    counts = result.get_counts(index)

    sim_matrix.loc[probe_name, fault_name] = (
        expectation_from_counts(counts, observable)
    )

print("\n=== SIMULATED SIGNATURE MATRIX ===")
print(sim_matrix.round(3))

max_error = float(
    (sim_matrix - matrix).abs().to_numpy().max()
)

print(f"\nMaximum absolute error: {max_error:.6f}")

assert max_error < 0.06

print("PASS: Analytical and simulated signatures agree.")
print("PASS: All five hypotheses are identifiable using the full probe set.")
