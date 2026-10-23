
import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import pauli_error

# ==================================================
# Q-TRACE | Mission 01: Pauli Fault Identifiability
# ==================================================

P_ERROR = 0.10
SHOTS = 4096
SEED = 42

OBSERVABLES = ["ZZ", "XX"]

# (Pauli error, physical qubit)
HYPOTHESES = {
    "None": None,
    "X0": ("X", 0),
    "X1": ("X", 1),
    "Z0": ("Z", 0),
    "Z1": ("Z", 1),
}


# --------------------------------------------------
# 1. Prepare a Bell state
# --------------------------------------------------

def bell_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


# --------------------------------------------------
# 2. Insert a stochastic Pauli error
# --------------------------------------------------

def build_experiment(fault, observable):
    qc = bell_circuit()

    if fault is not None:
        gate, qubit = fault

        error = pauli_error([
            ("I", 1 - P_ERROR),
            (gate, P_ERROR),
        ])

        # Error occurs after Bell-state preparation
        qc.append(error.to_instruction(), [qubit])

    # Measurement basis
    if observable == "XX":
        qc.h([0, 1])
    elif observable != "ZZ":
        raise ValueError("Unsupported observable")

    qc.measure_all()
    return qc


# --------------------------------------------------
# 3. Compute expectation from measurement counts
# --------------------------------------------------

def expectation_from_counts(counts):
    total = sum(counts.values())
    weighted_sum = 0

    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")

        # Eigenvalue of Z tensor Z:
        # even parity -> +1, odd parity -> -1
        eigenvalue = (-1) ** bits.count("1")
        weighted_sum += eigenvalue * count

    return weighted_sum / total


# --------------------------------------------------
# 4. Analytical prediction using Pauli commutation
# --------------------------------------------------

def theoretical_expectation(fault, observable):
    if fault is None:
        return 1.0

    gate, qubit = fault

    # Qiskit Pauli labels are ordered q1, q0
    labels = ["I", "I"]
    labels[1 - qubit] = gate

    error_pauli = Pauli("".join(labels))
    target_pauli = Pauli(observable)

    if error_pauli.anticommutes(target_pauli):
        return 1 - 2 * P_ERROR

    return 1.0


# --------------------------------------------------
# 5. Validate ideal Bell-state correlations
# --------------------------------------------------

ideal_state = Statevector.from_instruction(bell_circuit())

for observable in OBSERVABLES:
    ideal_value = ideal_state.expectation_value(
        Pauli(observable)
    ).real

    assert np.isclose(ideal_value, 1.0)


# --------------------------------------------------
# 6. Run experiments on Qiskit Aer
# --------------------------------------------------

simulator = AerSimulator(method="stabilizer")

experiments = []
labels = []

for name, fault in HYPOTHESES.items():
    for observable in OBSERVABLES:
        experiments.append(
            build_experiment(fault, observable)
        )
        labels.append((name, fault, observable))

result = simulator.run(
    experiments,
    shots=SHOTS,
    seed_simulator=SEED,
).result()

rows = []

for index, (name, fault, observable) in enumerate(labels):
    counts = result.get_counts(index)

    measured = expectation_from_counts(counts)
    expected = theoretical_expectation(fault, observable)

    rows.append({
        "Fault": name,
        "Observable": observable,
        "Theory": expected,
        "Simulation": measured,
        "Abs Error": abs(expected - measured),
    })

df = pd.DataFrame(rows)

print("\nQ-TRACE — Pauli Fault Identifiability\n")
print(df.to_string(index=False))

# Sampling-tolerant validation
assert (df["Abs Error"] < 0.06).all(), \
    "Simulation differs unexpectedly from theory"

print("\nPASS: All experiments agree with theory.")


# --------------------------------------------------
# 7. Compare fault signatures
# --------------------------------------------------

signatures = df.pivot(
    index="Fault",
    columns="Observable",
    values="Theory",
)

print("\nTheoretical fault signatures:")
print(signatures)

print("\nIdentifiability analysis:")
print("X0 and X1 have identical signatures.")
print("Z0 and Z1 have identical signatures.")
print("Error type is distinguishable; location is not.")
