
"""
Q-TRACE | Mission 10
Gate-Level Pauli Fault Localization via Heisenberg Backpropagation

Scope:
- 3-qubit Clifford circuits
- Single stochastic Pauli fault
- Ideal preparation and measurement
- No readout errors

Run:
    python notebooks/10_pauli_backpropagation.py
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import pauli_error


# ============================================================
# 1. CONFIGURATION
# ============================================================

N = 3
P_ERROR = 0.12
SHOTS = 4096
SEED = 2026

OUT = Path("results")

# Each tuple:
# (gate name, operated-on qubits)

GATES = [
    ("h", (0,)),
    ("cx", (0, 1)),
    ("h", (2,)),
    ("cx", (1, 2)),
    ("s", (1,)),
    ("cx", (0, 2)),
    ("h", (1,)),
    ("s", (2,)),
]

PREPARATIONS = (
    "zero",
    "plus",
    "yplus",
)


# ============================================================
# 2. FAULT AND PROBE REPRESENTATIONS
# ============================================================

@dataclass(frozen=True)
class Fault:

    gate_index: int
    qubit: int
    axis: str

    @property
    def name(self):
        return (
            f"g{self.gate_index:02d}_"
            f"{self.axis}{self.qubit}"
        )


@dataclass(frozen=True)
class Probe:

    preparation: str
    observable: str
    ideal: int

    @property
    def name(self):
        return (
            f"{self.preparation}_"
            f"{self.observable}"
        )


# ============================================================
# 3. CIRCUIT CONSTRUCTION
# ============================================================

def apply_gate(qc, gate):

    name, qubits = gate

    getattr(qc, name)(*qubits)


def prepare(qc, mode):

    if mode == "plus":

        qc.h(range(N))

    elif mode == "yplus":

        qc.h(range(N))
        qc.s(range(N))

    elif mode != "zero":

        raise ValueError(mode)


def circuit(
    mode,
    fault=None,
    stochastic=False,
):

    qc = QuantumCircuit(N)

    prepare(qc, mode)

    for idx, gate in enumerate(GATES):

        apply_gate(qc, gate)

        if (
            fault is not None
            and idx == fault.gate_index
        ):

            if stochastic:

                channel = pauli_error([
                    ("I", 1.0 - P_ERROR),
                    (fault.axis, P_ERROR),
                ])

                qc.append(
                    channel.to_instruction(),
                    [fault.qubit],
                )

            else:

                getattr(
                    qc,
                    fault.axis.lower()
                )(fault.qubit)

    return qc


# ============================================================
# 4. PAULI SYMPLECTIC REPRESENTATION
# ============================================================

def pauli_bits(label):
    """
    Convert a Pauli string into binary symplectic form.

    Qiskit labels:
        q_(N-1) ... q_1 q_0

    X = (1, 0)
    Z = (0, 1)
    Y = (1, 1)
    I = (0, 0)
    """

    x = np.array(
        [
            s in "XY"
            for s in label[::-1]
        ],
        dtype=np.uint8,
    )

    z = np.array(
        [
            s in "ZY"
            for s in label[::-1]
        ],
        dtype=np.uint8,
    )

    return x, z


def bits_to_pauli(x, z):

    labels = []

    for a, b in zip(
        x[::-1],
        z[::-1],
    ):

        if a and b:
            labels.append("Y")

        elif a:
            labels.append("X")

        elif b:
            labels.append("Z")

        else:
            labels.append("I")

    return "".join(labels)


# ============================================================
# 5. CLIFFORD PAULI PROPAGATION
# ============================================================

def clifford_update(x, z, gate):
    """
    Transform Pauli symplectic bits.

    Supported Clifford gates:
        H, S, CX

    Global Pauli phases are intentionally ignored.

    These x/z transformation rules apply to both
    forward and backward propagation for these gates.
    They are not sufficient to track Pauli signs.
    """

    x = x.copy()
    z = z.copy()

    name, q = gate

    if name == "h":

        a = q[0]

        x[a], z[a] = z[a], x[a]

    elif name == "s":

        a = q[0]

        z[a] ^= x[a]

    elif name == "cx":

        control, target = q

        x[target] ^= x[control]

        z[control] ^= z[target]

    else:

        raise NotImplementedError(name)

    return x, z


# ============================================================
# 6. HEISENBERG BACKPROPAGATION
# ============================================================

def heisenberg_at_fault(
    output_observable,
    gate_index,
):
    """
    Backpropagate output observable O through
    every gate AFTER the fault insertion point.

    If the fault is inserted after gate i:

        O_i = U_suffix^dagger O U_suffix

    The returned x/z bits represent O_i
    up to an irrelevant global Pauli phase.
    """

    x, z = pauli_bits(
        output_observable
    )

    suffix = GATES[
        gate_index + 1:
    ]

    for gate in reversed(suffix):

        x, z = clifford_update(
            x,
            z,
            gate,
        )

    return x, z


# ============================================================
# 7. EFFECTIVE OUTPUT PAULI
# ============================================================

def effective_output_pauli(fault):
    """
    Propagate a local Pauli fault forward
    through the remaining Clifford circuit.

    Multiple physical faults may result in
    the same effective output Pauli.

    Such faults cannot be distinguished by
    final-state measurements in this model.
    """

    if fault is None:

        return "I" * N

    x = np.zeros(
        N,
        dtype=np.uint8,
    )

    z = np.zeros(
        N,
        dtype=np.uint8,
    )

    q = fault.qubit

    x[q] = int(
        fault.axis in "XY"
    )

    z[q] = int(
        fault.axis in "ZY"
    )

    suffix = GATES[
        fault.gate_index + 1:
    ]

    for gate in suffix:

        x, z = clifford_update(
            x,
            z,
            gate,
        )

    return bits_to_pauli(x, z)


# ============================================================
# 8. PAULI FAULT SIGNATURE
# ============================================================

def flip_signature(fault, probe):
    """
    Return:
        0 if fault commutes with backpropagated observable
        1 if fault anticommutes

    For stochastic Pauli fault probability p:

        E_noisy = E_ideal * (1 - 2p * flip_signature)
    """

    if fault is None:

        return 0

    x, z = heisenberg_at_fault(
        probe.observable,
        fault.gate_index,
    )

    q = fault.qubit

    error_x = int(
        fault.axis in "XY"
    )

    error_z = int(
        fault.axis in "ZY"
    )

    anticommutation = (
        error_x * int(z[q])
        + error_z * int(x[q])
    ) % 2

    return int(
        anticommutation
    )


# ============================================================
# 9. GENERATE QUANTUM DIAGNOSTIC PROBES
# ============================================================

def generate_probes():
    """
    Generate all non-identity Pauli observables
    with deterministic ideal expectation +/-1.

    For the selected 3-qubit Clifford circuit,
    each stabilizer state has seven nonidentity
    Pauli stabilizer elements.
    """

    probes = []

    for mode in PREPARATIONS:

        state = Statevector.from_instruction(
            circuit(mode)
        )

        for letters in product(
            "IXYZ",
            repeat=N,
        ):

            label = "".join(
                letters
            )

            if label == "I" * N:
                continue

            expectation = float(
                np.real(
                    state.expectation_value(
                        Pauli(label)
                    )
                )
            )

            if np.isclose(
                abs(expectation),
                1.0,
                atol=1e-9,
            ):

                probes.append(
                    Probe(
                        mode,
                        label,
                        int(
                            round(expectation)
                        ),
                    )
                )

    return probes


# ============================================================
# 10. ENUMERATE GATE-LEVEL FAULT HYPOTHESES
# ============================================================

def build_faults():

    return [None] + [

        Fault(
            idx,
            qubit,
            axis,
        )

        for idx, (_, qubits) in enumerate(GATES)

        for qubit in qubits

        for axis in "XYZ"

    ]


def fault_name(fault):

    if fault is None:
        return "None"

    return fault.name


# ============================================================
# 11. BUILD SENSITIVITY MATRIX
# ============================================================

def signature_matrix(
    probes,
    faults,
):

    return np.array([

        [
            flip_signature(
                fault,
                probe,
            )
            for fault in faults
        ]

        for probe in probes

    ], dtype=np.uint8)


# ============================================================
# 12. IDENTIFIABILITY GROUPS
# ============================================================

def grouped_hypotheses(
    matrix,
    probe_indices,
    faults,
):
    """
    Group fault hypotheses with identical
    measurement signatures.
    """

    groups = defaultdict(list)

    for j, fault in enumerate(faults):

        signature = tuple(

            int(matrix[i, j])

            for i in probe_indices

        )

        groups[signature].append(
            fault_name(fault)
        )

    return list(
        groups.values()
    )


# ============================================================
# 13. GREEDY PROBE SELECTION
# ============================================================

def greedy_probe_selection(matrix):
    """
    Greedy set-cover algorithm.

    Universe:
        All pairs of distinguishable fault hypotheses.

    A probe covers pair (a,b) if their signatures
    differ under that probe.

    Select probes until every distinguishable pair
    has been separated.

    This finds a heuristic small probe set,
    not a guaranteed globally minimum set.
    """

    n_probes, n_faults = (
        matrix.shape
    )

    all_pairs = {

        (a, b)

        for a in range(n_faults)

        for b in range(
            a + 1,
            n_faults,
        )

        if np.any(
            matrix[:, a] != matrix[:, b]
        )
    }

    uncovered = all_pairs.copy()

    selected = []

    while uncovered:

        scores = [

            -1 if i in selected

            else sum(

                matrix[i, a] != matrix[i, b]

                for a, b in uncovered

            )

            for i in range(n_probes)

        ]

        best = int(
            np.argmax(scores)
        )

        assert scores[best] > 0

        selected.append(best)

        uncovered = {

            (a, b)

            for a, b in uncovered

            if matrix[best, a] == matrix[best, b]

        }

    return (
        selected,
        len(all_pairs),
    )


# ============================================================
# 14. PAULI MEASUREMENT BASIS ROTATION
# ============================================================

def measure_pauli(qc, observable):
    """
    Measure Pauli product observable.

    Z: computational basis
    X: H before measurement
    Y: Sdg followed by H before measurement
    I: ignored during parity calculation
    """

    for q, axis in enumerate(
        observable[::-1]
    ):

        if axis == "X":

            qc.h(q)

        elif axis == "Y":

            qc.sdg(q)
            qc.h(q)

    qc.measure_all()

    return qc


def expectation_from_counts(
    counts,
    observable,
):

    total = sum(
        counts.values()
    )

    weighted = 0

    for bitstring, frequency in counts.items():

        bits = bitstring.replace(
            " ",
            "",
        )

        parity = sum(

            int(bits[-1 - q])

            for q, axis in enumerate(
                observable[::-1]
            )

            if axis != "I"

        )

        eigenvalue = (
            -1 if parity % 2
            else 1
        )

        weighted += (
            eigenvalue * frequency
        )

    return weighted / total


# ============================================================
# 15. STATEVECTOR AND AER VALIDATION
# ============================================================

def run_validation(
    probes,
    faults,
    selected,
):

    # Check several fault hypotheses and
    # all selected diagnostic probes.

    test_faults = (
        [None]
        + [
            f for f in faults
            if f is not None
        ][::7][:6]
    )

    test_probes = [
        probes[k]
        for k in selected
    ]

    # --------------------------------------------
    # Deterministic fault test with Statevector
    # --------------------------------------------

    for fault in test_faults:

        for probe in test_probes:

            state = Statevector.from_instruction(
                circuit(
                    probe.preparation,
                    fault,
                    stochastic=False,
                )
            )

            actual = float(
                np.real(
                    state.expectation_value(
                        Pauli(probe.observable)
                    )
                )
            )

            predicted = (
                probe.ideal
                * (
                    -1
                    if flip_signature(fault, probe)
                    else 1
                )
            )

            assert np.isclose(
                actual,
                predicted,
                atol=1e-9,
            ), (
                fault_name(fault),
                probe.name,
                actual,
                predicted,
            )

    # --------------------------------------------
    # Stochastic Pauli noise with Qiskit Aer
    # --------------------------------------------

    experiments = []
    metadata = []

    for fault in test_faults:

        for probe in test_probes:

            qc = circuit(
                probe.preparation,
                fault,
                stochastic=True,
            )

            qc = measure_pauli(
                qc,
                probe.observable,
            )

            experiments.append(qc)

            metadata.append(
                (fault, probe)
            )

    simulator = AerSimulator(
        method="stabilizer"
    )

    result = simulator.run(
        experiments,
        shots=SHOTS,
        seed_simulator=SEED,
    ).result()

    rows = []

    for index, (fault, probe) in enumerate(metadata):

        counts = result.get_counts(
            index
        )

        observed = expectation_from_counts(
            counts,
            probe.observable,
        )

        predicted = (
            probe.ideal
            * (
                1.0
                - 2.0 * P_ERROR
                * flip_signature(fault, probe)
            )
        )

        rows.append({
            "fault": fault_name(fault),
            "probe": probe.name,
            "theory": predicted,
            "aer": observed,
            "abs_error": abs(
                observed - predicted
            ),
        })

    validation = pd.DataFrame(rows)

    assert (
        validation["abs_error"] < 0.07
    ).all(), "Unexpected Aer mismatch"

    return validation


# ============================================================
# 16. MAIN EXPERIMENT
# ============================================================

def main():

    OUT.mkdir(
        exist_ok=True
    )

    probes = generate_probes()

    faults = build_faults()

    matrix = signature_matrix(
        probes,
        faults,
    )

    selected, distinguishable_pairs = (
        greedy_probe_selection(matrix)
    )

    # --------------------------------------------
    # Structural equivalence classes
    # --------------------------------------------

    effective_groups = defaultdict(list)

    for fault in faults:

        output_pauli = (
            effective_output_pauli(fault)
        )

        effective_groups[
            output_pauli
        ].append(
            fault_name(fault)
        )

    observed_groups = grouped_hypotheses(
        matrix,
        range(len(probes)),
        faults,
    )

    selected_groups = grouped_hypotheses(
        matrix,
        selected,
        faults,
    )

    # The full probe library must distinguish
    # all different effective output Paulis.

    assert (
        len(observed_groups)
        == len(effective_groups)
    ), (
        "Probe library does not separate all "
        "distinct output Pauli errors"
    )

    assert (
        len(selected_groups)
        == len(observed_groups)
    )

    # --------------------------------------------
    # Save complete signature matrix
    # --------------------------------------------

    matrix_df = pd.DataFrame(

        matrix,

        index=[
            p.name for p in probes
        ],

        columns=[
            fault_name(f)
            for f in faults
        ],
    )

    matrix_df.to_csv(
        OUT / "mission10_signatures.csv"
    )

    # --------------------------------------------
    # Save selected probes
    # --------------------------------------------

    chosen = pd.DataFrame([

        {
            "order": order,

            "preparation":
                probes[i].preparation,

            "observable":
                probes[i].observable,

            "ideal_expectation":
                probes[i].ideal,
        }

        for order, i in enumerate(
            selected,
            start=1,
        )

    ])

    chosen.to_csv(
        OUT / "mission10_selected_probes.csv",
        index=False,
    )

    # --------------------------------------------
    # Save structurally equivalent fault classes
    # --------------------------------------------

    classes = pd.DataFrame([

        {
            "effective_output_pauli": p,

            "n_hypotheses": len(members),

            "hypotheses": ", ".join(
                members
            ),
        }

        for p, members in effective_groups.items()

    ])

    classes.to_csv(
        OUT / "mission10_equivalence_classes.csv",
        index=False,
    )

    # --------------------------------------------
    # Verify analytical results with Qiskit
    # --------------------------------------------

    validation = run_validation(
        probes,
        faults,
        selected,
    )

    validation.to_csv(
        OUT / "mission10_aer_validation.csv",
        index=False,
    )

    # --------------------------------------------
    # Report results
    # --------------------------------------------

    print(
        "\n=== Q-TRACE MISSION 10 ==="
    )

    print(
        "Physical fault hypotheses:",
        len(faults),
    )

    print(
        "Candidate quantum probes:",
        len(probes),
    )

    print(
        "Distinguishable pairs:",
        distinguishable_pairs,
    )

    print(
        "Identifiable output-error classes:",
        len(effective_groups),
    )

    print(
        "Selected probe settings:",
        len(selected),
    )

    print(
        "\n=== GREEDY-SELECTED PROBES ==="
    )

    print(
        chosen.to_string(index=False)
    )

    print(
        "\n=== NON-IDENTIFIABLE "
        "GATE-LEVEL FAULT CLASSES ==="
    )

    print(
        classes[
            classes.n_hypotheses > 1
        ].to_string(index=False)
    )

    print(
        "\n=== AER VALIDATION ==="
    )

    print(
        "Circuits verified:",
        len(validation),
    )

    print(
        "Maximum absolute error:",
        round(
            validation.abs_error.max(),
            6,
        ),
    )

    print(
        "PASS: Analytical signatures agree "
        "with Statevector and Aer."
    )

    print(
        "\nIMPORTANT: Equal effective output "
        "Paulis are structurally indistinguishable "
        "by final-state measurements in this model."
    )

    print(
        "\nSaved results/mission10_*.csv"
    )


# ============================================================
# 17. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
