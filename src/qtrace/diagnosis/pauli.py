
"""Pauli utilities shared by Q-TRACE diagnosis experiments."""

from qiskit.quantum_info import Pauli


def fault_pauli(fault):
    """Represent a single-qubit Pauli fault on a two-qubit register.

    Qiskit labels are ordered q1, q0.
    """
    gate, qubit = fault

    labels = ["I", "I"]
    labels[1 - qubit] = gate

    return Pauli("".join(labels))


def expectation_from_counts(counts, observable):
    """Estimate a Pauli expectation from measurement-basis counts.

    Assumes the circuit has already rotated the measured X
    observables into the computational basis.
    """
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
