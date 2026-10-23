
import unittest

from qtrace.diagnosis.pauli import (
    fault_pauli,
    expectation_from_counts,
)


class TestPauliUtilities(unittest.TestCase):

    def test_qiskit_qubit_order(self):
        self.assertEqual(fault_pauli(("X", 0)).to_label(), "IX")
        self.assertEqual(fault_pauli(("X", 1)).to_label(), "XI")
        self.assertEqual(fault_pauli(("Z", 0)).to_label(), "IZ")
        self.assertEqual(fault_pauli(("Z", 1)).to_label(), "ZI")

    def test_expectation_zz(self):
        counts = {"00": 70, "11": 20, "01": 10}
        self.assertAlmostEqual(
            expectation_from_counts(counts, "ZZ"),
            0.8,
        )

    def test_expectation_single_qubit(self):
        counts = {"00": 70, "01": 30}

        self.assertAlmostEqual(
            expectation_from_counts(counts, "IZ"),
            0.4,
        )
        self.assertAlmostEqual(
            expectation_from_counts(counts, "ZI"),
            1.0,
        )

    def test_identity_observable(self):
        counts = {"00": 30, "11": 70}
        self.assertAlmostEqual(
            expectation_from_counts(counts, "II"),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
