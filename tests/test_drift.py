import math
import unittest

import numpy as np

from qtrace.qec.drift import (
    fisher_p_two_sided,
    detect_shift,
)


class TestDriftDetection(unittest.TestCase):

    def test_identical_distributions(self):
        self.assertEqual(
            fisher_p_two_sided(5, 5, 5, 5),
            1.0,
        )

    def test_extreme_distribution_shift(self):
        self.assertAlmostEqual(
            fisher_p_two_sided(0, 10, 10, 0),
            2 / math.comb(20, 10),
        )

    def test_negative_counts_rejected(self):
        with self.assertRaises(ValueError):
            fisher_p_two_sided(-1, 3, 2, 4)

    def test_empty_row(self):
        self.assertEqual(
            fisher_p_two_sided(0, 0, 5, 5),
            1.0,
        )

    def test_insufficient_history(self):
        history = [np.zeros(8, dtype=int)]

        self.assertEqual(
            detect_shift(history, 2, 0.01),
            (False, 1.0),
        )

    def test_strong_shift_detected(self):
        old = np.zeros(8, dtype=int)
        recent = np.zeros(8, dtype=int)

        old[1] = 10
        recent[6] = 10

        alert, p_value = detect_shift(
            [old, old, recent, recent],
            window=2,
            alpha=0.01,
        )

        self.assertTrue(alert)
        self.assertLess(p_value, 0.01)

    def test_no_shift(self):
        counts = np.zeros(8, dtype=int)
        counts[1] = 10

        alert, p_value = detect_shift(
            [counts.copy() for _ in range(4)],
            window=2,
            alpha=0.01,
        )

        self.assertFalse(alert)
        self.assertEqual(p_value, 1.0)


if __name__ == "__main__":
    unittest.main()
