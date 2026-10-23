
import unittest

import numpy as np

from qtrace.qec.decoder_policies import (
    STANDARD,
    ALTERNATIVE,
    alternative_confidence,
    guarded_decoder,
    posterior_moments,
    cost_decoder,
)


class TestDecoderPolicies(unittest.TestCase):

    def test_standard_and_alternative_masks(self):
        np.testing.assert_array_equal(
            STANDARD, [0, 1, 4, 2]
        )
        np.testing.assert_array_equal(
            ALTERNATIVE, [7, 6, 3, 5]
        )

    def test_bayesian_confidence(self):
        self.assertEqual(
            alternative_confidence(0, 0), 0.5
        )
        self.assertEqual(
            alternative_confidence(3, 0), 0.9375
        )
        self.assertEqual(
            alternative_confidence(4, 0), 0.96875
        )

    def test_guarded_threshold(self):
        counts = np.zeros(8, dtype=int)
        counts[6] = 3

        table, _ = guarded_decoder(counts)
        self.assertEqual(table[1], 1)

        counts[6] = 4
        table, confidence = guarded_decoder(counts)

        self.assertEqual(table[1], 6)
        self.assertEqual(confidence[1], 0.96875)

    def test_guarded_preserves_standard_on_tie(self):
        counts = np.zeros(8, dtype=int)
        table, _ = guarded_decoder(counts)

        np.testing.assert_array_equal(
            table, STANDARD
        )

    def test_posterior_moments(self):
        gain, harm = posterior_moments(0, 0)

        self.assertAlmostEqual(gain, 0.25)
        self.assertAlmostEqual(harm, 0.25)

    def test_cost_greedy_equivalence(self):
        rng = np.random.default_rng(290001)

        for _ in range(100):
            counts = rng.integers(0, 100, size=8)

            greedy = np.where(
                counts[ALTERNATIVE] > counts[STANDARD],
                ALTERNATIVE,
                STANDARD,
            )

            np.testing.assert_array_equal(
                cost_decoder(counts, 1),
                greedy,
            )

    def test_cost_penalty_ordering(self):
        rng = np.random.default_rng(290002)

        for _ in range(100):
            counts = rng.integers(0, 100, size=8)

            greedy = cost_decoder(counts, 1)
            cost2 = cost_decoder(counts, 2)
            cost5 = cost_decoder(counts, 5)

            changed2 = cost2 != STANDARD
            changed5 = cost5 != STANDARD
            changed1 = greedy != STANDARD

            self.assertTrue(
                np.all(~changed5 | changed2)
            )
            self.assertTrue(
                np.all(~changed2 | changed1)
            )

    def test_standard_not_mutated(self):
        original = STANDARD.copy()

        counts = np.zeros(8, dtype=int)
        counts[6] = 20

        guarded_decoder(counts)
        cost_decoder(counts, 5)

        np.testing.assert_array_equal(
            STANDARD, original
        )


if __name__ == "__main__":
    unittest.main()
