
"""Tests for Q-TRACE repetition-code decoder utilities."""

import unittest

from qtrace.qec.repetition import (
    STANDARD,
    syndrome,
)
from qtrace.qec.decoder import learn_decoder


class TestRepetitionDecoder(unittest.TestCase):

    def test_syndrome_mapping(self):
        expected = [0, 1, 3, 2, 2, 3, 1, 0]

        actual = [
            syndrome(mask)
            for mask in range(8)
        ]

        self.assertEqual(actual, expected)

    def test_standard_decoder(self):
        self.assertEqual(
            STANDARD,
            [0, 1, 4, 2],
        )

        for s, mask in enumerate(STANDARD):
            self.assertEqual(syndrome(mask), s)

    def test_single_x_recovery(self):
        for mask in [1, 2, 4]:
            with self.subTest(mask=mask):
                s = syndrome(mask)
                recovered = mask ^ STANDARD[s]
                self.assertEqual(recovered, 0)

    def test_all_masks_have_two_candidates(self):
        for s in range(4):
            candidates = [
                mask
                for mask in range(8)
                if syndrome(mask) == s
            ]

            self.assertEqual(
                len(candidates), 2
            )

            self.assertEqual(
                candidates[0] ^ candidates[1],
                7,
            )

    def test_empty_calibration_uses_standard(self):
        learned, evidence = learn_decoder({})

        self.assertEqual(learned, STANDARD)
        self.assertEqual(len(evidence), 4)

    def test_ties_prefer_standard(self):
        counts = {
            (2, 3): 5,
            (2, 4): 5,
        }

        learned, _ = learn_decoder(counts)

        self.assertEqual(learned[2], 4)

    def test_correlated_calibration(self):
        probabilities = [
            .65, .03, .03, .22,
            .02, .02, .02, .01,
        ]

        counts = {
            (syndrome(mask), mask): int(
                round(100 * p)
            )
            for mask, p in enumerate(probabilities)
        }

        learned, evidence = learn_decoder(counts)

        self.assertEqual(
            learned,
            [0, 1, 3, 2],
        )

        self.assertEqual(
            evidence["2"]["chosen_mask"],
            3,
        )

        self.assertEqual(
            evidence["2"]["candidate_counts"],
            {"3": 22, "4": 2},
        )

    def test_incompatible_observations_reported(self):
        counts = {
            (0, 0): 2,
            (0, 1): 7,
        }

        learned, evidence = learn_decoder(counts)

        self.assertEqual(learned[0], 0)

        self.assertEqual(
            evidence["0"]["incompatible_observations"],
            7,
        )

    def test_standard_table_not_mutated(self):
        original = STANDARD.copy()

        learn_decoder({
            (1, 6): 20
        })

        self.assertEqual(STANDARD, original)


if __name__ == "__main__":
    unittest.main()
