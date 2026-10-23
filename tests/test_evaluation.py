
"""Tests for Q-TRACE repetition-code evaluation metrics."""

import unittest

from qtrace.evaluation.metrics import score_repetition


class TestRepetitionMetrics(unittest.TestCase):

    def test_zero_state(self):
        result = score_repetition(
            {"state": "0"},
            {"000": 9, "001": 3},
        )

        self.assertEqual(result["shots"], 12)
        self.assertAlmostEqual(
            result["logical_success"], 0.75
        )
        self.assertAlmostEqual(
            result["decoded_full_success"], 0.75
        )

    def test_logical_vs_full_success(self):
        result = score_repetition(
            {"state": "1"},
            {"001": 6, "011": 4},
        )

        self.assertAlmostEqual(
            result["logical_success"], 1.0
        )
        self.assertAlmostEqual(
            result["decoded_full_success"], 0.6
        )

    def test_all_six_input_eigenstates(self):
        expected_bits = {
            "0": "000",
            "1": "001",
            "+": "000",
            "-": "001",
            "+i": "000",
            "-i": "001",
        }

        for state, bits in expected_bits.items():
            with self.subTest(state=state):
                result = score_repetition(
                    {"state": state},
                    {bits: 10},
                )
                self.assertEqual(
                    result["logical_success"], 1.0
                )
                self.assertEqual(
                    result["decoded_full_success"], 1.0
                )

    def test_bitstring_whitespace(self):
        result = score_repetition(
            {"state": "0"},
            {"00 0": 7, "00 1": 3},
        )

        self.assertAlmostEqual(
            result["logical_success"], 0.7
        )

    def test_metadata_preserved(self):
        metadata = {
            "state": "+",
            "fault": "X0",
            "recovery": True,
        }

        result = score_repetition(
            metadata,
            {"000": 10},
        )

        for key, value in metadata.items():
            self.assertEqual(result[key], value)

        self.assertEqual(
            metadata,
            {
                "state": "+",
                "fault": "X0",
                "recovery": True,
            },
        )

    def test_shot_weighting(self):
        result = score_repetition(
            {"state": "0"},
            {"000": 1, "001": 3},
        )

        self.assertEqual(result["shots"], 4)
        self.assertAlmostEqual(
            result["logical_success"], 0.25
        )
        self.assertAlmostEqual(
            result["decoded_full_success"], 0.25
        )


if __name__ == "__main__":
    unittest.main()
