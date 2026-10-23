import unittest

import numpy as np

from qtrace.qec.temporal import (
    SYNDROME,
    WEIGHT,
    validate,
    temporal_lookup,
    packed_history,
    decode_one,
)


class TestTemporalMAP(unittest.TestCase):

    def test_syndrome_and_weight_conventions(self):
        self.assertEqual(
            SYNDROME.tolist(),
            [0, 1, 3, 2, 2, 3, 1, 0],
        )
        self.assertEqual(
            WEIGHT.tolist(),
            [0, 1, 1, 2, 1, 2, 2, 3],
        )

    def test_history_packing(self):
        self.assertEqual(
            packed_history([0, 1, 2, 3]),
            27,
        )
        self.assertEqual(
            packed_history([3, 2, 1]),
            57,
        )

    def test_invalid_history_and_data(self):
        for history in [[-1], [4], [0, 2, 5]]:
            with self.subTest(history=history):
                with self.assertRaises(ValueError):
                    packed_history(history)

        with self.assertRaises(ValueError):
            decode_one([0], 8, .02, .08, .04)

    def test_invalid_noise_and_rounds(self):
        for p in [-0.1, 0.5, float("nan"), float("inf")]:
            with self.subTest(p=p):
                with self.assertRaises(ValueError):
                    validate(p, "test")

        for rounds in [0, 8]:
            with self.subTest(rounds=rounds):
                with self.assertRaises(ValueError):
                    temporal_lookup(
                        rounds, .02, .08, .04
                    )

    def test_lookup_shapes(self):
        for rounds in [1, 2, 3]:
            with self.subTest(rounds=rounds):
                table = temporal_lookup(
                    rounds, .05, .08, .04
                )
                self.assertEqual(
                    table.shape,
                    (8 * 4 ** rounds,),
                )
                self.assertEqual(
                    table.dtype,
                    np.dtype("uint8"),
                )

    def test_perfect_channel(self):
        for logical_bit in [0, 1]:
            actual = decode_one(
                [0, 0, 0],
                7 * logical_bit,
                0.0,
                0.0,
                0.0,
            )
            self.assertEqual(
                actual,
                logical_bit,
            )

    def test_single_x_errors(self):
        for logical_bit in [0, 1]:
            for error in [0, 1, 2, 4]:
                syndrome = int(SYNDROME[error])
                observed = error ^ (7 * logical_bit)

                actual = decode_one(
                    [syndrome] * 3,
                    observed,
                    .02,
                    0.0,
                    0.0,
                )

                self.assertEqual(
                    actual,
                    logical_bit,
                )

    def test_cache_and_read_only(self):
        first = temporal_lookup(
            2, .05, .08, .04
        )
        second = temporal_lookup(
            2, .05, .08, .04
        )

        self.assertIs(first, second)
        self.assertFalse(first.flags.writeable)

        with self.assertRaises(ValueError):
            first[0] = 1

    def test_one_round_brute_force_map(self):
        """Cross-check forward MAP against direct mask enumeration."""
        pd, ps, pf = .13, .09, .07
        table = temporal_lookup(1, pd, ps, pf)

        for observed_syndrome in range(4):
            for final_data in range(8):
                scores = []

                for logical_bit in [0, 1]:
                    probability = 0.0

                    for mask in range(8):
                        data_weight = int(WEIGHT[mask])

                        prior = (
                            pd ** data_weight
                            * (1 - pd) ** (3 - data_weight)
                        )

                        syndrome_distance = (
                            int(SYNDROME[mask])
                            ^ observed_syndrome
                        ).bit_count()

                        syndrome_likelihood = (
                            ps ** syndrome_distance
                            * (1 - ps) ** (
                                2 - syndrome_distance
                            )
                        )

                        expected_data = (
                            mask ^ (7 * logical_bit)
                        )

                        distance = (
                            expected_data ^ final_data
                        ).bit_count()

                        final_likelihood = (
                            pf ** distance
                            * (1 - pf) ** (3 - distance)
                        )

                        probability += (
                            prior
                            * syndrome_likelihood
                            * final_likelihood
                        )

                    scores.append(probability)

                expected = int(scores[1] > scores[0])

                actual = int(
                    table[
                        8 * observed_syndrome
                        + final_data
                    ]
                )

                self.assertEqual(
                    actual,
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
