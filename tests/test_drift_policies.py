import unittest

import numpy as np

from qtrace.qec.drift_policies import (
    policy,
    initial_tables,
)

from qtrace.qec.decoder_policies import (
    STANDARD,
    guarded_decoder,
    cost_decoder,
)


class TestDriftPolicies(unittest.TestCase):

    def test_standard_policy(self):
        counts = np.zeros(8, dtype=int)

        np.testing.assert_array_equal(
            policy(counts, "standard"),
            STANDARD,
        )

    def test_greedy_reuses_core(self):
        rng = np.random.default_rng(320101)

        for _ in range(100):
            counts = rng.integers(0, 100, size=8)

            np.testing.assert_array_equal(
                policy(counts, "greedy"),
                cost_decoder(counts, 1),
            )

    def test_guarded_reuses_core(self):
        rng = np.random.default_rng(320102)

        for _ in range(100):
            counts = rng.integers(0, 100, size=8)

            expected, _ = guarded_decoder(
                counts,
                threshold=0.95,
            )

            np.testing.assert_array_equal(
                policy(counts, "guarded95"),
                expected,
            )

    def test_cost5_reuses_core(self):
        rng = np.random.default_rng(320103)

        for _ in range(100):
            counts = rng.integers(0, 100, size=8)

            np.testing.assert_array_equal(
                policy(counts, "cost5"),
                cost_decoder(counts, 5),
            )

    def test_invalid_counts(self):
        bad_inputs = [
            np.zeros(7, dtype=int),
            np.zeros(9, dtype=int),
            np.array([-1] + [0] * 7),
        ]

        for counts in bad_inputs:
            with self.subTest(counts=counts.tolist()):
                with self.assertRaises(ValueError):
                    policy(counts, "greedy")

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            policy(
                np.zeros(8, dtype=int),
                "nonexistent",
            )

    def test_initial_tables(self):
        counts = np.zeros(8, dtype=int)
        counts[6] = 10

        tables = initial_tables(counts)

        self.assertEqual(
            set(tables),
            {
                "standard",
                "fixed_greedy",
                "fixed_guarded95",
                "fixed_cost5",
                "periodic_cost5",
                "drift_cost5",
            },
        )

        np.testing.assert_array_equal(
            tables["fixed_greedy"],
            policy(counts, "greedy"),
        )

        np.testing.assert_array_equal(
            tables["fixed_guarded95"],
            policy(counts, "guarded95"),
        )

        for name in [
            "fixed_cost5",
            "periodic_cost5",
            "drift_cost5",
        ]:
            np.testing.assert_array_equal(
                tables[name],
                policy(counts, "cost5"),
            )

    def test_initial_tables_are_independent(self):
        counts = np.zeros(8, dtype=int)
        tables = initial_tables(counts)

        original = tables["fixed_cost5"].copy()

        tables["periodic_cost5"][0] = 7

        np.testing.assert_array_equal(
            tables["fixed_cost5"],
            original,
        )


if __name__ == "__main__":
    unittest.main()
