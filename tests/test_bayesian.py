
import unittest

import numpy as np

from qtrace.diagnosis.bayesian import (
    entropy,
    information_gain,
    update_posterior,
)


class TestBayesianDiagnosis(unittest.TestCase):

    def test_entropy(self):
        self.assertAlmostEqual(
            entropy(np.array([0.5, 0.5])),
            np.log(2),
        )
        self.assertAlmostEqual(
            entropy(np.array([1.0, 0.0])),
            0.0,
        )

    def test_uninformative_measurement(self):
        prior = np.array([0.5, 0.5])
        likelihood = np.array([
            [0.8, 0.8],
            [0.2, 0.2],
        ])

        self.assertAlmostEqual(
            information_gain(prior, likelihood),
            0.0,
        )

    def test_perfectly_informative_measurement(self):
        prior = np.array([0.5, 0.5])
        likelihood = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        self.assertAlmostEqual(
            information_gain(prior, likelihood),
            np.log(2),
        )

    def test_posterior_update(self):
        prior = np.array([0.6, 0.4])
        likelihood = np.array([0.25, 0.75])

        posterior = update_posterior(prior, likelihood)

        np.testing.assert_allclose(
            posterior,
            [1 / 3, 2 / 3],
        )

    def test_impossible_observation(self):
        prior = np.array([1.0, 0.0])
        likelihood = np.array([0.0, 1.0])

        with self.assertRaises(RuntimeError):
            update_posterior(prior, likelihood)


if __name__ == "__main__":
    unittest.main()
