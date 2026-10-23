
import unittest

import numpy as np

from qtrace.policies.stopping import BATCH, M2, Plan
from qtrace.policies.mitigation import physical_risk


class TestPolicyCore(unittest.TestCase):

    def test_constants(self):
        self.assertEqual(BATCH, 16)
        self.assertAlmostEqual(
            M2, 0.26497528958302274
        )

    def test_physical_risk_without_attenuation(self):
        risk = physical_risk(
            gain=1.0,
            attenuation=1.0,
            application_shots=128,
        )
        self.assertAlmostEqual(
            risk, (1.0 - M2) / 128
        )

    def test_physical_risk_with_zero_gain(self):
        risk = physical_risk(
            gain=0.0,
            attenuation=0.5,
            application_shots=128,
        )
        self.assertAlmostEqual(risk, M2)

    def test_vectorized_physical_risk(self):
        attenuation = np.array([1.0, 0.8, 0.6])
        risks = physical_risk(
            1.1, attenuation, 128
        )
        self.assertEqual(risks.shape, (3,))
        self.assertTrue(np.isfinite(risks).all())

    def test_plan_shapes(self):
        for prior in ["old", "effective"]:
            with self.subTest(prior=prior):
                plan = Plan(prior, 128)
                self.assertEqual(plan.last, 7)
                self.assertEqual(len(plan.tables), 8)

                state = plan.state(0, 0)
                self.assertTrue(
                    np.isfinite(state["gain"])
                )
                self.assertGreaterEqual(
                    state["gain"], 1.0
                )

    def test_initial_risk_matches_physical_formula(self):
        for prior in ["old", "effective"]:
            with self.subTest(prior=prior):
                plan = Plan(prior, 128)
                state = plan.state(0, 0)

                expected_risk = np.sum(
                    plan.prior * physical_risk(
                        state["gain"],
                        plan.lam,
                        128,
                    )
                )

                np.testing.assert_allclose(
                    state["risk"],
                    expected_risk,
                    rtol=1e-10,
                    atol=1e-12,
                )


if __name__ == "__main__":
    unittest.main()
