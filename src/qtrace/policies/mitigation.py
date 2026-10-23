
"""Physical risk calculation for Q-TRACE mitigation policies.

Extracted from Mission 19's independent DP verification.
"""

from .stopping import M2


def physical_risk(gain, attenuation, application_shots):
    """
    Average over independent targets with E[ideal**2] = M2
    and over application measurement outcomes.
    """
    return (
        M2 * (gain * attenuation - 1.0) ** 2
        + gain ** 2
        * (1.0 - M2 * attenuation ** 2)
        / application_shots
    )
