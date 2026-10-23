
"""Three-qubit bit-flip repetition-code utilities.

Q-TRACE's convention:
    mask bit i = X fault on data qubit i

The measured stabilizer parities are:
    syn[0] = q0 XOR q1
    syn[1] = q1 XOR q2

Syndrome integer:
    syn[0] + 2 * syn[1]

This module provides classical mapping utilities.
Quantum encoding and recovery circuits remain in the
experiment scripts during this refactor.
"""

STANDARD = [0, 1, 4, 2]


def syndrome(mask):
    """Return the two-bit syndrome for a three-qubit X mask.

    Preserves the exact convention used in Mission 24.
    """
    b0, b1, b2 = [
        (mask >> i) & 1
        for i in range(3)
    ]

    return (b0 ^ b1) + 2 * (b1 ^ b2)
