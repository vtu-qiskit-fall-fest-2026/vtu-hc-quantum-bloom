
"""Evaluation metrics for Q-TRACE repetition-code experiments.

Extracted from Mission 21.

The input counts must refer to the three-bit `out` register,
after decoding and the appropriate measurement-basis rotation.

This scorer does not account for calibration resources and does
not establish a quantum error-correction threshold.
"""


def score_repetition(metadata, counts):
    """Evaluate logical and full decoded-state success.

    The expected output for each prepared input eigenstate is:
        0, +, +i -> 0
        1, -, -i -> 1

    logical_success checks decoded q0 in the selected basis.
    decoded_full_success additionally requires q1=q2=0.

    Parameters
    ----------
    metadata : dict
        Must contain the input-state label under "state".
        Other metadata fields are preserved.
    counts : dict[str, int]
        Measurement counts for the three-bit output register.

    Returns
    -------
    dict
        Original metadata plus shots, logical_success and
        decoded_full_success.
    """
    expected = int(metadata["state"] in ["1", "-", "-i"])
    total = sum(counts.values())
    logical_ok = 0
    full_ok = 0

    for bitstring, count in counts.items():
        value = int(bitstring.replace(" ", ""), 2)

        if (value & 1) == expected:
            logical_ok += count

        # out = q2 q1 q0; q1=q2=0 verifies return to
        # the decoded code space in addition to the
        # q0 eigenbasis outcome.
        if value == expected:
            full_ok += count

    return {
        **metadata,
        "shots": total,
        "logical_success": logical_ok / total,
        "decoded_full_success": full_ok / total,
    }
