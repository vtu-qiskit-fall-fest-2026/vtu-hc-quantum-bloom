
"""Calibration-learned repetition-code decoder.

Extracted from Q-TRACE Mission 24.

The learning function receives measured joint counts:
    (measured_syndrome, measured_data_word) -> count

It does not receive injected-fault labels, true channel
probabilities, or held-out test measurements.

The standard decoder is used to break exact ties.
"""

from .repetition import STANDARD, syndrome


def learn_decoder(joint_counts):
    """Learn one recovery mask per measured syndrome.

    Only measured (syndrome, data word) counts enter
    this function.

    Choose between the two X masks compatible with
    each syndrome.

    A symmetric pseudocount is used; ties keep the
    standard decoder.

    Returns
    -------
    table : list[int]
        Four recovery masks indexed by syndrome.

    evidence : dict
        Per-syndrome candidate counts, incompatible
        observation counts, and chosen recovery mask.
    """
    table = STANDARD.copy()
    evidence = {}

    for s in range(4):
        candidates = [
            m for m in range(8)
            if syndrome(m) == s
        ]

        scores = {
            m: 1 + joint_counts.get((s, m), 0)
            for m in candidates
        }

        best = max(scores.values())

        winners = [
            m for m in candidates
            if scores[m] == best
        ]

        table[s] = (
            STANDARD[s]
            if STANDARD[s] in winners
            else winners[0]
        )

        evidence[str(s)] = {
            "candidate_counts": {
                str(m): scores[m] - 1
                for m in candidates
            },
            "incompatible_observations": sum(
                n
                for (observed_s, m), n
                in joint_counts.items()
                if observed_s == s
                and m not in candidates
            ),
            "chosen_mask": table[s],
        }

    return table, evidence
