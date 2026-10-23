"""Budget-matched diagnosis-to-mitigation pilot; simulator research, not QEC.

No original mission is imported or modified. See README for assumptions.
Run: python notebooks/14_mitigation_pilot.py --backend aer --repeats 20
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import pauli_error


FAULTS = ['None', 'X0', 'X1', 'Z0', 'Z1']

PROBES = [
    ('bell', 'ZZ'),
    ('bell', 'XX'),
    ('zero', 'IZ'),
    ('zero', 'ZI'),
    ('plus', 'IX'),
    ('plus', 'XI'),
]

BATCH = 16

RATES = np.array([.02, .05, .10, .15, .20, .30, .40])

STATE_FAULT = np.array([
    0
] + [
    f
    for f in range(1, 5)
    for _ in RATES
])

STATE_RATE = np.r_[0., np.tile(RATES, 4)]

PRIOR = np.r_[
    .2,
    np.full(4 * len(RATES), .2 / len(RATES)),
]

SCHEDULES = {
    'bell2': [0, 1],
    'fixed4': [0, 1, 2, 4],
    'round6': list(range(6)),
}

STRATEGIES = [
    'raw',
    'fixed_always',
    'fixed_selective',
    'adaptive_always',
    'adaptive_selective',
    'oracle',
]


def pauli(f):
    label = ['I', 'I']

    if f:
        label[1 - int(FAULTS[f][1])] = FAULTS[f][0]

    return Pauli(''.join(label))


def prepare(mode):
    qc = QuantumCircuit(2)

    if mode == 'bell':
        qc.h(0)
        qc.cx(0, 1)

    elif mode == 'plus':
        qc.h([0, 1])

    return qc


SENS = np.array([
    [
        int(pauli(f).anticommutes(Pauli(o)))
        for f in range(5)
    ]
    for _, o in PROBES
])

LIKELIHOOD = binom.pmf(
    np.arange(BATCH + 1)[None, :, None],
    BATCH,
    (SENS[:, STATE_FAULT] * STATE_RATE)[..., None, :],
)


def attenuation(observable):
    sensitive = np.array([
        int(pauli(f).anticommutes(Pauli(observable)))
        for f in STATE_FAULT
    ])

    return 1 - 2 * sensitive * STATE_RATE


def terminal(
    posterior,
    lam,
    n,
    second_moment,
    selective=True,
):
    """Prior-predictive MSE for raw or plug-in inverse, before target shots.

    theta is independent of noise.
    E[theta^2] = second_moment, fixed from validation.

    For gain g, exact conditional MSE is:
        g^2 * (1 - lambda^2 * theta^2) / n
        + (g * lambda - 1)^2 * theta^2

    The method does not claim optimality over all estimators.
    """
    post = np.atleast_2d(posterior)

    mean = post @ lam
    square = post @ (lam * lam)

    # Grid bounds ensure mean >= .2.
    gain = 1 / mean

    def risk(g):
        return (
            g * g * (1 - second_moment * square) / n
            + second_moment * (
                g * g * square
                - 2 * g * mean
                + 1
            )
        )

    corrected = risk(gain)
    raw = risk(np.ones_like(gain))

    if selective:
        use = corrected < raw
    else:
        use = np.ones(len(post), dtype=bool)

    return (
        np.where(use, corrected, raw),
        np.where(use, gain, 1.),
    )


def next_action(
    post,
    lam,
    remaining,
    second_moment,
):
    """One-batch lookahead.

    Assume that, after the next diagnostic batch, all remaining
    shots measure the target.

    Recomputed after each diagnostic batch.
    May stop early on problems requiring several informative batches.
    """
    stop = terminal(
        post,
        lam,
        remaining,
        second_moment,
    )[0][0]

    if remaining < 2 * BATCH:
        return None

    scores = []

    for lk in LIKELIHOOD:
        joint = lk * post
        predictive = joint.sum(axis=1)

        valid = predictive > 1e-15

        after = (
            joint[valid]
            / predictive[valid, None]
        )

        risks = terminal(
            after,
            lam,
            remaining - BATCH,
            second_moment,
        )[0]

        scores.append(
            predictive[valid] @ risks
        )

    best = int(np.argmin(scores))

    if scores[best] < stop - 1e-12:
        return best

    return None


class Backend:
    def __init__(self, name):
        self.name = name

        self.sim = AerSimulator(
            method='density_matrix',
            max_parallel_threads=1,
        )

    def minus(
        self,
        qc,
        obs,
        fault,
        rate,
        shots,
        seed,
    ):
        if self.name == 'analytic':
            ideal = float(
                Statevector.from_instruction(qc)
                .expectation_value(Pauli(obs))
                .real
            )

            factor = (
                1
                - 2 * rate
                * int(
                    pauli(fault).anticommutes(
                        Pauli(obs)
                    )
                )
            )

            probability = np.clip(
                (1 - factor * ideal) / 2,
                0,
                1,
            )

            return int(
                np.random.default_rng(seed).binomial(
                    shots,
                    probability,
                )
            )

        circuit = qc.copy()

        if fault:
            channel = pauli_error([
                ('I', 1 - rate),
                (FAULTS[fault][0], rate),
            ])

            circuit.append(
                channel.to_instruction(),
                [int(FAULTS[fault][1])],
            )

        for q, axis in enumerate(obs[::-1]):
            if axis == 'X':
                circuit.h(q)

        circuit.measure_all()

        counts = self.sim.run(
            circuit,
            shots=shots,
            seed_simulator=int(seed),
        ).result().get_counts()

        active = [
            q
            for q, axis in enumerate(obs[::-1])
            if axis != 'I'
        ]

        return sum(
            v
            for bits, v in counts.items()
            if sum(
                int(bits.replace(' ', '')[-1 - q])
                for q in active
            ) % 2
        )


def target(angles):
    qc = QuantumCircuit(2)

    qc.ry(float(angles[0]), 0)
    qc.ry(float(angles[1]), 1)

    qc.cx(0, 1)

    qc.ry(float(angles[2]), 0)
    qc.ry(float(angles[3]), 1)

    return qc


def trial(
    backend,
    qc,
    obs,
    fault,
    rate,
    budget,
    seed,
    strategy,
    config,
    m2,
):
    """Hidden fault/rate accessed only by measurement backend and oracle."""
    post = PRIOR.copy()
    used = 0
    step = 0

    lam = attenuation(obs)

    fixed_steps = min(
        int(budget * config['fraction']) // BATCH,
        budget // BATCH - 1,
    )

    while strategy.startswith(('fixed', 'adaptive')):
        if strategy.startswith('fixed'):
            if step >= fixed_steps:
                break

            sequence = SCHEDULES[
                config['schedule']
            ]

            probe = sequence[
                step % len(sequence)
            ]

        else:
            probe = next_action(
                post,
                lam,
                budget - used,
                m2,
            )

            if probe is None:
                break

        mode, measured = PROBES[probe]

        k = backend.minus(
            prepare(mode),
            measured,
            fault,
            rate,
            BATCH,
            seed + step * 101 + probe,
        )

        post *= LIKELIHOOD[probe, k]

        if post.sum() <= 0:
            raise RuntimeError(
                'Observation impossible under inference model'
            )

        post /= post.sum()

        used += BATCH
        step += 1

    n = budget - used

    if strategy == 'raw':
        gain = 1.

    elif strategy == 'oracle':
        gain = 1 / (
            1
            - 2 * rate
            * int(
                pauli(fault).anticommutes(
                    Pauli(obs)
                )
            )
        )

    else:
        gain = terminal(
            post,
            lam,
            n,
            m2,
            strategy.endswith('selective'),
        )[1][0]

    k = backend.minus(
        qc,
        obs,
        fault,
        rate,
        n,
        seed + 1000003,
    )

    raw = 1 - 2 * k / n

    assert used + n == budget and n > 0

    return {
        'estimate': gain * raw,
        'raw_same_shots': raw,
        'diagnostic_shots': used,
        'application_shots': n,
        'mitigated': int(gain > 1 + 1e-10),
        'gain': gain,
    }


def make_cases(seed, count, rates):
    rng = np.random.default_rng(seed)
    cases = []

    for i in range(count):
        qc = target(
            rng.uniform(-np.pi, np.pi, 4)
        )

        obs = ['ZZ', 'XX'][i % 2]

        ideal = float(
            Statevector.from_instruction(qc)
            .expectation_value(Pauli(obs))
            .real
        )

        # Identical hidden channel shared by diagnostics
        # and application within each case.
        for f in range(5):
            p = (
                0.
                if f == 0
                else rates[i % len(rates)]
            )

            cases.append((
                f't{i}_f{f}',
                qc,
                obs,
                f,
                p,
                ideal,
            ))

    return cases


def checks():
    assert np.allclose(
        LIKELIHOOD.sum(axis=1),
        1,
    )

    for mode, obs in PROBES:
        assert np.isclose(
            Statevector.from_instruction(
                prepare(mode)
            ).expectation_value(
                Pauli(obs)
            ),
            1,
        )

    # Validate attenuation on arbitrary non-stabilizer states,
    # including commuting errors.
    for _, qc, obs, _, _, ideal in make_cases(
        9,
        4,
        [.1],
    ):
        state = Statevector.from_instruction(qc)

        for f in range(5):
            faulted = state.evolve(pauli(f))

            noisy = (
                .83 * ideal
                + .17
                * faulted.expectation_value(
                    Pauli(obs)
                ).real
            )

            expected = ideal * (
                1
                - .34
                * int(
                    pauli(f).anticommutes(
                        Pauli(obs)
                    )
                )
            )

            assert np.isclose(noisy, expected)

    # Check risk formula by direct summation of
    # the full sampling distribution.
    lam = attenuation('ZZ')
    n = 32
    theta = .6

    risk, gain = terminal(
        PRIOR,
        lam,
        n,
        theta ** 2,
        False,
    )

    k = np.arange(n + 1)

    mse = sum(
        PRIOR[j]
        * np.sum(
            binom.pmf(
                k,
                n,
                (1 - lam[j] * theta) / 2,
            )
            * (
                gain[0] * (1 - 2 * k / n)
                - theta
            ) ** 2
        )
        for j in range(len(PRIOR))
    )

    assert np.isclose(mse, risk[0])

    assert next_action(
        PRIOR,
        lam,
        BATCH,
        .3,
    ) is None

    # Hidden truth must not affect a policy
    # given identical observations.
    class FixedObservations:
        def minus(
            self,
            qc,
            obs,
            fault,
            rate,
            shots,
            seed,
        ):
            return 0

    qc = target([.3, .5, .7, .9])

    cfg = {
        'schedule': 'round6',
        'fraction': .25,
    }

    for policy in STRATEGIES:
        if policy != 'oracle':
            a = trial(
                FixedObservations(),
                qc,
                'ZZ',
                0,
                0,
                128,
                1,
                policy,
                cfg,
                .3,
            )

            b = trial(
                FixedObservations(),
                qc,
                'ZZ',
                4,
                .4,
                128,
                1,
                policy,
                cfg,
                .3,
            )

            assert a == b, policy


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        '--backend',
        choices=['aer', 'analytic'],
        default='aer',
    )

    parser.add_argument(
        '--repeats',
        type=int,
        default=20,
    )

    parser.add_argument(
        '--targets',
        type=int,
        default=12,
    )

    parser.add_argument(
        '--budgets',
        type=int,
        nargs='+',
        default=[128, 512],
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/mitigation_pilot'),
    )

    args = parser.parse_args()

    if (
        args.repeats < 1
        or args.targets < 2
        or min(args.budgets) < 32
    ):
        parser.error(
            'Require repeats>=1, targets>=2, budgets>=32'
        )

    checks()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    backend = Backend(args.backend)

    validation = make_cases(
        1711,
        8,
        [.05, .15, .30],
    )

    test = make_cases(
        9823,
        args.targets,
        [.04, .12, .24],
    )

    # Validation-derived prior second moment,
    # NOT a test-circuit ideal value.
    m2 = float(
        np.mean([
            row[-1] ** 2
            for row in validation
        ])
    )

    tuning = []
    configs = {}

    # Tune fixed baselines on validation only.
    for budget in args.budgets:
        for strategy in [
            'fixed_always',
            'fixed_selective',
        ]:
            best = (float('inf'), None)

            for schedule in SCHEDULES:
                for fraction in [.125, .25, .5]:
                    cfg = {
                        'schedule': schedule,
                        'fraction': fraction,
                    }

                    errors = []

                    for idx, (
                        _,
                        qc,
                        obs,
                        f,
                        p,
                        ideal,
                    ) in enumerate(validation):
                        for rep in range(3):
                            r = trial(
                                backend,
                                qc,
                                obs,
                                f,
                                p,
                                budget,
                                200000 + idx * 1000 + rep,
                                strategy,
                                cfg,
                                m2,
                            )

                            errors.append(
                                (r['estimate'] - ideal) ** 2
                            )

                    score = float(
                        np.mean(errors)
                    )

                    tuning.append(dict(
                        budget=budget,
                        strategy=strategy,
                        **cfg,
                        mse=score,
                    ))

                    if score < best[0]:
                        best = (score, cfg)

            configs[
                f'{budget}_{strategy}'
            ] = best[1]

        print(
            'Validation locked for budget',
            budget,
            flush=True,
        )

    # Evaluate locked methods on test cases.
    rows = []

    for budget in args.budgets:
        for idx, (
            case,
            qc,
            obs,
            f,
            p,
            ideal,
        ) in enumerate(test):
            for rep in range(args.repeats):
                seed = (
                    700000
                    + idx * 1000
                    + rep
                )

                for strategy in STRATEGIES:
                    cfg = configs.get(
                        f'{budget}_{strategy}',
                        {
                            'schedule': 'round6',
                            'fraction': .25,
                        },
                    )

                    r = trial(
                        backend,
                        qc,
                        obs,
                        f,
                        p,
                        budget,
                        seed,
                        strategy,
                        cfg,
                        m2,
                    )

                    rows.append(dict(
                        budget=budget,
                        case=case,
                        rep=rep,
                        strategy=strategy,
                        fault=FAULTS[f],
                        rate=p,
                        observable=obs,
                        ideal=ideal,
                        **r,
                    ))

        print(
            'Test complete for budget',
            budget,
            flush=True,
        )

    df = pd.DataFrame(rows)

    df['error'] = df.estimate - df.ideal
    df['squared_error'] = df.error ** 2

    df['raw_squared_error_same_shots'] = (
        df.raw_same_shots - df.ideal
    ) ** 2

    df.to_csv(
        args.output / 'trials.csv',
        index=False,
    )

    summary = df.groupby(
        ['budget', 'strategy']
    ).agg(
        mse=('squared_error', 'mean'),
        mean_error=('error', 'mean'),
        diagnostic_shots=('diagnostic_shots', 'mean'),
        application_shots=('application_shots', 'mean'),
        mitigation_rate=('mitigated', 'mean'),
    )

    summary.to_csv(
        args.output / 'summary.csv'
    )

    # Bias/variance decomposed WITHIN each target/noise case.
    # Population variance convention: ddof=0.
    by_case = df.groupby(
        ['budget', 'strategy', 'case']
    ).agg(
        mse=('squared_error', 'mean'),
        bias=('error', 'mean'),
        variance=(
            'estimate',
            lambda x: np.var(x, ddof=0),
        ),
    )

    assert np.allclose(
        by_case.mse,
        by_case.bias ** 2 + by_case.variance,
    )

    by_case.to_csv(
        args.output / 'by_case.csv'
    )

    comparisons = []
    rng = np.random.default_rng(20261022)

    for budget in args.budgets:
        wide = df[
            df.budget == budget
        ].pivot(
            index=['case', 'rep'],
            columns='strategy',
            values='squared_error',
        )

        for baseline in [
            'raw',
            'fixed_always',
            'fixed_selective',
            'adaptive_always',
            'oracle',
        ]:
            delta = (
                wide.adaptive_selective
                - wide[baseline]
            )

            # Stratified by fixed target/noise case.
            # CI covers shot randomness, not generalization
            # to all circuits or noise models.
            matrix = np.stack([
                g.to_numpy()
                for _, g in delta.groupby(level='case')
            ])

            boot = np.zeros(2000)

            for group in matrix:
                boot += (
                    rng.choice(
                        group,
                        size=(2000, len(group)),
                        replace=True,
                    ).mean(axis=1)
                    / len(matrix)
                )

            lo, hi = np.quantile(
                boot,
                [.025, .975],
            )

            comparisons.append(dict(
                budget=budget,
                baseline=baseline,
                delta_mse=delta.mean(),
                ci_low=lo,
                ci_high=hi,
            ))

    pd.DataFrame(comparisons).to_csv(
        args.output / 'comparisons.csv',
        index=False,
    )

    pd.DataFrame(tuning).to_csv(
        args.output / 'validation.csv',
        index=False,
    )

    configuration = dict(
        backend=args.backend,
        repeats=args.repeats,
        targets=args.targets,
        budgets=args.budgets,
        validation_second_moment=m2,
        fixed_configs=configs,
        seed_validation=1711,
        seed_test=9823,
    )

    (
        args.output / 'config.json'
    ).write_text(
        json.dumps(
            configuration,
            indent=2,
        )
    )

    print(
        summary.round(6).to_string()
    )

    print(
        pd.DataFrame(comparisons)
        .round(6)
        .to_string(index=False)
    )


if __name__ == '__main__':
    main()